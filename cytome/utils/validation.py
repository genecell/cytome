"""Validation and repair helpers for Cytome datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class ValidationReport:
    """Validation result summary."""

    passed: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


_REQUIRED_TABLES = {
    "_manifest",
    "_provenance",
    "_schema_migrations",
    "_metadata",
    "cells",
    "genes",
    "GA_genes",
    "peaks",
    "samples",
    "proteins",
    "matrix_chunks",
    "matrix_meta",
    "dense_chunks",
    "embedding_meta",
}


def validate(ds) -> ValidationReport:
    """Run structural and consistency checks on a dataset.

    Parameters
    ----------
    ds
        Open ``CytomeDataset`` instance.

    Returns
    -------
    ValidationReport
        Validation outcomes.
    """
    conn = ds._conn
    passed: list[str] = []
    failed: list[str] = []

    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity == "ok":
        passed.append("sqlite_integrity")
    else:
        failed.append(f"sqlite_integrity: {integrity}")

    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    missing = sorted(_REQUIRED_TABLES - tables)
    if missing:
        failed.append(f"missing_tables: {missing}")
    else:
        passed.append("required_tables")

    entity_counts = {
        "cells": int(conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0]),
        "genes": int(conn.execute("SELECT COUNT(*) FROM genes").fetchone()[0]),
        "GA_genes": int(conn.execute("SELECT COUNT(*) FROM GA_genes").fetchone()[0]),
        "peaks": int(conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0]),
        "proteins": int(conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]),
    }

    for row in conn.execute(
        "SELECT matrix_name, n_rows, n_cols, row_entity, col_entity, n_chunks FROM matrix_meta"
    ).fetchall():
        name, n_rows, n_cols, row_entity, col_entity, n_chunks = row
        if row_entity in entity_counts and int(n_rows) != entity_counts[row_entity]:
            failed.append(f"matrix_rows:{name}")
        if col_entity in entity_counts and entity_counts[col_entity] > 0:
            if int(n_cols) != entity_counts[col_entity]:
                failed.append(f"matrix_cols:{name}")
        count = conn.execute(
            "SELECT COUNT(*) FROM matrix_chunks WHERE matrix_name = ?", (name,)
        ).fetchone()[0]
        if int(count) != int(n_chunks):
            failed.append(f"matrix_chunk_count:{name}")

    if not any(item.startswith("matrix_") for item in failed):
        passed.append("matrix_consistency")

    for row in conn.execute(
        "SELECT array_name, n_rows, n_chunks, entity FROM embedding_meta"
    ).fetchall():
        name, n_rows, n_chunks, entity = row
        if entity in entity_counts and entity_counts[entity] > 0 and int(n_rows) != entity_counts[entity]:
            failed.append(f"embedding_rows:{name}")
        count = conn.execute(
            "SELECT COUNT(*) FROM dense_chunks WHERE array_name = ?", (name,)
        ).fetchone()[0]
        if int(count) != int(n_chunks):
            failed.append(f"embedding_chunk_count:{name}")

    if not any(item.startswith("embedding_") for item in failed):
        passed.append("embedding_consistency")

    return ValidationReport(passed=(len(failed) == 0), checks_passed=passed, checks_failed=failed)


def repair(ds) -> None:
    """Repair recoverable integrity issues.

    Fixes:
    - Orphan matrix/embedding chunks
    - Matrix row count mismatches (matrix_meta.n_rows != cells table count)
    - Embedding row count mismatches

    Parameters
    ----------
    ds
        Open ``CytomeDataset`` instance.
    """
    conn = ds._conn
    with conn:
        conn.execute("REINDEX")
        conn.execute(
            """
            DELETE FROM matrix_chunks
            WHERE matrix_name NOT IN (SELECT matrix_name FROM matrix_meta)
            """
        )
        conn.execute(
            """
            DELETE FROM dense_chunks
            WHERE array_name NOT IN (SELECT array_name FROM embedding_meta)
            """
        )

        # Fix matrix row and column count mismatches
        entity_counts = {
            "cells": int(conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0]),
            "genes": int(conn.execute("SELECT COUNT(*) FROM genes").fetchone()[0]),
            "GA_genes": int(conn.execute("SELECT COUNT(*) FROM GA_genes").fetchone()[0]),
            "peaks": int(conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0]),
        }

        for row in conn.execute(
            "SELECT matrix_name, n_rows, n_cols, row_entity, col_entity FROM matrix_meta"
        ).fetchall():
            name, n_rows, n_cols, row_entity, col_entity = row
            n_rows, n_cols = int(n_rows), int(n_cols)

            # Fix row mismatches
            if row_entity in entity_counts:
                expected = entity_counts[row_entity]
                if n_rows != expected:
                    _repair_matrix_rows(conn, name, n_rows, expected)

            # Fix column mismatches (metadata-only update)
            if col_entity in entity_counts and entity_counts[col_entity] > 0:
                expected_cols = entity_counts[col_entity]
                if n_cols != expected_cols:
                    conn.execute(
                        "UPDATE matrix_meta SET n_cols = ? WHERE matrix_name = ?",
                        (expected_cols, name),
                    )

        # Fix embedding row count mismatches
        for row in conn.execute(
            "SELECT array_name, n_rows, entity FROM embedding_meta"
        ).fetchall():
            name, n_rows, entity = row
            n_rows = int(n_rows)
            if entity not in entity_counts:
                continue
            expected = entity_counts[entity]
            if n_rows == expected:
                continue
            _repair_embedding_rows(conn, name, n_rows, expected)


def _repair_matrix_rows(conn, matrix_name: str, current_n_rows: int, target_n_rows: int) -> None:
    """Rebuild a matrix to match the target row count.

    If the matrix has more rows than the entity table, truncates excess rows
    from the last chunk(s). If fewer, updates metadata only (missing rows
    are treated as all-zero).
    """
    from cytome.io.chunked_io import write_sparse_chunked
    from cytome.io.compression import decompress_blob

    if current_n_rows <= target_n_rows:
        # Matrix has fewer rows — just update metadata
        conn.execute(
            "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = ?",
            (target_n_rows, matrix_name),
        )
        return

    # Matrix has more rows than cells — rebuild chunks truncated to target_n_rows
    meta = conn.execute(
        "SELECT n_cols, dtype, chunk_size, row_entity, col_entity FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    if meta is None:
        return
    n_cols, dtype_str, chunk_size, row_entity, col_entity = meta
    n_cols = int(n_cols)
    chunk_size = int(chunk_size)
    dtype = np.dtype(dtype_str)

    # Read all chunks, stitch into full CSR, truncate, rewrite
    chunks = conn.execute(
        """SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression
           FROM matrix_chunks WHERE matrix_name = ? ORDER BY chunk_idx""",
        (matrix_name,),
    ).fetchall()

    parts = []
    for row_start, row_end, data_blob, indices_blob, indptr_blob, compression in chunks:
        row_start, row_end = int(row_start), int(row_end)
        chunk_data = np.frombuffer(decompress_blob(data_blob, compression), dtype=dtype)
        chunk_indices = np.frombuffer(decompress_blob(indices_blob, compression), dtype=np.int32)
        chunk_indptr = np.frombuffer(decompress_blob(indptr_blob, compression), dtype=np.int32)
        chunk_csr = sp.csr_matrix(
            (chunk_data, chunk_indices, chunk_indptr),
            shape=(row_end - row_start, n_cols),
        )
        # Only keep rows up to target
        if row_start >= target_n_rows:
            continue
        effective_end = min(row_end, target_n_rows)
        local_end = effective_end - row_start
        parts.append(chunk_csr[:local_end])

    if parts:
        full = sp.vstack(parts, format="csr")
    else:
        full = sp.csr_matrix((target_n_rows, n_cols), dtype=dtype)

    # Rewrite via standard chunked writer
    write_sparse_chunked(
        conn, matrix_name, full, chunk_size, "zstd",
        row_entity=row_entity, col_entity=col_entity,
    )


def _repair_embedding_rows(conn, array_name: str, current_n_rows: int, target_n_rows: int) -> None:
    """Fix embedding row count to match entity table."""
    from cytome.io.compression import decompress_blob, compress_blob

    if current_n_rows <= target_n_rows:
        conn.execute(
            "UPDATE embedding_meta SET n_rows = ? WHERE array_name = ?",
            (target_n_rows, array_name),
        )
        return

    # Read, truncate, rewrite
    meta = conn.execute(
        "SELECT n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    if meta is None:
        return
    n_cols, dtype_str = int(meta[0]), np.dtype(meta[1])

    chunks = conn.execute(
        """SELECT chunk_idx, row_start, row_end, data_blob, compression
           FROM dense_chunks WHERE array_name = ? ORDER BY chunk_idx""",
        (array_name,),
    ).fetchall()

    parts = []
    for chunk_idx, row_start, row_end, data_blob, compression in chunks:
        row_start, row_end = int(row_start), int(row_end)
        if row_start >= target_n_rows:
            conn.execute(
                "DELETE FROM dense_chunks WHERE array_name = ? AND chunk_idx = ?",
                (array_name, chunk_idx),
            )
            continue
        effective_end = min(row_end, target_n_rows)
        chunk = np.frombuffer(decompress_blob(data_blob, compression), dtype=dtype_str).reshape(
            (row_end - row_start, n_cols)
        )
        local_end = effective_end - row_start
        truncated = chunk[:local_end]
        # Update chunk in-place
        conn.execute(
            """UPDATE dense_chunks SET row_end = ?, data_blob = ?
               WHERE array_name = ? AND chunk_idx = ?""",
            (effective_end, compress_blob(truncated.tobytes(), method=compression),
             array_name, chunk_idx),
        )

    # Update metadata
    n_chunks = conn.execute(
        "SELECT COUNT(*) FROM dense_chunks WHERE array_name = ?", (array_name,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE embedding_meta SET n_rows = ?, n_chunks = ? WHERE array_name = ?",
        (target_n_rows, int(n_chunks), array_name),
    )
