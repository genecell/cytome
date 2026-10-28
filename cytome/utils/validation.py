"""Validation and repair helpers for Cytome datasets.

``validate`` checks what the readers rely on: every matrix has the rows of its
row table and the columns of its own feature table, its chunks are where its
metadata says, row chunks cover its rows once, column chunks cover its
columns; embeddings, graphs and fragments have the chunks their metadata
counts; a cell is named once by its sample and barcode. ``repair`` fixes what
can be fixed without guessing, and never makes a matrix unreadable.

A matrix's own feature table: the raw matrix (``adata.raw``) follows
``_raw_var`` and a sparse ``obsm`` stored as a matrix follows none. Converters
before 0.3.9 recorded both against ``genes``; they are recognised by the
``_anndata_raw`` and ``_anndata_obsm_as_matrix`` records, so such a store
validates, and ``repair`` no longer cuts their columns to the gene count (which
left a raw matrix of 10 columns as 4, unreadable).
"""

from __future__ import annotations

import json
import sqlite3
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

#: entity tables whose row count a matrix or embedding is checked against
_ENTITY_TABLES = ("cells", "genes", "GA_genes", "peaks", "tiles", "proteins", "_raw_var")


def _q(sql_conn, sql, args=()):
    try:
        return sql_conn.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []


def _entity_counts(conn) -> dict[str, int]:
    out = {}
    for t in _ENTITY_TABLES:
        rows = _q(conn, f'SELECT COUNT(*) FROM "{t}"')
        if rows:
            out[t] = int(rows[0][0])
    return out


def _metadata_json(conn, key):
    rows = _q(conn, "SELECT value FROM _metadata WHERE key = ?", (key,))
    if not rows:
        return None
    try:
        return json.loads(rows[0][0])
    except (TypeError, ValueError):
        return None


def matrix_column_entities(conn) -> dict[str, str | None]:
    """``{matrix: the table its columns follow, or None}``: the recorded
    ``col_entity``, except for the raw matrix (``_raw_var``) and sparse obsm
    stored as matrices (no table), whatever an older converter recorded."""
    out = {name: ce for name, ce in _q(conn, "SELECT matrix_name, col_entity FROM matrix_meta")}
    raw = _metadata_json(conn, "_anndata_raw")
    if isinstance(raw, dict) and raw.get("matrix_name") in out:
        out[raw["matrix_name"]] = "_raw_var"
    obsm = _metadata_json(conn, "_anndata_obsm_as_matrix")
    if isinstance(obsm, dict):
        for name in obsm:
            if name in out:
                out[name] = None
    for name, ce in list(out.items()):
        if ce == "_obsm":
            out[name] = None
    return out


def validate(ds, integrity: bool = True) -> ValidationReport:
    """Run structural and consistency checks on a dataset.

    Parameters
    ----------
    ds
        Open ``CytomeDataset`` instance.
    integrity
        Run SQLite's own ``PRAGMA integrity_check`` first (default), and check
        that every matrix chunk is stored in its matrix's dtype. Both read the
        whole file: two hours for an 80 GB store that is not in the page cache.
        ``False`` keeps cytome's own checks, which read no blob.

    Returns
    -------
    ValidationReport
        Validation outcomes.
    """
    conn = ds._conn
    passed: list[str] = []
    failed: list[str] = []

    if integrity:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if result == "ok":
            passed.append("sqlite_integrity")
        else:
            failed.append(f"sqlite_integrity: {result}")

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

    entity_counts = _entity_counts(conn)
    col_entities = matrix_column_entities(conn)

    for row in conn.execute(
        "SELECT matrix_name, n_rows, n_cols, row_entity, n_chunks, dtype, "
        "COALESCE(has_csc, 0), csc_n_chunks FROM matrix_meta"
    ).fetchall():
        name, n_rows, n_cols, row_entity, n_chunks, dtype, has_csc, csc_n_chunks = row
        col_entity = col_entities.get(name)
        if row_entity in entity_counts and int(n_rows) != entity_counts[row_entity]:
            failed.append(f"matrix_rows:{name}")
        if col_entity in entity_counts and entity_counts[col_entity] > 0:
            if int(n_cols) != entity_counts[col_entity]:
                failed.append(f"matrix_cols:{name}")
        spans = conn.execute(
            "SELECT row_start, row_end FROM matrix_chunks WHERE matrix_name = ? "
            "ORDER BY row_start, chunk_idx", (name,)).fetchall()
        if len(spans) != int(n_chunks or 0):
            failed.append(f"matrix_chunk_count:{name}")
        if spans:
            # row chunks cover the rows once: no overlap, no gap, nothing past the end
            at, ok = 0, True
            for rs, re_ in spans:
                if int(rs) != at or int(re_) < int(rs):
                    ok = False
                    break
                at = int(re_)
            if not ok or at != int(n_rows):
                failed.append(f"matrix_row_coverage:{name}")
        if has_csc:
            cspans = conn.execute(
                "SELECT col_start, col_end FROM matrix_csc_chunks WHERE matrix_name = ? "
                "ORDER BY col_start, chunk_idx", (name,)).fetchall()
            if csc_n_chunks is not None and len(cspans) != int(csc_n_chunks):
                failed.append(f"matrix_csc_chunk_count:{name}")
            at = 0
            for cs, ce in cspans:
                if int(cs) != at:
                    break
                at = int(ce)
            if at != int(n_cols):
                failed.append(f"matrix_col_coverage:{name}")
        if integrity:
            for table in ("matrix_chunks", "matrix_csc_chunks"):
                other = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE matrix_name = ? AND dtype != ?",
                    (name, str(dtype))).fetchone()[0]
                if other:
                    failed.append(f"matrix_chunk_dtype:{name}")
                    break

    for table in ("matrix_chunks", "matrix_csc_chunks"):
        orphans = _q(conn, f"SELECT COUNT(*) FROM {table} WHERE matrix_name NOT IN "
                           f"(SELECT matrix_name FROM matrix_meta)")
        if orphans and orphans[0][0]:
            failed.append("matrix_orphan_chunks")
            break
    if not any(item.startswith("matrix_") for item in failed):
        passed.append("matrix_consistency")

    for row in conn.execute(
        "SELECT array_name, n_rows, n_chunks, entity, dtype FROM embedding_meta"
    ).fetchall():
        name, n_rows, n_chunks, entity, dtype = row
        if entity in entity_counts and entity_counts[entity] > 0 and int(n_rows) != entity_counts[entity]:
            failed.append(f"embedding_rows:{name}")
        count = conn.execute(
            "SELECT COUNT(*) FROM dense_chunks WHERE array_name = ?", (name,)
        ).fetchone()[0]
        if int(count) != int(n_chunks):
            failed.append(f"embedding_chunk_count:{name}")
        try:
            kind = np.dtype(str(dtype)).kind
        except TypeError:
            kind = "O"
        if kind not in "biufc":
            failed.append(f"embedding_dtype:{name}")

    if not any(item.startswith("embedding_") for item in failed):
        passed.append("embedding_consistency")

    # graphs: the chunks graph_meta counts
    for g, axis, n in _q(conn, "SELECT graph_name, axis, n_chunks FROM graph_meta"):
        have = conn.execute("SELECT COUNT(*) FROM graph_chunks WHERE graph_name = ? AND axis = ?",
                            (g, axis)).fetchone()[0]
        if int(have) != int(n):
            failed.append(f"graph_chunk_count:{g}")
    if not any(item.startswith("graph_") for item in failed):
        passed.append("graph_consistency")

    # fragments: fragment_meta counts what the chunks hold, per chromosome
    if _q(conn, "SELECT 1 FROM fragment_chunks LIMIT 1"):
        held = dict(_q(conn, "SELECT chrom, SUM(n_fragments) FROM fragment_chunks GROUP BY chrom"))
        for chrom, n in _q(conn, "SELECT chrom, n_fragments FROM fragment_meta"):
            if int(held.get(chrom, 0) or 0) != int(n or 0):
                failed.append(f"fragment_count:{chrom}")
        recorded = {c for c, in _q(conn, "SELECT chrom FROM fragment_meta")}
        for chrom in held:
            if chrom not in recorded:
                failed.append(f"fragment_count:{chrom}")
    if not any(item.startswith("fragment_") for item in failed):
        passed.append("fragment_consistency")

    # every metadata value reads back (with integrity: it reads every value).
    # Record arrays were written so that numpy could not read their dtype, and
    # validate() passed on the store.
    if integrity:
        from cytome.core.metadata import MetadataStore
        meta = MetadataStore(conn)
        for (key,) in _q(conn, "SELECT key FROM _metadata"):
            try:
                meta[key]
            except Exception as exc:                          # noqa: BLE001
                failed.append(f"metadata_unreadable:{key} ({type(exc).__name__})")
        if not any(item.startswith("metadata_") for item in failed):
            passed.append("metadata_readable")

    # a cell is its sample and its barcode: the pair names one cell
    dup = _q(conn, "SELECT COUNT(*) FROM (SELECT 1 FROM cells GROUP BY sample_id, barcode "
                   "HAVING COUNT(*) > 1)")
    if dup and dup[0][0]:
        failed.append(f"cells_duplicate_barcodes: {int(dup[0][0])} barcodes name more than one "
                      f"cell of a sample")
    else:
        passed.append("cell_identity")

    return ValidationReport(passed=(len(failed) == 0), checks_passed=passed, checks_failed=failed)


#: failures repair() can fix (the rest need the data rebuilt)
REPAIRABLE = ("matrix_rows:", "matrix_cols:", "matrix_chunk_count:", "matrix_row_coverage:",
              "matrix_csc_chunk_count:", "matrix_chunk_dtype:", "matrix_orphan_chunks",
              "embedding_rows:", "embedding_chunk_count:")


def repair(ds) -> list[str]:
    """Repair recoverable integrity issues; returns what was done.

    Fixes:
    - Orphan matrix (row and column), embedding and graph chunks.
    - Matrix row counts that disagree with the row table: a matrix with more
      rows is cut; one with fewer gets zero rows for the missing ones, as an
      empty chunk every reader sees. Rows no chunk holds get one too.
    - Matrix column counts that disagree with the matrix's own feature table,
      only where no stored entry lies past the new count: a column count never
      goes below the largest column index + 1.
    - Chunks stored in another dtype than their matrix (a merge of int32 and
      float32 inputs by 0.3.8): the matrix is rewritten in one dtype, the
      promotion of all.
    - Embedding row count mismatches.

    Parameters
    ----------
    ds
        Open ``CytomeDataset`` instance.
    """
    conn = ds._conn
    ds.flush()
    done: list[str] = []
    with conn:
        conn.execute("REINDEX")
        for table in ("matrix_chunks", "matrix_csc_chunks"):
            cur = conn.execute(
                f"DELETE FROM {table} WHERE matrix_name NOT IN (SELECT matrix_name FROM matrix_meta)")
            if cur.rowcount:
                done.append(f"removed {cur.rowcount} orphan {table}")
        cur = conn.execute(
            "DELETE FROM dense_chunks WHERE array_name NOT IN (SELECT array_name FROM embedding_meta)")
        if cur.rowcount:
            done.append(f"removed {cur.rowcount} orphan dense_chunks")
        try:
            cur = conn.execute(
                "DELETE FROM graph_chunks WHERE NOT EXISTS (SELECT 1 FROM graph_meta m WHERE "
                "m.graph_name = graph_chunks.graph_name AND m.axis = graph_chunks.axis)")
            if cur.rowcount:
                done.append(f"removed {cur.rowcount} orphan graph_chunks")
        except sqlite3.OperationalError:
            pass

        entity_counts = _entity_counts(conn)
        col_entities = matrix_column_entities(conn)

        for row in conn.execute(
            "SELECT matrix_name, n_rows, n_cols, row_entity, dtype, COALESCE(n_chunks, 0), "
            "COALESCE(has_csc, 0) FROM matrix_meta"
        ).fetchall():
            name, n_rows, n_cols, row_entity, dtype, n_chunks, has_csc = row
            n_rows, n_cols = int(n_rows), int(n_cols)
            column_major = int(n_chunks) == 0 and bool(has_csc)

            # one dtype per matrix
            if not column_major and conn.execute(
                    "SELECT COUNT(*) FROM matrix_chunks WHERE matrix_name = ? AND dtype != ?",
                    (name, str(dtype))).fetchone()[0]:
                _rewrite_in_one_dtype(conn, name)
                done.append(f"{name}: rewritten in one dtype")

            # Fix row mismatches
            if row_entity in entity_counts:
                expected = entity_counts[row_entity]
                if n_rows != expected:
                    _repair_matrix_rows(conn, name, n_rows, expected)
                    done.append(f"{name}: {n_rows} rows -> {expected}")
            if not column_major and _fill_row_gaps(conn, name):
                done.append(f"{name}: rows no chunk held are zero rows")

            # Fix column mismatches (metadata-only update), never below what is stored
            col_entity = col_entities.get(name)
            if col_entity in entity_counts and entity_counts[col_entity] > 0:
                expected_cols = entity_counts[col_entity]
                if n_cols != expected_cols:
                    needed = _columns_in_use(conn, name)
                    if needed <= expected_cols:
                        conn.execute(
                            "UPDATE matrix_meta SET n_cols = ? WHERE matrix_name = ?",
                            (expected_cols, name),
                        )
                        done.append(f"{name}: {n_cols} columns -> {expected_cols}")
                    else:
                        import warnings
                        warnings.warn(
                            f"repair: {name} has entries in {needed} columns and its feature "
                            f"table {col_entity} has {expected_cols} rows; its column count is "
                            f"left at {n_cols} (cutting it would make the matrix unreadable)",
                            RuntimeWarning, stacklevel=2)

            # the column chunks' count, from the chunks themselves
            if has_csc:
                have = conn.execute("SELECT COUNT(*) FROM matrix_csc_chunks WHERE matrix_name = ?",
                                    (name,)).fetchone()[0]
                conn.execute("UPDATE matrix_meta SET csc_n_chunks = ? WHERE matrix_name = ?",
                             (int(have), name))
            have = conn.execute("SELECT COUNT(*) FROM matrix_chunks WHERE matrix_name = ?",
                                (name,)).fetchone()[0]
            if int(have) != int(n_chunks) and not column_major:
                conn.execute("UPDATE matrix_meta SET n_chunks = ? WHERE matrix_name = ?",
                             (int(have), name))

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
            done.append(f"{name}: {n_rows} rows -> {expected}")
    return done


def _columns_in_use(conn, matrix_name: str) -> int:
    """The largest column index any stored entry of the matrix uses, plus one."""
    from cytome.io.chunked_io import _frombuffer_index
    from cytome.io.compression import decompress_blob
    top = 0
    row = conn.execute("SELECT MAX(col_end) FROM matrix_csc_chunks WHERE matrix_name = ? "
                       "AND n_nonzero > 0", (matrix_name,)).fetchone()
    if row and row[0] is not None:
        # a column chunk's entries lie in its columns
        top = int(row[0])
    for nnz, blob, comp in conn.execute(
            "SELECT n_nonzero, indices_blob, compression FROM matrix_chunks WHERE matrix_name = ? "
            "AND n_nonzero > 0", (matrix_name,)):
        idx = _frombuffer_index(decompress_blob(blob, comp), int(nnz), f"{matrix_name} indices")
        if idx.size:
            top = max(top, int(idx.max()) + 1)
    return top


def _decoded_chunks(conn, matrix_name: str, n_cols: int):
    """``(row_start, row_end, csr block in the chunk's own dtype)`` in row order."""
    from cytome.io.chunked_io import _decode_block, _np_dtype
    for rs, re_, d, i, p, comp, cdt in conn.execute(
            "SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression, dtype "
            "FROM matrix_chunks WHERE matrix_name = ? ORDER BY row_start, chunk_idx", (matrix_name,)):
        own = _np_dtype(str(cdt))
        yield int(rs), int(re_), _decode_block(int(re_) - int(rs), n_cols, d, i, p, comp, cdt, own,
                                               f"{matrix_name} rows {rs}-{int(re_) - 1}")


def _rewrite_in_one_dtype(conn, matrix_name: str) -> None:
    """Rewrite a row-major matrix whose chunks hold several dtypes in the
    promotion of them all, so that every reader (R, older cytome) reads it."""
    from cytome.io.chunked_io import write_sparse_chunked
    meta = conn.execute(
        "SELECT n_rows, n_cols, chunk_size, row_entity, col_entity, dtype FROM matrix_meta "
        "WHERE matrix_name = ?", (matrix_name,)).fetchone()
    n_rows, n_cols, chunk_size, row_entity, col_entity, dtype = meta
    blocks, at = [], 0
    out = np.dtype(str(dtype))
    for rs, re_, blk in _decoded_chunks(conn, matrix_name, int(n_cols)):
        if rs > at:
            blocks.append(sp.csr_matrix((rs - at, int(n_cols)), dtype=blk.dtype))
        blocks.append(blk)
        out = np.result_type(out, blk.dtype)
        at = re_
    if at < int(n_rows):
        blocks.append(sp.csr_matrix((int(n_rows) - at, int(n_cols)), dtype=out))
    full = sp.vstack([b.astype(out) for b in blocks], format="csr") if blocks else \
        sp.csr_matrix((int(n_rows), int(n_cols)), dtype=out)
    write_sparse_chunked(conn, matrix_name, full, int(chunk_size), "zstd",
                         row_entity=row_entity, col_entity=col_entity, dtype=out)


def _empty_chunk(conn, matrix_name: str, chunk_idx: int, r0: int, r1: int, dtype: str) -> None:
    from cytome.io.chunked_io import _INSERT_CHUNK_SQL
    from cytome.io.compression import compress_blob
    conn.execute(_INSERT_CHUNK_SQL, (
        matrix_name, int(chunk_idx), int(r0), int(r1), 0,
        compress_blob(b"", "zstd"), compress_blob(b"", "zstd"),
        compress_blob(np.zeros(int(r1) - int(r0) + 1, dtype="<i4").tobytes(), "zstd"),
        str(dtype), "zstd"))


def _fill_row_gaps(conn, matrix_name: str) -> bool:
    """Give the rows of a row-major matrix that no chunk holds an empty chunk
    (readers count them as zero rows already; now the store says so)."""
    meta = conn.execute("SELECT n_rows, dtype FROM matrix_meta WHERE matrix_name = ?",
                        (matrix_name,)).fetchone()
    n_rows, dtype = int(meta[0]), meta[1]
    spans = conn.execute("SELECT row_start, row_end FROM matrix_chunks WHERE matrix_name = ? "
                         "ORDER BY row_start, chunk_idx", (matrix_name,)).fetchall()
    if not spans:
        return False
    gaps, at = [], 0
    for rs, re_ in spans:
        if int(rs) > at:
            gaps.append((at, int(rs)))
        at = max(at, int(re_))
    if at < n_rows:
        gaps.append((at, n_rows))
    if not gaps:
        return False
    nxt = int(conn.execute("SELECT MAX(chunk_idx) FROM matrix_chunks WHERE matrix_name = ?",
                           (matrix_name,)).fetchone()[0]) + 1
    for r0, r1 in gaps:
        _empty_chunk(conn, matrix_name, nxt, r0, r1, dtype)
        nxt += 1
    conn.execute("UPDATE matrix_meta SET n_chunks = (SELECT COUNT(*) FROM matrix_chunks "
                 "WHERE matrix_name = ?) WHERE matrix_name = ?", (matrix_name, matrix_name))
    return True


def _repair_matrix_rows(conn, matrix_name: str, current_n_rows: int, target_n_rows: int) -> None:
    """Rebuild a matrix to match the target row count.

    If the matrix has more rows than the entity table, truncates excess rows
    from the last chunk(s). If fewer, the missing rows are zero rows: the
    metadata says so, and :func:`_fill_row_gaps` gives them an empty chunk.
    """
    from cytome.io.chunked_io import write_sparse_chunked

    layout = conn.execute(
        "SELECT COALESCE(n_chunks, 0), COALESCE(has_csc, 0) FROM matrix_meta "
        "WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()
    column_major = layout is not None and layout[0] == 0 and layout[1]

    if current_n_rows <= target_n_rows:
        conn.execute(
            "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = ?",
            (target_n_rows, matrix_name),
        )
        return

    # A column-major matrix has no rows in `matrix_chunks`, so the stitch
    # below would find nothing, build an all-zero CSR of the target shape and
    # write THAT over the real data -- silent, total loss of the matrix. There
    # is no honest truncation to do here either: the row count that needs
    # fixing is metadata, and the column chunks are already correct. Refuse,
    # loudly, and say what to do.
    if column_major:
        raise ValueError(
            f"{matrix_name!r} is stored column-major and has {current_n_rows} "
            f"rows against {target_n_rows} in its row entity table. Repair "
            f"cannot truncate a column-major matrix by rewriting rows -- it "
            f"has none -- and rewriting it as an empty row-major matrix would "
            f"destroy it. Fix the entity table to match the matrix, or "
            f"re-create the matrix."
        )
    meta = conn.execute(
        "SELECT n_cols, dtype, chunk_size, row_entity, col_entity FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    if meta is None:
        return
    n_cols, dtype_str, chunk_size, row_entity, col_entity = meta
    n_cols = int(n_cols)
    out = np.dtype(str(dtype_str))

    parts = []
    at = 0
    for row_start, row_end, chunk_csr in _decoded_chunks(conn, matrix_name, n_cols):
        # Only keep rows up to target
        if row_start >= target_n_rows:
            continue
        if row_start > at:
            parts.append(sp.csr_matrix((row_start - at, n_cols), dtype=out))
        effective_end = min(row_end, target_n_rows)
        parts.append(chunk_csr[:effective_end - row_start].astype(out))
        at = effective_end
    if at < target_n_rows:
        parts.append(sp.csr_matrix((target_n_rows - at, n_cols), dtype=out))
    full = sp.vstack(parts, format="csr") if parts else sp.csr_matrix((target_n_rows, n_cols),
                                                                       dtype=out)

    # Rewrite via standard chunked writer
    write_sparse_chunked(
        conn, matrix_name, full, int(chunk_size), "zstd",
        row_entity=row_entity, col_entity=col_entity, dtype=out,
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
