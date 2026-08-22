"""Chunked sparse and dense array IO for Cytome."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, Tuple

import warnings

import numpy as np
import scipy.sparse as sp
import sqlite3

from .compression import compress_blob, decompress_blob  # noqa: F401

_INSERT_CHUNK_SQL = """
INSERT INTO matrix_chunks(
    matrix_name, chunk_idx, row_start, row_end, n_nonzero,
    data_blob, indices_blob, indptr_blob, dtype, compression
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChunkedLayerWriter:
    """Write a sparse layer incrementally, chunk by chunk.

    Each call to :meth:`write_chunk` splits the compute-sized chunk into
    storage-sized sub-chunks, compresses them with zstd, and writes them
    to the ``matrix_chunks`` table immediately via ``executemany``.

    Parameters
    ----------
    conn
        Open SQLite connection (WAL mode recommended).
    matrix_name
        Layer identifier (e.g. ``"RNA_infog"``).
    n_rows
        Total number of rows that will be written.
    n_cols
        Number of columns (features / genes).
    dtype
        Numpy dtype for values (default ``float64``).
    compression
        Blob compression method (default ``"zstd"``).
    row_entity
        Row entity table name.
    col_entity
        Column entity table name.
    overwrite
        If *True*, delete any existing data for *matrix_name* first.
    storage_chunk_size
        Rows per storage blob.  Default 128 gives 8x fewer blobs than
        the previous default of 16.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        matrix_name: str,
        n_rows: int,
        n_cols: int,
        dtype: np.dtype | str = np.float64,
        compression: str = "zstd",
        row_entity: str = "cells",
        col_entity: str = "genes",
        overwrite: bool = True,
        storage_chunk_size: int = 128,
    ) -> None:
        self._conn = conn
        self._matrix_name = matrix_name
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._dtype_str = str(np.dtype(dtype))
        self._compression = compression
        self._row_entity = row_entity
        self._col_entity = col_entity
        self._storage_chunk_size = storage_chunk_size

        self._chunk_idx = 0
        self._total_nnz = 0
        self._finalized = False

        if overwrite:
            conn.execute(
                "DELETE FROM matrix_chunks WHERE matrix_name = ?",
                (matrix_name,),
            )
            conn.execute(
                "DELETE FROM matrix_meta WHERE matrix_name = ?",
                (matrix_name,),
            )

    def write_chunk(self, sparse_chunk: sp.spmatrix, row_offset: int) -> None:
        """Write a compute-sized chunk, splitting into storage sub-chunks.

        All sub-chunks from one compute batch are inserted via a single
        ``executemany`` call.

        Parameters
        ----------
        sparse_chunk
            CSR (or convertible) sparse matrix of shape
            ``(chunk_rows, n_cols)``.
        row_offset
            Global row index of the first row in *sparse_chunk*.
        """
        if self._finalized:
            raise RuntimeError("Cannot write after finalize()")

        csr = sparse_chunk.tocsr()
        scs = self._storage_chunk_size
        compression = self._compression
        matrix_name = self._matrix_name
        dtype_str = self._dtype_str
        dtype_np = np.dtype(dtype_str)

        # Build all sub-chunk rows, then insert in one executemany call
        rows = []
        for sub_start in range(0, csr.shape[0], scs):
            sub_end = min(sub_start + scs, csr.shape[0])
            sub = csr[sub_start:sub_end]

            rows.append((
                matrix_name,
                self._chunk_idx,
                row_offset + sub_start,
                row_offset + sub_end,
                int(sub.nnz),
                compress_blob(sub.data.astype(dtype_np).tobytes(), method=compression),
                compress_blob(sub.indices.tobytes(), method=compression),
                compress_blob(sub.indptr.tobytes(), method=compression),
                dtype_str,
                compression,
            ))
            self._total_nnz += sub.nnz
            self._chunk_idx += 1

        self._conn.executemany(_INSERT_CHUNK_SQL, rows)

    def finalize(self, provenance_id: int | None = None) -> None:
        """Write matrix metadata and checkpoint WAL."""
        if self._finalized:
            raise RuntimeError("Already finalized")
        self._finalized = True

        self._conn.execute(
            """
            INSERT INTO matrix_meta(
                matrix_name, n_rows, n_cols, n_nonzero, dtype,
                row_entity, col_entity, chunk_size, n_chunks,
                created_at, provenance_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._matrix_name,
                int(self._n_rows),
                int(self._n_cols),
                int(self._total_nnz),
                self._dtype_str,
                self._row_entity,
                self._col_entity,
                int(self._storage_chunk_size),
                int(self._chunk_idx),
                _now_iso(),
                provenance_id,
            ),
        )
        self._conn.commit()
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def write_sparse_chunked(
    conn: sqlite3.Connection,
    matrix_name: str,
    csr_matrix: sp.csr_matrix,
    chunk_size: int,
    compression: str,
    row_entity: str = "cells",
    col_entity: str = "genes",
    provenance_id: int | None = None,
) -> None:
    """Write a CSR matrix into chunked compressed storage.

    Parameters
    ----------
    conn
        Database connection.
    matrix_name
        Matrix identifier.
    csr_matrix
        Sparse matrix in CSR format.
    chunk_size
        Number of rows per chunk.
    compression
        Compression method.
    row_entity
        Row entity table name.
    col_entity
        Column entity table name.
    provenance_id
        Optional provenance row id.
    """
    csr = csr_matrix.tocsr()
    conn.execute("DELETE FROM matrix_chunks WHERE matrix_name = ?", (matrix_name,))
    conn.execute("DELETE FROM matrix_meta WHERE matrix_name = ?", (matrix_name,))

    n_rows, n_cols = csr.shape
    chunk_idx = 0
    dtype_str = str(csr.data.dtype)
    batch: list[tuple] = []
    for row_start in range(0, n_rows, chunk_size):
        row_end = min(row_start + chunk_size, n_rows)
        chunk = csr[row_start:row_end]
        batch.append((
            matrix_name,
            chunk_idx,
            row_start,
            row_end,
            int(chunk.nnz),
            compress_blob(chunk.data.tobytes(), method=compression),
            compress_blob(chunk.indices.tobytes(), method=compression),
            compress_blob(chunk.indptr.tobytes(), method=compression),
            dtype_str,
            compression,
        ))
        chunk_idx += 1

    conn.executemany(_INSERT_CHUNK_SQL, batch)

    conn.execute(
        """
        INSERT INTO matrix_meta(
            matrix_name, n_rows, n_cols, n_nonzero, dtype,
            row_entity, col_entity, chunk_size, n_chunks, created_at, provenance_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            matrix_name,
            int(n_rows),
            int(n_cols),
            int(csr.nnz),
            str(csr.data.dtype),
            row_entity,
            col_entity,
            int(chunk_size),
            int(chunk_idx),
            _now_iso(),
            provenance_id,
        ),
    )


def read_sparse_chunked(conn: sqlite3.Connection, matrix_name: str) -> sp.csr_matrix:
    """Read a full sparse matrix from chunk storage."""
    meta = conn.execute(
        "SELECT n_rows, n_cols, dtype, n_nonzero FROM matrix_meta "
        "WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Matrix not found: {matrix_name}")
    n_rows, n_cols, dtype = int(meta[0]), int(meta[1]), np.dtype(meta[2])
    n_nonzero = int(meta[3]) if meta[3] is not None else None

    rows = conn.execute(
        """
        SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression
        FROM matrix_chunks
        WHERE matrix_name = ?
        ORDER BY chunk_idx
        """,
        (matrix_name,),
    ).fetchall()
    if not rows:
        return sp.csr_matrix((n_rows, n_cols), dtype=dtype)

    # Preallocate from matrix_meta.n_nonzero and fill in place. Accumulating
    # every chunk in a list and then np.concatenate-ing meant both the parts
    # and the result were alive at once, so a full read peaked at about twice
    # the matrix. n_nonzero is recorded at write time, so the destination size
    # is known before the first chunk is decompressed.
    if n_nonzero is not None:
        data = np.empty(n_nonzero, dtype=dtype)
        indices = np.empty(n_nonzero, dtype=np.int32)
        indptr_arr = np.empty(n_rows + 1, dtype=np.int32)
        indptr_arr[0] = 0
        nnz_at = 0
        row_at = 0
        for _row_start, _row_end, data_blob, indices_blob, indptr_blob, compression in rows:
            chunk_data = np.frombuffer(
                decompress_blob(data_blob, compression), dtype=dtype)
            chunk_indices = np.frombuffer(
                decompress_blob(indices_blob, compression), dtype=np.int32)
            chunk_indptr = np.frombuffer(
                decompress_blob(indptr_blob, compression), dtype=np.int32)
            k = chunk_data.shape[0]
            data[nnz_at:nnz_at + k] = chunk_data
            indices[nnz_at:nnz_at + k] = chunk_indices
            n_chunk_rows = chunk_indptr.shape[0] - 1
            indptr_arr[row_at + 1:row_at + 1 + n_chunk_rows] = chunk_indptr[1:] + nnz_at
            nnz_at += k
            row_at += n_chunk_rows
        if nnz_at != n_nonzero or row_at != n_rows:
            # The recorded totals disagree with what is on disk. Trust the
            # blobs, not the metadata, and say so rather than returning a
            # matrix padded with uninitialised memory.
            warnings.warn(
                f"{matrix_name}: matrix_meta says {n_nonzero} nonzeros over "
                f"{n_rows} rows but the chunks hold {nnz_at} over {row_at}. "
                f"Using the chunks.", stacklevel=2)
            data = data[:nnz_at]
            indices = indices[:nnz_at]
            indptr_arr = indptr_arr[:row_at + 1]
            n_rows = row_at
        return sp.csr_matrix((data, indices, indptr_arr), shape=(n_rows, n_cols))

    # No n_nonzero recorded (older file): fall back to the two-pass form.
    data_parts = []
    indices_parts = []
    indptr = [0]
    for row_start, row_end, data_blob, indices_blob, indptr_blob, compression in rows:
        del row_start, row_end
        chunk_data = np.frombuffer(decompress_blob(data_blob, compression), dtype=dtype)
        chunk_indices = np.frombuffer(
            decompress_blob(indices_blob, compression), dtype=np.int32
        )
        chunk_indptr = np.frombuffer(
            decompress_blob(indptr_blob, compression), dtype=np.int32
        )
        data_parts.append(chunk_data)
        indices_parts.append(chunk_indices)
        base = indptr[-1]
        indptr.extend((chunk_indptr[1:] + base).tolist())

    data = np.concatenate(data_parts) if data_parts else np.array([], dtype=dtype)
    indices = (
        np.concatenate(indices_parts) if indices_parts else np.array([], dtype=np.int32)
    )
    indptr_arr = np.asarray(indptr, dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr_arr), shape=(n_rows, n_cols))


def read_sparse_slice(
    conn: sqlite3.Connection,
    matrix_name: str,
    row_start: int,
    row_end: int,
) -> sp.csr_matrix:
    """Read a row slice from a chunked sparse matrix."""
    meta = conn.execute(
        "SELECT n_cols, dtype FROM matrix_meta WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()
    if meta is None:
        raise KeyError(f"Matrix not found: {matrix_name}")
    n_cols, dtype = int(meta[0]), np.dtype(meta[1])

    rows = conn.execute(
        """
        SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression
        FROM matrix_chunks
        WHERE matrix_name = ? AND row_end > ? AND row_start < ?
        ORDER BY chunk_idx
        """,
        (matrix_name, row_start, row_end),
    ).fetchall()

    parts = []
    for c_start, c_end, data_blob, indices_blob, indptr_blob, compression in rows:
        chunk_rows = c_end - c_start
        chunk_data = np.frombuffer(decompress_blob(data_blob, compression), dtype=dtype)
        chunk_indices = np.frombuffer(
            decompress_blob(indices_blob, compression), dtype=np.int32
        )
        chunk_indptr = np.frombuffer(
            decompress_blob(indptr_blob, compression), dtype=np.int32
        )
        chunk = sp.csr_matrix(
            (chunk_data, chunk_indices, chunk_indptr), shape=(chunk_rows, n_cols)
        )
        local_start = max(0, row_start - c_start)
        local_end = min(chunk_rows, row_end - c_start)
        parts.append(chunk[local_start:local_end])

    if not parts:
        return sp.csr_matrix((max(0, row_end - row_start), n_cols), dtype=dtype)
    return sp.vstack(parts, format="csr")


def read_sparse_rows(
    conn: sqlite3.Connection,
    matrix_name: str,
    row_indices: np.ndarray,
) -> sp.csr_matrix:
    """Read specific rows from a chunked sparse matrix.

    Only decompresses chunks that contain at least one requested row.
    Returns a CSR matrix with shape ``(len(row_indices), n_cols)``.

    Parameters
    ----------
    conn
        Database connection.
    matrix_name
        Matrix identifier.
    row_indices
        Sorted 0-based row indices to read.
    """
    row_indices = np.asarray(row_indices, dtype=np.int64)
    meta = conn.execute(
        "SELECT n_cols, dtype FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Matrix not found: {matrix_name}")
    n_cols, dtype = int(meta[0]), np.dtype(meta[1])

    if row_indices.size == 0:
        return sp.csr_matrix((0, n_cols), dtype=dtype)

    # Fetch chunk boundaries
    chunks = conn.execute(
        """
        SELECT chunk_idx, row_start, row_end
        FROM matrix_chunks
        WHERE matrix_name = ?
        ORDER BY chunk_idx
        """,
        (matrix_name,),
    ).fetchall()

    # Determine which chunks contain requested rows
    needed_chunks = []
    for chunk_idx, c_start, c_end in chunks:
        # Use searchsorted to find rows in [c_start, c_end)
        lo = int(np.searchsorted(row_indices, c_start, side="left"))
        hi = int(np.searchsorted(row_indices, c_end, side="left"))
        if lo < hi:
            needed_chunks.append((chunk_idx, c_start, c_end, row_indices[lo:hi]))

    if not needed_chunks:
        return sp.csr_matrix((0, n_cols), dtype=dtype)

    parts = []
    for chunk_idx, c_start, c_end, local_rows in needed_chunks:
        row = conn.execute(
            """
            SELECT data_blob, indices_blob, indptr_blob, compression
            FROM matrix_chunks
            WHERE matrix_name = ? AND chunk_idx = ?
            """,
            (matrix_name, chunk_idx),
        ).fetchone()
        data_blob, indices_blob, indptr_blob, compression = row
        chunk_data = np.frombuffer(decompress_blob(data_blob, compression), dtype=dtype)
        chunk_indices = np.frombuffer(
            decompress_blob(indices_blob, compression), dtype=np.int32
        )
        chunk_indptr = np.frombuffer(
            decompress_blob(indptr_blob, compression), dtype=np.int32
        )
        chunk = sp.csr_matrix(
            (chunk_data, chunk_indices, chunk_indptr),
            shape=(c_end - c_start, n_cols),
        )
        # Extract only the requested rows (convert global indices to local)
        local_idx = (local_rows - c_start).astype(np.int32)
        parts.append(chunk[local_idx])

    if len(parts) == 1:
        return parts[0]
    return sp.vstack(parts, format="csr")


def read_dense_rows(
    conn: sqlite3.Connection,
    array_name: str,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Read specific rows from a chunked dense array.

    Only decompresses chunks that contain at least one requested row.

    Parameters
    ----------
    conn
        Database connection.
    array_name
        Array identifier.
    row_indices
        Sorted 0-based row indices to read.
    """
    row_indices = np.asarray(row_indices, dtype=np.int64)
    meta = conn.execute(
        "SELECT n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Embedding not found: {array_name}")
    n_cols, dtype = int(meta[0]), np.dtype(meta[1])

    if row_indices.size == 0:
        return np.empty((0, n_cols), dtype=dtype)

    chunks = conn.execute(
        """
        SELECT chunk_idx, row_start, row_end
        FROM dense_chunks
        WHERE array_name = ?
        ORDER BY chunk_idx
        """,
        (array_name,),
    ).fetchall()

    parts = []
    for chunk_idx, c_start, c_end in chunks:
        lo = int(np.searchsorted(row_indices, c_start, side="left"))
        hi = int(np.searchsorted(row_indices, c_end, side="left"))
        if lo >= hi:
            continue
        local_rows = row_indices[lo:hi]
        row = conn.execute(
            """
            SELECT data_blob, compression
            FROM dense_chunks
            WHERE array_name = ? AND chunk_idx = ?
            """,
            (array_name, chunk_idx),
        ).fetchone()
        blob, compression = row
        chunk = np.frombuffer(decompress_blob(blob, compression), dtype=dtype).reshape(
            (c_end - c_start, n_cols)
        )
        local_idx = (local_rows - c_start).astype(np.int32)
        parts.append(chunk[local_idx])

    if not parts:
        return np.empty((0, n_cols), dtype=dtype)
    return np.vstack(parts)


def read_sparse_rows_iter(
    conn: sqlite3.Connection,
    matrix_name: str,
    row_filter: "np.ndarray | None" = None,
) -> Iterator[Tuple[int, int, sp.csr_matrix]]:
    """Iterate over sparse row chunks.

    ``row_filter`` is a sorted array of global row indices the caller wants.
    Chunks holding none of them are never fetched, so the blobs are neither
    read off disk nor decompressed.

    That matters more than it sounds. A per-batch masked read used to
    decompress every chunk and discard the ones with no selected rows, so a
    35-batch GDR over 200,061 cells read the whole matrix 35 times, 17 passes
    each. Measured on that file, a median batch needs 45 of 1,563 chunks: 2.9%
    of what the full scan reads.
    """
    meta = conn.execute(
        "SELECT n_cols, dtype FROM matrix_meta WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()
    if meta is None:
        raise KeyError(f"Matrix not found: {matrix_name}")
    n_cols, dtype = int(meta[0]), np.dtype(meta[1])

    if row_filter is not None:
        wanted = np.asarray(row_filter)
        if wanted.dtype == bool:
            wanted = np.flatnonzero(wanted)
        wanted = np.sort(np.asarray(wanted, dtype=np.int64))
        # Ask for the row ranges only. Selecting the blob columns here would
        # make SQLite read them, which is the cost being avoided.
        spans = conn.execute(
            "SELECT chunk_idx, row_start, row_end FROM matrix_chunks "
            "WHERE matrix_name = ? ORDER BY chunk_idx",
            (matrix_name,),
        ).fetchall()
        needed = [
            int(ci) for ci, rs, re_ in spans
            if np.searchsorted(wanted, int(rs), side="left")
            < np.searchsorted(wanted, int(re_), side="left")
        ]
        if not needed:
            return
        placeholders = ",".join("?" * len(needed))
        rows = conn.execute(
            f"""
            SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression
            FROM matrix_chunks WHERE matrix_name = ? AND chunk_idx IN ({placeholders})
            ORDER BY chunk_idx
            """,
            (matrix_name, *needed),
        )
    else:
        rows = conn.execute(
            """
            SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression
            FROM matrix_chunks WHERE matrix_name = ? ORDER BY chunk_idx
            """,
            (matrix_name,),
        )
    for row_start, row_end, data_blob, indices_blob, indptr_blob, compression in rows:
        chunk_data = np.frombuffer(decompress_blob(data_blob, compression), dtype=dtype)
        chunk_indices = np.frombuffer(
            decompress_blob(indices_blob, compression), dtype=np.int32
        )
        chunk_indptr = np.frombuffer(
            decompress_blob(indptr_blob, compression), dtype=np.int32
        )
        chunk = sp.csr_matrix(
            (chunk_data, chunk_indices, chunk_indptr), shape=(row_end - row_start, n_cols)
        )
        yield int(row_start), int(row_end), chunk


def write_dense_chunked(
    conn: sqlite3.Connection,
    array_name: str,
    ndarray: np.ndarray,
    chunk_size: int,
    compression: str,
    entity: str = "cells",
    provenance_id: int | None = None,
) -> None:
    """Write a dense 2D array as row chunks."""
    arr = np.asarray(ndarray)
    if arr.ndim != 2:
        raise ValueError("Dense chunked write expects a 2D array")

    conn.execute("DELETE FROM dense_chunks WHERE array_name = ?", (array_name,))
    conn.execute("DELETE FROM embedding_meta WHERE array_name = ?", (array_name,))

    n_rows, n_cols = arr.shape
    chunk_idx = 0
    for row_start in range(0, n_rows, chunk_size):
        row_end = min(row_start + chunk_size, n_rows)
        chunk = arr[row_start:row_end]
        conn.execute(
            """
            INSERT INTO dense_chunks(
                array_name, chunk_idx, row_start, row_end, n_cols,
                data_blob, dtype, compression
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                array_name,
                chunk_idx,
                row_start,
                row_end,
                n_cols,
                compress_blob(chunk.tobytes(), method=compression),
                str(chunk.dtype),
                compression,
            ),
        )
        chunk_idx += 1

    conn.execute(
        """
        INSERT INTO embedding_meta(
            array_name, n_rows, n_cols, dtype, entity, chunk_size,
            n_chunks, created_at, provenance_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            array_name,
            int(n_rows),
            int(n_cols),
            str(arr.dtype),
            entity,
            int(chunk_size),
            int(chunk_idx),
            _now_iso(),
            provenance_id,
        ),
    )


def read_dense_chunked(conn: sqlite3.Connection, array_name: str) -> np.ndarray:
    """Read full dense array from chunk storage."""
    meta = conn.execute(
        "SELECT n_rows, n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Embedding not found: {array_name}")
    n_rows, n_cols, dtype = int(meta[0]), int(meta[1]), np.dtype(meta[2])

    rows = conn.execute(
        """
        SELECT row_start, row_end, data_blob, compression
        FROM dense_chunks WHERE array_name = ? ORDER BY chunk_idx
        """,
        (array_name,),
    ).fetchall()
    arr = np.empty((n_rows, n_cols), dtype=dtype)
    for row_start, row_end, blob, compression in rows:
        chunk = np.frombuffer(decompress_blob(blob, compression), dtype=dtype).reshape(
            (row_end - row_start, n_cols)
        )
        arr[row_start:row_end] = chunk
    return arr


def read_dense_slice(
    conn: sqlite3.Connection,
    array_name: str,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """Read row slice of a dense chunked array."""
    meta = conn.execute(
        "SELECT n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Embedding not found: {array_name}")
    n_cols, dtype = int(meta[0]), np.dtype(meta[1])

    rows = conn.execute(
        """
        SELECT row_start, row_end, data_blob, compression
        FROM dense_chunks
        WHERE array_name = ? AND row_end > ? AND row_start < ?
        ORDER BY chunk_idx
        """,
        (array_name, row_start, row_end),
    ).fetchall()

    parts = []
    for c_start, c_end, blob, compression in rows:
        chunk = np.frombuffer(decompress_blob(blob, compression), dtype=dtype).reshape(
            (c_end - c_start, n_cols)
        )
        local_start = max(0, row_start - c_start)
        local_end = min(c_end - c_start, row_end - c_start)
        parts.append(chunk[local_start:local_end])

    if not parts:
        return np.empty((max(0, row_end - row_start), n_cols), dtype=dtype)
    return np.vstack(parts)
