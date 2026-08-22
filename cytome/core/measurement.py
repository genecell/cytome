"""Lazy sparse matrix access for Cytome measurements."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import scipy.sparse as sp
import sqlite3

from cytome.io.chunked_io import (
    read_sparse_chunked,
    read_sparse_rows,
    read_sparse_rows_iter,
    read_sparse_slice,
)
from cytome.io.compression import compress_blob, decompress_blob


class MeasurementLayer:
    """Lazy wrapper around chunked sparse matrix storage."""

    def __init__(self, conn: sqlite3.Connection, matrix_name: str) -> None:
        self._conn = conn
        self._matrix_name = matrix_name
        self._meta = self._load_meta()

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._meta["n_rows"]), int(self._meta["n_cols"])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._meta["dtype"])

    def __getitem__(self, key: object) -> sp.csr_matrix:
        if not isinstance(key, tuple):
            key = (key, slice(None))
        row_sel, col_sel = key
        n_rows, n_cols = self.shape

        row_start, row_end = _slice_bounds(row_sel, n_rows)
        matrix = read_sparse_slice(self._conn, self._matrix_name, row_start, row_end)

        if isinstance(col_sel, slice):
            c_start, c_end = _slice_bounds(col_sel, n_cols)
            return matrix[:, c_start:c_end].tocsr()
        if isinstance(col_sel, (list, np.ndarray)):
            return matrix[:, col_sel].tocsr()
        if isinstance(col_sel, int):
            return matrix[:, [col_sel]].tocsr()
        raise TypeError(f"Unsupported column selector: {type(col_sel)}")

    def to_memory(self) -> sp.csr_matrix:
        """Load entire matrix into memory."""
        return read_sparse_chunked(self._conn, self._matrix_name)

    def rows(self, indices: np.ndarray) -> sp.csr_matrix:
        """Read specific rows using chunk-selective I/O.

        Only decompresses chunks containing at least one requested row.
        """
        return read_sparse_rows(self._conn, self._matrix_name, indices)

    def iter_rows(self, row_filter=None) -> Iterator[Tuple[int, int, sp.csr_matrix]]:
        """Iterate row chunks as CSR matrices.

        ``row_filter``: sorted global row indices to keep. Chunks containing
        none of them are skipped without being fetched or decompressed.
        """
        yield from read_sparse_rows_iter(self._conn, self._matrix_name,
                                         row_filter=row_filter)

    def column(self, idx: int) -> sp.csr_matrix:
        """Read one feature column across all rows."""
        if self.has_feature_index:
            for col_start, col_end, chunk in self.iter_columns():
                if col_start <= idx < col_end:
                    return chunk[:, idx - col_start].tocsr()
        return self[:, idx]

    def columns(self, indices: list[int] | np.ndarray) -> sp.csr_matrix:
        """Read selected feature columns across all rows."""
        idx_list = list(indices)
        if self.has_feature_index and idx_list:
            pieces = [self.column(i) for i in idx_list]
            return sp.hstack(pieces, format="csr")
        return self[:, list(indices)]

    def build_feature_index(self, chunk_size: int | None = None, compression: str = "zstd") -> None:
        """Build optional CSC chunk index for feature-wise iteration."""
        full_csc = self.to_memory().tocsc()
        chunk_size = int(chunk_size or self._meta.get("chunk_size", 256))
        self._conn.execute(
            "DELETE FROM matrix_csc_chunks WHERE matrix_name = ?", (self._matrix_name,)
        )

        chunk_idx = 0
        for col_start in range(0, full_csc.shape[1], chunk_size):
            col_end = min(col_start + chunk_size, full_csc.shape[1])
            chunk = full_csc[:, col_start:col_end]
            self._conn.execute(
                """
                INSERT INTO matrix_csc_chunks(
                    matrix_name, chunk_idx, col_start, col_end, n_nonzero,
                    data_blob, indices_blob, indptr_blob, dtype, compression
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._matrix_name,
                    chunk_idx,
                    col_start,
                    col_end,
                    int(chunk.nnz),
                    compress_blob(chunk.data.tobytes(), compression),
                    compress_blob(chunk.indices.astype(np.int32).tobytes(), compression),
                    compress_blob(chunk.indptr.astype(np.int32).tobytes(), compression),
                    str(chunk.data.dtype),
                    compression,
                ),
            )
            chunk_idx += 1

        self._conn.execute(
            """
            UPDATE matrix_meta
            SET has_csc = 1, csc_chunk_size = ?, csc_n_chunks = ?
            WHERE matrix_name = ?
            """,
            (chunk_size, chunk_idx, self._matrix_name),
        )
        self._meta = self._load_meta()

    @property
    def has_feature_index(self) -> bool:
        return bool(self._meta.get("has_csc", 0))

    def iter_columns(self) -> Iterator[Tuple[int, int, sp.csc_matrix]]:
        """Iterate CSC chunks for feature-major streaming."""
        if not self.has_feature_index:
            raise RuntimeError("Feature index not built. Call build_feature_index() first.")
        rows = self._conn.execute(
            """
            SELECT col_start, col_end, data_blob, indices_blob, indptr_blob, dtype, compression
            FROM matrix_csc_chunks
            WHERE matrix_name = ?
            ORDER BY chunk_idx
            """,
            (self._matrix_name,),
        )
        n_rows = self.shape[0]
        for col_start, col_end, data_blob, indices_blob, indptr_blob, dtype, compression in rows:
            data = np.frombuffer(decompress_blob(data_blob, compression), dtype=np.dtype(dtype))
            indices = np.frombuffer(
                decompress_blob(indices_blob, compression), dtype=np.int32
            )
            indptr = np.frombuffer(decompress_blob(indptr_blob, compression), dtype=np.int32)
            chunk = sp.csc_matrix(
                (data, indices, indptr), shape=(n_rows, int(col_end) - int(col_start))
            )
            yield int(col_start), int(col_end), chunk

    def drop_feature_index(self) -> None:
        """Drop CSC chunks and metadata flags."""
        self._conn.execute(
            "DELETE FROM matrix_csc_chunks WHERE matrix_name = ?", (self._matrix_name,)
        )
        self._conn.execute(
            """
            UPDATE matrix_meta
            SET has_csc = 0, csc_chunk_size = NULL, csc_n_chunks = NULL
            WHERE matrix_name = ?
            """,
            (self._matrix_name,),
        )
        self._meta = self._load_meta()

    def _load_meta(self) -> dict[str, object]:
        row = self._conn.execute(
            """
            SELECT n_rows, n_cols, dtype, chunk_size, has_csc, csc_chunk_size, csc_n_chunks
            FROM matrix_meta WHERE matrix_name = ?
            """,
            (self._matrix_name,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Matrix not found: {self._matrix_name}")
        return {
            "n_rows": row[0],
            "n_cols": row[1],
            "dtype": row[2],
            "chunk_size": row[3],
            "has_csc": row[4],
            "csc_chunk_size": row[5],
            "csc_n_chunks": row[6],
        }


def _slice_bounds(sel: object, size: int) -> tuple[int, int]:
    if isinstance(sel, slice):
        start = 0 if sel.start is None else sel.start
        stop = size if sel.stop is None else sel.stop
        return max(0, start), min(size, stop)
    if isinstance(sel, int):
        if sel < 0:
            sel += size
        if sel < 0 or sel >= size:
            raise IndexError("index out of range")
        return sel, sel + 1
    raise TypeError(f"Unsupported slice selector: {type(sel)}")
