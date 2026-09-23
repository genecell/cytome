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
        if self.is_column_major:
            matrix = self.to_memory()[row_start:row_end]
        else:
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
        """Load entire matrix into memory.

        A column-major matrix is assembled from its CSC chunks; row-major
        from its row chunks. Either way the result is CSR.
        """
        if self.is_column_major:
            blocks = [chunk for _, _, chunk in self.iter_columns()]
            if not blocks:
                return sp.csr_matrix(self.shape, dtype=self.dtype)
            return sp.hstack(blocks, format="csc").tocsr()
        return read_sparse_chunked(self._conn, self._matrix_name)

    def rows(self, indices: np.ndarray) -> sp.csr_matrix:
        """Read specific rows using chunk-selective I/O.

        Only decompresses chunks containing at least one requested row.
        """
        if self.is_column_major:
            return self.to_memory()[np.asarray(indices)]
        return read_sparse_rows(self._conn, self._matrix_name, indices)

    def iter_rows(self, row_filter=None) -> Iterator[Tuple[int, int, sp.csr_matrix]]:
        """Iterate row chunks as CSR matrices.

        ``row_filter``: sorted global row indices to keep. Chunks containing
        none of them are skipped without being fetched or decompressed.
        """
        if self.is_column_major:
            # Silence here would be an empty matrix: iter_chunks would yield
            # nothing and every consumer would compute on nothing.
            raise ValueError(
                f"{self._matrix_name!r} is stored column-major (CSC chunks only); "
                f"it has no row chunks to iterate. Read it by columns with "
                f"iter_columns(), or use a consumer that supports the CSC layout."
            )
        yield from read_sparse_rows_iter(self._conn, self._matrix_name,
                                         row_filter=row_filter)

    def column(self, idx: int) -> sp.csr_matrix:
        """Read one feature column across all rows."""
        return self.columns([idx])

    def columns(self, indices: list[int] | np.ndarray) -> sp.csr_matrix:
        """Read selected feature columns across all rows, in the order given.

        With a feature index this touches only the chunks that actually hold
        the requested columns, each one once. The previous form called
        ``column()`` per index and each of those re-iterated every chunk in
        the matrix, so asking for k features read the whole store k times --
        fine for one marker, quadratic for a marker panel.
        """
        idx_list = [int(i) for i in indices]
        if not idx_list:
            return sp.csr_matrix((self.shape[0], 0), dtype=self.dtype)
        if not self.has_feature_index:
            return self[:, idx_list]

        wanted = np.asarray(idx_list, dtype=np.int64)
        out = sp.lil_matrix((self.shape[0], wanted.shape[0]), dtype=self.dtype)
        remaining = wanted.shape[0]
        for col_start, col_end, chunk in self.iter_columns():
            hits = np.flatnonzero((wanted >= col_start) & (wanted < col_end))
            if hits.size == 0:
                continue
            local = chunk[:, (wanted[hits] - col_start).tolist()].tocsc()
            for slot, j in enumerate(hits):
                out[:, int(j)] = local[:, slot]
            remaining -= hits.size
            if remaining <= 0:
                break
        return out.tocsr()

    def build_feature_index(self, chunk_size: int | None = None, compression: str = "zstd") -> None:
        """Build the column-major (CSC) chunks for feature-wise iteration.

        **This materialises the whole matrix in memory** (``to_memory().tocsc()``)
        before writing the chunks, so it is for matrices that fit in RAM -- a
        gene matrix, a small peak set. For a large peak or tile matrix do not
        call this: re-quantify with ``--layout csc`` (the importer's and the
        quantifier's default), which writes column-major in one bounded pass
        without ever holding the matrix.

        Why it matters: on a matrix with far more features than cells the SVD
        engine's transpose product is about 8x faster on column-major storage,
        because it scatters into a ``cells x block`` buffer that fits in cache
        rather than a ``features x block`` one that does not.
        """
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
            SELECT n_rows, n_cols, dtype, chunk_size, has_csc, csc_chunk_size, csc_n_chunks,
                   n_chunks
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
            "n_chunks": row[7],
        }

    @property
    def is_column_major(self) -> bool:
        """True when the matrix is stored as CSC chunks only (no row chunks)."""
        return bool(self._meta.get("has_csc")) and not self._meta.get("n_chunks")


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
