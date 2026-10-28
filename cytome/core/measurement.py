"""Lazy sparse matrix access for Cytome measurements."""

from __future__ import annotations

from cytome.core import _handles

from typing import Iterator, Tuple

import numpy as np
import scipy.sparse as sp
import sqlite3

from cytome.io.chunked_io import (
    _canonical_csr,
    _chunk_payload,
    _decode_block,
    _np_dtype,
    read_sparse_chunked,
    read_sparse_rows,
    read_sparse_rows_iter,
    read_sparse_slice,
    storage_dtype,
)
from cytome.io.compression import compress_blob, decompress_blob  # noqa: F401


class MeasurementLayer:
    """Lazy wrapper around chunked sparse matrix storage."""

    def __init__(self, conn: sqlite3.Connection, matrix_name: str) -> None:
        self._conn = conn
        _handles.bind(self, conn)
        self._matrix_name = matrix_name
        self._meta = self._load_meta()

    def _rebound(self) -> None:
        """Called after the dataset reopened its connection (``filter_cells``,
        ``reopen``): the matrix may have changed shape, so its metadata is read
        again. A matrix that no longer exists keeps its old metadata, and the
        next read raises ``KeyError``."""
        try:
            self._meta = self._load_meta()
        except (KeyError, sqlite3.Error):
            pass

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

        if isinstance(row_sel, slice):
            start, stop, step = row_sel.indices(n_rows)
            lo, hi = (start, stop) if step > 0 else (stop + 1, start + 1)
            lo, hi = max(lo, 0), max(hi, max(lo, 0))
            if self.is_column_major:
                matrix = self.to_memory()[lo:hi]
            else:
                matrix = read_sparse_slice(self._conn, self._matrix_name, lo, hi)
            if step != 1:
                # rows lo..hi-1 read once; the step (and direction) applied here
                matrix = matrix[np.arange(start - lo, stop - lo, step)]
        elif _is_int(row_sel):
            i = int(row_sel)
            if i < 0:
                i += n_rows
            if i < 0 or i >= n_rows:
                raise IndexError("index out of range")
            matrix = self.rows(np.array([i]))
        elif isinstance(row_sel, (list, np.ndarray)):
            matrix = self.rows(np.asarray(row_sel))
        else:
            raise TypeError(f"Unsupported row selector: {type(row_sel)}")

        if isinstance(col_sel, slice):
            return matrix[:, col_sel].tocsr()
        if isinstance(col_sel, (list, np.ndarray)):
            return matrix[:, col_sel].tocsr()
        if _is_int(col_sel):
            return matrix[:, [int(col_sel)]].tocsr()
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
        """Read specific rows, in the order given, using chunk-selective I/O.

        Only decompresses chunks containing at least one requested row. The
        indices need not be sorted or unique; an index outside the matrix
        raises ``IndexError``.
        """
        idx = np.asarray(indices)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        if self.is_column_major:
            idx = np.asarray(idx, dtype=np.int64)
            n = self.shape[0]
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise IndexError(f"{self._matrix_name}: row index outside its {n} rows")
            return self.to_memory()[idx]
        return read_sparse_rows(self._conn, self._matrix_name, idx)

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
        fine for one marker, quadratic for a marker panel. The columns are
        joined in one ``hstack`` (a LIL matrix filled column by column took
        2.4 s for 50 genes over 160,000 cells, against 0.4 s).
        """
        idx_list = [int(i) for i in indices]
        n_rows, n_cols = self.shape
        if not idx_list:
            return sp.csr_matrix((n_rows, 0), dtype=self.dtype)
        wanted = np.asarray(idx_list, dtype=np.int64)
        if wanted.min() < -n_cols or wanted.max() >= n_cols:
            raise IndexError(f"{self._matrix_name}: column index outside its {n_cols} columns")
        wanted = np.where(wanted < 0, wanted + n_cols, wanted)
        if not self.has_feature_index:
            return self[:, wanted.tolist()]

        picked: dict[int, sp.csc_matrix] = {}
        remaining = set(wanted.tolist())
        for col_start, col_end, chunk in self.iter_columns():
            hits = [j for j in remaining if col_start <= j < col_end]
            if not hits:
                continue
            local = chunk[:, [j - col_start for j in hits]].tocsc()
            for slot, j in enumerate(hits):
                picked[j] = local[:, [slot]]
            remaining.difference_update(hits)
            if not remaining:
                break
        dtype = next(iter(picked.values())).dtype if picked else self.dtype
        cols = [picked.get(int(j), sp.csc_matrix((n_rows, 1), dtype=dtype)) for j in wanted]
        return sp.hstack(cols, format="csc").tocsr()

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

        Committed on return, as one transaction: another connection sees the
        index, and a failure leaves the matrix as it was.
        """
        full_csc = self.to_memory().tocsc()
        chunk_size = int(chunk_size or self._meta.get("chunk_size", 256))
        dtype = storage_dtype(full_csc.dtype)
        # a CSC matrix is the CSR of its transpose: the same canonical arrays
        data, indices, indptr = _canonical_csr(full_csc.T.tocsr(), dtype, self._matrix_name)
        with self._conn:
            self._conn.execute(
                "DELETE FROM matrix_csc_chunks WHERE matrix_name = ?", (self._matrix_name,)
            )
            chunk_idx = 0
            for col_start in range(0, full_csc.shape[1], chunk_size):
                col_end = min(col_start + chunk_size, full_csc.shape[1])
                nnz, d_blob, i_blob, p_blob = _chunk_payload(
                    data, indices, indptr, col_start, col_end, compression, self._matrix_name)
                self._conn.execute(
                    """
                    INSERT INTO matrix_csc_chunks(
                        matrix_name, chunk_idx, col_start, col_end, n_nonzero,
                        data_blob, indices_blob, indptr_blob, dtype, compression
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self._matrix_name, chunk_idx, col_start, col_end, int(nnz),
                     d_blob, i_blob, p_blob, str(dtype), compression),
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
        meta_dtype = self.dtype
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
            c0, c1 = int(col_start), int(col_end)
            chunk = _decode_block(c1 - c0, n_rows, data_blob, indices_blob, indptr_blob,
                                  compression, dtype, meta_dtype,
                                  f"{self._matrix_name} columns {c0}-{c1 - 1}", fmt="csc",
                                  matrix_name=self._matrix_name)
            yield c0, c1, chunk

    def drop_feature_index(self) -> None:
        """Drop CSC chunks and metadata flags (committed on return)."""
        with self._conn:
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


def _is_int(sel) -> bool:
    return isinstance(sel, (int, np.integer)) and not isinstance(sel, (bool, np.bool_))


def _slice_bounds(sel: object, size: int) -> tuple[int, int]:
    """``(start, stop)`` of a step-1 slice or an integer over ``size`` rows,
    negative positions counted from the end."""
    if isinstance(sel, slice):
        start, stop, step = sel.indices(size)
        if step != 1:
            raise ValueError("only step-1 slices are supported here")
        return start, max(start, stop)
    if _is_int(sel):
        sel = int(sel)
        if sel < 0:
            sel += size
        if sel < 0 or sel >= size:
            raise IndexError("index out of range")
        return sel, sel + 1
    raise TypeError(f"Unsupported slice selector: {type(sel)}")
