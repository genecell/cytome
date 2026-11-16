"""Lazy sparse matrix access for Cytome measurements."""

from __future__ import annotations

from cytome.core import _handles

from typing import Iterator, Tuple

import warnings

import numpy as np
import scipy.sparse as sp
import sqlite3

from cytome.io.chunked_io import (
    _canonical_csr,
    _chunk_payload,
    _decode_block,
    _np_dtype,
    read_sparse_axis,
    read_sparse_chunked,
    read_sparse_chunked_csc,
    read_sparse_rows_iter,
    read_sparse_slice,
    storage_dtype,
    touched_nonzeros,
)
from cytome.io.compression import compress_blob, decompress_blob  # noqa: F401


class FullPassWarning(UserWarning):
    """A read along the axis a matrix is not stored by: every chunk is read once.

    Rows of a column-major matrix (a peak matrix as the quantifier writes it), or
    columns of a row-major one without a column index. Memory stays at the
    result plus one chunk, but the time is a pass over the whole matrix, so a
    loop of such reads (one per batch of cells) repeats that pass per batch.
    """


#: (store file, matrix, axis) already warned about, one warning each per process
_FULL_PASS_REPORTED: set = set()
#: a pass over fewer chunks than this is quick, and not worth a warning
_FULL_PASS_MIN_CHUNKS = 64


def _warn_full_pass(conn, matrix_name: str, read_by: str, stored_by: str, n_chunks: int) -> None:
    if n_chunks < _FULL_PASS_MIN_CHUNKS:
        return
    try:
        path = next((r[2] for r in conn.execute("PRAGMA database_list") if r[1] == "main"), "")
    except sqlite3.Error:
        path = ""
    key = (path, matrix_name, read_by)
    if key in _FULL_PASS_REPORTED:
        return
    _FULL_PASS_REPORTED.add(key)
    warnings.warn(
        f"{matrix_name} is stored by {stored_by}: reading it by {read_by} reads each of its "
        f"{n_chunks:,} chunks once (memory: the result plus one chunk). For repeated reads by "
        f"{read_by}, keep a copy of the matrix stored by {read_by}.",
        FullPassWarning, stacklevel=4)


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
        """``X[rows]`` or ``X[rows, cols]``, as CSR.

        Each selector is an integer, a slice (any step), a sequence of integers
        (any order, repeats allowed) or a boolean mask of the axis's length;
        negative positions count from the end. Reads go through one reader
        (:meth:`_select`): along the axis the matrix is stored by, only the
        chunks that hold a wanted position are read; along the other, every
        chunk is read once and only the wanted positions are kept.
        """
        if not isinstance(key, tuple):
            key = (key, slice(None))
        if len(key) != 2:
            raise IndexError(f"{self._matrix_name}: a matrix takes one or two indices, not {len(key)}")
        row_sel, col_sel = key
        n_rows, n_cols = self.shape
        # a step-1 range of rows from row chunks keeps its range read
        if isinstance(row_sel, slice) and self._has_row_chunks:
            start, stop, step = row_sel.indices(n_rows)
            if step == 1 and not (start == 0 and stop >= n_rows):
                cols = _positions(col_sel, n_cols, "column", self._matrix_name)
                matrix = read_sparse_slice(self._conn, self._matrix_name, start, max(start, stop))
                return matrix if cols is None else matrix[:, cols].tocsr()
        return self._select(_positions(row_sel, n_rows, "row", self._matrix_name),
                            _positions(col_sel, n_cols, "column", self._matrix_name))

    @property
    def _has_row_chunks(self) -> bool:
        return bool(self._meta.get("n_chunks"))

    @property
    def _has_col_chunks(self) -> bool:
        return bool(self._meta.get("has_csc"))

    def _select(self, rows, cols) -> sp.csr_matrix:
        """The one reader behind ``rows()``, ``columns()``, ``X[...]`` and
        ``to_memory()``. ``rows`` and ``cols`` are int64 positions (any order,
        repeats allowed, already in range) or ``None`` for all.

        - Along the axis the matrix is stored by, only the chunks that hold a
          wanted position are read, and the other axis is masked inside each
          chunk. A matrix stored both ways is read by the layout whose chunks
          holding the selection hold fewer entries.
        - With a selection only along the other axis, every chunk is read once
          and the wanted positions kept (a :class:`FullPassWarning`, once, for a
          matrix of 64 chunks or more).
        """
        conn, name = self._conn, self._matrix_name
        n_rows, n_cols = self.shape
        if rows is None and cols is None:
            return self.to_memory()
        shape = (n_rows if rows is None else rows.size, n_cols if cols is None else cols.size)
        if 0 in shape:
            return sp.csr_matrix(shape, dtype=self.dtype)
        by_rows, by_cols = self._has_row_chunks, self._has_col_chunks
        if not (by_rows or by_cols):                     # no chunks: an all-zero matrix
            return sp.csr_matrix(shape, dtype=self.dtype)
        if rows is not None and cols is not None and by_rows and by_cols:
            use = "rows" if touched_nonzeros(conn, name, "rows", rows) <= \
                touched_nonzeros(conn, name, "cols", cols) else "cols"
        elif rows is not None and by_rows:
            use = "rows"
        elif cols is not None and by_cols:
            use = "cols"
        else:
            # the selection is only along the axis the matrix is not stored by
            stored = "cols" if by_cols else "rows"
            n_chunks = int(self._meta.get("csc_n_chunks") or 0) if stored == "cols" else \
                int(self._meta.get("n_chunks") or 0)
            _warn_full_pass(conn, name, "rows" if rows is not None else "columns",
                            "columns" if stored == "cols" else "rows", n_chunks)
            out = read_sparse_axis(conn, name, stored, major=None,
                                   minor=rows if stored == "cols" else cols)
            return out.tocsr()
        if use == "rows":
            return read_sparse_axis(conn, name, "rows", major=rows, minor=cols).tocsr()
        return read_sparse_axis(conn, name, "cols", major=cols, minor=rows).tocsr()

    def to_memory(self) -> sp.csr_matrix:
        """Load entire matrix into memory, as CSR.

        A column-major matrix is read from its column chunks
        (:func:`~cytome.io.chunked_io.read_sparse_chunked_csc`: each placed by
        its span, the destination sized once), then converted; a row-major one
        from its row chunks.
        """
        if self.is_column_major:
            return read_sparse_chunked_csc(self._conn, self._matrix_name).tocsr()
        return read_sparse_chunked(self._conn, self._matrix_name)

    def rows(self, indices: np.ndarray) -> sp.csr_matrix:
        """Read specific rows, in the order given, as CSR.

        ``indices``: integers (any order, repeats allowed; negative ones count
        from the end) or a boolean mask over every row. An index outside the
        matrix raises ``IndexError``.

        From row chunks, only the chunks holding a requested row are read. A
        column-major matrix (CSC chunks only) has no row chunks: every column
        chunk is read once and the requested rows kept, so memory stays at the
        result plus one chunk, but the time is a pass over the whole matrix
        (a :class:`FullPassWarning`, once per matrix and process).
        """
        n_rows = self.shape[0]
        idx = _positions(indices if not _is_int(indices) else [indices], n_rows, "row",
                         self._matrix_name, everything_is_none=False)
        return self._select(idx, None)

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
        """Read selected feature columns across all rows, in the order given, as CSR.

        ``indices``: integers (any order, repeats allowed; negative ones count
        from the end) or a boolean mask over every column. An index outside
        the matrix raises ``IndexError``.

        From column chunks (a column-major matrix, or a feature index), only
        the chunks that hold a requested column are fetched and decoded, each
        once; they are found from the chunks' recorded spans, before any blob
        is read. A matrix without column chunks is read once, row chunk by row
        chunk, keeping the requested columns (a :class:`FullPassWarning`).
        """
        n_cols = self.shape[1]
        idx = _positions(indices if not _is_int(indices) else [indices], n_cols, "column",
                         self._matrix_name, everything_is_none=False)
        return self._select(None, idx)

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


def _positions(sel, n: int, what: str, matrix_name: str = "", everything_is_none: bool = True):
    """The positions ``sel`` names along an axis of ``n``, as an int64 array.

    numpy's rules, the same for rows and columns: an integer, a slice (any
    step), a sequence of integers (any order, repeats allowed) or a boolean
    mask with one value per position; negative positions count from the end;
    a position outside the axis raises ``IndexError``. A full slice gives
    ``None`` (every position, in order) when ``everything_is_none``.
    """
    if isinstance(sel, slice):
        start, stop, step = sel.indices(n)
        if everything_is_none and step == 1 and start == 0 and stop >= n:
            return None
        return np.arange(start, stop, step, dtype=np.int64)
    if _is_int(sel):
        i = int(sel)
        j = i + n if i < 0 else i
        if j < 0 or j >= n:
            raise IndexError(f"{matrix_name}: {what} {i} is outside its {n} {what}s")
        return np.array([j], dtype=np.int64)
    if sel is None or isinstance(sel, (str, bytes, float, bool, np.bool_)):
        raise TypeError(f"Unsupported {what} selector: {type(sel)}")
    arr = np.asarray(sel)
    if arr.dtype == bool:
        if arr.ndim != 1 or arr.size != n:
            raise IndexError(f"{matrix_name}: a boolean {what} mask needs {n} values, not {arr.size}")
        return np.flatnonzero(arr).astype(np.int64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{matrix_name}: {what} positions must be integers or a boolean mask, "
                        f"not {arr.dtype}")
    asked = arr.astype(np.int64).ravel()
    pos = np.where(asked < 0, asked + n, asked)
    bad = (pos < 0) | (pos >= n)
    if bad.any():
        raise IndexError(f"{matrix_name}: {what} {int(asked[bad][0])} is outside its {n} {what}s")
    return pos


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
