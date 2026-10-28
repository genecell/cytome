"""Lazy dense embedding access."""

from __future__ import annotations

from cytome.core import _handles

import numpy as np
import sqlite3

from cytome.io.chunked_io import read_dense_chunked, read_dense_rows, read_dense_slice


class EmbeddingArray:
    """Lazy wrapper around chunked dense arrays."""

    def __init__(self, conn: sqlite3.Connection, array_name: str) -> None:
        self._conn = conn
        _handles.bind(self, conn)
        self._array_name = array_name
        self._load_meta()

    def _load_meta(self) -> None:
        row = self._conn.execute(
            "SELECT n_rows, n_cols, dtype FROM embedding_meta WHERE array_name = ?",
            (self._array_name,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Embedding not found: {self._array_name}")
        self._shape = (int(row[0]), int(row[1]))
        self._dtype = np.dtype(row[2])

    def _rebound(self) -> None:
        """The dataset reopened its connection: read the shape again (a
        ``filter_cells`` changes it)."""
        try:
            self._load_meta()
        except (KeyError, sqlite3.Error):
            pass

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple):
            row_sel, col_sel = key
        else:
            row_sel, col_sel = key, slice(None)

        n = self._shape[0]
        if isinstance(row_sel, slice):
            start, stop, step = row_sel.indices(n)
            if step == 1:
                arr = read_dense_slice(self._conn, self._array_name, start, max(start, stop))
            else:
                arr = read_dense_rows(self._conn, self._array_name,
                                      np.arange(start, stop, step, dtype=np.int64))
        elif isinstance(row_sel, (int, np.integer)) and not isinstance(row_sel, (bool, np.bool_)):
            i = int(row_sel)
            if i < 0:
                i += n
            if i < 0 or i >= n:
                raise IndexError("index out of range")
            arr = read_dense_slice(self._conn, self._array_name, i, i + 1)
        elif isinstance(row_sel, (list, np.ndarray)):
            idx = np.asarray(row_sel)
            if idx.dtype == bool:
                if idx.shape[0] != n:
                    raise IndexError(f"boolean mask of length {idx.shape[0]} for {n} rows")
                idx = np.flatnonzero(idx)
            idx = np.asarray(idx, dtype=np.int64)
            idx = np.where(idx < 0, idx + n, idx)
            arr = read_dense_rows(self._conn, self._array_name, idx)
        else:
            raise TypeError(f"Unsupported row selector: {type(row_sel)}")
        return arr[:, col_sel]

    def to_memory(self) -> np.ndarray:
        """Load full embedding array into memory."""
        return read_dense_chunked(self._conn, self._array_name)


def _slice_bounds(sel: object, size: int) -> tuple[int, int]:
    """``(start, stop)`` of a step-1 slice or an integer over ``size`` rows,
    negative positions counted from the end."""
    if isinstance(sel, slice):
        start, stop, step = sel.indices(size)
        if step != 1:
            raise ValueError("only step-1 slices are supported here")
        return start, max(start, stop)
    if isinstance(sel, (int, np.integer)) and not isinstance(sel, (bool, np.bool_)):
        sel = int(sel)
        if sel < 0:
            sel += size
        if sel < 0 or sel >= size:
            raise IndexError("index out of range")
        return sel, sel + 1
    raise TypeError(f"Unsupported row selector: {type(sel)}")
