"""Lazy dense embedding access."""

from __future__ import annotations

import numpy as np
import sqlite3

from cytome.io.chunked_io import read_dense_chunked, read_dense_slice


class EmbeddingArray:
    """Lazy wrapper around chunked dense arrays."""

    def __init__(self, conn: sqlite3.Connection, array_name: str) -> None:
        self._conn = conn
        self._array_name = array_name
        row = conn.execute(
            "SELECT n_rows, n_cols, dtype FROM embedding_meta WHERE array_name = ?",
            (array_name,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Embedding not found: {array_name}")
        self._shape = (int(row[0]), int(row[1]))
        self._dtype = np.dtype(row[2])

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

        row_start, row_end = _slice_bounds(row_sel, self._shape[0])
        arr = read_dense_slice(self._conn, self._array_name, row_start, row_end)
        return arr[:, col_sel]

    def to_memory(self) -> np.ndarray:
        """Load full embedding array into memory."""
        return read_dense_chunked(self._conn, self._array_name)


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
    raise TypeError(f"Unsupported row selector: {type(sel)}")
