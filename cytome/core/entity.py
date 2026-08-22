"""Entity table wrappers for Cytome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import sqlite3


@dataclass
class _EntityWrite:
    table_name: str
    column_name: str
    values: np.ndarray


class EntityTable:
    """SQL-backed metadata table for cells/genes/peaks/samples.

    Parameters
    ----------
    conn
        Open SQLite connection.
    table_name
        Backing SQL table name.
    enqueue_write
        Optional callback for write-behind caching.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        enqueue_write: Optional[Callable[[str, object], None]] = None,
    ) -> None:
        self._conn = conn
        self._table_name = table_name
        self._enqueue_write = enqueue_write

    @property
    def columns(self) -> list[str]:
        """Return list of table column names."""
        rows = self._conn.execute(f"PRAGMA table_info({self._table_name})").fetchall()
        return [r[1] for r in rows]

    @property
    def n(self) -> int:
        """Return row count."""
        return int(
            self._conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()[0]
        )

    def __contains__(self, column_name: str) -> bool:
        """Return True if the column exists in the table."""
        return column_name in self.columns

    def has_column(self, column_name: str) -> bool:
        """Return True if the column exists in the table.

        Equivalent to ``column_name in entity_table``.
        """
        return column_name in self.columns

    def __getitem__(self, column_name: str) -> np.ndarray:
        """Return a column as a NumPy array."""
        if column_name not in self.columns:
            raise KeyError(f"Column {column_name!r} not in {self._table_name}")
        rows = self._conn.execute(
            f"SELECT {_quote_ident(column_name)} FROM {self._table_name} ORDER BY ROWID"
        ).fetchall()
        return np.asarray([r[0] for r in rows])

    def __setitem__(self, column_name: str, values: Iterable[object]) -> None:
        """Add or update a column.

        Parameters
        ----------
        column_name
            Name of the column.
        values
            Values with length equal to row count.
        """
        values_arr = np.asarray(list(values))
        if values_arr.shape[0] != self.n:
            raise ValueError(
                f"Length mismatch for {self._table_name}.{column_name}: "
                f"expected {self.n}, got {values_arr.shape[0]}"
            )
        if self._enqueue_write is not None:
            self._enqueue_write(
                f"entity:{self._table_name}:{column_name}",
                _EntityWrite(self._table_name, column_name, values_arr),
            )
            return
        self._apply_column_write(column_name, values_arr)

    def query(self, expression: str) -> pd.DataFrame:
        """Run SQL WHERE query and return matching rows as DataFrame."""
        sql = f"SELECT * FROM {self._table_name} WHERE {expression}"
        return pd.read_sql_query(sql, self._conn)

    def query_mask(self, expression: str) -> np.ndarray:
        """Return boolean mask for rows satisfying SQL expression."""
        key_col = _infer_primary_key(self._conn, self._table_name)
        all_ids = self._conn.execute(
            f"SELECT {key_col} FROM {self._table_name} ORDER BY ROWID"
        ).fetchall()
        hit_ids = self._conn.execute(
            f"SELECT {key_col} FROM {self._table_name} WHERE {expression}"
        ).fetchall()
        id_set = {h[0] for h in hit_ids}
        return np.asarray([i[0] in id_set for i in all_ids], dtype=bool)

    def to_pandas(self) -> pd.DataFrame:
        """Materialize table as pandas DataFrame."""
        return pd.read_sql_query(f"SELECT * FROM {self._table_name}", self._conn)

    def _apply_column_write(self, column_name: str, values: np.ndarray) -> None:
        if column_name not in self.columns:
            sqlite_type = _numpy_to_sqlite_type(values.dtype)
            try:
                self._conn.execute(
                    f"ALTER TABLE {self._table_name} ADD COLUMN {_quote_ident(column_name)} {sqlite_type}"
                )
            except Exception:
                pass  # column may already exist
        key_col = _infer_primary_key(self._conn, self._table_name)
        keys = self._conn.execute(
            f"SELECT {_quote_ident(key_col)} FROM {self._table_name} ORDER BY ROWID"
        ).fetchall()
        self._conn.executemany(
            f"UPDATE {self._table_name} SET {_quote_ident(column_name)} = ? WHERE {_quote_ident(key_col)} = ?",
            [(v.item() if hasattr(v, "item") else v, k[0]) for v, k in zip(values, keys)],
        )


def _numpy_to_sqlite_type(dtype: np.dtype) -> str:
    # bool BEFORE integer: np.issubdtype(np.bool_, np.integer) is False, so a
    # boolean column used to fall through to TEXT and store '1'/'0'. Read back
    # that is a <U1 array, and a non-empty string is truthy, so
    # `col.astype(bool)` returned ALL TRUE -- which silently turned
    # highly_variable into "every gene" on any file where INFOG happened to
    # create the column. Which files were affected depended on write order,
    # because ALTER TABLE ADD COLUMN only fires when the column is absent, so
    # whichever writer got there first fixed the type.
    if np.issubdtype(dtype, np.bool_):
        return "INTEGER"
    if np.issubdtype(dtype, np.integer):
        return "INTEGER"
    if np.issubdtype(dtype, np.floating):
        return "REAL"
    return "TEXT"


def _infer_primary_key(conn: sqlite3.Connection, table_name: str) -> str:
    info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for col in info:
        if col[5] == 1:
            return col[1]
    return info[0][1]


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'
