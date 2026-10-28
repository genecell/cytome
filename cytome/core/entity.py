"""Entity table wrappers for Cytome."""

from __future__ import annotations

from cytome.core import _handles

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
    #: the SQL type of a column the write creates; None infers it from ``values``
    sql_type: Optional[str] = None


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
        _handles.bind(self, conn)
        self._table_name = table_name
        self._enqueue_write = enqueue_write

    @property
    def columns(self) -> list[str]:
        """Return list of table column names, including columns written and
        not yet flushed."""
        from cytome.io.sqlite_engine import visible_columns
        rows = visible_columns(self._conn, self._table_name)
        cols = [r[1] for r in rows]
        lowered = {c.lower() for c in cols}
        cols += [c for c in self._pending_columns() if c.lower() not in lowered]
        return cols

    def _pending(self):
        """``(dataset, keys)``: the dataset that queued writes for this
        table, and the keys of those still waiting for its flush()."""
        owner = getattr(self._enqueue_write, "__self__", None)
        waiting = getattr(owner, "_pending_writes", None) or {}
        mine = (f"entity:{self._table_name}:", f"entity_table:{self._table_name}")
        return owner, [k for k in waiting if k.startswith(mine)]

    def _pending_columns(self) -> list[str]:
        prefix = f"entity:{self._table_name}:"
        return [k[len(prefix):] for k in self._pending()[1] if k.startswith(prefix)]

    def _settle(self) -> None:
        """Reads go to SQL: write what is waiting for this table first, so a
        column reads back the same before and after an explicit flush()."""
        owner, keys = self._pending()
        if keys:
            owner.flush()

    @property
    def n(self) -> int:
        """Return row count."""
        return int(
            self._conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()[0]
        )

    def resolve_column(self, column_name: str, warn: bool = False):
        """Resolve ``column_name`` to the stored casing, or None if absent.

        SQLite identifiers are case-insensitive: a table holding ``Leiden``
        will happily accept ``UPDATE ... SET leiden = ?`` and route it to
        ``Leiden``. An API that checks membership case-sensitively can only
        disagree with its own storage — which is how ``cells['leiden'] = ...``
        overwrote ``Leiden`` silently while ``cells['leiden']`` raised
        ``KeyError`` on the very column it had just written. Every name-taking
        method resolves through here so Python and SQLite agree.

        With ``warn=True`` a case mismatch emits a warning naming both
        spellings; membership checks resolve silently.
        """
        cols = self.columns
        if column_name in cols:
            return column_name
        lowered = {c.lower(): c for c in cols}
        actual = lowered.get(column_name.lower())
        if actual is not None and warn:
            import warnings
            warnings.warn(
                f"{self._table_name}[{column_name!r}] resolves to existing "
                f"column {actual!r} (SQLite column names are "
                f"case-insensitive); using {actual!r}.",
                stacklevel=3,
            )
        return actual

    def __contains__(self, column_name: str) -> bool:
        """Return True if the column exists in the table (case-insensitive,
        matching SQLite's own identifier semantics)."""
        if not isinstance(column_name, str):
            return False
        return self.resolve_column(column_name) is not None

    def __iter__(self):
        """The column names, as a DataFrame iterates its columns. Without it
        ``list(table)`` fell back to ``table[0]``, ``table[1]``, ..., which
        raised on the first index."""
        return iter(self.columns)

    def has_column(self, column_name: str) -> bool:
        """Return True if the column exists in the table.

        Equivalent to ``column_name in entity_table``.
        """
        return column_name in self

    def __getitem__(self, column_name: str) -> np.ndarray:
        """Return a column as a NumPy array."""
        actual = self.resolve_column(column_name, warn=True)
        if actual is None:
            raise KeyError(f"Column {column_name!r} not in {self._table_name}")
        column_name = actual
        self._settle()
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
        from cytome.core.column_meta import column_values
        # missing values become NULL (they were the text 'nan', or failed the
        # flush for pd.NA); a categorical keeps its categories' type
        values_arr, sql_type = column_values(values)
        if values_arr.shape[0] != self.n:
            raise ValueError(
                f"Length mismatch for {self._table_name}.{column_name}: "
                f"expected {self.n}, got {values_arr.shape[0]}"
            )
        # Canonicalise BEFORE either write path, so the enqueue key, the
        # category-invalidation lookup and the SQL all see the stored casing.
        actual = self.resolve_column(column_name, warn=True)
        if actual is not None:
            column_name = actual
        key_col = _infer_primary_key(self._conn, self._table_name)
        if column_name.lower() == str(key_col).lower():
            raise ValueError(
                f"{column_name!r} is the key of {self._table_name}: the rows of every matrix, "
                f"embedding and fragment refer to it, so it is not written as a column")
        if self._enqueue_write is not None:
            self._enqueue_write(
                f"entity:{self._table_name}:{column_name}",
                _EntityWrite(self._table_name, column_name, values_arr, sql_type),
            )
            return
        self._apply_column_writes([(column_name, values_arr, sql_type)])

    def query(self, expression: str) -> pd.DataFrame:
        """Run SQL WHERE query and return matching rows as DataFrame."""
        self._settle()
        sql = f"SELECT * FROM {self._table_name} WHERE {expression}"
        return self._hide_shim_rowid(pd.read_sql_query(sql, self._conn))

    def _hide_shim_rowid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the rowid a read shim's view exposes (not a column of the table)."""
        if self._table_name in getattr(self._conn, "shim_views", ()) and "rowid" in df.columns:
            return df.drop(columns="rowid")
        return df

    def query_mask(self, expression: str) -> np.ndarray:
        """Return boolean mask for rows satisfying SQL expression."""
        self._settle()
        key_col = _infer_primary_key(self._conn, self._table_name)
        all_ids = self._conn.execute(
            f"SELECT {key_col} FROM {self._table_name} ORDER BY ROWID"
        ).fetchall()
        hit_ids = self._conn.execute(
            f"SELECT {key_col} FROM {self._table_name} WHERE {expression}"
        ).fetchall()
        id_set = {h[0] for h in hit_ids}
        return np.asarray([i[0] in id_set for i in all_ids], dtype=bool)

    def to_pandas(self, restore_dtypes: bool = False) -> pd.DataFrame:
        """Materialize table as pandas DataFrame.

        ``restore_dtypes=True`` gives each column the pandas dtype recorded
        for it when it was written from a DataFrame (``from_anndata``,
        ``update_from_anndata``): categoricals with their categories in their
        order (and ``ordered``), nullable integers and booleans -- what
        ``to_anndata`` puts in ``obs`` and ``var``. By default the values come
        as SQLite stores them.
        """
        self._settle()
        df = self._hide_shim_rowid(pd.read_sql_query(f"SELECT * FROM {self._table_name}", self._conn))
        if restore_dtypes:
            from cytome.core.column_meta import restore_column_dtypes
            df = restore_column_dtypes(self._conn, self._table_name, df)
        return df

    def categories(self, column_name: str):
        """The categories of a categorical column, in their stored order.

        The order is the one the column had as a pandas categorical when it
        was written (``from_anndata`` of an ``obs`` column, say), kept in
        ``_column_meta``; ``None`` when the column was not written as a
        categorical. Not the display order and colors of
        :meth:`CytomeDataset.set_categories` (``ds.get_categories``), which a
        user sets for plotting.
        """
        actual = self.resolve_column(column_name)
        if actual is None:
            raise KeyError(f"Column {column_name!r} not in {self._table_name}")
        from cytome.core.column_meta import read_records
        import json
        for col, dtype, cats in read_records(self._conn, self._table_name):
            if col == actual and dtype in ("categorical", "ordered_categorical"):
                return json.loads(cats) if cats else []
        return None

    def _apply_column_write(self, column_name: str, values: np.ndarray) -> None:
        self._apply_column_writes([(column_name, values)])

    def _apply_column_writes(self, writes) -> None:
        """Write several columns ``[(name, values), ...]`` in one pass.

        SQLite stores a row's columns together, so changing one value
        rewrites the whole row: a column write costs a full copy of the
        table, whatever its width, and in WAL mode a second one at the
        checkpoint. One UPDATE setting all the columns pays that once.
        A name given twice keeps its last values, as in sequence.
        """
        writes = [tuple(w) + (None,) * (3 - len(w)) for w in writes]
        names = []
        for column_name, values, sql_type in writes:
            actual = self.resolve_column(column_name)
            if actual is None:
                sqlite_type = sql_type or _numpy_to_sqlite_type(np.asarray(values).dtype)
                # No blanket except here. The old `except Exception: pass` was
                # load-bearing for a bug: it swallowed SQLite's *duplicate column*
                # error when a case-variant of the column already existed, and the
                # UPDATE below then overwrote that column silently. It also
                # swallowed a locked database and a full disk. Resolution above
                # makes the duplicate case unreachable; everything else should be
                # heard about.
                self._conn.execute(
                    f"ALTER TABLE {self._table_name} ADD COLUMN {_quote_ident(column_name)} {sqlite_type}"
                )
                actual = column_name
            names.append(actual)
        last = {}
        for i, name in enumerate(names):
            last[name.lower()] = i
        keep = sorted(last.values())
        names = [names[i] for i in keep]
        arrays = [np.asarray(writes[i][1]) for i in keep]
        key_col = _infer_primary_key(self._conn, self._table_name)
        keys = [k[0] for k in self._conn.execute(
            f"SELECT {_quote_ident(key_col)} FROM {self._table_name} ORDER BY ROWID"
        )]
        sets = ", ".join(f"{_quote_ident(n)} = ?" for n in names)
        self._conn.executemany(
            f"UPDATE {self._table_name} SET {sets} WHERE {_quote_ident(key_col)} = ?",
            _rows(arrays, keys),
        )


def _as_sql_values(a: np.ndarray) -> list:
    # tolist() gives each element as .item() would, for every dtype but
    # object, whose elements it returns as they are (np.int64 among them,
    # which sqlite3 cannot bind).
    if a.dtype == object:
        return [v.item() if hasattr(v, "item") else v for v in a]
    return a.tolist()


def _rows(arrays, keys, block: int = 65536):
    """``(v1, v2, ..., key)`` per row, converted a block at a time so the
    Python objects of all the rows never exist at once."""
    for lo in range(0, len(keys), block):
        hi = min(lo + block, len(keys))
        cols = [_as_sql_values(a[lo:hi]) for a in arrays]
        yield from zip(*cols, keys[lo:hi])


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
    # the base table's: a read shim's view (temp) reports no primary key
    info = conn.execute(f"PRAGMA main.table_info({table_name})").fetchall() or \
        conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for col in info:
        if col[5] == 1:
            return col[1]
    return info[0][1]


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'
