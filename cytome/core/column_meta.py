"""The pandas types of a table's columns, recorded beside the table.

SQLite keeps a column's values and a declared affinity, not a pandas dtype:
categories, their order, nullable integers and booleans would come back as
plain values. ``_column_meta(table_name, column_name, dtype, categories)``
records them, and :func:`restore_column_dtypes` rebuilds them on read.

A categorical column stores its *values* in the type of its categories
(INTEGER for integer and boolean categories, REAL for float, TEXT otherwise),
so every reader that does not know this table -- raw SQL, the R package, an
older cytome -- still sees the labels themselves, never codes. Stores written
before cytome 0.3.9 kept such values as TEXT (``"1"`` for the category ``1``,
``"1"`` for ``True``); the restore matches a stored value to its category by
value first, then by the text SQLite made of it, so those stores come back
exactly too. (They came back all NaN: a categorical with integer, boolean or
float categories lost the whole column in every ``to_anndata``.)

The record can be stale: a later write of the column's values does not update
it. A value outside the recorded categories is kept, as a category after the
recorded ones, with a warning that names the column and the values; the R
package reads it the same way.
"""
from __future__ import annotations

import json
import sqlite3
import warnings

import numpy as np
import pandas as pd


def sql_type_of(series: pd.Series) -> str:
    """The SQLite column type a pandas column is stored in.

    **TEXT is the fallback, not INTEGER**: an unrecognised dtype loses its
    type rather than its value, which is the recoverable direction (pandas 3's
    ``StringDtype`` once missed every branch and was declared INTEGER).
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return sql_type_of(pd.Series(dtype.categories))
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    return "TEXT"


def column_values(values) -> tuple[np.ndarray, str]:
    """``(array, sql_type)``: a column's values as SQLite binds them.

    Missing values (``None``, ``NaN``, ``pd.NA``, ``NaT``) become ``None``,
    which SQL spells NULL. An object or categorical column with NaN was stored
    as the text ``'nan'`` (a fake group in COSG and in plots), and a nullable
    ``Int64``, ``string`` or ``boolean`` column holding ``pd.NA`` failed the
    flush. A categorical keeps its category values (stored in the type of its
    categories); a plain numeric array goes through unchanged.
    """
    if isinstance(values, np.ndarray) and values.ndim == 1 and values.dtype.kind in "biufUS":
        # what np.asarray(list(values)) gives for such an array, without one
        # Python object per row; a copy, so a later change to the caller's
        # array does not reach the pending write
        arr = values.copy()
        return arr, sql_type_of(pd.Series(arr[:0]))
    if isinstance(values, pd.Series):
        series = values.reset_index(drop=True)
    elif isinstance(values, (pd.Index, pd.Categorical, pd.api.extensions.ExtensionArray, np.ndarray)):
        series = pd.Series(values)
    else:
        series = pd.Series(list(values))
    sql_type = sql_type_of(series)
    missing = series.isna().to_numpy()
    if series.dtype.kind in "biuf" and not missing.any():
        return series.to_numpy().copy(), sql_type
    obj = series.astype(object).to_numpy(copy=True)
    if missing.any():
        obj[missing] = None
    return obj, sql_type


def dtype_record(series: pd.Series):
    """``(dtype tag, categories JSON or None)`` of a pandas column, or None
    for a dtype that is not recorded (object columns keep their JSON types)."""
    categories = None
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = [_jsonable(c) for c in series.cat.categories.tolist()]
        categories = json.dumps(cats, default=str)
        return ("ordered_categorical" if bool(series.cat.ordered) else "categorical"), categories
    if str(series.dtype) == "boolean":
        return "boolean", None                     # nullable: NA stays NA
    if pd.api.types.is_bool_dtype(series.dtype):
        return "bool", None
    if pd.api.types.is_integer_dtype(series.dtype):
        return str(series.dtype), None
    if pd.api.types.is_float_dtype(series.dtype):
        return str(series.dtype), None
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime", None
    if pd.api.types.is_string_dtype(series.dtype):
        return "string", None
    return None


def column_records(df: pd.DataFrame) -> list:
    """``[(column, dtype tag, categories JSON)]`` for the columns of ``df``."""
    out = []
    for col in df.columns:
        rec = dtype_record(df[col])
        if rec is not None:
            out.append((str(col), rec[0], rec[1]))
    return out


def store_column_meta(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    """Record the pandas dtype of each column of ``df`` as stored in ``table_name``."""
    store_records(conn, table_name, column_records(df))


def store_records(conn: sqlite3.Connection, table_name: str, records) -> None:
    conn.execute("DELETE FROM _column_meta WHERE table_name = ?", (table_name,))
    conn.executemany(
        """
        INSERT OR REPLACE INTO _column_meta(table_name, column_name, dtype, categories)
        VALUES (?, ?, ?, ?)
        """,
        [(table_name, c, t, cats) for c, t, cats in records],
    )


def _jsonable(value):
    """A category as JSON can hold it (numpy scalars to Python ones)."""
    return value.item() if hasattr(value, "item") else value


def _sqlite_text(conn: sqlite3.Connection, values) -> list:
    """The text SQLite makes of each value when it is stored in a TEXT column
    (``CAST(? AS TEXT)``: 15 significant digits for a float, ``'1'`` for True)."""
    out = []
    values = list(values)
    for i in range(0, len(values), 400):
        part = values[i:i + 400]
        sql = " UNION ALL ".join(["SELECT CAST(? AS TEXT)"] * len(part))
        out.extend(r[0] for r in conn.execute(sql, [_bindable(v) for v in part]))
    return out


def _bindable(v):
    v = v.item() if hasattr(v, "item") else v
    return v if isinstance(v, (int, float, str, bytes, type(None))) else str(v)


def restore_categorical(conn, values: pd.Series, categories: list, ordered: bool,
                        what: str | None = None) -> pd.Categorical:
    """The stored ``values`` of a categorical column as the categorical they were.

    Matched to a category by value (keeping ``True`` and ``1`` apart), then,
    for what does not match (stores written before 0.3.9 kept every category
    as text), by the text SQLite made of the category.

    A value that matches no recorded category is kept: it becomes a category
    after the recorded ones, in the order the values first appear, with a
    warning that names the column (``what``, else the Series' name) and the
    values. The recorded categories keep their positions, so a colour list
    stored for them still fits them. (Such a value was made missing: it was
    lost, and could not be told from a missing label.) A missing value stays
    missing.
    """
    cats = list(categories)
    bool_cats = bool(cats) and all(isinstance(c, bool) for c in cats)
    exact = {}
    for i, c in enumerate(cats):
        exact.setdefault((isinstance(c, bool), c), i)
    by_text = None
    extra = []
    codes_of_unique = []
    uniq_codes, uniques = pd.factorize(values, use_na_sentinel=True)
    for u in uniques:
        u = u.item() if hasattr(u, "item") else u
        if isinstance(u, float) and u.is_integer() and not any(isinstance(c, float) for c in cats):
            u = int(u)                                 # an INTEGER column read with NULLs
        as_bool = bool_cats and not isinstance(u, str) and u in (0, 1)
        if as_bool:
            code = exact.get((True, bool(u)), -1)
        else:
            code = exact.get((isinstance(u, bool), u), -1)
        if code < 0:
            if by_text is None:
                by_text = {}
                for i, t in enumerate(_sqlite_text(conn, cats)):
                    by_text.setdefault(t, i)
            t = u if isinstance(u, str) else _sqlite_text(conn, [u])[0]
            code = by_text.get(t, -1)
        if code < 0:                                   # in no recorded category: kept
            code = len(cats) + len(extra)
            extra.append(bool(u) if as_bool else u)
        codes_of_unique.append(code)
    lookup = np.asarray(codes_of_unique + [-1], dtype=np.int64)    # index -1 -> missing
    codes = lookup[np.asarray(uniq_codes, dtype=np.int64)]
    if extra:
        name = what if what is not None else getattr(values, "name", None)
        shown = ", ".join(str(v) for v in extra[:3]) + (", ..." if len(extra) > 3 else "")
        warnings.warn(
            f"cytome: {int((codes >= len(cats)).sum())} values of "
            f"{name if name is not None else 'a categorical column'} are not among its "
            f"recorded categories ({shown}); they are kept as extra categories, after the "
            f"recorded ones", UserWarning, stacklevel=2)
    cats = cats + extra
    return pd.Categorical.from_codes(codes, categories=pd.Index(cats) if cats else
                                     pd.Index([], dtype=object), ordered=ordered)


def read_records(conn: sqlite3.Connection, table_name: str) -> list:
    try:
        return conn.execute(
            "SELECT column_name, dtype, categories FROM _column_meta WHERE table_name = ?",
            (table_name,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def restore_series(conn, series: pd.Series, dtype: str, categories_json,
                   what: str | None = None) -> pd.Series:
    """One column (or index, as a Series) back to its recorded dtype.

    ``what`` names the column in a warning (``"cells.cell_type"``)."""
    if dtype in {"categorical", "ordered_categorical"}:
        cats = json.loads(categories_json) if categories_json else []
        return pd.Series(restore_categorical(conn, series, cats, dtype == "ordered_categorical",
                                             what=what),
                         index=series.index, name=series.name)
    try:
        if dtype in ("bool", "boolean"):
            if dtype == "boolean" or series.isna().any():
                # a missing value stays missing: astype(bool) made NaN True
                return series.astype("boolean")
            return series.astype(bool)
        if dtype.startswith("Int") or dtype.startswith("UInt"):
            return series.astype(dtype)
        if dtype in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}:
            if series.isna().any():
                return series.astype(dtype.upper().replace("INT", "Int"))
            return series.astype(dtype)
        if dtype in {"float32", "float64"}:
            return series.astype(dtype)
        if dtype == "datetime":
            if pd.api.types.is_numeric_dtype(series.dtype):
                return pd.to_datetime(series, unit="ms")      # to_json writes epoch ms
            return pd.to_datetime(series)
    except Exception:
        pass
    return series


def apply_records(conn, df: pd.DataFrame, records, table: str | None = None) -> pd.DataFrame:
    for col, dtype, categories_json in records:
        if col in df.columns:
            df[col] = restore_series(conn, df[col], dtype, categories_json,
                                     what=f"{table}.{col}" if table else str(col))
    return df


def restore_column_dtypes(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """``df`` (read from ``table_name``) with the pandas dtypes recorded for it."""
    return apply_records(conn, df, read_records(conn, table_name), table=table_name)
