"""Arbitrary metadata (uns-equivalent) storage."""

from __future__ import annotations

from cytome.core import _handles

import io
import json
import base64
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import numpy as np
import pandas as pd
import sqlite3


@dataclass
class _MetadataWrite:
    key: str
    value_json: str
    value_type: str
    delete: bool = False
    #: dtype records of a table or a Series (see _dtype_records), else None
    records: Optional[list] = None


#: ``_column_meta`` holds, under ``_metadata:<key>``, the pandas dtypes of a
#: DataFrame or Series stored as metadata. The value itself stays the JSON every
#: earlier cytome reads; only a reader of this version applies the records.
_RECORDS_PREFIX = "_metadata:"
_INDEX = "\x00index"            # the record of the index (no column has this name)
_VALUES = "\x00values"          # the record of a Series' values


class MetadataStore:
    """JSON-backed metadata key-value store."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        enqueue_write: Optional[Callable[[str, object], None]] = None,
    ) -> None:
        self._conn = conn
        _handles.bind(self, conn)
        self._enqueue_write = enqueue_write

    def _pending(self) -> dict:
        """``{key: _MetadataWrite}`` queued in the owning dataset and not yet
        flushed: reads see them, as they see a pending entity column."""
        owner = getattr(self._enqueue_write, "__self__", None)
        waiting = getattr(owner, "_pending_writes", None) or {}
        return {k[len("metadata:"):]: v for k, v in waiting.items()
                if k.startswith("metadata:") and isinstance(v, _MetadataWrite)}

    def _records(self, key: str, value_type: str):
        if value_type not in ("dataframe", "series"):
            return None
        from cytome.core.column_meta import read_records
        return read_records(self._conn, _RECORDS_PREFIX + key) or None

    def __getitem__(self, key: str) -> Any:
        waiting = self._pending().get(key)
        if waiting is not None:
            if waiting.delete:
                raise KeyError(key)
            return _deserialize_value(waiting.value_json, waiting.value_type,
                                      records=waiting.records, conn=self._conn)
        row = self._conn.execute(
            "SELECT value, value_type FROM _metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return _deserialize_value(row[0], row[1], records=self._records(key, row[1]),
                                  conn=self._conn)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` if not present."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        waiting = self._pending().get(key)
        if waiting is not None:
            return not waiting.delete
        row = self._conn.execute(
            "SELECT 1 FROM _metadata WHERE key = ?", (key,)
        ).fetchone()
        return row is not None

    #: Past this, say so. The metadata table stores JSON text, and an ndarray
    #: serialises at roughly 20 bytes per float32 element -- a full-resolution
    #: tissue image lands in the hundreds of megabytes, which SQLite will
    #: store but which makes the file slow and enormous for something that
    #: belongs in a blob.
    WARN_JSON_BYTES = 64 * 1024 * 1024

    #: Past this, refuse. SQLite's own ``SQLITE_MAX_LENGTH`` (1 GB by default)
    #: rejects the row with "string or blob too big", and above 2 GB Python's
    #: ``sqlite3`` raises ``OverflowError: string longer than INT_MAX bytes``
    #: from deep inside the write -- which is how a 6000x6000x3 float32 image
    #: used to fail while the same image as uint8 squeaked under, a size cliff
    #: that reads as a mysterious dtype dependency. Fail before either, and
    #: name the place bulk arrays belong. Deliberately not lower: a payload
    #: SQLite would have accepted must keep working, with a warning.
    MAX_JSON_BYTES = 900 * 1024 * 1024

    _WHERE_BULK_GOES = (
        "The metadata table stores JSON values, not bulk arrays. For tissue "
        "images use ds.add_spatial_image(library, key, array) -- stored as a "
        "compressed blob and returned by ds.spatial_images.as_uns() in the "
        "scanpy convention; for other large arrays use a matrix or an "
        "embedding.")

    def __setitem__(self, key: str, value: Any) -> None:
        value_json, value_type = _serialize_value(value)
        records = _dtype_records(value)
        n_bytes = len(value_json)
        if n_bytes > self.MAX_JSON_BYTES:
            raise ValueError(
                f"metadata[{key!r}] serialises to {n_bytes / 1e6:.0f} MB of "
                f"JSON, past what SQLite will store in one value. "
                f"{self._WHERE_BULK_GOES}")
        if n_bytes > self.WARN_JSON_BYTES:
            import warnings as _warnings
            _warnings.warn(
                f"metadata[{key!r}] serialises to {n_bytes / 1e6:.0f} MB of "
                f"JSON. It will be stored, but slowly and at that size on "
                f"disk. {self._WHERE_BULK_GOES}",
                stacklevel=2)
        payload = _MetadataWrite(key=key, value_json=value_json, value_type=value_type,
                                 records=records)
        if self._enqueue_write is not None:
            self._enqueue_write(f"metadata:{key}", payload)
            return
        self._apply_write(payload)

    def __delitem__(self, key: str) -> None:
        payload = _MetadataWrite(key=key, value_json="", value_type="", delete=True)
        if self._enqueue_write is not None:
            self._enqueue_write(f"metadata:{key}", payload)
            return
        self._apply_write(payload)

    def keys(self) -> list[str]:
        rows = self._conn.execute("SELECT key FROM _metadata ORDER BY key").fetchall()
        keys = {r[0] for r in rows}
        for k, w in self._pending().items():
            if w.delete:
                keys.discard(k)
            else:
                keys.add(k)
        return sorted(keys)

    def items(self) -> Iterator[tuple[str, Any]]:
        waiting = self._pending()
        if not waiting:
            rows = self._conn.execute(
                "SELECT key, value, value_type FROM _metadata ORDER BY key").fetchall()
            for key, value, value_type in rows:
                yield key, _deserialize_value(value, value_type,
                                              records=self._records(key, value_type),
                                              conn=self._conn)
            return
        for key in self.keys():
            yield key, self[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        if self._pending():
            return len(self.keys())
        row = self._conn.execute("SELECT COUNT(*) FROM _metadata").fetchone()
        return int(row[0]) if row else 0

    def _apply_write(self, payload: _MetadataWrite) -> None:
        from cytome.core.column_meta import store_records
        self._conn.execute("DELETE FROM _column_meta WHERE table_name = ?",
                           (_RECORDS_PREFIX + payload.key,))
        if payload.delete:
            self._conn.execute("DELETE FROM _metadata WHERE key = ?", (payload.key,))
            return
        self._conn.execute(
            """
            INSERT OR REPLACE INTO _metadata(key, value, value_type)
            VALUES (?, ?, ?)
            """,
            (payload.key, payload.value_json, payload.value_type),
        )
        if payload.records:
            store_records(self._conn, _RECORDS_PREFIX + payload.key, payload.records)


def _dtype_records(value: Any):
    """The dtypes JSON cannot carry, for a DataFrame or a Series: its columns'
    (categoricals, nullable types, datetimes) and its index's."""
    from cytome.core.column_meta import column_records, dtype_record
    if isinstance(value, pd.DataFrame):
        out = column_records(value)
    elif isinstance(value, pd.Series):
        rec = dtype_record(value.reset_index(drop=True))
        out = [] if rec is None else [(_VALUES, rec[0], rec[1])]
    else:
        return None
    rec = dtype_record(pd.Series(value.index))
    out.append((_INDEX, *(rec if rec is not None else ("object", None))))
    return out


def _array_dtype_spec(dtype: np.dtype):
    """A dtype as JSON can hold it: its name, or for a structured dtype (a
    record array such as scanpy's ``uns['rank_genes_groups']['names']``) its
    fields. ``str(dtype)`` of a structured dtype is no name numpy reads back."""
    return dtype.descr if dtype.fields is not None else str(dtype)


def _array_dtype(spec):
    """The dtype of :func:`_array_dtype_spec`, or of what 0.3.8 and earlier wrote
    for a structured dtype (its ``str()``, a Python literal of its fields)."""
    if spec is None:
        return None
    if isinstance(spec, list):
        return np.dtype([_tuplify(f) for f in spec])
    if isinstance(spec, str) and spec.lstrip().startswith("["):
        import ast
        return np.dtype([_tuplify(f) for f in ast.literal_eval(spec)])
    return np.dtype(spec)


def _tuplify(field):
    """A field of a dtype descr read from JSON: lists back to tuples."""
    name, kind, *shape = field
    if isinstance(kind, list):
        kind = [_tuplify(f) for f in kind]
    return (name, kind, *[tuple(x) if isinstance(x, list) else x for x in shape])


def _as_record(row, dtype: np.dtype):
    if dtype.fields is None:
        return row
    return tuple(_as_record(v, dtype.fields[n][0]) for v, n in zip(row, dtype.names))


def _array_from(data, spec) -> np.ndarray:
    dtype = _array_dtype(spec)
    if dtype is not None and dtype.fields is not None:
        return np.array([_as_record(r, dtype) for r in data], dtype=dtype)
    return np.array(data, dtype=dtype)


def _serialize_value(value: Any) -> tuple[str, str]:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S", "O"):
            return json.dumps({"data": value.tolist(), "dtype": "str"}), "ndarray_str"
        return (json.dumps({"data": value.tolist(), "dtype": _array_dtype_spec(value.dtype)},
                           default=_json_fallback), "ndarray")
    if isinstance(value, bytes):
        return json.dumps({"base64": base64.b64encode(value).decode("ascii")}), "bytes"
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient="split"), "dataframe"
    if isinstance(value, pd.Series):
        return value.to_json(orient="index"), "series"
    if isinstance(value, (dict, list, str, int, float, bool, type(None))):
        return json.dumps(value, default=_json_fallback), _infer_type(value)
    try:
        return json.dumps(value, default=_json_fallback), "object"
    except TypeError as exc:
        raise TypeError(
            f"Cannot serialize value of type {type(value).__name__} to JSON. "
            "Convert to a dict, list, or numpy array first. "
            f"Original error: {exc}"
        ) from exc


def _deserialize_value(value_json: str, value_type: str, records=None, conn=None) -> Any:
    if value_type == "ndarray_str":
        obj = json.loads(value_json)
        if isinstance(obj, dict) and "data" in obj:
            return np.array(obj["data"], dtype=str)
        return np.array(obj, dtype=str)
    if value_type == "ndarray":
        obj = json.loads(value_json)
        if isinstance(obj, dict) and "data" in obj:
            return _array_from(obj["data"], obj.get("dtype"))
        return np.array(obj)
    if value_type == "bytes":
        obj = json.loads(value_json)
        if isinstance(obj, dict) and "base64" in obj:
            return base64.b64decode(obj["base64"].encode("ascii"))
        return value_json.encode("utf-8")
    # read_json takes a buffer, not the JSON text itself: pandas 3 reads a
    # string as a path, and a stored DataFrame came back as FileNotFoundError.
    # Values and labels are read as the JSON holds them (no guessing: "01" came
    # back as 1, and a table keyed by cluster labels stopped matching them);
    # the records restore what JSON cannot hold (categoricals, nullable types).
    if value_type == "dataframe":
        df = pd.read_json(io.StringIO(value_json), orient="split", dtype=False,
                          convert_axes=False, convert_dates=False)
        if records:
            from cytome.core.column_meta import apply_records, restore_series
            cols = [r for r in records if r[0] != _INDEX]
            df = apply_records(conn, df, cols)
            for name, tag, cats in records:
                if name == _INDEX and tag != "object":
                    df.index = pd.Index(restore_series(conn, pd.Series(df.index), tag, cats))
        return df
    if value_type == "series":
        if not records:
            # written before the records: its index went out as JSON keys
            # (text), and only conversion can bring numbers back
            return pd.read_json(io.StringIO(value_json), orient="index", typ="series")
        from cytome.core.column_meta import restore_series
        out = pd.read_json(io.StringIO(value_json), orient="index", typ="series", dtype=False,
                           convert_axes=False, convert_dates=False)
        for name, tag, cats in records:
            if name == _VALUES:
                out = restore_series(conn, out, tag, cats)
            elif name == _INDEX and tag not in ("object", "string"):
                out.index = pd.Index(restore_series(conn, pd.Series(out.index), tag, cats))
        return out
    return _restore_json_types(json.loads(value_json))


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("U", "S", "O"):
            return {"__ndarray__": obj.tolist(), "dtype": "str"}
        return {"__ndarray__": obj.tolist(), "dtype": _array_dtype_spec(obj.dtype)}
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _infer_type(value: Any) -> str:
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return "scalar"


def _restore_json_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            dtype = obj.get("dtype")
            if dtype == "str":
                return np.array(obj["__ndarray__"], dtype=str)
            return _array_from(obj["__ndarray__"], dtype)
        if "__bytes__" in obj:
            return base64.b64decode(obj["__bytes__"].encode("ascii"))
        return {k: _restore_json_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_json_types(v) for v in obj]
    return obj
