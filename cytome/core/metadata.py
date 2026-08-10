"""Arbitrary metadata (uns-equivalent) storage."""

from __future__ import annotations

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


class MetadataStore:
    """JSON-backed metadata key-value store."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        enqueue_write: Optional[Callable[[str, object], None]] = None,
    ) -> None:
        self._conn = conn
        self._enqueue_write = enqueue_write

    def __getitem__(self, key: str) -> Any:
        row = self._conn.execute(
            "SELECT value, value_type FROM _metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return _deserialize_value(row[0], row[1])

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` if not present."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM _metadata WHERE key = ?", (key,)
        ).fetchone()
        return row is not None

    def __setitem__(self, key: str, value: Any) -> None:
        value_json, value_type = _serialize_value(value)
        payload = _MetadataWrite(key=key, value_json=value_json, value_type=value_type)
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
        return [r[0] for r in rows]

    def items(self) -> Iterator[tuple[str, Any]]:
        rows = self._conn.execute("SELECT key, value, value_type FROM _metadata ORDER BY key")
        for key, value, value_type in rows:
            yield key, _deserialize_value(value, value_type)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM _metadata").fetchone()
        return int(row[0]) if row else 0

    def _apply_write(self, payload: _MetadataWrite) -> None:
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


def _serialize_value(value: Any) -> tuple[str, str]:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S", "O"):
            return json.dumps({"data": value.tolist(), "dtype": "str"}), "ndarray_str"
        return json.dumps({"data": value.tolist(), "dtype": str(value.dtype)}), "ndarray"
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


def _deserialize_value(value_json: str, value_type: str) -> Any:
    if value_type == "ndarray_str":
        obj = json.loads(value_json)
        if isinstance(obj, dict) and "data" in obj:
            return np.array(obj["data"], dtype=str)
        return np.array(obj, dtype=str)
    if value_type == "ndarray":
        obj = json.loads(value_json)
        if isinstance(obj, dict) and "data" in obj:
            return np.array(obj["data"], dtype=obj.get("dtype"))
        return np.array(obj)
    if value_type == "bytes":
        obj = json.loads(value_json)
        if isinstance(obj, dict) and "base64" in obj:
            return base64.b64decode(obj["base64"].encode("ascii"))
        return value_json.encode("utf-8")
    if value_type == "dataframe":
        return pd.read_json(value_json, orient="split")
    if value_type == "series":
        return pd.read_json(value_json, orient="index", typ="series")
    return _restore_json_types(json.loads(value_json))


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("U", "S", "O"):
            return {"__ndarray__": obj.tolist(), "dtype": "str"}
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
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
            return np.array(obj["__ndarray__"], dtype=dtype)
        if "__bytes__" in obj:
            return base64.b64decode(obj["__bytes__"].encode("ascii"))
        return {k: _restore_json_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_json_types(v) for v in obj]
    return obj
