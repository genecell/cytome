"""Frozen stores: a store that must not change, for a benchmark or a published figure, is frozen.

``cytome.freeze(path, reason)`` records the freeze in the store's manifest and in its provenance, folds the
write-ahead log into the file, and takes the write permission off the file. Every writer is then refused, by
the operating system and not only by cytome: a cytome open is read-only and a write raises
:class:`cytome.ReadOnlyError` naming the freeze; PIASO's Rust quantifier and importer, or the ``sqlite3``
shell, cannot open the file for writing; ``create(path, force=True)`` and every function that writes a store
with ``force=True`` refuse to replace it. ``cytome.unfreeze(path, reason)`` gives the permission back and
records that too.

Why: nine stores had been edited in place in one day, without backups, including a benchmark substrate; nothing
broke, and nothing would have stopped it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

__all__ = ["freeze", "unfreeze", "frozen_info", "describe"]

_KEY = "frozen"


def frozen_info(path) -> Optional[dict]:
    """The freeze record of the store at ``path`` (``at``, ``reason``, ``mode``), or None if it is not frozen.

    Read without writing anything, from the manifest.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        con = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = con.execute("SELECT value FROM _manifest WHERE key = ?", (_KEY,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if not row or row[0] is None:
        return None
    try:
        info = json.loads(row[0])
    except (TypeError, ValueError):
        return {"reason": str(row[0])}
    return info if isinstance(info, dict) else {"reason": str(info)}


def describe(info: dict, path) -> str:
    """How an error names a freeze."""
    at = info.get("at", "an unrecorded time")
    why = f": {info['reason']}" if info.get("reason") else ""
    return f"it is frozen (since {at}{why}); cytome.unfreeze({str(path)!r}, reason=...) makes it writable"


def _log(conn, operation: str, path: Path, parameters: dict) -> None:
    from cytome.core.provenance import ProvenanceLog
    from cytome.io.sqlite_engine import _package_version

    ProvenanceLog(conn).log(
        operation=operation, function_name=f"cytome.{operation}", parameters=parameters,
        package_name="cytome", package_version=_package_version(), input_objects=[path.name],
        output_objects=[path.name])


def freeze(path, reason: str) -> dict:
    """Freeze the store at ``path``: from now on nothing can write it until :func:`unfreeze`.

    Parameters
    ----------
    path : str or Path
        The store. Nothing else may have it open for writing (a running pipeline would fail at its next
        write); readers may keep it open.
    reason : str
        Why, recorded with the time in the manifest and the provenance (``'GRN benchmark substrate'``).

    Returns
    -------
    dict
        The freeze record: ``at``, ``reason`` and ``mode`` (the file's permission bits before).
    """
    p = Path(path)
    if not reason or not str(reason).strip():
        raise ValueError("freeze needs a reason, recorded with the store")
    if frozen_info(p) is not None:
        raise ValueError(f"{p.name} is frozen already")
    from cytome.io.sqlite_engine import _why_not_writable

    why = _why_not_writable(p)
    if why is not None:
        raise PermissionError(f"{p.name} cannot be written, so it cannot be frozen: {why}")
    # a plain connection: the freeze writes its record and nothing else (a cytome open would bring an older
    # store's schema up to this version's)
    conn = sqlite3.connect(str(p), timeout=60)
    mode = stat.S_IMODE(p.stat().st_mode)
    info = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": str(reason),
            "mode": oct(mode)}
    try:
        conn.execute("INSERT OR REPLACE INTO _manifest(key, value) VALUES (?, ?)", (_KEY, json.dumps(info)))
        conn.commit()
        _log(conn, "freeze", p, {"reason": str(reason)})
        conn.commit()
        # everything into the main file, so the file alone holds the store
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return info


def unfreeze(path, reason: str) -> dict:
    """Give a frozen store its write permission back, and record why.

    Returns the freeze record that was lifted.
    """
    p = Path(path)
    if not reason or not str(reason).strip():
        raise ValueError("unfreeze needs a reason, recorded with the store")
    info = frozen_info(p)
    if info is None:
        raise ValueError(f"{p.name} is not frozen")
    try:
        before = int(str(info.get("mode", "")), 8)
    except ValueError:
        before = stat.S_IMODE(p.stat().st_mode) | stat.S_IWUSR
    os.chmod(p, before | stat.S_IWUSR)
    # SQLite gives the -wal and -shm files a reader creates the main file's permissions, so while the store
    # was frozen they came up read-only as well
    for sfx in ("-wal", "-shm"):
        side = Path(str(p) + sfx)
        if side.exists():
            os.chmod(side, stat.S_IMODE(side.stat().st_mode) | (before & 0o222) | stat.S_IWUSR)
    from cytome.io.sqlite_engine import _why_not_writable

    why = _why_not_writable(p)
    if why is not None:
        raise PermissionError(f"{p.name} still cannot be written: {why}")
    conn = sqlite3.connect(str(p), timeout=60)
    try:
        conn.execute("DELETE FROM _manifest WHERE key = ?", (_KEY,))
        conn.commit()
        _log(conn, "unfreeze", p, {"reason": str(reason), "frozen": info})
        conn.commit()
    finally:
        conn.close()
    return info
