"""Which objects hold which connection, and which datasets are open.

Every accessor a dataset hands out (``ds.cells``, ``ds.ATAC.counts``,
``ds.embeddings`` ...) keeps the ``sqlite3.Connection`` it was made from.
When the dataset reopens its connection -- around a writer that needs the
file to itself, or after ``filter_cells`` replaces the file -- an accessor
the caller kept in a variable would otherwise hold a closed connection.
Accessors register here when they are made; ``rebind`` moves every live one
to the new connection.

Keyed by ``id(conn)``: the accessors hold the connection strongly, so the id
cannot be reused while any of them is alive, and the entry is dropped when
the dataset closes for good.

An iterator already running (``ds.iter_chunks``) holds a cursor, not an
accessor, and cannot be moved: it stops with SQLite's "Cannot operate on a
closed database".
"""
from __future__ import annotations

import os
import weakref

_USERS: dict[int, "weakref.WeakSet"] = {}
_OPEN: "weakref.WeakSet" = weakref.WeakSet()


def bind(obj, conn) -> None:
    """Record that ``obj`` reads through ``conn`` (as ``obj._conn``)."""
    try:
        _USERS.setdefault(id(conn), weakref.WeakSet()).add(obj)
    except TypeError:  # an object without weakref support: left unbound
        pass


def rebind(old, new) -> None:
    """Point every live accessor of ``old`` at ``new``."""
    users = _USERS.pop(id(old), None)
    if not users:
        return
    for obj in list(users):
        obj._conn = new
        bind(obj, new)


def forget(conn) -> None:
    _USERS.pop(id(conn), None)


def track(dataset) -> None:
    _OPEN.add(dataset)


def open_on(path) -> list:
    """The datasets of this process that have ``path`` open."""
    target = os.path.realpath(str(path))
    out = []
    for ds in list(_OPEN):
        try:
            if os.path.realpath(str(ds.path)) == target and not ds.is_closed:
                out.append(ds)
        except Exception:
            pass
    return out
