"""Hand a matrix's stored chunks, as compressed bytes, to a compute kernel.

A kernel in another library (PIASO's Rust SVD engine) must not open a store
that this Python process holds: it would bring a second copy of SQLite, which
cannot see this process's locks, and closing its file handle drops them.
Instead the kernel asks for chunks and receives the blobs exactly as stored;
it decompresses and computes on its own threads. Reading goes through SQLite
connections of this process -- the same library as the dataset's -- so no
lock is lost and uncommitted work is never visible half-written.

Reads run on ``n_readers`` threads, each with its own read-only connection
(a ``sqlite3`` connection is bound to the thread that made it). While the
kernel works on one batch the readers fetch the next. With ``cache=True``
the blobs are kept for later passes, within a memory budget.
"""
from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_TABLE = {"csr": ("matrix_chunks", "row_start", "row_end"),
          "csc": ("matrix_csc_chunks", "col_start", "col_end")}


def available_memory_bytes() -> int | None:
    """Memory this process can still use: the smaller of the machine's
    available memory and what is left under a cgroup limit (a cluster job's
    ``--mem``). ``None`` if neither can be read."""
    vals = []
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    vals.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    for lim, use in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                      "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit = Path(lim).read_text().strip()
            if limit == "max" or int(limit) >= 1 << 60:
                continue
            vals.append(max(int(limit) - int(Path(use).read_text().strip()), 0))
            break
        except (OSError, ValueError):
            continue
    return min(vals) if vals else None


class ChunkSource:
    """The chunks of one stored matrix, fetched as compressed blobs.

    Parameters
    ----------
    ds
        An open dataset. Pending writes are flushed and committed here, so
        everything written so far is visible to the readers.
    matrix_name
        ``"<modality>_<layer>"``, e.g. ``"ATAC_counts"``.
    n_readers
        Threads that read from the store, each on its own connection.
        More help only when the store is not in memory.
    cache
        Keep fetched blobs for later requests. The cache holds the matrix
        as stored on disk, compressed. If it would exceed
        ``cache_budget_bytes`` (default: half the memory available to this
        process) it is dropped, reads go back to the store, and a warning
        says so.
    cache_budget_bytes
        See ``cache``.

    Protocol used by the kernel: :meth:`meta`, :meth:`chunks`, then
    :meth:`fetch` batch after batch, naming the next batch so it can be read
    while the current one is computed.
    """

    def __init__(self, ds, matrix_name: str, n_readers: int = 4, cache: bool = False,
                 cache_budget_bytes: int | None = None):
        self._ds = ds
        self.path = str(ds.path)
        # A store opened read-only is read the same way by every reader: the
        # URI decides between mode=ro and immutable, see open_read_only.
        self._uri = getattr(ds._conn, "uri", None)
        self.matrix_name = matrix_name
        self.n_readers = max(1, int(n_readers))
        # Everything written so far must be in the file before anything is
        # read, by this connection or the readers'.
        ds.flush()
        ds._conn.commit()
        row = ds._conn.execute(
            "SELECT n_rows, n_cols, dtype, COALESCE(n_chunks, 0), COALESCE(has_csc, 0) "
            "FROM matrix_meta WHERE matrix_name = ?", (matrix_name,)).fetchone()
        if row is None:
            raise KeyError(f"no matrix {matrix_name!r} in {self.path}")
        self._meta = (int(row[0]), int(row[1]), str(row[2]), int(row[3]) > 0, bool(row[4]))
        self.cache = bool(cache)
        if cache_budget_bytes is None and self.cache:
            avail = available_memory_bytes()
            cache_budget_bytes = avail // 2 if avail else None
        self.cache_budget_bytes = cache_budget_bytes
        self._cache: dict = {}
        self.cached_bytes = 0
        self.cache_declined = None
        self.bytes_read = 0
        self._pool = None
        self._local = threading.local()
        self._conns: list = []
        self._lock = threading.Lock()
        self._pending = None
        #: bumped by release_connections(): a reader thread's connection of an
        #: older generation is closed and opened again on its next read
        self._gen = 0
        from cytome.core import _handles
        _handles.track_source(self)

    # -- protocol ---------------------------------------------------------
    def meta(self):
        """``(n_rows, n_cols, dtype, has_row_chunks, has_column_chunks)``."""
        return self._meta

    def chunks(self, layout: str):
        """``(chunk_idx, start, end, n_nonzero)`` as int64 arrays, in chunk
        order; ``start``/``end`` are rows for ``"csr"``, columns for ``"csc"``."""
        table, a, b = _TABLE[layout]
        rows = self._ds._conn.execute(
            f"SELECT chunk_idx, {a}, {b}, COALESCE(n_nonzero, 0) FROM {table} "
            f"WHERE matrix_name = ? ORDER BY chunk_idx", (self.matrix_name,)).fetchall()
        arr = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
        return tuple(np.ascontiguousarray(arr[:, i]) for i in range(4))

    def fetch(self, layout: str, idx, next_idx=None):
        """The blobs of chunks ``idx``, in that order, as
        ``(data, indices, indptr, compression)`` tuples; then start reading
        ``next_idx`` in the background."""
        idx = [int(i) for i in idx]
        pending, self._pending = self._pending, None
        if pending is not None and pending[0] == (layout, tuple(idx)):
            got = pending[1].result()
        else:
            if pending is not None:
                pending[1].result()  # never leave a read running unobserved
            got = self._read(layout, idx)
        if next_idx is not None and len(next_idx):
            nxt = [int(i) for i in next_idx]
            self._pending = ((layout, tuple(nxt)), self._executor().submit(self._read, layout, nxt))
        return got

    def release_connections(self) -> None:
        """Close the readers' connections (they open again on the next fetch).

        :func:`cytome.released` calls it, so a program that must have the file
        to itself does not meet this source's SQLite handles: they were left
        open across it before.
        """
        if self._pending is not None:
            try:
                self._pending[1].result()
            except Exception:
                pass
            self._pending = None
        with self._lock:
            conns, self._conns = self._conns, []
            self._gen += 1
        for c in conns:
            try:
                c.close()
            except Exception:
                pass

    def close(self) -> None:
        self.release_connections()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._cache.clear()
        self.cached_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- reading ----------------------------------------------------------
    def _executor(self):
        if self._pool is None:
            # one extra thread so a background read never waits on the
            # threads a foreground read is using
            self._pool = ThreadPoolExecutor(max_workers=self.n_readers + 1,
                                            thread_name_prefix="cytome-chunks")
        return self._pool

    def _conn(self):
        c = getattr(self._local, "conn", None)
        if c is not None and getattr(self._local, "gen", None) != self._gen:
            c = None                       # closed by release_connections()
        if c is None:
            import sqlite3
            if self._uri is not None:
                c = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
            else:
                try:
                    c = sqlite3.connect(f"file:{Path(self.path).resolve().as_posix()}?mode=ro",
                                        uri=True, check_same_thread=False)
                except sqlite3.OperationalError:
                    c = sqlite3.connect(self.path, check_same_thread=False)
            c.execute("PRAGMA busy_timeout=60000")
            self._local.conn = c
            self._local.gen = self._gen
            with self._lock:
                self._conns.append(c)
        return c

    def _read_run(self, table, idx):
        c = self._conn()
        out = {}
        # One statement for a run of chunks: one read transaction, and one
        # lock on the small index file, not one per chunk.
        for start in range(0, len(idx), 500):
            part = idx[start:start + 500]
            q = (f"SELECT chunk_idx, data_blob, indices_blob, indptr_blob, compression FROM {table} "
                 f"WHERE matrix_name = ? AND chunk_idx IN ({','.join('?' * len(part))})")
            for i, d, ind, ptr, comp in c.execute(q, (self.matrix_name, *part)):
                out[int(i)] = (bytes(d), bytes(ind), bytes(ptr), str(comp or "none"))
        return out

    def _read(self, layout, idx):
        table = _TABLE[layout][0]
        want = [i for i in idx if (layout, i) not in self._cache]
        got = {}
        if want:
            n = min(self.n_readers, len(want))
            if n == 1:
                got = self._read_run(table, want)
            else:
                parts = [want[j * len(want) // n:(j + 1) * len(want) // n] for j in range(n)]
                for d in self._executor().map(lambda p: self._read_run(table, p), parts):
                    got.update(d)
            missing = [i for i in want if i not in got]
            if missing:
                raise KeyError(f"{self.matrix_name}: chunks {missing[:5]} not in {table}")
            nbytes = sum(len(v[0]) + len(v[1]) + len(v[2]) for v in got.values())
            with self._lock:
                self.bytes_read += nbytes
                if self.cache:
                    if self.cache_budget_bytes is not None and self.cached_bytes + nbytes > self.cache_budget_bytes:
                        self.cache = False
                        self.cache_declined = (self.cached_bytes + nbytes, self.cache_budget_bytes)
                        self._cache.clear()
                        self.cached_bytes = 0
                        warnings.warn(
                            f"{self.matrix_name}: the chunk cache would pass its budget of "
                            f"{self.cache_budget_bytes / 1e9:.1f} GB; it is dropped and every pass "
                            f"reads the store.", RuntimeWarning, stacklevel=3)
                    else:
                        for i, v in got.items():
                            self._cache[(layout, i)] = v
                        self.cached_bytes += nbytes
        return [self._cache[(layout, i)] if (layout, i) in self._cache else got[i] for i in idx]


def released(target):
    """Close every dataset of this process on ``target``'s file, and reopen
    them afterwards.

    For a program that must have the file to itself -- a writer from another
    library, which cannot see this process's connections or their locks.
    Pending writes are flushed first. Accessors kept from those datasets
    (``cells = ds.cells``) keep working after the reopen; an iterator that was
    running does not.

    ``target`` is a path or a dataset. Datasets closed before the call stay
    closed.
    """
    from contextlib import contextmanager
    from cytome.core import _handles
    from cytome.io.sqlite_engine import close_database

    @contextmanager
    def _cm():
        path = getattr(target, "path", target)
        for source in _handles.sources_on(path):
            source.release_connections()
        held = _handles.open_on(path)
        for ds in held:
            ds._rollback_open_writers()
            ds.flush()
            close_database(ds._conn)
        try:
            yield held
        finally:
            for ds in held:
                ds.reopen()
    return _cm()
