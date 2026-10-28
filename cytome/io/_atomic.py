"""Build a store under a temporary name and give it its name when it is whole.

A conversion killed half way left a store under its final name that opened and
passed ``validate()`` (audit 2026-10-27, L1 improvement 6): a converter now
writes ``<output>.partial``, without anything else seeing it, and renames it to
``<output>`` only after the last write. A failure removes the partial file, and
an older store at ``<output>`` (``force=True``) stays as it was until the new
one replaces it.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


def remove_store_files(path) -> None:
    """Delete a store file and its sidecars, whichever exist."""
    for sfx in ("", "-wal", "-shm", "-journal"):
        Path(str(path) + sfx).unlink(missing_ok=True)


def sync_path(path) -> None:
    """fsync a file or directory; file systems that cannot sync a directory are skipped."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def move_into_place(partial, output) -> None:
    """Sync ``partial`` and rename it to ``output``, dropping an older output's
    sidecars first (SQLite would replay them into the new file)."""
    partial, output = Path(partial), Path(output)
    sync_path(partial)
    for sfx in ("-wal", "-shm", "-journal"):
        Path(str(output) + sfx).unlink(missing_ok=True)
    os.replace(partial, output)
    for sfx in ("-wal", "-shm"):
        Path(str(partial) + sfx).unlink(missing_ok=True)
    sync_path(output.parent)


@contextmanager
def building(output, force: bool = False):
    """``with building(output, force) as partial:`` write and close a store at
    ``partial``; on leaving the block it becomes ``output``. An existing
    ``output`` is refused before anything is written unless ``force``."""
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(
            f"Cytome already exists: {output}. Pass force=True to overwrite "
            f"(guards against accidentally truncating an existing cytome).")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    remove_store_files(partial)
    try:
        yield partial
    except BaseException:
        remove_store_files(partial)
        raise
    move_into_place(partial, output)
