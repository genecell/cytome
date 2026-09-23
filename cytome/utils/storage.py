"""Where a store lives, and the one thing worth saying about it.

A network file system keeps a client-side cache of a file and drops the
whole of it whenever the file may have changed under it: on every file
lock, and whenever the file's modification time moves. A cytome is one
file, read many times and written to between the reads, so on such a mount
its cache is dropped by every write (a quantified matrix, a metadata entry,
a checkpoint) and re-read from the server on the next pass. Measured on the
same node and cores, a 31k-cell store from a network mount against
node-local scratch: PICCO 50.8 s against 12.8 s, the quantifier 41.5 s
against 15.5 s, COSG over five million tiles 2 min 58 s against 1 min 51 s.
The SVD engine reads without locks and is within a factor of 1.5; the rest
is the first read and the writes. Users on a cluster rarely know which mount
they are on. This tells them, once.
"""
from __future__ import annotations

import os
import sys

__all__ = ["filesystem_type", "is_network_filesystem", "network_filesystem_notice"]

#: File-system types that go over a network. ``fuse.*`` covers sshfs and
#: friends; ``nfs`` covers nfs4 by prefix.
NETWORK_FS_PREFIXES = ("nfs", "lustre", "gpfs", "beegfs", "cifs", "smb", "ceph", "panfs",
                       "glusterfs", "fuse.", "afs", "9p", "virtiofs", "weka")

_said: set[str] = set()


def filesystem_type(path: str | os.PathLike, mounts: str = "/proc/self/mounts") -> str | None:
    """The mount type the path's real location is on, or ``None`` if unknown.

    Linux only: reads the mount table and takes the longest mount point that
    is a prefix of the resolved path. Symlinks are resolved first, so a link
    in ``$HOME`` to a network volume is reported as the volume.
    """
    try:
        real = os.path.realpath(os.fspath(path))
        with open(mounts, "r", encoding="utf-8", errors="replace") as fh:
            best, best_type = "", None
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount, fstype = parts[1].replace("\\040", " "), parts[2]
                if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
                    best, best_type = mount, fstype
            return best_type
    except OSError:
        return None


def is_network_filesystem(fstype: str | None) -> bool:
    return bool(fstype) and fstype.lower().startswith(NETWORK_FS_PREFIXES)


def network_filesystem_notice(path: str | os.PathLike, mounts: str = "/proc/self/mounts",
                              stream=None) -> bool:
    """Print one line, once per process per store, when the store is on a
    network file system. Returns whether it printed."""
    fstype = filesystem_type(path, mounts=mounts)
    if not is_network_filesystem(fstype):
        return False
    key = os.path.realpath(os.fspath(path))
    if key in _said:
        return False
    _said.add(key)
    print(f"cytome: {os.path.basename(key)} is on a network file system ({fstype}). Reads "
          f"are served from this node's cache after the first pass, but the first read "
          f"and every write go over the network, and a write drops the cache of the "
          f"whole file. For a long job, copy it to node-local scratch ($TMPDIR) first "
          f"and open the copy; copy it back if you write to it.",
          file=stream or sys.stderr)
    return True
