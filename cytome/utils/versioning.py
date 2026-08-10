"""Schema version compatibility helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

CURRENT_VERSION = "1.0.0"


@dataclass
class Migration:
    from_version: str
    to_version: str
    description: str

    def apply(self, conn: sqlite3.Connection) -> None:
        """Apply migration SQL steps."""
        del conn


MIGRATIONS: list[Migration] = []


def check_compatibility(conn: sqlite3.Connection) -> None:
    """Check whether current reader can open this dataset."""
    row = conn.execute(
        "SELECT value FROM _manifest WHERE key = 'min_reader_version'"
    ).fetchone()
    if row is None:
        raise RuntimeError("Manifest key 'min_reader_version' missing")
    min_reader = json.loads(row[0])
    if _version_tuple(CURRENT_VERSION) < _version_tuple(min_reader):
        raise RuntimeError(
            f"Dataset requires reader version {min_reader}, current is {CURRENT_VERSION}"
        )


def migrate_if_needed(conn: sqlite3.Connection) -> None:
    """Run pending migrations if file version is behind current."""
    row = conn.execute("SELECT value FROM _manifest WHERE key = 'format_version'").fetchone()
    if row is None:
        raise RuntimeError("Manifest key 'format_version' missing")
    current = json.loads(row[0])
    if _version_tuple(current) >= _version_tuple(CURRENT_VERSION):
        return

    for migration in MIGRATIONS:
        if migration.from_version == current:
            migration.apply(conn)
            current = migration.to_version

    conn.execute(
        "INSERT OR REPLACE INTO _manifest(key, value) VALUES ('format_version', ?)",
        (json.dumps(current),),
    )


def _version_tuple(version: str) -> tuple[int, int, int]:
    return tuple(int(p) for p in version.split("."))  # type: ignore[return-value]
