"""Provenance logging for Cytome operations."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import logging
import platform
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ProvenanceLog:
    """Access and record provenance information."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def log(
        self,
        operation: str,
        function_name: str,
        parameters: dict[str, Any],
        package_name: str,
        package_version: str,
        input_objects: list[str],
        output_objects: list[str],
        random_seed: int | None = None,
        notes: str | None = None,
    ) -> int:
        """Write a provenance record and return its ID."""
        deps = _capture_dependency_versions()
        hardware = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_build": platform.python_build(),
        }
        cursor = self._conn.execute(
            """
            INSERT INTO _provenance(
                timestamp, operation, package_name, package_version,
                python_version, function_name, parameters, dependency_versions,
                input_objects, output_objects, random_seed, hardware_info, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                operation,
                package_name,
                package_version,
                sys.version,
                function_name,
                json.dumps(parameters),
                json.dumps(deps),
                json.dumps(input_objects),
                json.dumps(output_objects),
                random_seed,
                json.dumps(hardware),
                notes,
            ),
        )
        return int(cursor.lastrowid)

    def show(self) -> str:
        """Return formatted provenance history text."""
        rows = self._conn.execute(
            """
            SELECT id, timestamp, function_name, package_name, package_version, parameters
            FROM _provenance
            ORDER BY id
            """
        ).fetchall()
        lines = []
        for row in rows:
            pid, timestamp, func, pkg, ver, params = row
            lines.append(f"[{pid}] {timestamp} | {func} | {pkg} {ver} | {params}")
        text = "\n".join(lines)
        if text:
            logger.info("\n%s", text)
        return text

    def get_for_object(self, object_name: str) -> list[dict[str, Any]]:
        """Return provenance rows that reference an object in inputs/outputs."""
        rows = self._conn.execute(
            """
            SELECT * FROM _provenance
            WHERE input_objects LIKE ? OR output_objects LIKE ?
            ORDER BY id
            """,
            (f'%"{object_name}"%', f'%"{object_name}"%'),
        ).fetchall()
        cols = [d[1] for d in self._conn.execute("PRAGMA table_info(_provenance)").fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def export_methods_text(self) -> str:
        """Generate concise methods description text."""
        rows = self._conn.execute(
            """
            SELECT timestamp, function_name, package_name, package_version, parameters
            FROM _provenance ORDER BY id
            """
        ).fetchall()
        if not rows:
            return "No provenance records available."
        parts = []
        for timestamp, func, pkg, ver, params in rows:
            parts.append(
                f"At {timestamp}, `{func}` was run with {pkg} {ver} and parameters {params}."
            )
        return " ".join(parts)


def _capture_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        versions[name] = dist.version
    return versions
