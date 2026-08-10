"""Helpers to build and rebuild R-tree indices."""

from __future__ import annotations

import sqlite3

from cytome.index.rtree import (
    create_fragment_rtree,
    create_peak_rtree,
    populate_fragment_rtree,
    populate_peak_rtree,
)


def build_fragment_indices(conn: sqlite3.Connection) -> None:
    """Build R-tree indices for all chromosome fragment tables."""
    rows = conn.execute(
        "SELECT chrom FROM fragment_meta WHERE n_fragments > 0 ORDER BY chrom"
    ).fetchall()
    if rows:
        chroms = [str(r[0]) for r in rows]
    else:
        table_rows = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table'
              AND name LIKE 'fragments_chr%'
              AND name NOT LIKE '%_rtree%'
              AND name NOT LIKE '%_node'
              AND name NOT LIKE '%_parent'
              AND name NOT LIKE '%_rowid'
            ORDER BY name
            """
        ).fetchall()
        chroms = [str(name)[len("fragments_") :] for (name,) in table_rows]

    for chrom in chroms:
        create_fragment_rtree(conn, chrom)
        populate_fragment_rtree(conn, chrom)


def build_peak_index(conn: sqlite3.Connection) -> None:
    """Build the peak coordinate R-tree index."""
    create_peak_rtree(conn)
    populate_peak_rtree(conn)


def rebuild_indices(conn: sqlite3.Connection) -> None:
    """Drop and rebuild all R-tree indices."""
    rows = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE 'fragments_chr%_rtree'
        """
    ).fetchall()
    for (name,) in rows:
        conn.execute(f"DROP TABLE IF EXISTS {name}")

    conn.execute("DROP TABLE IF EXISTS peaks_rtree")
    build_fragment_indices(conn)
    build_peak_index(conn)
