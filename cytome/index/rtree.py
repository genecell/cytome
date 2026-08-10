"""SQLite R-tree index management for genomic coordinates."""

from __future__ import annotations

import sqlite3
from typing import List, Sequence, Tuple

from cytome.utils.genome import chrom_to_int


FragmentHit = Tuple[int, int, int, int]
PeakHit = Tuple[int, str, int, int]


def create_fragment_rtree(conn: sqlite3.Connection, chrom: str) -> None:
    """Create chromosome-specific fragment R-tree virtual table."""
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS fragments_{chrom}_rtree USING rtree(
            id,
            min_start, max_start,
            min_end, max_end
        )
        """
    )


def populate_fragment_rtree(conn: sqlite3.Connection, chrom: str) -> None:
    """Populate a chromosome fragment R-tree from fragment table."""
    create_fragment_rtree(conn, chrom)
    conn.execute(f"DELETE FROM fragments_{chrom}_rtree")
    conn.execute(
        f"""
        INSERT INTO fragments_{chrom}_rtree(id, min_start, max_start, min_end, max_end)
        SELECT rowid, start, start, end_, end_
        FROM fragments_{chrom}
        """
    )


def query_fragment_rtree(
    conn: sqlite3.Connection,
    chrom: str,
    start: int,
    end: int,
) -> List[FragmentHit]:
    """Range query fragments using chromosome-specific R-tree.

    Tries three strategies in order:
    1. R-tree index join with per-row table (fastest)
    2. Linear scan of per-row table (no R-tree)
    3. Compressed chunk scan (for subset/chunk-only datasets)

    Falls through to chunk scan when per-row tables exist but are empty
    (common for subset outputs where schema creates empty tables).
    """
    # Strategy 1: R-tree + per-row table
    try:
        rows = conn.execute(
            f"""
            SELECT f.start, f.end_, f.cell_idx, f.dup_count
            FROM fragments_{chrom} AS f
            INNER JOIN fragments_{chrom}_rtree AS r ON f.rowid = r.id
            WHERE r.max_start < ? AND r.min_end > ?
            ORDER BY f.start
            """,
            (end, start),
        ).fetchall()
        if rows:
            return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]
    except sqlite3.OperationalError:
        pass

    # Strategy 2: per-row table without R-tree
    try:
        rows = conn.execute(
            f"""
            SELECT start, end_, cell_idx, dup_count
            FROM fragments_{chrom}
            WHERE start < ? AND end_ > ?
            ORDER BY start
            """,
            (end, start),
        ).fetchall()
        if rows:
            return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]
    except sqlite3.OperationalError:
        pass

    # Strategy 3: compressed fragment_chunks (subset outputs)
    return _query_region_from_chunks(conn, chrom, start, end)


def _query_region_from_chunks(
    conn: sqlite3.Connection,
    chrom: str,
    start: int,
    end: int,
) -> List[FragmentHit]:
    """Scan compressed fragment_chunks for overlapping fragments.

    Uses min_start per chunk to skip non-overlapping chunks.
    Within each chunk, fragments are sorted by start position.
    """
    import numpy as np
    from cytome.io.compression import decompress_blob, decode_starts, decode_ends

    try:
        rows = conn.execute(
            "SELECT starts_blob, ends_blob, cell_idx_blob, compression, "
            "COALESCE(encoding, 0) as encoding, min_start "
            "FROM fragment_chunks WHERE chrom = ? ORDER BY chunk_idx",
            (chrom,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    hits = []
    for starts_b, ends_b, cells_b, comp, enc, min_start in rows:
        # Skip chunk if all starts are past query end
        if min_start is not None and min_start >= end:
            continue

        starts = decode_starts(starts_b, comp, enc)
        ends = decode_ends(ends_b, comp, starts, enc)
        cells = np.frombuffer(decompress_blob(cells_b, comp), dtype=np.int32)

        # Overlap: fragment.start < query.end AND fragment.end > query.start
        mask = (starts < end) & (ends > start)
        if mask.any():
            for s, e, c in zip(starts[mask], ends[mask], cells[mask]):
                hits.append((int(s), int(e), int(c), 1))

    hits.sort(key=lambda x: x[0])
    return hits


def create_peak_rtree(conn: sqlite3.Connection) -> None:
    """Create peaks R-tree virtual table."""
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS peaks_rtree USING rtree(
            id,
            min_chr, max_chr,
            min_start, max_start,
            min_end, max_end
        )
        """
    )


def populate_peak_rtree(conn: sqlite3.Connection) -> None:
    """Populate peaks R-tree from peaks table.

    Peaks on chromosomes not in :data:`cytome.utils.genome.CHROM_ORDER` (e.g. unplaced
    scaffolds like ``GL456233.1``) are **skipped** with a warning rather than crashing;
    use ``keep_chroms`` at import time to drop them from the dataset entirely.
    """
    from cytome.utils.genome import CHROM_ORDER
    create_peak_rtree(conn)
    conn.execute("DELETE FROM peaks_rtree")
    rows = conn.execute("SELECT peak_idx, chr, start, end_ FROM peaks").fetchall()
    records, skipped = [], 0
    for peak_idx, chr_name, start, end_ in rows:
        if chr_name is None:
            continue
        if chr_name not in CHROM_ORDER:
            skipped += 1
            continue
        ci = chrom_to_int(chr_name)
        records.append((int(peak_idx), ci, ci, int(start), int(start), int(end_), int(end_)))
    if skipped:
        import warnings
        warnings.warn(
            f"populate_peak_rtree: skipped {skipped} peak(s) on non-standard chromosomes "
            f"(not in CHROM_ORDER). Pass keep_chroms='standard' at import to drop them cleanly.",
            stacklevel=2,
        )
    conn.executemany(
        """
        INSERT INTO peaks_rtree(id, min_chr, max_chr, min_start, max_start, min_end, max_end)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )


def query_peak_rtree(
    conn: sqlite3.Connection,
    chrom: str,
    start: int,
    end: int,
) -> List[PeakHit]:
    """Query peaks overlapping a region."""
    chr_i = chrom_to_int(chrom)
    try:
        rows = conn.execute(
            """
            SELECT p.peak_idx, p.chr, p.start, p.end_
            FROM peaks AS p
            INNER JOIN peaks_rtree AS r ON p.peak_idx = r.id
            WHERE r.min_chr = ? AND r.max_start < ? AND r.min_end > ?
            ORDER BY p.start
            """,
            (chr_i, end, start),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(
            """
            SELECT peak_idx, chr, start, end_
            FROM peaks
            WHERE chr = ? AND start < ? AND end_ > ?
            ORDER BY start
            """,
            (chrom, end, start),
        ).fetchall()
    return [(int(r[0]), str(r[1]), int(r[2]), int(r[3])) for r in rows]
