"""Genomic region parsing and manipulation helpers."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

Region = Tuple[str, int, int]

_REGION_RE = re.compile(r"^\s*([^:]+):([^\-]+)-([^\s]+)\s*$")


def parse_region(s: str) -> Region:
    """Parse region string like ``chr1:1,000-2M``.

    Parameters
    ----------
    s
        Region string.

    Returns
    -------
    tuple
        ``(chrom, start, end)``
    """
    m = _REGION_RE.match(s)
    if not m:
        raise ValueError(f"Invalid region format: {s}")
    chrom, start_s, end_s = m.group(1), m.group(2), m.group(3)
    start = _parse_coord(start_s)
    end = _parse_coord(end_s)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"Invalid region coordinates: {s}")
    return chrom, start, end


def regions_overlap(r1: Region, r2: Region) -> bool:
    """Return whether two regions overlap."""
    if r1[0] != r2[0]:
        return False
    return r1[1] < r2[2] and r2[1] < r1[2]


def merge_regions(regions: Iterable[Region]) -> List[Region]:
    """Merge overlapping or touching regions by chromosome."""
    sorted_regions = sorted(regions, key=lambda r: (r[0], r[1], r[2]))
    if not sorted_regions:
        return []

    merged: List[Region] = [sorted_regions[0]]
    for chrom, start, end in sorted_regions[1:]:
        m_chrom, m_start, m_end = merged[-1]
        if chrom == m_chrom and start <= m_end:
            merged[-1] = (m_chrom, m_start, max(m_end, end))
        else:
            merged.append((chrom, start, end))
    return merged


def extend_region(region: Region, upstream: int, downstream: int) -> Region:
    """Extend a region by upstream and downstream base pairs."""
    chrom, start, end = region
    return chrom, max(0, start - int(upstream)), end + int(downstream)


def _parse_coord(text: str) -> int:
    t = text.strip().replace(",", "")
    unit = 1
    if t[-1] in {"K", "k", "M", "m", "G", "g"}:
        suffix = t[-1].upper()
        t = t[:-1]
        unit = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}[suffix]
    return int(float(t) * unit)
