from __future__ import annotations

from cytome.utils.regions import extend_region, merge_regions, parse_region, regions_overlap


def test_parse_region():
    assert parse_region("chr1:1000-2000") == ("chr1", 1000, 2000)
    assert parse_region("chr1:1,000-2,000") == ("chr1", 1000, 2000)
    assert parse_region("chr1:1M-2M") == ("chr1", 1_000_000, 2_000_000)


def test_overlap():
    assert regions_overlap(("chr1", 100, 200), ("chr1", 150, 250))
    assert not regions_overlap(("chr1", 100, 200), ("chr1", 200, 300))
    assert not regions_overlap(("chr1", 100, 200), ("chr2", 150, 250))


def test_merge():
    merged = merge_regions([("chr1", 10, 20), ("chr1", 15, 30), ("chr2", 5, 8)])
    assert merged == [("chr1", 10, 30), ("chr2", 5, 8)]


def test_extend_region():
    assert extend_region(("chr1", 100, 200), 20, 30) == ("chr1", 80, 230)
    assert extend_region(("chr1", 10, 20), 20, 0) == ("chr1", 0, 20)
