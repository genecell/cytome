from __future__ import annotations

from cytome.index.rtree import (
    create_fragment_rtree,
    populate_fragment_rtree,
    query_fragment_rtree,
)
from cytome.io.sqlite_engine import create_database, close_database


class TestRTree:
    def test_create_and_populate(self, tmp_cytome):
        conn = create_database(tmp_cytome)
        conn.execute(
            "INSERT INTO fragments_chr1(start, end_, cell_idx, dup_count) VALUES (100, 200, 0, 1)"
        )
        create_fragment_rtree(conn, "chr1")
        populate_fragment_rtree(conn, "chr1")
        out = query_fragment_rtree(conn, "chr1", 120, 180)
        assert len(out) == 1
        close_database(conn)

    def test_range_query_accuracy(self, tmp_cytome):
        conn = create_database(tmp_cytome)
        conn.executemany(
            "INSERT INTO fragments_chr1(start, end_, cell_idx, dup_count) VALUES (?, ?, ?, 1)",
            [(100, 150, 0), (160, 220, 0), (300, 350, 1)],
        )
        populate_fragment_rtree(conn, "chr1")
        out = query_fragment_rtree(conn, "chr1", 140, 170)
        assert len(out) == 2
        close_database(conn)

    def test_empty_region(self, tmp_cytome):
        conn = create_database(tmp_cytome)
        conn.execute(
            "INSERT INTO fragments_chr1(start, end_, cell_idx, dup_count) VALUES (100, 150, 0, 1)"
        )
        populate_fragment_rtree(conn, "chr1")
        out = query_fragment_rtree(conn, "chr1", 200, 300)
        assert out == []
        close_database(conn)

    def test_overlapping_fragments(self, tmp_cytome):
        conn = create_database(tmp_cytome)
        conn.executemany(
            "INSERT INTO fragments_chr1(start, end_, cell_idx, dup_count) VALUES (?, ?, ?, 1)",
            [(90, 210, 0), (180, 260, 1)],
        )
        populate_fragment_rtree(conn, "chr1")
        out = query_fragment_rtree(conn, "chr1", 200, 220)
        assert len(out) == 2
        close_database(conn)
