"""2026-06-04: fragment export must read the compressed fragment_chunks layout
(used by subset/filter_cells cytomes), not only the legacy per-row
fragments_{chrom} tables (which a subset cytome does not have)."""
import os
import sqlite3
import tempfile

import numpy as np
import pytest

from cytome.io.sqlite_engine import open_database
from cytome.io.compression import compress_blob
from cytome.io.convert_fragments import _iter_export_lines


def _enc(arr):
    return compress_blob(np.asarray(arr, np.int32).tobytes(), "zstd")


def _chunks_only_conn(tmp):
    """A cytome with fragments ONLY in fragment_chunks (no per-row tables) —
    exactly what subset/filter_cells produces."""
    path = str(tmp / "chunks.cytome")
    sqlite3.connect(path).close()
    conn = open_database(path)
    starts = [100, 300, 200]
    ends = [150, 350, 250]
    cells = [0, 1, 0]
    conn.execute(
        "INSERT INTO fragment_chunks (chrom, chunk_idx, row_start, row_end, "
        "n_fragments, min_start, starts_blob, ends_blob, cell_idx_blob, "
        "compression, encoding) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("chr1", 0, 0, 3, 3, 100, _enc(starts), _enc(ends), _enc(cells), "zstd", 0),
    )
    conn.execute(
        "INSERT INTO fragment_meta (chrom, n_fragments, table_name, rtree_name, "
        "min_start, max_end) VALUES (?,?,?,?,?,?)",
        ("chr1", 3, "fragments_chr1", "fragments_chr1_rtree", 100, 350),
    )
    conn.commit()
    return conn


def test_export_reads_fragment_chunks(tmp_path):
    conn = _chunks_only_conn(tmp_path)
    # The per-row fragments_chr1 table is empty/absent on a subset cytome, so the
    # old code path yielded 0 lines (or raised); the fix decodes fragment_chunks.
    per_row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name='fragments_chr1'").fetchone()[0]
    if per_row:
        assert conn.execute("SELECT COUNT(*) FROM fragments_chr1").fetchone()[0] == 0

    names = {0: "BC0", 1: "BC1"}
    lines = list(_iter_export_lines(conn, names, None, None))
    conn.close()
    # 3 fragments, start-sorted (100, 200, 300), 5 tab fields each.
    assert len(lines) == 3
    starts = [int(ln.split("\t")[1]) for ln in lines]
    assert starts == [100, 200, 300]
    bcs = [ln.split("\t")[3] for ln in lines]
    assert bcs == ["BC0", "BC0", "BC1"]   # cell 0 @100/200, cell 1 @300


def test_export_chunks_respects_barcode_filter(tmp_path):
    conn = _chunks_only_conn(tmp_path)
    names = {0: "BC0", 1: "BC1"}
    lines = list(_iter_export_lines(conn, names, {"BC1"}, None))
    conn.close()
    assert len(lines) == 1 and lines[0].split("\t")[3] == "BC1"
