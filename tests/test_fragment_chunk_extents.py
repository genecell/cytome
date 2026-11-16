"""Fragment chunks get their largest end recorded once, explicitly; region queries are the same either way.

PIASO's importer wrote no ``max_end``, so a region query had to read every earlier chunk of the chromosome.
The rule:
fill it in an explicit step (``fill_fragment_chunk_extents``, also run by ``repair``), never during a read, so
read-only and frozen stores are left alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cytome
from cytome.io.compression import compress_blob
from cytome.io.sqlite_engine import ReadOnlyError
from cytome.utils.validation import repair

# three chunks on chr1, sorted by start; the second holds a long fragment, so its largest end is not its last
CHUNKS = [
    ([100, 150, 400], [180, 300, 420], [0, 1, 2]),
    ([1000, 1100, 1200], [1050, 9000, 1260], [1, 0, 2]),
    ([5000, 5100], [5050, 5150], [2, 1]),
]


def _store(path, encoding=1):
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b", "c"]}))
    ds.flush()
    row = 0
    for k, (s, e, c) in enumerate(CHUNKS):
        s, e, c = (np.asarray(x, dtype=np.int32) for x in (s, e, c))
        sb = np.diff(s, prepend=0).astype(np.int32) if encoding == 1 else s
        eb = (e - s).astype(np.int32) if encoding == 1 else e
        ds._conn.execute(
            "INSERT INTO fragment_chunks (chrom, chunk_idx, row_start, row_end, n_fragments, min_start, "
            "starts_blob, ends_blob, cell_idx_blob, compression, encoding) VALUES ('chr1',?,?,?,?,?,?,?,?,'lz4',?)",
            (k, row, row + len(s), len(s), int(s[0]), compress_blob(sb.tobytes(), "lz4"),
             compress_blob(eb.tobytes(), "lz4"), compress_blob(c.tobytes(), "lz4"), encoding))
        row += len(s)
    ds._conn.execute(
        "INSERT OR REPLACE INTO fragment_meta (chrom, n_fragments, table_name, rtree_name, min_start, max_end) "
        "VALUES ('chr1', ?, 'fragments_chr1', 'fragments_chr1_rtree', 100, 9000)", (row,))
    ds._conn.commit()
    ds.flush()
    return ds


def _extents(ds):
    return [r[0] for r in ds._conn.execute("SELECT max_end FROM fragment_chunks ORDER BY chunk_idx")]


@pytest.mark.parametrize("encoding", [0, 1])
def test_extents_are_filled_once_and_queries_do_not_change(tmp_path, encoding):
    ds = _store(tmp_path / f"e{encoding}.cytome", encoding=encoding)
    regions = [(0, 200), (2000, 3000), (8000, 8500), (5100, 5120), (0, 10000)]
    before = [ds.fragments.query_region("chr1", a, b) for a, b in regions]
    assert _extents(ds) == [None, None, None]
    assert cytome.fill_fragment_chunk_extents(ds) == {"chunks_filled": 3, "chromosomes": 1}
    assert _extents(ds) == [420, 9000, 5150]
    after = [ds.fragments.query_region("chr1", a, b) for a, b in regions]
    for x, y in zip(before, after):
        for k in ("start", "end_", "cell_idx"):
            np.testing.assert_array_equal(np.asarray(x[k]), np.asarray(y[k]))
    # (2000, 3000) lies inside the long fragment 1100-9000 only
    assert after[1]["start"].tolist() == [1100]
    assert cytome.fill_fragment_chunk_extents(ds)["chunks_filled"] == 0
    ds.close()


def test_reads_never_fill_and_repair_does(tmp_path):
    path = tmp_path / "r.cytome"
    _store(path).close()
    ro = cytome.open(path, mode="r")
    ro.fragments.query_region("chr1", 0, 10000)
    with pytest.raises(ReadOnlyError):
        cytome.fill_fragment_chunk_extents(ro)
    ro.close()
    ds = cytome.open(path)
    assert _extents(ds) == [None, None, None]                 # the read wrote nothing
    assert any("max_end" in d for d in repair(ds))
    assert _extents(ds) == [420, 9000, 5150]
    ds.close()
