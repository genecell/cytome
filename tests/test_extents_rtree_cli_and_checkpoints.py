"""Coverage sized by the data, chromosome filters that keep what they say,
R*-tree queries that do not miss edges, read-only CLI commands, checkpoints that
report.

Audit 2026-10-27, part 4 majors 28 and 29 and the storage-core and converter
minors (L1 findings 21, 22, 27 and improvement 5; L2 minor on from_10x_h5):
export_coverage assumed hg38 when no genome was recorded, so mouse fragments
past hg38's lengths piled into the last bin; keep_chroms='all' kept fewer
fragments than 'standard' and other names were dropped silently; the R*-tree's
float32 bounds made boundary peaks, fragments and cells go missing; a region
query held a chromosome's blobs; the CLI opened stores read-write to read them,
always ran SQLite's integrity check, overwrote with init, defaulted merge to
the intersection, and wrote a 0-cell store for an empty sample column;
checkpoint() returned nothing and compact() swallowed errors; from_10x_h5
stored 0.4 as 0.
"""
import gzip
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.index.builder import build_peak_index
from cytome.index.rtree import _query_region_from_chunks, query_peak_rtree
from cytome.io.convert_fragments import (
    _update_fragment_meta,
    _write_block_chunks,
    import_fragments_streaming,
)
from cytome.io.export_bigwig import export_coverage


def _cells(tmp_path, groups, name="s.cytome"):
    ds = cytome.create(str(tmp_path / name))
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(len(groups))],
                                         "g": groups}))
    ds.flush()
    return ds


def _frags(ds, chrom, starts, cells, chunk=10):
    s = np.asarray(starts, dtype=np.int32)
    _write_block_chunks(ds._conn, chrom, s, s + 100, np.asarray(cells, dtype=np.int32),
                        chunk_size=chunk, compression="lz4", compression_level=1,
                        chunk_idx_start=0, encoding=1)
    _update_fragment_meta(ds._conn, chrom, s.size)
    ds._conn.commit()


def test_coverage_without_a_genome_follows_the_fragments(tmp_path):
    ds = _cells(tmp_path, ["Alzheimer's", "x"])
    _frags(ds, "chrX", [160_000_000], [0])          # past hg38's chrX (156,040,895)
    _frags(ds, "2L", [5_000], [0])                   # a chromosome hg38 lacks
    out = export_coverage(ds, "g", str(tmp_path / "cov"), format="bedgraph",
                          normalize="raw", bin_size=1000)
    lines = [l.split("\t") for p in out for l in open(p)]
    assert ("chrX", "160000000") in [(l[0], l[1]) for l in lines]
    assert "2L" in {l[0] for l in lines}
    ds.close()


def _fragment_file(tmp_path):
    p = tmp_path / "f.tsv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("chr1\t10\t60\tc0\t1\nchrM\t5\t50\tc0\t1\n2L\t1\t40\tc0\t1\n")
    return p


def test_chr_filter_all_keeps_every_chromosome_and_a_filter_reports(tmp_path):
    from cytome.io.sqlite_engine import _create_schema
    from cytome.utils.genome import CHROM_ORDER
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    assert import_fragments_streaming(conn, _fragment_file(tmp_path), {"c0": 0},
                                      chr_filter="all") == {"chr1": 1, "chrM": 1, "2L": 1}
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    with pytest.warns(UserWarning, match="2L"):
        got = import_fragments_streaming(conn, _fragment_file(tmp_path), {"c0": 0},
                                         chr_filter=set(CHROM_ORDER))
    assert got == {"chr1": 1, "chrM": 1}


def test_rtree_queries_find_features_at_the_edge(tmp_path):
    ds = _cells(tmp_path, ["a"])
    ds.set_entity("peaks", pd.DataFrame({"peak_id": ["p"], "chr": ["chr1"],
                                         "start": [200_000_001], "end_": [200_000_101]}))
    ds.flush()
    build_peak_index(ds._conn)
    ds._conn.commit()
    for start, end in ((200_000_100, 200_000_200), (199_999_900, 200_000_002)):
        assert [h[0] for h in query_peak_rtree(ds._conn, "chr1", start, end)] == [0]
    assert query_peak_rtree(ds._conn, "chr1", 200_000_101, 200_000_300) == []
    ds.set_spatial_coords(np.array([[16_777_217.0, 3.0]]))          # 2**24 + 1
    assert ds.cells_in_region((16_777_217.0, 16_777_218.0), (0, 10)).tolist() == [0]
    ds.close()


def test_a_region_query_reads_only_the_chunks_that_can_overlap(tmp_path, monkeypatch):
    import cytome.io.compression as comp
    ds = _cells(tmp_path, ["a", "b"])
    _frags(ds, "chr1", np.arange(0, 100_000, 1_000), [0, 1] * 50, chunk=10)   # 10 chunks
    decoded = []
    real = comp.decode_starts

    def counting(*a, **k):
        decoded.append(1)
        return real(*a, **k)
    monkeypatch.setattr(comp, "decode_starts", counting)
    hits = _query_region_from_chunks(ds._conn, "chr1", 15_050, 15_060)
    assert [h[0] for h in hits] == [15_000]
    assert len(decoded) == 2                     # the chunks starting before 15,060
    ds.close()


def _run(*args):
    return subprocess.run([sys.executable, "-m", "cytome.cli.main", *args],
                          capture_output=True, text=True, timeout=300)


def test_cli_reading_commands_do_not_change_the_file(tmp_path):
    ds = _cells(tmp_path, ["a", "b"])
    ds.close()
    p = tmp_path / "s.cytome"
    con = sqlite3.connect(str(p))
    con.execute("DROP TABLE cell_sets")                          # as an older store
    con.commit()
    con.close()
    stamp = (os.path.getsize(p), os.path.getmtime(p))
    for args in (("info", str(p)), ("validate", "--quick", str(p)),
                 ("provenance", str(p))):
        r = _run(*args)
        assert r.returncode == 0, r.stderr
    assert (os.path.getsize(p), os.path.getmtime(p)) == stamp
    con = sqlite3.connect(str(p))
    assert not con.execute("SELECT 1 FROM sqlite_master WHERE name = 'cell_sets'").fetchone()
    con.close()


def test_cli_subset_refuses_an_empty_sample_column_and_selects_rows(tmp_path):
    ds = _cells(tmp_path, ["a", "b", "a"])
    ds.close()
    p = tmp_path / "s.cytome"
    r = _run("subset", str(p), "-o", str(tmp_path / "o.cytome"), "--sample", "S1")
    assert r.returncode != 0 and "no sample_id" in r.stderr
    assert not (tmp_path / "o.cytome").exists()
    r = _run("subset", str(p), "-o", str(tmp_path / "q.cytome"), "--query", "g = 'a'")
    assert r.returncode == 0, r.stderr
    q = cytome.open(str(tmp_path / "q.cytome"))
    assert q.cells["barcode"].tolist() == ["c0", "c2"]
    q.close()


def test_cli_init_needs_force_to_overwrite(tmp_path):
    bc = tmp_path / "bc.tsv"
    bc.write_text("AAA\nCCC\n")
    out = tmp_path / "i.cytome"
    assert _run("init", "--barcodes", str(bc), "-o", str(out)).returncode == 0
    r = _run("init", "--barcodes", str(bc), "-o", str(out))
    assert r.returncode != 0 and "exists" in r.stderr
    assert _run("init", "--barcodes", str(bc), "-o", str(out), "--force").returncode == 0


def test_checkpoint_reports_and_compact_raises(tmp_path):
    ds = _cells(tmp_path, ["a"])
    res = ds.checkpoint("PASSIVE")
    assert set(res) == {"busy", "log_frames", "checkpointed"}
    with pytest.raises(ValueError, match="mode"):
        ds.checkpoint("NONSENSE")
    assert ds.compact()["busy"] in (0, 1)
    ds.close()


def test_from_10x_h5_keeps_float_values(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "f.h5"
    M = sp.csc_matrix(np.array([[0.4, 0], [0, 2.0]], dtype=np.float32))   # features x cells
    with h5py.File(p, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = M.data, M.indices, M.indptr
        g["shape"] = np.array(M.shape)
        g["barcodes"] = np.array([b"c0", b"c1"])
        feat = g.create_group("features")
        feat["id"] = np.array([b"g0", b"g1"])
        feat["name"] = np.array([b"G0", b"G1"])
        feat["feature_type"] = np.array([b"Gene Expression"] * 2)
    ds = cytome.from_10x_h5(str(p), str(tmp_path / "o.cytome"))
    m = ds.RNA.counts.to_memory().toarray()
    assert m[0, 0] == pytest.approx(0.4) and ds.matrix_meta("RNA_counts")["is_integer"] == 0
    ds.close()
