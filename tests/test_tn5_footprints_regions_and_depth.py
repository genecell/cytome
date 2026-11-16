"""Tn5 footprints: regions inside one another are counted in full, and no depth is assumed.

Two defects of the first version of ``get_tn5_footprints``:
- fragment chunks were skipped by a binary search over the regions' ends, which are sorted only when no
  region lies inside another. A wide region given with a narrow one inside it lost most of its cut sites;
- with ``groupby`` and no fragment count per cell in the store, ``normalize='cpm'`` divided by an assumed
  5,000 fragments per cell.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

import cytome
from cytome.io.convert_fragments import import_fragments_streaming

N_CELLS = 12


def _store(tmp_path, seed=3):
    rng = np.random.default_rng(seed)
    barcodes = [f"c{i:02d}" for i in range(N_CELLS)]
    ds = cytome.create(tmp_path / "f.cytome")
    ds.set_entity("cells", pd.DataFrame({"barcode": barcodes, "kind": ["a"] * 4 + ["b"] * 8}))
    ds.flush()
    starts = np.sort(rng.integers(0, 150_000, 4000))
    ends = starts + rng.integers(40, 400, len(starts))
    cells = rng.integers(0, N_CELLS, len(starts))
    path = tmp_path / "fragments.tsv.gz"
    with gzip.open(path, "wt") as f:
        for s, e, c in zip(starts, ends, cells):
            f.write(f"chr1\t{s}\t{e}\t{barcodes[c]}\t1\n")
    mapping = {str(b): int(i) for i, b in ds._conn.execute("SELECT cell_idx, barcode FROM cells")}
    import_fragments_streaming(ds._conn, path, mapping, chunk_size=200)   # twenty chunks, each with its extent
    ds._conn.commit()
    ds.flush()
    assert ds._conn.execute("SELECT COUNT(*), COUNT(max_end) FROM fragment_chunks").fetchone() == (20, 20)
    return ds, starts, ends, cells


def _cuts_in(starts, ends, lo, hi, keep=None):
    keep = np.ones(len(starts), bool) if keep is None else keep
    c5, c3 = starts[keep], ends[keep] - 1
    return int(((c5 >= lo) & (c5 < hi)).sum() + ((c3 >= lo) & (c3 < hi)).sum())


def test_a_region_is_counted_in_full_beside_a_region_inside_it(tmp_path):
    ds, starts, ends, _ = _store(tmp_path)
    wide, narrow = ("chr1", 0, 150_000), ("chr1", 500, 900)
    alone = ds.get_tn5_footprints(regions=[wide], normalize="raw")
    both = ds.get_tn5_footprints(regions=[wide, narrow], normalize="raw")
    flipped = ds.get_tn5_footprints(regions=[narrow, wide], normalize="raw")
    expected = _cuts_in(starts, ends, 0, 150_000)
    assert int(np.asarray(alone).sum()) == expected
    assert int(both[0].sum()) == expected                 # was a fraction of it
    assert int(flipped[1].sum()) == expected
    assert int(both[1].sum()) == int(flipped[0].sum()) == _cuts_in(starts, ends, 500, 900)
    np.testing.assert_array_equal(both[0], np.asarray(alone)[0])
    ds.close()


def test_overlapping_and_nested_regions_match_a_direct_count(tmp_path):
    ds, starts, ends, cells = _store(tmp_path)
    regions = [("chr1", 10_000, 90_000), ("chr1", 20_000, 21_000), ("chr1", 20_500, 140_000), ("chr1", 60_000, 60_200),
               ("chr1", 149_000, 150_500)]
    out = ds.get_tn5_footprints(regions=regions, groupby="kind", normalize="raw")
    for (_, lo, hi), arr in zip(regions, out):
        assert arr.shape == (2, hi - lo)
        assert int(arr[0].sum()) == _cuts_in(starts, ends, lo, hi, cells < 4)
        assert int(arr[1].sum()) == _cuts_in(starts, ends, lo, hi, cells >= 4)
    ds.close()


def test_cuts_per_million_use_the_cells_own_fragment_counts(tmp_path):
    ds, starts, ends, cells = _store(tmp_path)
    region = [("chr1", 0, 150_000)]
    raw = ds.get_tn5_footprints(regions=region, groupby="kind", normalize="raw")
    cpm = ds.get_tn5_footprints(regions=region, groupby="kind", normalize="cpm", return_dict=True)
    n_frag = np.array([(cells < 4).sum(), (cells >= 4).sum()], dtype=np.float64)
    np.testing.assert_array_equal(cpm["total_fragments"], n_frag)
    np.testing.assert_allclose(cpm["footprints"][0], raw[0] * (1e6 / (2.0 * n_frag))[:, None], rtol=1e-5)
    ds.close()


def test_no_depth_is_assumed_where_the_store_records_none(tmp_path):
    ds, *_ = _store(tmp_path)
    ds._conn.execute("UPDATE cells SET n_fragments = NULL")
    ds._conn.commit()
    region = [("chr1", 0, 1000)]
    with pytest.raises(ValueError, match="n_fragments"):
        ds.get_tn5_footprints(regions=region, groupby="kind")          # normalize='cpm' is the default
    raw = ds.get_tn5_footprints(regions=region, groupby="kind", normalize="raw", return_dict=True)
    assert raw["footprints"].shape == (1, 2, 1000)
    assert raw["total_fragments"].tolist() == [0.0, 0.0]               # not 5,000 per cell
    per_cell = ds.get_tn5_footprints(regions=region, groupby="kind", normalize="per_cell")
    np.testing.assert_allclose(per_cell[0], raw["footprints"][0] / np.array([4.0, 8.0])[:, None], rtol=1e-6)
    ds.close()


def test_an_unknown_normalization_is_refused_before_any_read(tmp_path):
    ds, *_ = _store(tmp_path)
    with pytest.raises(ValueError, match="normalize"):
        ds.get_tn5_footprints(regions=[("chr1", 0, 1000)], normalize="rpkm")
    ds.close()
