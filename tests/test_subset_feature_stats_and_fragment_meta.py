"""2026-10-16: what a subset carries about the cells it removed, and the extent of each chromosome.

- A feature table can cache sums over the cells (``n_cells_accessible``,
  ``tile_total_counts``), with the importer's guard over them in the metadata. A subset
  that removes cells computes them again for the kept cells, in either layout, with a new
  guard; columns without a guard are left out, and so are a split's. A subset that keeps
  every cell carries both as they are. Per-cell sums on the cells table stay valid.
- ``fragment_meta.max_end`` is the largest end of a chromosome, not the last fragment's
  end: an earlier, longer fragment can end after it. Import, subset and merge record it.
- A source whose chunks turn out unsorted after the streaming subset has written some of
  them is redone from the start, not appended to.
"""
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
import cytome.io.subset as subset_mod
from cytome.core.dataset import CytomeDataset
from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks


def _fragments_of(rng, n_cells, m):
    """Sorted fragments with one long fragment early on that ends after all others."""
    s = np.sort(rng.integers(1_000, 900_000, size=m)).astype(np.int32)
    e = (s + rng.integers(50, 400, size=m)).astype(np.int32)
    e[3] = 2_000_000                                      # starts early, ends last
    c = rng.integers(0, n_cells, size=m).astype(np.int32)
    c[3] = 0                                              # a cell that every subset keeps
    return s, e, c


def _store(path, n=40, t=300, seed=1, guard=True, csc=False, density=0.1):
    rng = np.random.default_rng(seed)
    ds = CytomeDataset(path, mode="w")
    T = sp.random(n, t, density=density, format="csr", dtype=np.float32, random_state=rng)
    T.data = np.ceil(T.data * 5).astype(np.float32)
    nnz_col = np.diff(T.tocsc().indptr).astype(np.int64)
    sum_col = np.asarray(T.sum(axis=0)).ravel().astype(np.int64)
    ds.set_entity("cells", pd.DataFrame({
        "barcode": [f"BC{i:03d}" for i in range(n)],
        "n_tiles_accessible": np.diff(T.indptr).astype(np.int64),
        "tile_total_counts": np.asarray(T.sum(axis=1)).ravel().astype(np.int64)}))
    ds.set_entity("tiles", pd.DataFrame({
        "tile_id": [f"chr1:{i * 500}-{i * 500 + 500}" for i in range(t)],
        "chr": "chr1", "start": np.arange(t) * 500, "end_": np.arange(t) * 500 + 500,
        "n_cells_accessible": nnz_col, "tile_total_counts": sum_col}))
    ds.flush()
    ds.add_matrix("tiles_counts", T)
    ds.flush()
    ds._conn.execute("UPDATE matrix_meta SET col_entity = 'tiles' WHERE matrix_name = 'tiles_counts'")
    if csc:
        from cytome.core.measurement import MeasurementLayer
        MeasurementLayer(ds._conn, "tiles_counts").build_feature_index(chunk_size=64)
        ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name='tiles_counts'")
        ds._conn.execute("UPDATE matrix_meta SET n_chunks = 0 WHERE matrix_name='tiles_counts'")
    if guard:
        ds._conn.execute("INSERT OR REPLACE INTO _metadata (key, value, value_type) VALUES (?, ?, "
                         "'scalar')", ("importer_stats:tiles_counts", json.dumps(
                             {"matrix": "tiles_counts", "n_rows": n, "n_cols": t,
                              "n_nonzero": int(T.nnz), "counting": "fragment"})))
    ds.metadata["counting:tiles_counts"] = "fragment"
    frags = {}
    for chrom, m in (("chr1", 700), ("chr2", 300)):
        s, e, c = _fragments_of(rng, n, m)
        _write_block_chunks(ds._conn, chrom, s, e, c, chunk_size=64, compression="lz4",
                            compression_level=1, chunk_idx_start=0, encoding=1)
        _update_fragment_meta(ds._conn, chrom, m)
        frags[chrom] = (s, e, c)
    ds._conn.commit()
    ds.close()
    return str(path), T, frags


def _columns(ds, table):
    return {r[1] for r in ds._conn.execute(f"PRAGMA table_info({table})")}


def _meta(ds):
    return dict((r[0], (r[1], r[2])) for r in ds._conn.execute(
        "SELECT chrom, n_fragments, max_end FROM fragment_meta"))


def test_update_fragment_meta_takes_the_largest_end(tmp_path):
    p, _, frags = _store(tmp_path / "s.cytome")
    ds = cytome.open(p, mode="r")
    for chrom, (s, e, _c) in frags.items():
        n, max_end = _meta(ds)[chrom]
        assert n == len(s) and max_end == int(e.max()) == 2_000_000 > int(e[-1])
        assert ds._conn.execute("SELECT min_start FROM fragment_meta WHERE chrom = ?",
                                (chrom,)).fetchone()[0] == int(s[0])
    ds.close()


def _sums(path):
    ds = cytome.open(path, mode="r")
    try:
        rows = ds._conn.execute("SELECT n_cells_accessible, tile_total_counts FROM tiles "
                                "ORDER BY tile_idx").fetchall()
        guard = ds._conn.execute("SELECT value FROM _metadata WHERE key = "
                                 "'importer_stats:tiles_counts'").fetchone()
        return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=float),
                json.loads(guard[0]) if guard else None)
    finally:
        ds.close()


@pytest.mark.parametrize("csc", [False, True])
def test_a_subset_that_removes_cells_recomputes_the_guarded_sums(tmp_path, csc):
    p, T, frags = _store(tmp_path / "s.cytome", csc=csc)
    keep = np.ones(T.shape[0], dtype=bool)
    keep[[5, 6, 17, 39]] = False
    ds = cytome.open(p)
    out = ds.subset(keep, output=str(tmp_path / "sub.cytome"))
    ids = "SELECT tile_id FROM tiles ORDER BY tile_idx"
    assert out._conn.execute(ids).fetchall() == ds._conn.execute(ids).fetchall()
    assert out.metadata["counting:tiles_counts"] == "fragment"
    cells = out.cells.to_pandas()                      # per-cell sums stay valid
    assert cells["n_tiles_accessible"].tolist() == np.diff(T[keep].indptr).tolist()
    meta = _meta(out)
    for chrom, (s, e, c) in frags.items():
        k = keep[c]
        assert meta[chrom] == (int(k.sum()), int(e[k].max()))
    out.close(); ds.close()
    nca, tot, guard = _sums(str(tmp_path / "sub.cytome"))
    Tk = T[keep].tocsc()
    assert nca.tolist() == np.asarray((Tk > 0).sum(axis=0)).ravel().tolist()
    assert np.array_equal(tot, np.asarray(Tk.sum(axis=0), dtype=float).ravel())
    assert guard == {"matrix": "tiles_counts", "n_rows": int(keep.sum()), "n_cols": T.shape[1],
                     "n_nonzero": int(Tk.nnz), "counting": "fragment"}


def test_sums_without_a_guard_are_left_out(tmp_path):
    p, T, _ = _store(tmp_path / "s.cytome", guard=False)
    keep = np.ones(T.shape[0], dtype=bool)
    keep[:3] = False
    ds = cytome.open(p)
    out = ds.subset(keep, output=str(tmp_path / "sub.cytome"))
    assert not {"n_cells_accessible", "tile_total_counts"} & _columns(out, "tiles")
    assert "importer_stats:tiles_counts" not in out.metadata
    out.close(); ds.close()


def test_the_old_behaviour_leaves_them_out(tmp_path, monkeypatch):
    """The switch the benchmark of the two behaviours uses."""
    monkeypatch.setattr(subset_mod, "_RECOMPUTE_FEATURE_SUMS", False)
    p, T, _ = _store(tmp_path / "s.cytome")
    keep = np.ones(T.shape[0], dtype=bool)
    keep[:3] = False
    ds = cytome.open(p)
    out = ds.subset(keep, output=str(tmp_path / "sub.cytome"))
    assert not {"n_cells_accessible", "tile_total_counts"} & _columns(out, "tiles")
    assert "importer_stats:tiles_counts" not in out.metadata
    out.close(); ds.close()


def test_a_subset_of_every_cell_keeps_them(tmp_path):
    p, T, _ = _store(tmp_path / "s.cytome")
    ds = cytome.open(p)
    out = ds.subset(np.ones(T.shape[0], dtype=bool), output=str(tmp_path / "all.cytome"))
    a = ds._conn.execute("SELECT n_cells_accessible, tile_total_counts FROM tiles "
                         "ORDER BY tile_idx").fetchall()
    b = out._conn.execute("SELECT n_cells_accessible, tile_total_counts FROM tiles "
                          "ORDER BY tile_idx").fetchall()
    assert a == b and len(a) == T.shape[1]
    assert out.metadata["importer_stats:tiles_counts"]["n_rows"] == T.shape[0]
    out.close(); ds.close()


def test_split_stores_leave_out_the_feature_sums(tmp_path):
    p, T, _ = _store(tmp_path / "s.cytome")
    labels = np.where(np.arange(T.shape[0]) % 3 == 0, "x", "y")
    ds = cytome.open(p)
    stores = ds.split(labels, {"x": str(tmp_path / "x.cytome"), "y": str(tmp_path / "y.cytome")},
                      ["tiles_counts"])
    for st in stores.values():
        assert not {"n_cells_accessible", "tile_total_counts"} & _columns(st, "tiles")
        assert "importer_stats:tiles_counts" not in st.metadata
        st.close()
    ds.close()


def test_merge_records_the_largest_end(tmp_path):
    pa, _, fa = _store(tmp_path / "a.cytome", seed=2)
    pb, _, fb = _store(tmp_path / "b.cytome", seed=3)
    a, b = cytome.open(pa, mode="r"), cytome.open(pb, mode="r")
    out = cytome.merge([a, b], output=str(tmp_path / "m.cytome"), batch_labels=["A", "B"])
    meta = _meta(out)
    for chrom in ("chr1", "chr2"):
        assert meta[chrom] == (len(fa[chrom][0]) + len(fb[chrom][0]), 2_000_000)
    out.close(); a.close(); b.close()


def test_an_unsorted_source_is_redone_after_chunks_were_written(tmp_path):
    """The streaming pass flushes full chunks as it goes; disorder found later restarts
    the chromosome from an empty slate instead of colliding with those chunks."""
    p, T, frags = _store(tmp_path / "s.cytome")
    ds = cytome.open(p)
    s, e, c = frags["chr1"]
    s2 = s.copy()
    s2[650] = 10                                          # out of order in a late chunk
    ds._conn.execute("DELETE FROM fragment_chunks WHERE chrom = 'chr1'")
    _write_block_chunks(ds._conn, "chr1", s2, e, c, chunk_size=64, compression="lz4",
                        compression_level=1, chunk_idx_start=0, encoding=1)
    ds._conn.commit()
    out = CytomeDataset(tmp_path / "o.cytome", mode="w")
    out._conn.execute("""CREATE TABLE IF NOT EXISTS fragment_meta (chrom TEXT PRIMARY KEY,
        n_fragments INTEGER NOT NULL, table_name TEXT NOT NULL, rtree_name TEXT NOT NULL,
        min_start INTEGER, max_end INTEGER)""")
    remap = np.arange(T.shape[0], dtype=np.int32)       # keep every cell
    with pytest.warns(RuntimeWarning, match="not start-sorted"):
        subset_mod._subset_from_chunks(ds.ATAC.fragments, out._conn, remap, T.shape[0], 32)
    out._conn.commit()
    parts = list(out.ATAC.fragments.iter_chromosome_chunks("chr1"))
    got = np.concatenate([q[0] for q in parts])
    assert np.array_equal(got, np.sort(s2, kind="mergesort")) and len(got) == len(s2)
    assert _meta(out)["chr1"] == (len(s2), 2_000_000)
    out.close(); ds.close()


def test_column_sums_handle_empty_columns_and_explicit_zeros():
    data = np.array([2, 0.5, 3, 0, 4], dtype=np.float32)       # one stored zero
    ptr = np.array([0, 0, 2, 2, 5, 5])                          # empty, 2, empty, 3, empty
    nca, tot = subset_mod._FeatureSums.column_sums(data, ptr)
    assert nca.tolist() == [0, 2, 0, 2, 0] and tot.tolist() == [0, 2.5, 0, 7, 0]
    nca, tot = subset_mod._FeatureSums.column_sums(np.array([], np.float32), np.zeros(4, np.int64))
    assert nca.tolist() == [0, 0, 0] and tot.tolist() == [0, 0, 0]


@pytest.mark.parametrize("csc", [False, True])
def test_sums_are_written_in_small_statements_too(tmp_path, monkeypatch, csc):
    """An SQLite older than 3.32 takes 999 bound parameters a statement; the writer splits."""
    real_init = subset_mod._FeatureSums.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self.rows_per_statement = 7
    monkeypatch.setattr(subset_mod._FeatureSums, "__init__", init)
    p, T, _ = _store(tmp_path / "s.cytome", csc=csc)
    keep = np.ones(T.shape[0], dtype=bool)
    keep[::4] = False
    ds = cytome.open(p)
    ds.subset(keep, output=str(tmp_path / "sub.cytome")).close()
    ds.close()
    nca, tot, guard = _sums(str(tmp_path / "sub.cytome"))
    Tk = T[keep].tocsc()
    assert nca.tolist() == np.asarray((Tk > 0).sum(axis=0)).ravel().tolist()
    assert np.array_equal(tot, np.asarray(Tk.sum(axis=0), dtype=float).ravel())
    assert guard["n_rows"] == int(keep.sum())
