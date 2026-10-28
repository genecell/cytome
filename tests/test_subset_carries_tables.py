"""subset and filter_cells carry every table that does not depend on the
cells kept, and var embeddings of every feature entity.

Audit 2026-10-27, part 4 major 22 and 18, improvement 3 (L2 findings 10, 15):
``_raw_var``, ``_column_meta``, spatial images and scale factors, cell sets and
the feature index of a matrix stored both ways were dropped, and the peak
R-tree was not built again: to_anndata of a filtered store whose source had
.raw raised, categoricals came back as plain values, Visium images vanished;
the samples table was copied with the counts of the cells before. ATAC, gene
activity and tile var embeddings were not listed by ``var_embeddings`` and
filter_cells dropped them. A store in a folder that cannot be written could
not be subset.
"""
import json
import os
import stat

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.index.rtree import query_peak_rtree

anndata = pytest.importorskip("anndata")


@pytest.fixture
def rich(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1, (20, 6)).astype(np.float32))
    obs = pd.DataFrame({"ct": pd.Categorical(list("ab") * 10)},
                       index=[f"c{i}" for i in range(20)])
    a = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(6)]))
    a.raw = a.copy()
    a.uns["spatial"] = {"lib": {"images": {"hires": np.arange(16, dtype=np.uint8).reshape(4, 4)},
                                "scalefactors": {"tissue_hires_scalef": 0.5}}}
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    c = ds._conn
    c.execute("UPDATE cells SET sample_id = CASE WHEN cell_idx < 12 THEN 'S1' ELSE 'S2' END")
    c.executemany("INSERT INTO samples(sample_id, n_cells) VALUES (?, ?)", [("S1", 12), ("S2", 8)])
    c.execute("INSERT INTO cell_sets VALUES ('picked', ?)", (json.dumps([1, 5, 15]),))
    c.commit()
    ds.RNA.counts.build_feature_index(chunk_size=2)
    ds.add_var_embedding("RNA_varm_pc", np.arange(12.0).reshape(6, 2))
    return ds, X


def test_filter_cells_carries_the_cell_independent_tables(rich):
    ds, X = rich
    keep = np.arange(20) % 4 != 3                       # 15 cells: 9 of S1, 6 of S2
    ds.filter_cells(keep)
    b = ds.to_anndata()                                  # raised: _raw_var was empty
    assert b.raw is not None and (b.raw.X != X[keep]).nnz == 0
    assert str(b.obs["ct"].dtype) == "category"
    assert ds.spatial_images.keys() == [("lib", "hires")]
    assert ds.spatial_images.scalefactors("lib") == {"tissue_hires_scalef": 0.5}
    assert dict(ds._conn.execute("SELECT sample_id, n_cells FROM samples")) == {"S1": 9, "S2": 6}
    kept_rows = np.flatnonzero(keep).tolist()
    assert json.loads(ds._conn.execute("SELECT cell_indices FROM cell_sets").fetchone()[0]) \
        == [kept_rows.index(1), kept_rows.index(5)]
    assert ds.var_embeddings.keys() == ["RNA_varm_pc"]
    ds.close()


def test_a_matrix_stored_both_ways_keeps_its_feature_index(rich, tmp_path):
    ds, X = rich
    keep = np.array([0, 3, 4, 9, 17])
    out = ds.subset(keep, output=str(tmp_path / "s.cytome"))
    ml = out.RNA.counts
    assert ml.has_feature_index and not ml.is_column_major
    assert (ml.columns([5, 0, 2]) != X[keep][:, [5, 0, 2]]).nnz == 0
    assert out.validate(integrity=False).passed
    out.close()
    ds.close()


def test_cells_columns_keep_their_declared_types(rich, tmp_path):
    ds, _ = rich
    vals = np.array([1, None, 3] * 6 + [4, 5], dtype=object)
    ds.cells["score"] = pd.array(vals, dtype="Int64")
    ds.flush()
    out = ds.subset(np.arange(10), output=str(tmp_path / "s.cytome"))
    decl = {r[1]: r[2] for r in out._conn.execute("PRAGMA table_info(cells)")}
    assert decl["score"] == "INTEGER"
    assert out.cells["score"].tolist()[:3] == [1, None, 3]
    out.close()
    ds.close()


def test_var_embeddings_of_every_entity_are_listed_and_carried(tmp_path):
    ds = cytome.create(str(tmp_path / "p.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(4)]}))
    ds.set_entity("peaks", pd.DataFrame({"peak_id": ["chr1:1-100", "chr1:200-300"],
                                         "chr": ["chr1"] * 2, "start": [1, 200],
                                         "end_": [100, 300]}))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(np.ones((4, 2), dtype=np.float32)))
    ds.flush()
    from cytome.index.builder import build_peak_index
    build_peak_index(ds._conn)
    ds._conn.commit()
    ds.add_var_embedding("ATAC_varm_x", np.ones((2, 3)), entity="peaks")
    assert ds.var_embeddings.keys() == ["ATAC_varm_x"] and "ATAC_varm_x" in ds.var_embeddings
    ds.filter_cells(np.array([True, True, False, True]))
    assert ds.var_embeddings.keys() == ["ATAC_varm_x"]
    assert ds.var_embeddings.entity_of("ATAC_varm_x") == "peaks"
    assert [h[0] for h in query_peak_rtree(ds._conn, "chr1", 150, 250)] == [1]
    ds.close()


def test_a_store_in_a_read_only_folder_can_be_subset(rich, tmp_path):
    ds, X = rich
    path = ds.path
    ds.close()
    folder = os.path.dirname(path)
    mode = os.stat(folder).st_mode
    os.chmod(folder, stat.S_IRUSR | stat.S_IXUSR)
    try:
        src = cytome.open(str(path), mode="r")
        out_dir = tmp_path.parent / (tmp_path.name + "_out")
        out_dir.mkdir()
        out = src.subset(np.arange(5), output=str(out_dir / "s.cytome"))
        assert out.n_cells == 5
        assert (out.RNA.counts.to_memory() != X[:5]).nnz == 0
        assert out._conn.execute("SELECT COUNT(*) FROM _raw_var").fetchone()[0] == 6
        out.close()
        src.close()
    finally:
        os.chmod(folder, mode)
