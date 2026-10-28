"""AnnData conversion keeps names, keys and matrices apart.

Audit 2026-10-27, part 4: critical 5 (to_anndata exported cell_idx and
gene_idx, and from_anndata kept them as keys: after a subset of genes, gene 2's
lookups read gene 4's column), critical 7 (a layer named like X's matrix
replaced X), major 24 (from_h5ad ignored counts_layer on both paths, and the
streaming path keyed genes differently and failed on duplicate names), major
25 (to_anndata made the barcode column the obs_names, the Ensembl IDs the
var_names of a 10x object, raw a layer, and failed on graphs with cell_mask).
L2 minors: compression ignored, "/" in an embedding name, update_from_anndata
without a count check, a dense X probed 20,000 rows at a time; improvement
4 (a layers= option).
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome

anndata = pytest.importorskip("anndata")


def _counts(n=6, m=5, seed=0):
    return sp.csr_matrix(np.random.default_rng(seed).poisson(1, (n, m)).astype(np.float32))


def _adata(**kw):
    X = kw.pop("X", _counts())
    obs = kw.pop("obs", pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]))
    var = kw.pop("var", pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]))
    return anndata.AnnData(X=X, obs=obs, var=var, **kw)


def _layers(a):
    """The layers' names; anndata >= 0.13 also lists X, under the key None."""
    return sorted(k for k in a.layers if k is not None)


def test_row_keys_are_not_exported_nor_taken_back(tmp_path):
    a = _adata()
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    b = ds.to_anndata()
    assert "cell_idx" not in b.obs.columns and "gene_idx" not in b.var.columns
    # an AnnData exported by an earlier version, subset: its keys are not positions
    b.obs["cell_idx"] = np.arange(6)
    b.var["gene_idx"] = np.arange(5)
    sub = b[1:, ["g2", "g3", "g4"]].copy()
    with pytest.warns(UserWarning, match="row key"):
        ds2 = cytome.from_anndata(sub, output=str(tmp_path / "b.cytome"))
    assert ds2.genes["gene_idx"].tolist() == [0, 1, 2]
    assert ds2.cells["cell_idx"].tolist() == [0, 1, 2, 3, 4]
    gi = int(ds2._conn.execute("SELECT gene_idx FROM genes WHERE gene_id = 'g2'").fetchone()[0])
    col = cytome.read_feature_column(ds2, "RNA", "counts", gi)
    assert np.array_equal(col, a.X[1:, 2].toarray().ravel())
    ds.close()
    ds2.close()


def test_stratified_downsample_returns_positions(tmp_path):
    a = _adata(obs=pd.DataFrame({"g": list("aabbcc")}, index=[f"c{i}" for i in range(6)]))
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    ds._conn.execute("UPDATE cells SET cell_idx = cell_idx + 100")   # keys that are not rows
    ds._conn.commit()
    keep = ds.downsample(n_cells=3, method="stratified", groupby="g", seed=1)
    assert keep.max() < 6 and np.all(np.diff(keep) > 0)
    ds.close()


def test_a_layer_named_like_x_is_refused_unless_it_is_x(tmp_path):
    rng = np.random.default_rng(1)
    norm = sp.csr_matrix(rng.random((6, 5)).astype(np.float32))
    a = _adata(X=norm.copy(), layers={"data": norm * 2, "counts": _counts()})
    with pytest.raises(ValueError, match="would both be stored as 'RNA_data'"):
        cytome.from_anndata(a, output=str(tmp_path / "x.cytome"))
    assert not (tmp_path / "x.cytome").exists()          # refused before writing
    ds = cytome.from_anndata(a, output=str(tmp_path / "y.cytome"), main_layer_name="scale_data")
    assert (ds.RNA.layer("scale_data").to_memory() != norm).nnz == 0
    assert (ds.RNA.layer("data").to_memory() != norm * 2).nnz == 0
    ds.close()
    # counts_layer= with main_layer_name='counts' named two matrices RNA_counts
    with pytest.raises(ValueError, match="RNA_counts"):
        cytome.from_anndata(a, output=str(tmp_path / "z.cytome"), counts_layer="counts",
                            main_layer_name="counts")


def test_a_layer_that_is_x_is_stored_once_and_comes_back(tmp_path):
    X = _counts()
    a = _adata(X=X.copy(), layers={"counts": X.copy()})
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    assert ds.list_matrices() == ["RNA_counts"]
    b = ds.to_anndata()
    assert (b.layers["counts"] != X).nnz == 0
    ds.close()


@pytest.mark.parametrize("backed", [False, True])
def test_from_h5ad_honours_counts_layer(tmp_path, backed):
    rng = np.random.default_rng(2)
    norm = sp.csr_matrix(rng.random((6, 5)).astype(np.float32))
    umi = _counts()
    p = tmp_path / "x.h5ad"
    _adata(X=norm, layers={"umi": umi}).write_h5ad(p)
    ds = cytome.from_h5ad(p, str(tmp_path / "o.cytome"), counts_layer="umi", backed=backed,
                          verbose=False)
    assert sorted(ds.list_matrices()) == ["RNA_counts", "RNA_data"]
    assert (ds.RNA.counts.to_memory() != umi).nnz == 0
    ds.close()
    _adata(X=norm, layers={"lognorm": norm}).write_h5ad(tmp_path / "y.h5ad")
    with pytest.raises(ValueError, match="integer"):
        cytome.from_h5ad(tmp_path / "y.h5ad", str(tmp_path / "p.cytome"), counts_layer="lognorm",
                         backed=backed, verbose=False)


def test_backed_and_in_memory_key_genes_alike_and_take_duplicates(tmp_path):
    var = pd.DataFrame({"gene_ids": [f"ENSG{i}" for i in range(5)]},
                       index=["A", "B", "B", "C", "D"])
    p = tmp_path / "x.h5ad"
    with pytest.warns(UserWarning):
        _adata(var=var).write_h5ad(p)
    out = {}
    for backed in (False, True):
        with pytest.warns(UserWarning, match="duplicate gene symbol"):
            ds = cytome.from_h5ad(p, str(tmp_path / f"{backed}.cytome"), backed=backed,
                                  verbose=False)
        out[backed] = ds.genes.to_pandas()[["gene_id", "symbol"]]
        b = ds.to_anndata()
        assert list(b.var_names) == ["A", "B", "B", "C", "D"]
        ds.close()
    pd.testing.assert_frame_equal(out[False], out[True])
    assert out[True]["gene_id"].tolist() == [f"ENSG{i}" for i in range(5)]


def test_obs_and_var_names_come_back(tmp_path):
    obs = pd.DataFrame({"barcode": [f"AAA{i}" for i in range(6)]},
                       index=[f"lib{i % 2}_AAA{i}" for i in range(6)])
    var = pd.DataFrame({"gene_ids": [f"ENSG{i}" for i in range(5)]},
                       index=[f"SYM{i}" for i in range(5)])
    a = _adata(obs=obs, var=var)
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    b = ds.to_anndata()
    assert list(b.obs_names) == list(obs.index)
    assert b.obs["barcode"].tolist() == obs["barcode"].tolist()
    assert "_obs_name" not in b.obs.columns
    assert list(b.var_names) == list(var.index)
    assert b.var["gene_ids"].tolist() == var["gene_ids"].tolist()
    ds.close()


def test_raw_is_raw_not_a_layer_and_cell_mask_cuts_graphs_and_raw(tmp_path):
    X = _counts()
    a = _adata(X=X)
    a.raw = a.copy()
    a.obsp["connectivities"] = sp.csr_matrix(np.eye(6, dtype=np.float32))
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    b = ds.to_anndata()
    assert _layers(b) == [] and b.raw is not None and b.raw.shape == (6, 5)
    keep = np.array([1, 4])
    c = ds.to_anndata(cell_mask=keep)
    assert c.shape == (2, 5) and c.obsp["connectivities"].shape == (2, 2)
    assert c.raw.shape == (2, 5)
    assert (c.raw.X != X[keep]).nnz == 0
    ds.close()


def test_to_anndata_layers_option(tmp_path):
    X = _counts()
    a = _adata(X=X, layers={"a": X * 2, "b": X * 3})
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    assert _layers(ds.to_anndata()) == ["a", "b"]
    assert _layers(ds.to_anndata(layers=["b"])) == ["b"]
    assert _layers(ds.to_anndata(layers=["RNA_a"])) == ["a"]
    assert _layers(ds.to_anndata(layers=[])) == []
    ds.close()


def test_compression_is_honoured(tmp_path):
    ds = cytome.from_anndata(_adata(), output=str(tmp_path / "a.cytome"), compression="lz4")
    assert {r[0] for r in ds._conn.execute("SELECT DISTINCT compression FROM matrix_chunks")} \
        == {"lz4"}
    ds.close()


def test_slash_in_embedding_names_writes_a_readable_h5ad(tmp_path):
    a = _adata()
    a.obsm["X_umap/2d"] = np.ones((6, 2))
    a.layers["log/norm"] = a.X.copy() * 0.5
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    b = ds.to_anndata()
    p = tmp_path / "o.h5ad"
    b.write_h5ad(p)
    c = anndata.read_h5ad(p)
    ds2 = cytome.from_anndata(c, output=str(tmp_path / "b.cytome"))
    assert "X_umap/2d" in ds2.metadata["_anndata_obsm_map"].values()
    assert "log/norm" in ds2.metadata["_anndata_layer_map"].values()
    ds.close()
    ds2.close()


def test_update_from_anndata_refuses_other_cells(tmp_path):
    ds = cytome.from_anndata(_adata(), output=str(tmp_path / "a.cytome"))
    b = ds.to_anndata()
    with pytest.raises(ValueError, match="cells"):
        cytome.io.convert_anndata.update_from_anndata(ds, b[:4].copy())
    b.obs_names = [f"x{i}" for i in range(6)]
    b.obs["barcode"] = [f"x{i}" for i in range(6)]
    with pytest.raises(ValueError, match="barcodes"):
        cytome.io.convert_anndata.update_from_anndata(ds, b)
    good = ds.to_anndata()
    good.obs["leiden"] = pd.Categorical(list("001122"))
    cytome.io.convert_anndata.update_from_anndata(ds, good)
    assert ds.cells["leiden"].tolist() == list("001122")
    ds.close()


def test_dense_x_probe_reads_a_bounded_slice():
    from cytome.io.convert_anndata import _h5_values_are_integer

    asked = []

    class Dense:
        ndim, shape = 2, (1_000_000, 50_000)

        def __getitem__(self, sl):
            n = sl.stop - (sl.start or 0)
            asked.append(n)
            return np.ones((min(n, 10), self.shape[1]))

    assert _h5_values_are_integer(Dense()) is True
    assert asked and max(asked) * 50_000 <= 50_000, f"rows read: {asked}"
