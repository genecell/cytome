from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome


anndata = pytest.importorskip("anndata")


def _make_rich_adata(n_obs: int = 80, n_vars: int = 40):
    X = sp.random(n_obs, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=7)
    obs = pd.DataFrame(
        {
            "barcode": [f"CELL{i:04d}" for i in range(n_obs)],
            "cell_type": pd.Categorical(
                np.random.choice(["T", "B", "NK"], n_obs),
                categories=["T", "B", "NK", "Mono"],
            ),
            "n_genes": np.random.randint(200, 5000, n_obs).astype(np.int32),
            "pct_mito": np.random.uniform(0, 20, n_obs).astype(np.float32),
            "is_doublet": np.random.choice([True, False], n_obs),
        }
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"ENSG{i:05d}" for i in range(n_vars)],
            "symbol": [f"GENE{i}" for i in range(n_vars)],
            "gene_type": pd.Categorical(
                np.random.choice(["protein_coding", "lincRNA"], n_vars),
                categories=["protein_coding", "lincRNA", "miRNA"],
            ),
        }
    )
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.layers["spliced"] = sp.random(
        n_obs, n_vars, density=0.08, format="csr", dtype=np.float32, random_state=8
    )
    adata.obsm["X_pca"] = np.random.randn(n_obs, 15).astype(np.float32)
    adata.obsm["protein_expression"] = np.random.randn(n_obs, 5).astype(np.float32)
    adata.varm["PCs"] = np.random.randn(n_vars, 15).astype(np.float32)
    adata.obsp["connectivities"] = sp.random(
        n_obs, n_obs, density=0.02, format="csr", dtype=np.float32, random_state=9
    )
    adata.varp["gene_corr"] = sp.random(
        n_vars, n_vars, density=0.03, format="csr", dtype=np.float32, random_state=10
    )
    adata.uns["cell_type_colors"] = np.array(["#ff0000", "#00ff00", "#0000ff", "#aaaaaa"])
    adata.uns["pca"] = {
        "variance": np.random.uniform(0, 10, 15).astype(np.float32),
        "variance_ratio": np.random.uniform(0, 0.1, 15).astype(np.float32),
    }
    return adata


def test_full_roundtrip_fidelity(tmp_path):
    adata = _make_rich_adata()
    path = tmp_path / "fidelity.cytome"
    ds = cytome.from_anndata(adata, output=path)
    ds.close()

    with cytome.open(path) as rd:
        back = rd.to_anndata()

    assert back.shape == adata.shape
    assert set(back.layers.keys()) == set(adata.layers.keys())
    assert set(back.obsm.keys()) == set(adata.obsm.keys())
    assert "PCs" in back.varm
    assert set(back.obsp.keys()) == set(adata.obsp.keys())
    assert set(back.varp.keys()) == set(adata.varp.keys())
    assert back.obs["cell_type"].dtype.name == "category"
    assert list(back.obs["cell_type"].cat.categories) == list(adata.obs["cell_type"].cat.categories)
    assert back.obs["is_doublet"].dtype == bool
    assert isinstance(back.uns["cell_type_colors"], np.ndarray)
    assert back.uns["pca"]["variance"].dtype == np.float32
    np.testing.assert_allclose(back.obsm["X_pca"], adata.obsm["X_pca"], rtol=1e-5)
    np.testing.assert_allclose(back.varm["PCs"], adata.varm["PCs"], rtol=1e-5)


def test_roundtrip_with_raw(tmp_path):
    adata = _make_rich_adata(n_obs=60, n_vars=30)
    adata_full = adata.copy()
    adata.raw = adata_full
    adata = adata[:, :20].copy()

    path = tmp_path / "raw.cytome"
    ds = cytome.from_anndata(adata, output=path)
    ds.close()
    with cytome.open(path) as rd:
        back = rd.to_anndata()

    assert back.raw is not None
    assert back.n_vars == 20
    assert back.raw.n_vars == 30
