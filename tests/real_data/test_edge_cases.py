"""Edge-case validation tests with real data patterns."""

from __future__ import annotations

import os

import pytest

DATA_DIR = "tests/real_data/data"
PANCREAS_PATH = os.path.join(DATA_DIR, "Pancreas_with_cc_anndata.h5ad")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PANCREAS_PATH),
    reason="Pancreas dataset not downloaded.",
)


def test_categorical_with_unused_categories(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(PANCREAS_PATH)
    cat_cols = [c for c in adata.obs.columns if hasattr(adata.obs[c], "cat")]
    if not cat_cols:
        pytest.skip("No categorical columns")

    col = cat_cols[0]
    first = adata.obs[col].cat.categories[0]
    adata_sub = adata[adata.obs[col] == first].copy()
    path = str(tmp_path / "cat_test.cytome")
    ds = cytome.from_anndata(adata_sub, modality="RNA", output=path)
    ds.flush()
    assert ds.n_cells == adata_sub.shape[0]
    ds.close()


def test_layers_different_dtypes(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(PANCREAS_PATH)
    if not adata.layers:
        pytest.skip("No layers")

    path = str(tmp_path / "dtype_test.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=path)
    ds.flush()
    for k in adata.layers.keys():
        arr = ds.RNA.layer(k)[:10, :10]
        assert arr.shape == (10, 10)
    ds.close()


def test_var_names_special_characters(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(PANCREAS_PATH)
    path = str(tmp_path / "special_test.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=path)
    g = ds.genes.to_pandas()
    assert g.shape[0] == adata.shape[1]
    ds.close()


def test_empty_uns_keys(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(PANCREAS_PATH)
    adata.uns["empty_dict"] = {}
    adata.uns["empty_list"] = []
    adata.uns["none_value"] = None

    path = str(tmp_path / "uns_test.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=path)
    ds.flush()
    # key preservation depends on converter policy; at minimum conversion should succeed
    assert ds.n_cells == adata.shape[0]
    ds.close()
