"""Validation on dentate gyrus real dataset."""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp

DATA_DIR = "tests/real_data/data"
DG_PATH = os.path.join(DATA_DIR, "DentateGyrus_anndata.h5ad")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DG_PATH),
    reason="DentateGyrus dataset not downloaded.",
)


def test_conversion_and_roundtrip(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(DG_PATH)
    path = str(tmp_path / "dg.cytome")

    ds = cytome.from_anndata(adata, modality="RNA", output=path)
    ds.flush()
    assert ds.n_cells == adata.shape[0]

    adata_rt = ds.to_anndata(modality="RNA")
    assert adata_rt.shape == adata.shape

    lhs = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
    rhs = adata_rt.X.tocsr() if sp.issparse(adata_rt.X) else sp.csr_matrix(np.asarray(adata_rt.X))
    diff = lhs - rhs
    max_diff = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
    assert max_diff < 1e-5

    ds.close()


def test_subset_and_merge(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(DG_PATH)
    # Keep this integration test bounded in runtime.
    adata = adata[: min(800, adata.n_obs), : min(3000, adata.n_vars)].copy()
    path = str(tmp_path / "dg.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=path)
    ds.flush()

    n_sub = min(500, ds.n_cells)
    mask = np.zeros(ds.n_cells, dtype=bool)
    mask[:n_sub] = True
    sub = ds.subset(mask, output=str(tmp_path / "dg_sub.cytome"))
    assert sub.n_cells == n_sub
    sub.close()

    merged = cytome.merge([path, path], output=str(tmp_path / "dg_merged.cytome"))
    assert merged.n_cells == 2 * ds.n_cells
    merged.close()
    ds.close()
