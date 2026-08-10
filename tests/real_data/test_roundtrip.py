"""Round-trip correctness tests on real datasets."""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp

DATA_DIR = "tests/real_data/data"
PANCREAS_PATH = os.path.join(DATA_DIR, "Pancreas_with_cc_anndata.h5ad")
DG_PATH = os.path.join(DATA_DIR, "DentateGyrus_anndata.h5ad")


@pytest.mark.parametrize("path", [PANCREAS_PATH, DG_PATH])
def test_roundtrip_real_dataset(path, tmp_path):
    pytest.importorskip("anndata")
    import anndata
    import cytome

    if not os.path.exists(path):
        pytest.skip(f"Dataset missing: {path}")

    adata = anndata.read_h5ad(path)
    out = str(tmp_path / "rt.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=out)
    adata_back = ds.to_anndata(modality="RNA")

    assert adata_back.shape == adata.shape
    a = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
    b = adata_back.X.tocsr() if sp.issparse(adata_back.X) else sp.csr_matrix(np.asarray(adata_back.X))
    diff = a - b
    max_diff = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
    assert max_diff < 1e-5
    ds.close()
