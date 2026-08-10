from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cytome


def _base_ds(path):
    ds = cytome.create(path)
    n = 100
    ds.set_entity("cells", {
        "barcode": [f"c{i}" for i in range(n)],
        "cell_type": np.where(np.arange(n) % 2 == 0, "A", "B"),
    })
    ds.set_entity("genes", {"gene_id": [f"G{i}" for i in range(10)], "symbol": [f"G{i}" for i in range(10)]})
    ds.add_matrix("RNA_counts", sp.random(n, 10, density=0.2, format="csr", dtype=np.float32, random_state=1))
    ds.add_embedding("RNA_pca", np.random.randn(n, 5).astype(np.float32))
    ds.flush()
    return ds


def test_subset_by_mask(tmp_path):
    ds = _base_ds(tmp_path / "a.cytome")
    mask = ds.cells["cell_type"] == "A"
    sub = ds.subset(mask, output=tmp_path / "sub.cytome")
    assert sub.n_cells == int(mask.sum())
    assert all(sub.cells["cell_type"] == "A")
    sub.close()
    ds.close()


def test_subset_by_indices_and_all(tmp_path):
    ds = _base_ds(tmp_path / "a.cytome")
    sub = ds.subset(np.arange(ds.n_cells), output=tmp_path / "all.cytome")
    assert sub.n_cells == ds.n_cells
    sub.close()
    ds.close()


def test_downsample_random_and_stratified(tmp_path):
    ds = _base_ds(tmp_path / "a.cytome")
    out = ds.downsample(n_cells=20, output=tmp_path / "d.cytome", seed=42)
    assert out.n_cells == 20
    out.close()
    out2 = ds.downsample(n_cells=20, method="stratified", groupby="cell_type", output=tmp_path / "s.cytome", seed=42)
    assert out2.n_cells <= 20
    out2.close()
    idx = ds.downsample(fraction=0.1, output=None, seed=42)
    assert len(idx) == 10
    ds.close()
