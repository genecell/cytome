"""Basic Cytome vs AnnData benchmark smoke tests on real data."""

from __future__ import annotations

import os
import time

import pytest

DATA_DIR = "tests/real_data/data"
PANCREAS_PATH = os.path.join(DATA_DIR, "Pancreas_with_cc_anndata.h5ad")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PANCREAS_PATH),
    reason="Pancreas dataset not downloaded.",
)


def test_open_time_comparison(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(PANCREAS_PATH)
    cytome_path = str(tmp_path / "bench.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=cytome_path)
    ds.flush()
    ds.close()
    del adata

    t0 = time.time()
    adata2 = anndata.read_h5ad(PANCREAS_PATH)
    t_anndata = time.time() - t0

    t0 = time.time()
    ds = cytome.open(cytome_path)
    t_cytome = time.time() - t0
    ds.close()

    print(f"\nOpen time — AnnData: {t_anndata*1000:.0f}ms, cytome: {t_cytome*1000:.0f}ms")
    print(f"Speedup: {t_anndata/max(t_cytome, 1e-6):.1f}x")
    del adata2


def test_metadata_query_comparison(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(PANCREAS_PATH)
    cytome_path = str(tmp_path / "bench.cytome")
    ds_tmp = cytome.from_anndata(adata, modality="RNA", output=cytome_path)
    ds_tmp.flush()
    ds_tmp.close()

    t0 = time.time()
    adata2 = anndata.read_h5ad(PANCREAS_PATH)
    obs_cols = list(adata2.obs.columns)
    _ = adata2.obs[obs_cols[0]].values if obs_cols else None
    t_anndata = time.time() - t0

    t0 = time.time()
    ds = cytome.open(cytome_path)
    if obs_cols:
        _ = ds.cells[obs_cols[0]]
    t_cytome = time.time() - t0
    ds.close()

    print(f"\nMetadata query — AnnData: {t_anndata*1000:.0f}ms, cytome: {t_cytome*1000:.0f}ms")
    del adata2


def test_file_size_comparison(tmp_path):
    import anndata
    import cytome

    adata = anndata.read_h5ad(PANCREAS_PATH)
    cytome_path = str(tmp_path / "bench.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=cytome_path)
    ds.flush()
    ds.close()

    h5ad_size = os.path.getsize(PANCREAS_PATH)
    cytome_size = os.path.getsize(cytome_path)
    print(f"\nFile size — h5ad: {h5ad_size/1024/1024:.1f} MB, cytome: {cytome_size/1024/1024:.1f} MB")
    print(f"Ratio: {cytome_size/max(h5ad_size,1):.2f}x")
