"""to_anndata must resolve X / layers / obsm per the REQUESTED modality.

Regression for the bug where `to_anndata(modality="GA")` returned the RNA X matrix
(wrong column count) because the global `_anndata_X_layer` / `_anndata_layer_map` /
`_anndata_obsm_map` metadata — recorded by the first from_anndata (RNA) — hijacked the
resolution for every other modality. The fix gates those fast-paths on the
`{modality}_` prefix (+ n_cols guards) and adds cross-modality cell-embedding support.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import pytest

import cytome


def _rna(n=24, g=8, seed=0):
    rng = np.random.RandomState(seed)
    A = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, (n, g)).astype(np.float32)))
    A.obs_names = [f"cell{i}" for i in range(n)]
    A.var_names = [f"RNAgene{i}" for i in range(g)]
    A.layers["infog"] = A.X.copy()
    A.obsm["X_umap"] = rng.randn(n, 2).astype(np.float32)
    return A


def _ga(obs_names, g=13, seed=1):
    rng = np.random.RandomState(seed)
    A = ad.AnnData(X=sp.csr_matrix(rng.poisson(0.5, (len(obs_names), g)).astype(np.float32)))
    A.obs_names = list(obs_names)
    A.var_names = [f"GAgene{i}" for i in range(g)]  # DIFFERENT count + names than RNA
    return A


def _build(tmp_path):
    rna = _rna()
    ds = cytome.from_anndata(rna, modality="RNA", output=str(tmp_path / "m.cytome"))
    ga = _ga(rna.obs_names)
    ds.add_modality("GA", ga)
    # add_modality writes GA_counts but leaves the GA_genes var table empty;
    # populate it the way the real _persist_gene_activity_to_cytome does.
    ds._conn.executemany(
        "INSERT INTO GA_genes (gene_idx, gene_id) VALUES (?, ?)",
        [(i, str(name)) for i, name in enumerate(ga.var_names)],
    )
    ds._conn.commit()
    ds.flush()
    return ds, rna


def test_to_anndata_ga_uses_ga_matrix(tmp_path):
    ds, rna = _build(tmp_path)
    a = ds.to_anndata(modality="GA")
    assert a.shape == (rna.n_obs, 13)                    # GA feature count, not RNA's 8
    assert list(a.var_names[:2]) == ["GAgene0", "GAgene1"]
    # no RNA layer leaked onto the GA AnnData (would be an 8-col mismatch)
    for lyr in a.layers.values():
        assert lyr.shape[1] == a.shape[1]
    ds.close()


def test_to_anndata_rna_still_correct(tmp_path):
    ds, rna = _build(tmp_path)
    r = ds.to_anndata(modality="RNA")
    assert r.shape == (rna.n_obs, 8)
    assert "infog" in r.layers and r.layers["infog"].shape[1] == 8
    assert "X_umap" in r.obsm
    ds.close()


def test_cross_modality_embeddings_toggle(tmp_path):
    ds, rna = _build(tmp_path)
    # GA has no own embedding; RNA's X_umap should attach when cross-modality is on
    on = ds.to_anndata(modality="GA")
    off = ds.to_anndata(modality="GA", include_cross_modality_embeddings=False)
    assert any("umap" in k.lower() for k in on.obsm)     # RNA X_umap pulled in
    assert len(off.obsm) == 0                            # GA alone has none
    ds.close()
