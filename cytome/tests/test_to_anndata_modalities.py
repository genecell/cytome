"""Tests for ``cytome.Dataset.to_anndata`` modality-aware var dispatch.

Pre-fix: the dispatch was hardcoded ``peaks`` for ATAC vs ``genes`` for
everything else, so ``to_anndata(modality='GA')`` would read the
``GA_counts`` matrix correctly but pull ``var`` from ``ds.genes``,
producing a shape mismatch when an RNA modality coexists.

Post-fix: a small modality→(entity, id_col) registry routes each known
modality (RNA, GA, ATAC, tiles) to its own var entity. Coexistence with
disjoint feature sets is handled correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_multimodal_cytome(path):
    """Build a tiny cytome with RNA + GA + ATAC + tiles modalities,
    each with a disjoint feature set."""
    import cytome
    ds = cytome.create(path)
    n = 5
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n),
        "barcode": [f"AAA-{i}" for i in range(n)],
    }))
    # RNA modality (3 genes)
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": [0, 1, 2],
        "gene_id": ["GeneR1", "GeneR2", "GeneR3"],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((n, 3), dtype=np.float32)))
    # GA modality (4 genes — disjoint from RNA)
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": [0, 1, 2, 3],
        "gene_id": ["GeneG1", "GeneG2", "GeneG3", "GeneG4"],
    }))
    ds.add_matrix("GA_counts", sp.csr_matrix(2 * np.ones((n, 4), dtype=np.float32)))
    # ATAC modality (2 peaks)
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": [0, 1],
        "peak_id": ["chr1:100-200", "chr2:300-400"],
        "chr": ["chr1", "chr2"],
        "start": [100, 300],
        "end_": [200, 400],
    }))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(3 * np.ones((n, 2), dtype=np.float32)))
    # tiles modality (2 tiles)
    ds.set_entity("tiles", pd.DataFrame({
        "tile_idx": [0, 1],
        "tile_id": ["tile-A", "tile-B"],
        "chr": ["chr1", "chr1"],
        "start": [0, 500],
        "end_": [500, 1000],
    }))
    ds.add_matrix("tiles_counts", sp.csr_matrix(4 * np.ones((n, 2), dtype=np.float32)))
    ds.flush()
    return ds


def test_to_anndata_rna_modality(tmp_path):
    """RNA → reads from ds.genes (3 features)."""
    import cytome
    ds = _build_multimodal_cytome(tmp_path / "multi.cytome")
    adata = ds.to_anndata(modality="RNA", layer="counts")
    assert adata.shape == (5, 3)
    assert list(adata.var_names) == ["GeneR1", "GeneR2", "GeneR3"]
    ds.close()


def test_to_anndata_ga_modality_uses_GA_genes(tmp_path):
    """GA → reads from ds.GA_genes (4 features), NOT ds.genes."""
    import cytome
    ds = _build_multimodal_cytome(tmp_path / "multi.cytome")
    adata = ds.to_anndata(modality="GA", layer="counts")
    assert adata.shape == (5, 4), (
        f"GA modality should yield (5, 4); got {adata.shape}. "
        f"Pre-fix, this would either crash on shape mismatch or return "
        f"(5, 3) reading from RNA's genes table."
    )
    assert list(adata.var_names) == ["GeneG1", "GeneG2", "GeneG3", "GeneG4"]
    # Values are 2.0 throughout (per construction)
    assert float(adata.X.sum()) == pytest.approx(5 * 4 * 2.0)
    ds.close()


def test_to_anndata_atac_modality(tmp_path):
    """ATAC → reads from ds.peaks (2 peaks)."""
    import cytome
    ds = _build_multimodal_cytome(tmp_path / "multi.cytome")
    adata = ds.to_anndata(modality="ATAC", layer="counts")
    assert adata.shape == (5, 2)
    assert list(adata.var_names) == ["chr1:100-200", "chr2:300-400"]
    ds.close()


def test_to_anndata_tiles_modality(tmp_path):
    """tiles → reads from ds.tiles (2 tiles)."""
    import cytome
    ds = _build_multimodal_cytome(tmp_path / "multi.cytome")
    adata = ds.to_anndata(modality="tiles", layer="counts")
    assert adata.shape == (5, 2)
    assert list(adata.var_names) == ["tile-A", "tile-B"]
    ds.close()


def test_to_anndata_unknown_modality_raises(tmp_path):
    """Unknown modality string raises ValueError naming the known set."""
    import cytome
    ds = _build_multimodal_cytome(tmp_path / "multi.cytome")
    with pytest.raises(ValueError) as exc_info:
        ds.to_anndata(modality="BANANA", layer="counts")
    msg = str(exc_info.value)
    assert "BANANA" in msg
    assert "RNA" in msg and "GA" in msg and "ATAC" in msg and "tiles" in msg
    ds.close()


def test_to_anndata_ga_and_rna_coexist(tmp_path):
    """End-to-end: a cytome with both RNA (genes) and GA (GA_genes) round-trips
    through to_anndata for both modalities without cross-contamination."""
    import cytome
    ds = _build_multimodal_cytome(tmp_path / "multi.cytome")
    rna = ds.to_anndata(modality="RNA", layer="counts")
    ga = ds.to_anndata(modality="GA", layer="counts")

    # RNA adata is unaffected by the existence of GA_genes
    assert rna.shape == (5, 3)
    assert set(rna.var_names) == {"GeneR1", "GeneR2", "GeneR3"}
    # GA adata is unaffected by the existence of genes
    assert ga.shape == (5, 4)
    assert set(ga.var_names) == {"GeneG1", "GeneG2", "GeneG3", "GeneG4"}
    # No overlap
    assert set(rna.var_names).isdisjoint(set(ga.var_names))
    ds.close()
