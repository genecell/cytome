"""Tests for ``cytome.utils.modality``.

The registry centralises modality → entity-table routing so future
modalities update one place. Public consumers: piaso plotting, COSG
streaming, and cytome's own ``to_anndata``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_multimodal_cytome(path, n_cells=8, seed=0):
    """RNA + GA + ATAC + tiles cytome with disjoint feature sets per modality."""
    import cytome
    rng = np.random.default_rng(seed)
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"AAA-{i}" for i in range(n_cells)],
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": [0, 1, 2],
        "gene_id": ["Sox2", "Pax6", "Foxg1"],
    }))
    ds.add_matrix(
        "RNA_counts",
        sp.csr_matrix(np.eye(n_cells, 3, dtype=np.float32) * 5.0),
    )
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": [0, 1, 2, 3],
        "gene_id": ["Olig2", "Nestin", "Tubb3", "Gfap"],
    }))
    ds.add_matrix(
        "GA_counts",
        sp.csr_matrix(np.ones((n_cells, 4), dtype=np.float32) * 7.0),
    )
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": [0, 1],
        "peak_id": ["chr1:100-200", "chr2:300-400"],
        "chr": ["chr1", "chr2"],
        "start": [100, 300],
        "end_": [200, 400],
    }))
    ds.add_matrix(
        "ATAC_counts",
        sp.csr_matrix(np.ones((n_cells, 2), dtype=np.float32) * 3.0),
    )
    ds.set_entity("tiles", pd.DataFrame({
        "tile_idx": [0, 1],
        "tile_id": ["chr1:0-500", "chr1:500-1000"],
        "chr": ["chr1", "chr1"],
        "start": [0, 500],
        "end_": [500, 1000],
    }))
    ds.add_matrix(
        "tiles_counts",
        sp.csr_matrix(np.ones((n_cells, 2), dtype=np.float32) * 2.0),
    )
    ds.flush()
    return ds


# --- Registry surface ---

def test_modality_registry_covers_4_modalities():
    from cytome.utils.modality import MODALITY_REGISTRY
    names = [e[0] for e in MODALITY_REGISTRY]
    assert names == ["RNA", "GA", "ATAC", "tiles"]


def test_modality_var_entity_each_modality():
    from cytome.utils import modality_var_entity
    assert modality_var_entity("RNA") == ("genes", "gene_id")
    assert modality_var_entity("GA") == ("GA_genes", "gene_id")
    assert modality_var_entity("ATAC") == ("peaks", "peak_id")
    assert modality_var_entity("tiles") == ("tiles", "tile_id")


def test_modality_var_entity_case_insensitive_for_canonical_three():
    from cytome.utils import modality_var_entity
    # rna / ga / atac (lowercase) should resolve same as upper
    assert modality_var_entity("rna") == ("genes", "gene_id")
    assert modality_var_entity("ga") == ("GA_genes", "gene_id")
    assert modality_var_entity("atac") == ("peaks", "peak_id")


def test_modality_var_entity_unknown_raises():
    from cytome.utils import modality_var_entity
    with pytest.raises(ValueError, match="Unknown modality"):
        modality_var_entity("BANANA")


# --- modality_feature_table_info ---

def test_modality_feature_table_info_routes_correctly(tmp_path):
    from cytome.utils import modality_feature_table_info
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    assert modality_feature_table_info(ds, "RNA") == ("genes", "gene_idx", "gene_id")
    assert modality_feature_table_info(ds, "GA") == ("GA_genes", "gene_idx", "gene_id")
    assert modality_feature_table_info(ds, "ATAC") == ("peaks", "peak_idx", "peak_id")
    assert modality_feature_table_info(ds, "tiles") == ("tiles", "tile_idx", "tile_id")
    ds.close()


# --- modality_has_feature ---

def test_modality_has_feature_finds_in_correct_modality(tmp_path):
    from cytome.utils import modality_has_feature
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    # Sox2 in RNA, Olig2 in GA, chr1:100-200 in ATAC
    assert modality_has_feature(ds, "RNA", "Sox2") == (0, "gene_id")
    assert modality_has_feature(ds, "GA", "Olig2") == (0, "gene_id")
    assert modality_has_feature(ds, "ATAC", "chr1:100-200") == (0, "peak_id")
    # Sox2 NOT in GA — should return None
    assert modality_has_feature(ds, "GA", "Sox2") is None
    # Made-up feature → None everywhere
    for mod in ("RNA", "GA", "ATAC", "tiles"):
        assert modality_has_feature(ds, mod, "NonExistent") is None
    ds.close()


# --- read_feature_column ---

def test_read_feature_column_returns_correct_values(tmp_path):
    from cytome.utils import read_feature_column
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    # GA_counts is all 7.0
    vals = read_feature_column(ds, "GA", "counts", feat_idx=0)
    assert vals.shape == (8,)
    assert float(vals.mean()) == pytest.approx(7.0)
    # RNA_counts is 5.0 along the diagonal (eye * 5.0)
    vals = read_feature_column(ds, "RNA", "counts", feat_idx=0)
    # diagonal: cell_idx=0 has Sox2=5.0, others 0
    assert float(vals[0]) == pytest.approx(5.0)
    assert float(vals[1:].max()) == 0.0
    ds.close()


# --- modality_cell_depth ---

def test_modality_cell_depth_caches_per_modality(tmp_path):
    from cytome.utils import modality_cell_depth
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    rna_depth = modality_cell_depth(ds, "RNA", use_cached_stats=True)
    ga_depth = modality_cell_depth(ds, "GA", use_cached_stats=True)
    # Both cached under their own keys
    assert "RNA_cell_depth" in {k for k in ds.metadata.keys()}
    assert "GA_cell_depth" in {k for k in ds.metadata.keys()}
    # RNA: diagonal of 5.0 → each cell has at most one nonzero of 5.0
    # (cell 0 has Sox2=5; cells 3-7 fall outside the eye(8,3) so are 0)
    assert float(rna_depth[0]) == pytest.approx(5.0)
    # GA: every cell has 4 features × 7.0 = 28.0
    assert float(ga_depth[0]) == pytest.approx(28.0)
    # Re-call with use_cached_stats=False recomputes (same result for static data)
    ga_depth_2 = modality_cell_depth(ds, "GA", use_cached_stats=False)
    np.testing.assert_allclose(ga_depth, ga_depth_2)
    ds.close()


def test_convert_anndata_still_uses_centralised_registry(tmp_path):
    """Backward-compat: cytome/io/convert_anndata.py's _MODALITY_VAR_ENTITY
    is now an alias for cytome.utils.modality.MODALITY_VAR_ENTITY. Code
    that imported the private name still works."""
    from cytome.io.convert_anndata import _MODALITY_VAR_ENTITY, _modality_var_entity
    from cytome.utils.modality import MODALITY_VAR_ENTITY
    assert _MODALITY_VAR_ENTITY is MODALITY_VAR_ENTITY
    assert _modality_var_entity("GA") == ("GA_genes", "gene_id")


def test_to_anndata_still_works_on_ga_modality(tmp_path):
    """End-to-end smoke: the existing to_anndata + GA modality path
    (the cytome ffea4e5 fix) still works through the refactored
    registry import."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    adata = ds.to_anndata(modality="GA", layer="counts")
    assert adata.shape == (8, 4)
    assert list(adata.var_names) == ["Olig2", "Nestin", "Tubb3", "Gfap"]
    ds.close()


# ----------------------------------------------------------------------
# Round 24: populated symbol-first name_col resolver + feature_name_col +
# manifest override + read_feature_columns batched reader.
# ----------------------------------------------------------------------

def _rna_cytome_with_symbols(path, symbol_populated=True):
    import cytome
    ds = cytome.create(path)
    n = 6
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(4),
        "gene_id": [f"ENSG{i:05d}" for i in range(4)],
        "symbol": ([f"Gene{i}" for i in range(4)] if symbol_populated else [None] * 4),
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.arange(n * 4, dtype=np.float32).reshape(n, 4)))
    ds.flush()
    return ds


def test_name_col_prefers_populated_symbol(tmp_path):
    from cytome.utils.modality import modality_feature_table_info as info
    ds = _rna_cytome_with_symbols(tmp_path / "a.cytome", symbol_populated=True)
    assert info(ds, "RNA")[2] == "symbol"                       # populated -> symbol
    assert info(ds, "RNA", feature_name_col="gene_id")[2] == "gene_id"  # explicit override
    ds.metadata["RNA_name_col"] = "gene_id"; ds.flush()
    assert info(ds, "RNA")[2] == "gene_id"                      # manifest override
    ds.close()


def test_name_col_falls_through_when_symbol_all_null(tmp_path):
    from cytome.utils.modality import modality_feature_table_info as info
    ds = _rna_cytome_with_symbols(tmp_path / "b.cytome", symbol_populated=False)
    assert info(ds, "RNA")[2] == "gene_id"                     # all-NULL symbol -> gene_id
    ds.close()


def test_read_feature_columns_matches_single(tmp_path):
    from cytome.utils.modality import read_feature_column, read_feature_columns
    ds = _rna_cytome_with_symbols(tmp_path / "c.cytome")
    cols = read_feature_columns(ds, "RNA", "counts", [0, 2, 3])
    for k, j in enumerate([0, 2, 3]):
        assert np.allclose(cols[:, k], read_feature_column(ds, "RNA", "counts", j))
    ds.close()
