"""Tests for the GA_genes per-modality entity table.

GA_genes is a sibling of ``genes`` reserved for inferred Gene Activity
matrices. It exists so an RNA modality (``RNA_counts``, col_entity='genes')
and an IGA-derived gene activity modality (``GA_counts``, col_entity='GA_genes')
can coexist in one cytome without colliding on the col_entity validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def test_ga_genes_table_exists_in_fresh_cytome(tmp_path):
    """Schema creates GA_genes alongside genes/peaks/etc."""
    import cytome
    ds = cytome.create(tmp_path / "fresh.cytome")
    tables = {
        r[0]
        for r in ds._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "GA_genes" in tables, f"GA_genes missing from fresh cytome; got: {sorted(tables)}"
    # And it's empty
    assert int(ds._conn.execute("SELECT COUNT(*) FROM GA_genes").fetchone()[0]) == 0
    ds.close()


def test_ga_genes_accessor(tmp_path):
    """ds.GA_genes returns an EntityTable, populates via set_entity."""
    import cytome
    ds = cytome.create(tmp_path / "acc.cytome")
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": [0, 1, 2],
        "gene_id": ["GeneA", "GeneB", "GeneC"],
    }))
    ds.flush()

    df = ds.GA_genes.to_pandas()
    assert list(df["gene_id"]) == ["GeneA", "GeneB", "GeneC"]
    ds.close()


def test_ga_and_rna_genes_independent(tmp_path):
    """RNA writes 'genes' and IGA writes 'GA_genes' — they don't collide."""
    import cytome
    from cytome.utils.validation import validate

    ds = cytome.create(tmp_path / "two.cytome")
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(5),
        "barcode": [f"AAA-{i}" for i in range(5)],
    }))
    # RNA modality — uses 'genes'
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": [0, 1, 2],
        "gene_id": ["GeneA", "GeneB", "GeneC"],
    }))
    # GA modality — uses 'GA_genes', a SUPERSET of RNA genes
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": [0, 1, 2, 3],
        "gene_id": ["GeneA", "GeneB", "GeneC", "GeneZ"],
    }))
    ds.flush()

    ds.add_matrix(
        "RNA_counts",
        sp.csr_matrix(np.ones((5, 3), dtype=np.float32)),
    )
    ds.add_matrix(
        "GA_counts",
        sp.csr_matrix(np.ones((5, 4), dtype=np.float32)),
    )
    ds.flush()

    # Validate: both matrices pass col_entity check
    report = validate(ds)
    matrix_col_fails = [c for c in report.checks_failed if c.startswith("matrix_cols:")]
    assert not matrix_col_fails, (
        f"RNA + GA coexistence should validate; got: {matrix_col_fails}"
    )

    # Independent: writing GA_genes does not touch genes
    assert list(ds.genes.to_pandas()["gene_id"]) == ["GeneA", "GeneB", "GeneC"]
    assert list(ds.GA_genes.to_pandas()["gene_id"]) == ["GeneA", "GeneB", "GeneC", "GeneZ"]
    ds.close()


def test_ga_first_then_rna_scenario(tmp_path):
    """Scenario the GA_genes table was designed for:
    1. User runs IGA on an ATAC-only cytome → writes GA_counts (col_entity='GA_genes').
    2. User later imports RNA from an h5ad via cytome.from_anndata → sets 'genes'.
    The GA_genes table must survive step 2, and validation must pass.
    """
    import cytome
    from cytome.utils.validation import validate

    work = tmp_path / "ga_first.cytome"
    ds = cytome.create(work)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(5),
        "barcode": [f"AAA-{i}" for i in range(5)],
    }))
    # ATAC peaks (so cytome looks ATAC-only at first)
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": [0, 1, 2],
        "peak_id": ["chr1:100-200", "chr1:300-400", "chr1:500-600"],
        "chr": ["chr1"] * 3,
        "start": [100, 300, 500],
        "end_": [200, 400, 600],
    }))
    ds.add_matrix(
        "ATAC_counts",
        sp.csr_matrix(np.ones((5, 3), dtype=np.float32)),
    )
    ds.flush()

    # Step 1: IGA writes GA_genes + GA_counts
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": [0, 1, 2, 3],
        "gene_id": ["Foxg1", "Sox2", "Pax6", "Nestin"],
    }))
    ds.add_matrix(
        "GA_counts",
        sp.csr_matrix(np.eye(5, 4, dtype=np.float32)),
    )
    ds.flush()

    # Step 2: RNA later imported with a DIFFERENT gene set
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": [0, 1],
        "gene_id": ["Pax6", "Olig2"],
    }))
    ds.add_matrix(
        "RNA_counts",
        sp.csr_matrix(np.ones((5, 2), dtype=np.float32)),
    )
    ds.flush()

    # GA_genes survived
    assert list(ds.GA_genes.to_pandas()["gene_id"]) == ["Foxg1", "Sox2", "Pax6", "Nestin"]
    # genes is the RNA-import-time set
    assert list(ds.genes.to_pandas()["gene_id"]) == ["Pax6", "Olig2"]

    # Validation: every col_entity matches its table's row count
    report = validate(ds)
    matrix_col_fails = [c for c in report.checks_failed if c.startswith("matrix_cols:")]
    assert not matrix_col_fails, (
        f"GA-first-then-RNA broke validation: {matrix_col_fails}"
    )
    ds.close()


def test_infer_col_entity_for_ga_matrices():
    """GA-prefixed matrix names route to GA_genes by default."""
    from cytome.core.dataset import _infer_col_entity
    assert _infer_col_entity("GA_counts") == "GA_genes"
    assert _infer_col_entity("GA_normalized") == "GA_genes"
    assert _infer_col_entity("GA") == "GA_genes"
    # Existing rules unaffected
    assert _infer_col_entity("RNA_counts") == "genes"
    assert _infer_col_entity("ATAC_counts") == "peaks"
    assert _infer_col_entity("tile_counts") == "tiles"


def test_ga_genes_table_added_to_legacy_cytome_on_open(tmp_path):
    """An older cytome file (created before GA_genes existed) gains the table
    automatically when opened, because open_database calls _create_schema."""
    import sqlite3
    import cytome

    legacy = tmp_path / "legacy.cytome"
    # Build a minimal cytome and then DROP GA_genes to simulate a pre-feature file
    ds = cytome.create(legacy)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(3),
        "barcode": ["a", "b", "c"],
    }))
    ds.flush()
    ds._conn.execute("DROP TABLE GA_genes")
    ds._conn.commit()
    ds.close()

    # Sanity: the table really is gone
    raw = sqlite3.connect(legacy)
    tables_before = {r[0] for r in raw.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    raw.close()
    assert "GA_genes" not in tables_before

    # Re-opening through cytome should re-create it
    ds = cytome.open(legacy)
    tables_after = {r[0] for r in ds._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "GA_genes" in tables_after, "GA_genes should be auto-created on open"
    ds.close()
