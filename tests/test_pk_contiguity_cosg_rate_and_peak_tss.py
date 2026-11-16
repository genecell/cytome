"""Tests for entity primary-key contiguity (0..n-1), Four-Axiom COSG-Rate, and Strand-Split TSS annotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import cytome
from cytome.core.peak_annotation import _dedup_tss_track, _nearest_tss_on_chrom, annotate_peaks_tss
from cytome.core.pseudobulk import PseudobulkCube, aggregate_pseudobulk
from cytome.utils.validation import repair, validate


def test_write_entity_table_enforces_pk_contiguity_and_repair(tmp_path):
    p = tmp_path / "sub_test.cytome"
    ds = cytome.create(p)
    cells_df = pd.DataFrame({
        "cell_idx": [10, 25, 50, 100],
        "barcode": ["bc_0", "bc_1", "bc_2", "bc_3"],
        "sample_id": ["s1", "s1", "s1", "s1"],
        "cluster": ["A", "A", "B", "B"],
    })
    genes_df = pd.DataFrame({
        "gene_idx": [5, 15, 25],
        "gene_id": ["G0", "G1", "G2"],
        "symbol": ["Sox2", "Pax6", "Gad1"],
    })
    ds.set_entity("cells", cells_df)
    ds.set_entity("genes", genes_df)
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((4, 3), dtype=np.float32)))
    ds.flush()

    assert list(ds.cells["cell_idx"]) == [0, 1, 2, 3]
    assert list(ds.genes["gene_idx"]) == [0, 1, 2]
    rep = validate(ds, integrity=False)
    assert rep.passed, rep.checks_failed

    # Simulate a legacy store with non-contiguous cell_idx and spatial_coords
    ds._conn.execute("PRAGMA defer_foreign_keys = ON")
    for new_ci, old_ci in [(100, 3), (50, 2), (25, 1), (10, 0)]:
        ds._conn.execute("UPDATE cells SET cell_idx = ? WHERE cell_idx = ?", (new_ci, old_ci))
    ds._conn.executemany(
        "INSERT INTO spatial_coords (cell_idx, x, y, z) VALUES (?,?,?,?)",
        [(10, 1.0, 2.0, 0.0), (25, 3.0, 4.0, 0.0), (50, 5.0, 6.0, 0.0), (100, 7.0, 8.0, 0.0)],
    )
    ds._conn.commit()

    # Single-pass positional query_mask works even on legacy non-contiguous primary keys
    mask = ds.cells.query_mask("cluster = 'B'")
    assert list(mask) == [False, False, True, True]

    # Subset preserves spatial_coords even on legacy non-contiguous primary keys
    sub_p = tmp_path / "sub_out.cytome"
    sub_ds = ds.subset(np.array([1, 3]), output=sub_p)
    sp_rows = sub_ds._conn.execute("SELECT cell_idx, x, y FROM spatial_coords ORDER BY cell_idx").fetchall()
    assert sp_rows == [(0, 3.0, 4.0), (1, 7.0, 8.0)]
    sub_ds.close()

    # validate catches non-contiguous cell_idx, and repair fixes both cells and spatial_coords
    rep_bad = validate(ds, integrity=False)
    assert not rep_bad.passed
    assert "entity_pk_contiguity:cells" in rep_bad.checks_failed

    actions = repair(ds)
    assert any("cells: compacted cell_idx" in a for a in actions)
    rep_fixed = validate(ds, integrity=False)
    assert rep_fixed.passed, rep_fixed.checks_failed
    assert list(ds.cells["cell_idx"]) == [0, 1, 2, 3]
    sp_after = ds._conn.execute("SELECT cell_idx, x, y FROM spatial_coords ORDER BY cell_idx").fetchall()
    assert sp_after == [(0, 1.0, 2.0), (1, 3.0, 4.0), (2, 5.0, 6.0), (3, 7.0, 8.0)]
    ds.close()


def test_strand_split_theorem_and_dedup(tmp_path):
    # Test _dedup_tss_track prioritizes protein-coding canonical symbol over Gm* at identical (tss, strand)
    recs = [
        (10000, 1, "Gm19999", "ENSMUSG00000099999", True),
        (10000, 1, "Sox2", "ENSMUSG00000074637", True),
        (10000, 1, "Sox2ot", "ENSMUSG00000100000", False),
    ]
    pos, sign, names, ids = _dedup_tss_track(recs)
    assert len(pos) == 1
    assert names[0] == "Sox2"

    # Test Strand-Split Theorem:
    # Summit at 11,000.
    # Gene A at 9,200 (+) -> signed distance = +1,800 bp (downstream, outside [-2000, +500] promoter window).
    # Gene B at 9,100 (-) -> signed distance = -(11000 - 9100) = -1,900 bp (upstream, INSIDE [-2000, +500] promoter!).
    # A pooled 1D searchsorted only sees left=9200(+) and misses 9100(-) completely!
    summits = np.array([11000], dtype=np.int64)
    tss_pos = np.array([9100, 9200], dtype=np.int64)
    tss_sign = np.array([-1, 1], dtype=np.int64)
    tss_names = np.array(["GeneB_minus_prom", "GeneA_plus_body"], dtype=object)
    tss_ids = np.array(["ID_B", "ID_A"], dtype=object)

    n_sym, n_id, s_dist = _nearest_tss_on_chrom(
        summits, tss_pos, tss_sign, tss_names, tss_ids, promoter_window=(2000, 500)
    )
    assert n_sym[0] == "GeneB_minus_prom"
    assert n_id[0] == "ID_B"
    assert int(s_dist[0]) == -1900


def test_modality_depths(tmp_path):
    # Modality-aware cube depths and their persistence. (The nearest-gene naming this test also checked is
    # in test_reference_is_passed_not_searched.py: genes are now named by the reference and joined by ID.)
    p = tmp_path / "ens_harm.cytome"
    ds = cytome.create(p)
    cells_df = pd.DataFrame({
        "cell_idx": [0, 1, 2, 3],
        "barcode": ["b0", "b1", "b2", "b3"],
        "sample_id": ["s1", "s1", "s1", "s1"],
        "cluster": ["A", "A", "B", "B"],
        "total_counts": [1000.0, 2000.0, 3000.0, 4000.0],
        "n_fragments": [5000.0, 6000.0, 7000.0, 8000.0],
    })
    # Simulate CellRanger GENCODE vM23 symbol '4933415F23Rik' with stable Ensembl ID 'ENSMUSG00000073730'
    genes_df = pd.DataFrame({
        "gene_idx": [0, 1],
        "gene_id": ["4933415F23Rik", "Sox2"],
        "symbol": ["4933415F23Rik", "Sox2"],
        "gene_ids": ["ENSMUSG00000073730", "ENSMUSG00000074637"],
    })
    peaks_df = pd.DataFrame({
        "peak_idx": [0, 1],
        "peak_id": ["chr1:10000-10500", "chr1:50000-50500"],
        "chr": ["chr1", "chr1"],
        "start": [10000, 50000],
        "end_": [10500, 50500],
    })
    ds.set_entity("cells", cells_df)
    ds.set_entity("genes", genes_df)
    ds.set_entity("peaks", peaks_df)
    ds.flush()
    # cells' totals: RNA 1, 2, 3, 4 and ATAC 5, 6, 7, 8
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.array([[1, 0], [1, 1], [2, 1], [3, 1]], dtype=np.float32)))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(np.array([[5, 0], [3, 3], [7, 0], [4, 4]], dtype=np.float32)))
    ds.flush()

    # Each cube's depth is its own matrix's cell totals (RNA 1..4, ATAC 5..8 above), whatever the
    # cells table holds; total_fragments is the cells' n_fragments, the same for both cubes
    rna_cube = aggregate_pseudobulk(ds, by="cluster", modality="RNA")
    atac_cube = aggregate_pseudobulk(ds, by="cluster", modality="ATAC")
    assert np.allclose(rna_cube.total_counts, [3.0, 7.0])
    assert np.allclose(rna_cube.total_counts_sq, [1.0 + 4.0, 9.0 + 16.0])
    assert np.allclose(atac_cube.total_counts, [11.0, 15.0])
    assert np.allclose(atac_cube.total_counts_sq, [25.0 + 36.0, 49.0 + 64.0])
    assert np.allclose(rna_cube.total_fragments, [11000.0, 15000.0])
    assert np.allclose(atac_cube.total_fragments, [11000.0, 15000.0])

    # The cached cube keeps total_counts_sq
    atac_cached = aggregate_pseudobulk(ds, by="cluster", modality="ATAC")
    assert np.allclose(atac_cached.total_counts_sq, [61.0, 113.0])
    ds.close()
