"""Unit tests for Cytome 0.4.0 primitives: the pseudobulk cube, peak sequences, GTF biotypes and TSS annotation,
Tn5 footprints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome


def _build_synthetic_multiome(tmp_path, n_cells=120, n_peaks=24, n_genes=16, seed=42):
    rng = np.random.default_rng(seed)
    gtf_path = tmp_path / "test_ref.gtf"
    fasta_path = tmp_path / "test_ref.fa"

    # Build a synthetic FASTA with chr1 (50,000 bp) and chr8 (50,000 bp)
    bases = np.array(list("ACGT"))
    seq_chr1 = "".join(rng.choice(bases, size=50000))
    seq_chr8 = "".join(rng.choice(bases, size=50000))
    with open(fasta_path, "wt") as f:
        f.write(">chr1\n")
        for i in range(0, len(seq_chr1), 80):
            f.write(seq_chr1[i : i + 80] + "\n")
        f.write(">chr8\n")
        for i in range(0, len(seq_chr8), 80):
            f.write(seq_chr8[i : i + 80] + "\n")

    # Build a synthetic NCBI-style GTF where chr1 has `gene` lines and chr8 lacks `gene` lines (has CDS)
    gtf_lines = [
        '#gtf-version 2.2',
        'chr1\tGnomon\tgene\t1001\t4000\t.\t+\t.\tgene_id "SATB2"; gene_name "SATB2"; gene_biotype "protein_coding";',
        'chr1\tGnomon\ttranscript\t1001\t4000\t.\t+\t.\tgene_id "SATB2"; transcript_id "T1"; gene_name "SATB2";',
        'chr1\tGnomon\texon\t1001\t1500\t.\t+\t.\tgene_id "SATB2"; transcript_id "T1"; exon_number "1";',
        'chr1\tGnomon\tCDS\t1100\t1450\t.\t+\t.\tgene_id "SATB2"; transcript_id "T1";',
        'chr1\tGnomon\tgene\t1501\t2200\t.\t-\t.\tgene_id "LINC001"; gene_name "LINC001"; gene_biotype "lncRNA";',
        'chr1\tGnomon\ttranscript\t1501\t2200\t.\t-\t.\tgene_id "LINC001"; transcript_id "T_L1"; transcript_type "lncRNA";',
        'chr1\tGnomon\texon\t1501\t2200\t.\t-\t.\tgene_id "LINC001"; transcript_id "T_L1"; exon_number "1";',
        # Syntenic block on chr8: LPAR6 (+) .. LOC1001 (+) .. ESD (+) .. LOC1002 (+) .. HTR2A (-)
        # where LOC1001 is RB1 (missing in query) and LOC1002 is between ESD and HTR2A (0-gap if adjacent)
        'chr8\tGnomon\ttranscript\t5001\t8000\t.\t+\t.\tgene_id "LPAR6"; transcript_id "TX1"; gene "LPAR6";',
        'chr8\tGnomon\texon\t5001\t5600\t.\t+\t.\tgene_id "LPAR6"; transcript_id "TX1";',
        'chr8\tGnomon\tCDS\t5100\t5500\t.\t+\t.\tgene_id "LPAR6"; transcript_id "TX1";',
        'chr8\tGnomon\ttranscript\t10001\t16000\t.\t+\t.\tgene_id "LOC1001"; transcript_id "TX2"; gene "LOC1001";',
        'chr8\tGnomon\texon\t10001\t11000\t.\t+\t.\tgene_id "LOC1001"; transcript_id "TX2";',
        'chr8\tGnomon\tCDS\t10100\t10900\t.\t+\t.\tgene_id "LOC1001"; transcript_id "TX2";',
        'chr8\tGnomon\ttranscript\t20001\t24000\t.\t+\t.\tgene_id "ESD"; transcript_id "TX3"; gene "ESD";',
        'chr8\tGnomon\texon\t20001\t21000\t.\t+\t.\tgene_id "ESD"; transcript_id "TX3";',
        'chr8\tGnomon\tCDS\t20100\t20900\t.\t+\t.\tgene_id "ESD"; transcript_id "TX3";',
        'chr8\tGnomon\ttranscript\t26001\t28000\t.\t+\t.\tgene_id "LOC1002"; transcript_id "TX4"; gene "LOC1002";',
        'chr8\tGnomon\texon\t26001\t27000\t.\t+\t.\tgene_id "LOC1002"; transcript_id "TX4";',
        'chr8\tGnomon\tCDS\t26100\t26900\t.\t+\t.\tgene_id "LOC1002"; transcript_id "TX4";',
        'chr8\tGnomon\ttranscript\t32001\t36000\t.\t-\t.\tgene_id "HTR2A"; transcript_id "TX5"; gene "HTR2A";',
        'chr8\tGnomon\texon\t32001\t33000\t.\t-\t.\tgene_id "HTR2A"; transcript_id "TX5";',
        'chr8\tGnomon\tCDS\t32100\t32900\t.\t-\t.\tgene_id "HTR2A"; transcript_id "TX5";',
    ]
    gtf_path.write_text("\n".join(gtf_lines) + "\n")

    ds = cytome.create(tmp_path / "multiome.cytome", force=True)
    subclasses = ["L2_3_IT", "Pvalb", "Sst"]
    age_bins = ["Young", "Aged"]
    donors = [f"D{d}" for d in range(12)]
    donor_ages = {f"D{d}": 25.0 + 5.0 * d for d in range(12)}

    cell_sub = [subclasses[i % 3] for i in range(n_cells)]
    cell_donor = [donors[i % 12] for i in range(n_cells)]
    cell_age = [donor_ages[d] for d in cell_donor]
    cell_age_bin = ["Young" if a < 50.0 else "Aged" for a in cell_age]

    ds.set_entity(
        "cells",
        pd.DataFrame({
            "cell_idx": np.arange(n_cells),
            "barcode": [f"BC{i:04d}" for i in range(n_cells)],
            "sample_id": cell_donor,
            "SampleID": cell_donor,
            "Subclass_predicted": cell_sub,
            "Age_bin": cell_age_bin,
            "Age": cell_age,
            "n_fragments": rng.integers(2000, 8000, size=n_cells),
        }),
    )

    # Create peaks: include one wide peak (>1,200 bp) with off-center summit!
    peak_rows = []
    for p in range(n_peaks):
        ch = "chr1" if p < (n_peaks // 2) else "chr8"
        s = 900 + p * 1200
        width = 1400 if p == 0 else 500
        e = s + width
        summit_off = 150 if p == 0 else 250
        peak_rows.append({
            "peak_idx": p,
            "peak_id": f"{ch}:{s}-{e}",
            "chr": ch,
            "start": s,
            "end_": e,
            "summit": summit_off,
            "score": float(100 + p),
            "signal": float(2.5 + 0.1 * p),
            "neg_log10_pvalue": float(10.0 + p),
            "neg_log10_qvalue": float(8.0 + p),
        })
    ds.set_entity("peaks", pd.DataFrame(peak_rows))

    gene_ids = ["SATB2", "LINC001", "LPAR6", "LOC1001", "ESD", "LOC1002", "HTR2A"] + [
        f"GENE{i}" for i in range(n_genes - 7)
    ]
    ds.set_entity(
        "genes",
        pd.DataFrame({
            "gene_idx": np.arange(n_genes),
            "gene_id": gene_ids,
            "symbol": gene_ids,
        }),
    )
    ds.set_entity(
        "GA_genes",
        pd.DataFrame({
            "gene_idx": np.arange(n_genes),
            "gene_id": gene_ids,
            "symbol": gene_ids,
        }),
    )
    ds.flush()

    X_atac = sp.csr_matrix(rng.poisson(1.5, size=(n_cells, n_peaks)).astype(np.float32))
    X_rna = sp.csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))
    ds.add_matrix("ATAC_counts", X_atac)
    ds.add_matrix("RNA_counts", X_rna)
    ds.flush()
    ds.ATAC.counts.build_feature_index(chunk_size=8)

    # Write compressed fragment chunks on chr1 and chr8
    from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks
    for ch in ("chr1", "chr8"):
        f_starts = np.sort(rng.integers(800, 35000, size=2000, dtype=np.int32))
        f_ends = f_starts + rng.integers(40, 220, size=2000, dtype=np.int32)
        f_cells = rng.integers(0, n_cells, size=2000, dtype=np.int32)
        _write_block_chunks(
            ds._conn, ch, f_starts, f_ends, f_cells,
            chunk_size=500, compression="lz4", compression_level=1,
            chunk_idx_start=0, encoding=1,
        )
        _update_fragment_meta(ds._conn, ch, 2000)
    ds._conn.commit()
    return ds, X_atac, X_rna, gtf_path, fasta_path, seq_chr1, seq_chr8


def test_t1_t2_pseudobulk_cube_and_marginalization(tmp_path):
    """T1 & T2: PseudobulkCube exactness (CSR and CSC) + additive marginalization identity."""
    ds, X_atac, _, _, _, _, _ = _build_synthetic_multiome(tmp_path)
    try:
        cube_joint = ds.aggregate_pseudobulk(
            modality="ATAC", layer="counts", by=["Subclass_predicted", "Age_bin"], cache=True
        )
        cube_marg = cube_joint.marginalize("Subclass_predicted")
        cube_direct = ds.aggregate_pseudobulk(
            modality="ATAC", layer="counts", by="Subclass_predicted", force=True, cache=False
        )

        np.testing.assert_array_equal(cube_marg.group_names, cube_direct.group_names)
        np.testing.assert_array_equal(cube_marg.n_cells, cube_direct.n_cells)
        np.testing.assert_allclose(cube_marg.total_counts, cube_direct.total_counts, rtol=1e-6)
        np.testing.assert_allclose(cube_marg.sum, cube_direct.sum, rtol=1e-6)
        np.testing.assert_allclose(cube_marg.sum_sq, cube_direct.sum_sq, rtol=1e-5)
        np.testing.assert_array_equal(cube_marg.nnz, cube_direct.nnz)

        # Verify against dense numpy groupby ground truth
        dense = X_atac.toarray()
        sub_col = np.asarray(ds.cells["Subclass_predicted"])
        for gi, gname in enumerate(cube_direct.group_names):
            mask = sub_col == gname
            np.testing.assert_allclose(cube_direct.sum[gi], dense[mask].sum(axis=0), rtol=1e-5)
            np.testing.assert_allclose(cube_direct.sum_sq[gi], (dense[mask] ** 2).sum(axis=0), rtol=1e-5)
            np.testing.assert_array_equal(cube_direct.nnz[gi], (dense[mask] > 0).sum(axis=0))
    finally:
        ds.close()


def test_t4_variable_length_peak_sequences_roundtrip(tmp_path):
    """T4: 2.5-bit variable-length peak sequence roundtrip for summit & midpoint windows and RC."""
    ds, _, _, _, fasta_path, seq_chr1, _ = _build_synthetic_multiome(tmp_path)
    try:
        info = ds.store_peak_sequences(fasta_path=fasta_path, max_flank=500, chunk_size=8)
        assert info["n_peaks"] == 24

        # Peak 0 on chr1 is [900, 2300) (1,400 bp wide!) with summit offset = 150 -> abs_summit = 1050, abs_mid = 1600
        seqs_sum = ds.get_peak_sequences(window=500, center="summit", encoding="str")
        seqs_mid = ds.get_peak_sequences(window=500, center="midpoint", encoding="str")
        assert seqs_sum[0] == seq_chr1[1050 - 250 : 1050 + 250]
        assert seqs_mid[0] == seq_chr1[1600 - 250 : 1600 + 250]

        # Test 1,000 bp window on both summit and midpoint
        seqs_1000_sum = ds.get_peak_sequences(window=1000, center="summit", peak_indices=[0], encoding="str")
        seqs_1000_mid = ds.get_peak_sequences(window=1000, center="midpoint", peak_indices=[0], encoding="str")
        assert seqs_1000_sum[0] == seq_chr1[1050 - 500 : 1050 + 500]
        assert seqs_1000_mid[0] == seq_chr1[1600 - 500 : 1600 + 500]

        # Reverse complement check
        rc_str = ds.get_peak_sequences(window=500, center="summit", peak_indices=[0], encoding="str", reverse_complement=True)[0]
        comp_map = str.maketrans("ACGTN", "TGCAN")
        assert rc_str == seqs_sum[0].translate(comp_map)[::-1]

        # One-hot shape check
        oh = ds.get_peak_sequences(window=360, center="summit", peak_indices=[0, 2, 5], encoding="onehot")
        assert oh.shape == (3, 4, 360) and oh.dtype == np.float32
        np.testing.assert_allclose(oh.sum(axis=1), 1.0, atol=1e-6)
    finally:
        ds.close()


def test_t5_t6_gtf_biotype_inference_and_dual_tss_annotation(tmp_path):
    """T5 & T6: GTF CDS biotype inference on chr8 + dual-track protein-coding vs lncRNA TSS assignment."""
    ds, _, _, gtf_path, fasta_path, _, _ = _build_synthetic_multiome(tmp_path)
    try:
        ds.store_peak_sequences(fasta_path=fasta_path, max_flank=500)
        ann_res = ds.annotate_peaks_tss(gtf_path=gtf_path)
        assert ann_res["n_peaks"] == 24
        assert ann_res["has_gc_content"] is True

        # Check that chr8 genes without `gene` lines inferred `protein_coding` from CDS rows!
        ga = ds.query_gene_annotation()
        chr8_types = dict(zip(ga["gene_id"], ga["gene_type"]))
        assert chr8_types["LPAR6"] == "protein_coding"
        assert chr8_types["LOC1001"] == "protein_coding"
        assert chr8_types["LINC001"] == "lncRNA"

        # Peak 1 on chr1 is at [2100, 2600), summit at 2350:
        # LINC001 (lncRNA, '-' strand, TSS at 2199) is only 151 bp away, while SATB2 (protein_coding, '+' strand, TSS at 1000) is 1,350 bp away!
        peaks_df = ds.peaks.to_pandas()
        p1 = peaks_df.iloc[1]
        assert p1["nearest_pc_gene"] == "SATB2"
        assert p1["nearest_any_gene"] == "LINC001"
        assert p1["nearest_gene"] == "SATB2"  # protein-coding preferred!
        assert 0.0 <= float(p1["gc_content"]) <= 1.0
    finally:
        ds.close()


def test_t8_t9_tn5_footprints_and_fragment_size_stratification(tmp_path):
    """T8 & T9: Multi-region base-pair Tn5 footprinting + fragment size stratification."""
    ds, _, _, _, fasta_path, _, _ = _build_synthetic_multiome(tmp_path)
    try:
        ds.store_peak_sequences(fasta_path=fasta_path, max_flank=500)
        fp_all = ds.get_tn5_footprints(
            regions=[0, 1, 2],
            groupby="Subclass_predicted",
            window=400,
            normalize="raw",
        )
        fp_nfr = ds.get_tn5_footprints(
            regions=[0, 1, 2],
            groupby="Subclass_predicted",
            window=400,
            fragment_size_range=(0, 120),
            normalize="raw",
        )
        fp_mono = ds.get_tn5_footprints(
            regions=[0, 1, 2],
            groupby="Subclass_predicted",
            window=400,
            fragment_size_range=(121, 500),
            normalize="raw",
        )
        assert fp_all.shape == (3, 3, 400)
        # Partition of fragment sizes [0, 120] + [121, 500] equals all fragments in [0, 500]!
        np.testing.assert_allclose(fp_all, fp_nfr + fp_mono, rtol=1e-5)

        # Bias-corrected footprints
        fp_corr = ds.get_tn5_footprints(
            regions=[0, 1, 2],
            groupby="Subclass_predicted",
            window=400,
            correct_tn5_bias=True,
            return_dict=True,
        )
        assert fp_corr["footprints"].shape == (3, 3, 400)
        assert fp_corr["tn5_bias"] is not None and fp_corr["tn5_bias"].shape == (3, 400)
    finally:
        ds.close()
