"""Tests for the cytome 0.2.2 GTF annotation tables.

Covers:
- Schema creation on a fresh cytome
- ``cytome.import_gtf()`` end-to-end
- 1-based-closed → 0-based-half-open coordinate conversion
- ``ds.query_gene_annotation()`` range / by-name queries
- ``ds.query_exon_annotation()`` range / by-gene_id queries
- Idempotency (force=False raises, force=True replaces)
- Empty input handling
- Schema migration: pre-0.2.2 cytomes get tables on open
"""
from __future__ import annotations

import gzip
import os

import pandas as pd
import pytest

import cytome


_TINY_GTF_LINES = [
    "##header line ignored\n",
    # Dlx2 — protein_coding gene on chr2:+ strand
    'chr2\tunknown\tgene\t71500001\t71510000\t.\t+\t.\t'
    'gene_id "G_Dlx2"; gene_name "Dlx2"; gene_type "protein_coding";\n',
    # Dlx2 exons
    'chr2\tunknown\texon\t71500001\t71501000\t.\t+\t.\t'
    'gene_id "G_Dlx2"; transcript_id "T_Dlx2"; exon_number "1";\n',
    'chr2\tunknown\texon\t71509001\t71510000\t.\t+\t.\t'
    'gene_id "G_Dlx2"; transcript_id "T_Dlx2"; exon_number "2";\n',
    # Dlx1 — different gene on chr2:- strand
    'chr2\tunknown\tgene\t71600001\t71610000\t.\t-\t.\t'
    'gene_id "G_Dlx1"; gene_name "Dlx1"; gene_type "protein_coding";\n',
    'chr2\tunknown\texon\t71600001\t71610000\t.\t-\t.\t'
    'gene_id "G_Dlx1"; transcript_id "T_Dlx1"; exon_number "1";\n',
    # Pvalb — gene on chr15 (out of Dlx range)
    'chr15\tunknown\tgene\t78130001\t78180000\t.\t+\t.\t'
    'gene_id "G_Pvalb"; gene_name "Pvalb"; gene_type "protein_coding";\n',
    # CDS feature for Dlx2 (tests non-exon feature handling)
    'chr2\tunknown\tCDS\t71500100\t71500200\t.\t+\t.\t'
    'gene_id "G_Dlx2"; transcript_id "T_Dlx2";\n',
]


def _write_tiny_gtf(path, gzipped=False):
    opener = gzip.open if gzipped else open
    with opener(path, "wt") as f:
        f.writelines(_TINY_GTF_LINES)


# ---------------------------------------------------------------------------
# Schema + import
# ---------------------------------------------------------------------------

def test_schema_creates_annotation_tables(tmp_path):
    """Empty cytome has the new tables ready for import (schema
    migration is automatic via _create_schema())."""
    p = str(tmp_path / "fresh.cytome")
    ds = cytome.create(p)
    # Tables exist and are empty
    n_g = ds._conn.execute("SELECT COUNT(*) FROM _gene_annotation").fetchone()[0]
    n_e = ds._conn.execute("SELECT COUNT(*) FROM _exon_annotation").fetchone()[0]
    assert n_g == 0 and n_e == 0
    assert ds.gene_annotation_info() is None
    ds.close()


def test_import_gtf_round_trip(tmp_path):
    """End-to-end import + summary info."""
    p = str(tmp_path / "imp.cytome")
    g = str(tmp_path / "tiny.gtf.gz")
    _write_tiny_gtf(g, gzipped=True)

    ds = cytome.create(p)
    # Round 8: default feature_filter now imports the FULL set
    # (transcript + CDS + UTR + 5'/3' UTR variants). Narrow it
    # here to match the legacy 'gene + exon' expectation.
    out = cytome.import_gtf(
        ds, g, feature_filter=("gene", "exon"), verbose=False,
    )
    assert out["n_genes"] == 3
    assert out["n_exons"] == 3
    info = ds.gene_annotation_info()
    assert info is not None
    assert info["n_genes"] == 3 and info["n_exons"] == 3
    ds.close()


def test_coordinate_conversion_gtf_1based_to_bed_0based(tmp_path):
    """GTF: 1-based closed. Cytome: 0-based half-open.
    GTF 71500001-71510000 → cytome 71500000-71510000."""
    p = str(tmp_path / "coord.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    dlx2 = ds.query_gene_annotation(gene_names=["Dlx2"]).iloc[0]
    assert dlx2["start"] == 71_500_000   # 71500001 - 1
    assert dlx2["end"] == 71_510_000     # unchanged (half-open)
    ds.close()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_query_gene_annotation_by_range(tmp_path):
    """Range query: chr2:71.49M-71.55M overlaps Dlx2 but not Dlx1."""
    p = str(tmp_path / "rng.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    df = ds.query_gene_annotation(
        chrom="chr2", start=71_490_000, end=71_550_000
    )
    assert list(df["gene_name"]) == ["Dlx2"]
    ds.close()


def test_query_gene_annotation_by_name(tmp_path):
    """Filter by gene_names returns just those genes."""
    p = str(tmp_path / "nm.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    df = ds.query_gene_annotation(gene_names=["Pvalb"])
    assert len(df) == 1
    assert df.iloc[0]["chrom"] == "chr15"
    ds.close()


def test_query_gene_annotation_empty_returns_empty_df(tmp_path):
    """No overlapping genes → empty DataFrame, not None."""
    p = str(tmp_path / "emp.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    df = ds.query_gene_annotation(chrom="chr99", start=0, end=1_000_000)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    # Schema preserved
    assert "gene_name" in df.columns
    ds.close()


def test_query_exon_annotation_filters_by_gene_id_and_feature(tmp_path):
    """Exon query filtered by gene_id returns only that gene's exons,
    and the feature filter separates exon from CDS."""
    p = str(tmp_path / "ex.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    # Import with extended feature_filter so CDS is also picked up
    cytome.import_gtf(
        ds, g,
        feature_filter=("gene", "exon", "CDS"),
        verbose=False,
    )

    exons = ds.query_exon_annotation(gene_ids=["G_Dlx2"], features=["exon"])
    assert len(exons) == 2
    assert (exons["feature"] == "exon").all()
    assert (exons["gene_id"] == "G_Dlx2").all()

    cds = ds.query_exon_annotation(gene_ids=["G_Dlx2"], features=["CDS"])
    assert len(cds) == 1
    assert cds.iloc[0]["feature"] == "CDS"
    ds.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_import_gtf_idempotent_without_force_raises(tmp_path):
    """A second import without force=True raises with a hint."""
    p = str(tmp_path / "idx.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    with pytest.raises(RuntimeError, match="force=True"):
        cytome.import_gtf(ds, g, verbose=False)
    ds.close()


def test_import_gtf_force_true_replaces(tmp_path):
    """force=True clears the existing tables and reimports."""
    p = str(tmp_path / "frc.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    # Re-import with force — same counts after
    cytome.import_gtf(ds, g, force=True, verbose=False)
    info = ds.gene_annotation_info()
    assert info["n_genes"] == 3
    ds.close()


# ---------------------------------------------------------------------------
# feature_filter
# ---------------------------------------------------------------------------

def test_feature_filter_can_include_cds_and_transcripts(tmp_path):
    """Custom feature_filter brings in extra GTF feature types."""
    p = str(tmp_path / "ff.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(
        ds, g,
        feature_filter=("gene", "exon", "CDS"),
        verbose=False,
    )
    cds = ds.query_exon_annotation(features=["CDS"])
    assert len(cds) == 1
    ds.close()


def test_feature_filter_includes_cds_by_default():
    """Round 8: default feature_filter is the FULL set —
    `cytome.import_gtf` without an explicit `feature_filter`
    should import CDS, transcript, and UTR variants."""
    import inspect
    from cytome.io.gtf_import import import_gtf
    default = inspect.signature(import_gtf).parameters[
        "feature_filter"
    ].default
    assert "CDS" in default
    assert "transcript" in default
    assert "UTR" in default
    assert "five_prime_utr" in default
    assert "three_prime_utr" in default


def test_feature_filter_narrow_still_works(tmp_path):
    """Users can still pass a narrow filter explicitly."""
    p = str(tmp_path / "noc.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(
        ds, g, feature_filter=("gene", "exon"), verbose=False,
    )
    cds = ds.query_exon_annotation(features=["CDS"])
    assert len(cds) == 0
    ds.close()


def test_import_gtf_populates_transcript_type(tmp_path):
    """Round 8: transcript_type / transcript_biotype attribute is
    saved per row in `_exon_annotation`."""
    p = str(tmp_path / "tt.cytome")
    g = str(tmp_path / "tt.gtf")
    with open(g, "w") as f:
        f.write(
            'chr1\tt\tgene\t100\t1000\t.\t+\t.\t'
            'gene_id "G"; gene_name "G"; gene_type "protein_coding";\n'
            'chr1\tt\ttranscript\t100\t1000\t.\t+\t.\t'
            'gene_id "G"; transcript_id "T1"; '
            'transcript_type "protein_coding";\n'
            'chr1\tt\texon\t100\t300\t.\t+\t.\t'
            'gene_id "G"; transcript_id "T1"; '
            'transcript_type "protein_coding";\n'
            'chr1\tt\ttranscript\t100\t1000\t.\t+\t.\t'
            'gene_id "G"; transcript_id "T2"; '
            'transcript_type "nonsense_mediated_decay";\n'
        )
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    df = ds.query_exon_annotation(features=["transcript"])
    assert "transcript_type" in df.columns
    types = sorted(t for t in df["transcript_type"].dropna().unique())
    assert types == ["nonsense_mediated_decay", "protein_coding"]
    ds.close()


# ---------------------------------------------------------------------------
# Compressed input
# ---------------------------------------------------------------------------

def test_import_gtf_handles_gz_and_uncompressed(tmp_path):
    """Both .gz and uncompressed input parse identically."""
    p1 = str(tmp_path / "g.cytome")
    p2 = str(tmp_path / "u.cytome")
    g_gz = str(tmp_path / "tg.gtf.gz")
    g_un = str(tmp_path / "tu.gtf")
    _write_tiny_gtf(g_gz, gzipped=True)
    _write_tiny_gtf(g_un, gzipped=False)

    ds1 = cytome.create(p1)
    cytome.import_gtf(ds1, g_gz, verbose=False)
    n_gz = ds1.gene_annotation_info()["n_genes"]
    ds1.close()

    ds2 = cytome.create(p2)
    cytome.import_gtf(ds2, g_un, verbose=False)
    n_un = ds2.gene_annotation_info()["n_genes"]
    ds2.close()

    assert n_gz == n_un == 3


# ---------------------------------------------------------------------------
# Source label
# ---------------------------------------------------------------------------

def test_source_label_defaults_to_filename_stem(tmp_path):
    """Default source label is the filename without .gtf/.gz."""
    p = str(tmp_path / "src.cytome")
    g = str(tmp_path / "gencode.vM25.basic.annotation.gtf.gz")
    _write_tiny_gtf(g, gzipped=True)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, verbose=False)
    assert (
        ds.gene_annotation_info()["source"]
        == "gencode.vM25.basic.annotation"
    )
    ds.close()


def test_source_label_explicit_override(tmp_path):
    p = str(tmp_path / "src2.cytome")
    g = str(tmp_path / "t.gtf")
    _write_tiny_gtf(g, gzipped=False)
    ds = cytome.create(p)
    cytome.import_gtf(ds, g, source_label="my_custom", verbose=False)
    assert ds.gene_annotation_info()["source"] == "my_custom"
    ds.close()


# ---------------------------------------------------------------------------
# Round 6a — transcript_id_pattern / transcript_id_prefixes filter
# ---------------------------------------------------------------------------

def _write_refseq_style_gtf(path, gzipped=False):
    """GTF with NM_ (curated) + XM_ (predicted) transcript_ids."""
    lines = [
        'chr1\trefseq\tgene\t1000\t5000\t.\t+\t.\t'
        'gene_id "G1"; gene_name "Foo";\n',
        # NM_001 — curated transcript + its exon
        'chr1\trefseq\ttranscript\t1000\t3000\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "NM_001"; gene_name "Foo";\n',
        'chr1\trefseq\texon\t1000\t1500\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "NM_001"; gene_name "Foo";\n',
        # XM_002 — predicted transcript + its exon
        'chr1\trefseq\ttranscript\t2000\t5000\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "XM_002"; gene_name "Foo";\n',
        'chr1\trefseq\texon\t4500\t5000\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "XM_002"; gene_name "Foo";\n',
    ]
    opener = gzip.open if gzipped else open
    with opener(path, 'wt') as f:
        f.writelines(lines)


def test_import_gtf_transcript_id_prefixes_filter_drops_xm(tmp_path):
    """transcript_id_prefixes=['NM_'] keeps NM_ child rows, drops XM_."""
    p = str(tmp_path / "filt.cytome")
    g = str(tmp_path / "refseq.gtf")
    _write_refseq_style_gtf(g)
    ds = cytome.create(p)
    cytome.import_gtf(
        ds, g, feature_filter=('gene', 'exon', 'transcript'),
        transcript_id_prefixes=['NM_'], verbose=False,
    )
    # 1 gene + 1 NM_ transcript + 1 NM_ exon = expected
    tx = ds.query_exon_annotation(features=['transcript'])
    ex = ds.query_exon_annotation(features=['exon'])
    assert len(tx) == 1 and tx.iloc[0]['transcript_id'] == 'NM_001', tx
    assert len(ex) == 1 and ex.iloc[0]['transcript_id'] == 'NM_001', ex
    ds.close()


def test_import_gtf_transcript_id_pattern_regex(tmp_path):
    """transcript_id_pattern accepts a raw regex."""
    p = str(tmp_path / "filt2.cytome")
    g = str(tmp_path / "refseq2.gtf")
    _write_refseq_style_gtf(g)
    ds = cytome.create(p)
    cytome.import_gtf(
        ds, g, feature_filter=('gene', 'exon', 'transcript'),
        transcript_id_pattern=r'^NM_|^NR_', verbose=False,
    )
    tx = ds.query_exon_annotation(features=['transcript'])
    assert all(t.startswith('NM_') or t.startswith('NR_')
               for t in tx['transcript_id']), tx['transcript_id'].tolist()
    ds.close()


def test_import_gtf_both_pattern_and_prefixes_raises(tmp_path):
    """Passing both must raise ValueError."""
    p = str(tmp_path / "x.cytome")
    g = str(tmp_path / "x.gtf")
    _write_refseq_style_gtf(g)
    ds = cytome.create(p)
    with pytest.raises(ValueError, match="not both"):
        cytome.import_gtf(
            ds, g, transcript_id_pattern=r'^NM_',
            transcript_id_prefixes=['NM_'],
        )
    ds.close()


# ---------------------------------------------------------------------------
# Round 6b — transcript_tags + gene_biotypes filters
# ---------------------------------------------------------------------------

def _write_gencode_style_gtf(path):
    """GTF with gene_biotype + transcript-level tag attributes."""
    lines = [
        'chr1\tgencode\tgene\t1000\t5000\t.\t+\t.\t'
        'gene_id "G1"; gene_name "PC"; gene_biotype "protein_coding";\n',
        'chr1\tgencode\ttranscript\t1000\t5000\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "T1"; tag "basic";\n',
        'chr1\tgencode\texon\t1000\t1500\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "T1";\n',
        # Same gene, non-basic transcript
        'chr1\tgencode\ttranscript\t2000\t5000\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "T2";\n',
        'chr1\tgencode\texon\t4500\t5000\t.\t+\t.\t'
        'gene_id "G1"; transcript_id "T2";\n',
        # Different gene — lncRNA
        'chr1\tgencode\tgene\t10000\t15000\t.\t+\t.\t'
        'gene_id "G2"; gene_name "LNC"; gene_biotype "lncRNA";\n',
        'chr1\tgencode\ttranscript\t10000\t15000\t.\t+\t.\t'
        'gene_id "G2"; transcript_id "T3"; tag "basic";\n',
        'chr1\tgencode\texon\t10000\t10500\t.\t+\t.\t'
        'gene_id "G2"; transcript_id "T3";\n',
    ]
    with open(path, 'wt') as f:
        f.writelines(lines)


def test_import_gtf_gene_biotypes_filter(tmp_path):
    """gene_biotypes=['protein_coding'] keeps PC gene + all its
    child rows; drops the lncRNA gene + its children."""
    p = str(tmp_path / "biotype.cytome")
    g = str(tmp_path / "gencode.gtf")
    _write_gencode_style_gtf(g)
    ds = cytome.create(p)
    cytome.import_gtf(
        ds, g, feature_filter=('gene', 'exon', 'transcript'),
        gene_biotypes=['protein_coding'], verbose=False,
    )
    g_df = ds.query_gene_annotation()
    assert len(g_df) == 1, g_df
    assert g_df.iloc[0]['gene_name'] == 'PC'
    # All child rows should belong to G1
    ex_df = ds.query_exon_annotation()
    assert len(ex_df) > 0
    assert all(ex_df['gene_id'] == 'G1')
    ds.close()


def test_import_gtf_transcript_tags_filter(tmp_path):
    """transcript_tags=['basic'] keeps only transcripts with that
    tag (and their child rows)."""
    p = str(tmp_path / "tag.cytome")
    g = str(tmp_path / "gencode.gtf")
    _write_gencode_style_gtf(g)
    ds = cytome.create(p)
    cytome.import_gtf(
        ds, g, feature_filter=('gene', 'exon', 'transcript'),
        transcript_tags=['basic'], verbose=False,
    )
    tx_df = ds.query_exon_annotation(features=['transcript'])
    assert len(tx_df) == 2, tx_df    # T1 + T3 (both tagged 'basic')
    assert set(tx_df['transcript_id']) == {'T1', 'T3'}
    # T2 (no basic tag) and its exon must be excluded
    ex_df = ds.query_exon_annotation(features=['exon'])
    assert all(ex_df['transcript_id'] != 'T2'), ex_df
    ds.close()


def test_import_gtf_combined_biotype_and_tag_filter(tmp_path):
    """Combining gene_biotypes + transcript_tags."""
    p = str(tmp_path / "combo.cytome")
    g = str(tmp_path / "gencode.gtf")
    _write_gencode_style_gtf(g)
    ds = cytome.create(p)
    cytome.import_gtf(
        ds, g, feature_filter=('gene', 'exon', 'transcript'),
        gene_biotypes=['protein_coding'],
        transcript_tags=['basic'],
        verbose=False,
    )
    # Only G1 (protein_coding) AND only T1 (basic) survives
    g_df = ds.query_gene_annotation()
    assert len(g_df) == 1 and g_df.iloc[0]['gene_id'] == 'G1'
    tx_df = ds.query_exon_annotation(features=['transcript'])
    assert len(tx_df) == 1 and tx_df.iloc[0]['transcript_id'] == 'T1'
    ds.close()
