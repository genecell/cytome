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


# ---------------------------------------------------------------------------
# 0.3.7 — one reader for every GTF convention
# ---------------------------------------------------------------------------

def _gtf(path, rows):
    with open(path, "wt") as f:
        for r in rows:
            f.write("\t".join(r[:8]) + "\t" + r[8] + "\n")


# NCBI RefSeq style, as in the Cell Ranger ARC Mmul_10 reference: names in `gene`, gene_id = the
# symbol (suffixed where it repeats), `gene` lines for some genes only, transcript_id "" on them.
_NCBI_ROWS = [
    ("chr1", "Gnomon", "gene", "1001", "5000", ".", "-", ".",
     'gene_id "PGBD2"; transcript_id ""; gbkey "Gene"; gene "PGBD2"; gene_biotype "protein_coding";'),
    ("chr1", "Gnomon", "transcript", "1001", "5000", ".", "-", ".",
     'gene_id "PGBD2"; transcript_id "XM_1"; gene "PGBD2"; transcript_biotype "mRNA";'),
    ("chr1", "Gnomon", "exon", "4001", "5000", ".", "-", ".",
     'gene_id "PGBD2"; transcript_id "XM_1"; gene "PGBD2";'),
    # no gene line: spans its two transcripts
    ("chr8", "Gnomon", "transcript", "27727647", "27747523", ".", "-", ".",
     'gene_id "CHRNA2"; transcript_id "XM_2"; gene "CHRNA2"; product "a; b"; transcript_biotype "mRNA";'),
    ("chr8", "Gnomon", "exon", "27747013", "27747523", ".", "-", ".",
     'gene_id "CHRNA2"; transcript_id "XM_2"; gene "CHRNA2";'),
    ("chr8", "Gnomon", "CDS", "27747013", "27747400", ".", "-", ".",
     'gene_id "CHRNA2"; transcript_id "XM_2"; gene "CHRNA2";'),
    ("chr8", "BestRefSeq", "transcript", "27729000", "27747600", ".", "-", ".",
     'gene_id "CHRNA2"; transcript_id "NM_3"; gene "CHRNA2";'),
    ("chr8", "BestRefSeq", "exon", "27747100", "27747600", ".", "-", ".",
     'gene_id "CHRNA2"; transcript_id "NM_3"; gene "CHRNA2";'),
    # one symbol, two genes
    ("chr2", "tRNAscan-SE", "transcript", "100", "172", ".", "+", ".",
     'gene_id "TRNAK-UUU"; transcript_id "tx-1"; gene "TRNAK-UUU";'),
    ("chr2", "tRNAscan-SE", "exon", "100", "172", ".", "+", ".",
     'gene_id "TRNAK-UUU"; transcript_id "tx-1"; gene "TRNAK-UUU";'),
    ("chr5", "tRNAscan-SE", "transcript", "900", "972", ".", "-", ".",
     'gene_id "TRNAK-UUU_1"; transcript_id "tx-2"; gene "TRNAK-UUU";'),
    ("chr5", "tRNAscan-SE", "exon", "900", "972", ".", "-", ".",
     'gene_id "TRNAK-UUU_1"; transcript_id "tx-2"; gene "TRNAK-UUU";'),
    # plain GTF 2.2: exons only, no transcript or gene line
    ("chr3", "x", "exon", "501", "600", ".", "+", ".", 'gene_id "EXONLY"; transcript_id "t9"; gene "EXONLY";'),
    ("chr3", "x", "exon", "801", "900", ".", "+", ".", 'gene_id "EXONLY"; transcript_id "t9"; gene "EXONLY";'),
]


def test_ncbi_style_gtf_every_gene_gets_a_named_row(tmp_path):
    g = str(tmp_path / "genes.gtf")
    _gtf(g, _NCBI_ROWS)
    ds = cytome.create(str(tmp_path / "n.cytome"))
    out = cytome.import_gtf(ds, g, verbose=False)
    assert out["n_genes"] == 5 and out["n_gene_lines"] == 1 and out["n_genes_inferred"] == 4
    assert out["n_genes_named"] == 5
    assert out["features"] == {"transcript": 5, "exon": 7, "CDS": 1} and out["n_exons"] == 7
    c = ds.query_gene_annotation(gene_names=["CHRNA2"])
    assert c[["chrom", "start", "end", "strand"]].values.tolist() == [["chr8", 27727646, 27747600, "-"]]
    p = ds.query_gene_annotation(gene_names=["PGBD2"])          # gene line present, named by `gene`
    assert p[["start", "end", "gene_type"]].values.tolist() == [[1000, 5000, "protein_coding"]]
    info = ds.gene_annotation_info()
    assert info["n_genes"] == 5 and info["n_genes_named"] == 5 and info["reader_version"] == 2
    assert info["n_feature_gene_ids"] == 5 and info["n_exons"] == 7 and info["n_features"] == 13
    ds.close()


def test_exon_only_gene_spans_its_exons_and_repeated_symbols_return_every_gene(tmp_path):
    g = str(tmp_path / "genes.gtf")
    _gtf(g, _NCBI_ROWS)
    ds = cytome.create(str(tmp_path / "n.cytome"))
    cytome.import_gtf(ds, g, verbose=False)
    e = ds.query_gene_annotation(gene_names=["EXONLY"])
    assert e[["start", "end"]].values.tolist() == [[500, 900]]
    t = ds.query_gene_annotation(gene_names=["TRNAK-UUU"])
    assert sorted(t.gene_id) == ["TRNAK-UUU", "TRNAK-UUU_1"] and sorted(t.chrom) == ["chr2", "chr5"]
    one = ds.query_gene_annotation(gene_ids=["TRNAK-UUU_1"])
    assert one.chrom.tolist() == ["chr5"]
    both = ds.query_gene_annotation(gene_names=["PGBD2"], gene_ids=["EXONLY"])
    assert sorted(both.gene_id) == ["EXONLY", "PGBD2"]
    ds.close()


def test_gene_line_after_its_rows_still_wins(tmp_path):
    g = str(tmp_path / "late.gtf")
    _gtf(g, [
        ("chr1", "x", "exon", "201", "300", ".", "+", ".", 'gene_id "G"; transcript_id "T"; gene "Gchild";'),
        ("chr1", "x", "gene", "101", "900", ".", "+", ".", 'gene_id "G"; gene_name "Gline"; gene_type "lncRNA";'),
    ])
    ds = cytome.create(str(tmp_path / "l.cytome"))
    out = cytome.import_gtf(ds, g, verbose=False)
    assert out["n_genes_inferred"] == 0
    r = ds.query_gene_annotation(gene_ids=["G"]).iloc[0]
    assert (r.start, r.end, r.gene_name, r.gene_type) == (100, 900, "Gline", "lncRNA")
    ds.close()


def test_gene_split_across_chromosomes_is_kept_on_the_locus_with_most_rows(tmp_path):
    g = str(tmp_path / "split.gtf")
    _gtf(g, [
        ("chr1", "x", "exon", "101", "200", ".", "+", ".", 'gene_id "S"; transcript_id "T1";'),
        ("chr1", "x", "exon", "301", "400", ".", "+", ".", 'gene_id "S"; transcript_id "T1";'),
        ("chr2", "x", "exon", "101", "900", ".", "+", ".", 'gene_id "S"; transcript_id "T2";'),
    ])
    ds = cytome.create(str(tmp_path / "s.cytome"))
    out = cytome.import_gtf(ds, g, verbose=False)
    assert out["genes_split_across_loci"] == 1
    r = ds.query_gene_annotation(gene_ids=["S"]).iloc[0]
    assert (r.chrom, r.start, r.end) == ("chr1", 100, 400) and r.gene_name is None   # never an ID as a name
    ds.close()


def test_biotype_filter_does_not_need_gene_lines(tmp_path):
    g = str(tmp_path / "bt.gtf")
    _gtf(g, _NCBI_ROWS + [
        # Ensembl style without a gene line: the biotype is on the rows
        ("chr4", "ensembl", "transcript", "11", "90", ".", "+", ".",
         'gene_id "ENSG1"; transcript_id "ENST1"; gene_name "EN"; gene_biotype "protein_coding";'),
        ("chr4", "ensembl", "exon", "11", "40", ".", "+", ".",
         'gene_id "ENSG1"; transcript_id "ENST1"; gene_name "EN"; gene_biotype "protein_coding";'),
    ])
    ds = cytome.create(str(tmp_path / "b.cytome"))
    out = cytome.import_gtf(ds, g, gene_biotypes=["protein_coding"], verbose=False)
    assert sorted(ds.query_gene_annotation().gene_id) == ["ENSG1", "PGBD2"]
    assert set(ds.query_exon_annotation().gene_id) == {"ENSG1", "PGBD2"}
    # CHRNA2, both TRNAK-UUU and EXONLY carry no biotype on any row: dropped, and counted
    assert out["genes_dropped_biotype"] == 4 and out["genes_dropped_no_biotype"] == 4
    ds.close()


def test_tag_filter_does_not_depend_on_line_order(tmp_path):
    g = str(tmp_path / "tags.gtf")
    _gtf(g, [
        ("chr1", "x", "exon", "101", "200", ".", "+", ".", 'gene_id "G"; transcript_id "T1"; gene_name "G";'),
        ("chr1", "x", "transcript", "101", "500", ".", "+", ".",
         'gene_id "G"; transcript_id "T1"; gene_name "G"; tag "basic";'),
        ("chr1", "x", "exon", "301", "500", ".", "+", ".", 'gene_id "G"; transcript_id "T2"; gene_name "G";'),
    ])
    ds = cytome.create(str(tmp_path / "t.cytome"))
    cytome.import_gtf(ds, g, transcript_tags=["basic"], verbose=False)
    ex = ds.query_exon_annotation(features=["exon"])
    assert ex.transcript_id.tolist() == ["T1"]                 # T1's exon precedes its transcript line
    assert len(ds.query_gene_annotation()) == 1
    ds.close()


def test_harmonize_chrom_maps_only_to_names_the_store_has():
    from cytome.io.gtf_import import harmonize_chrom
    assert harmonize_chrom("8", {"chr8"}) == "chr8"
    assert harmonize_chrom("chr8", {"8"}) == "8"
    assert harmonize_chrom("MT", {"chrM"}) == "chrM"
    assert harmonize_chrom("chrM", {"MT"}) == "MT"
    assert harmonize_chrom("8", {"8", "chr8"}) == "8"
    assert harmonize_chrom("KZ1234.1", {"chr1"}) == "KZ1234.1"   # no other spelling: unchanged


def _store_with_peaks(path, chroms):
    ds = cytome.create(path)
    for i, c in enumerate(chroms):
        ds._conn.execute("INSERT INTO peaks (peak_idx, peak_id, chr, start, end_) VALUES (?, ?, ?, 1, 100)",
                         (i, f"{c}:1-100", c))
    ds._conn.commit()
    return ds


def test_chrom_names_auto_follows_the_store_spelling(tmp_path):
    g = str(tmp_path / "ens.gtf")
    _gtf(g, [
        ("8", "ensembl", "gene", "1001", "2000", ".", "-", ".", 'gene_id "ENSG8"; gene_name "CHRNA2";'),
        ("8", "ensembl", "exon", "1001", "1200", ".", "-", ".", 'gene_id "ENSG8"; transcript_id "ENST8";'),
        ("MT", "ensembl", "gene", "1", "500", ".", "+", ".", 'gene_id "ENSGM"; gene_name "MT-ND1";'),
        ("KZ1.1", "ensembl", "gene", "1", "50", ".", "+", ".", 'gene_id "ENSGK"; gene_name "SCAF";'),
    ])
    ds = _store_with_peaks(str(tmp_path / "a.cytome"), ["chr8", "chrM", "chr1"])
    out = cytome.import_gtf(ds, g, verbose=False)
    assert out["chrom_renamed"] == {"8": "chr8", "MT": "chrM"}
    got = dict(zip(ds.query_gene_annotation().gene_id, ds.query_gene_annotation().chrom))
    assert got == {"ENSG8": "chr8", "ENSGM": "chrM", "ENSGK": "KZ1.1"}
    assert ds.query_exon_annotation().chrom.tolist() == ["chr8"]
    cytome.import_gtf(ds, g, chrom_names="keep", force=True, verbose=False)
    assert sorted(set(ds.query_gene_annotation().chrom)) == ["8", "KZ1.1", "MT"]
    ds.close()
    bare = cytome.create(str(tmp_path / "b.cytome"))           # no fragments, peaks or tiles
    assert cytome.import_gtf(bare, g, verbose=False)["chrom_renamed"] == {}
    assert sorted(set(bare.query_gene_annotation().chrom)) == ["8", "KZ1.1", "MT"]
    bare.close()


def test_read_gtf_gives_what_import_gtf_writes(tmp_path):
    g = str(tmp_path / "genes.gtf")
    _gtf(g, _NCBI_ROWS)
    genes, feats = cytome.read_gtf(g)
    ds = cytome.create(str(tmp_path / "r.cytome"))
    cytome.import_gtf(ds, g, verbose=False)
    a = ds.query_gene_annotation().sort_values("gene_id").reset_index(drop=True)
    b = genes.sort_values("gene_id").reset_index(drop=True)[a.columns]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    x = ds.query_exon_annotation()
    assert len(x) == len(feats) and sorted(x.gene_id) == sorted(feats.gene_id)
    ds.close()


def test_gene_annotation_info_marks_an_import_by_an_older_reader(tmp_path):
    ds = cytome.create(str(tmp_path / "o.cytome"))
    # what cytome <= 0.3.6 wrote for a GTF with gene lines for one gene only, unnamed
    ds._conn.execute("INSERT INTO _gene_annotation VALUES ('A', 'chr1', 0, 10, '+', NULL, NULL, 'genes')")
    ds._conn.executemany("INSERT INTO _exon_annotation (gene_id, chrom, start, end, feature) VALUES (?, 'chr1', 0, 5, 'exon')",
                         [("A",), ("B",), ("C",)])
    ds._conn.commit()
    info = ds.gene_annotation_info()
    assert info["reader_version"] is None and info["n_genes"] == 1 and info["n_genes_named"] == 0
    assert info["n_feature_gene_ids"] == 3
    ds.close()


def test_import_gtf_refuses_a_read_only_store_before_reading(tmp_path):
    p = str(tmp_path / "ro.cytome")
    g = str(tmp_path / "genes.gtf")
    _gtf(g, _NCBI_ROWS)
    cytome.create(p).close()
    ds = cytome.open(p, mode="r")
    with pytest.raises(cytome.ReadOnlyError):
        cytome.import_gtf(ds, g, verbose=False)
    ds.close()
