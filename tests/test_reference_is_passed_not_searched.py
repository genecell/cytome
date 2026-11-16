"""A store's reference is passed as data, never searched for, and peaks name their genes by ID.

What was wrong before, and the rule now:
- the GTF, the TSS BED and the FASTA were found by walking up from the store, the working directory and the
  package, by two environment variables, and by a genome guessed from the chromosome names (chr19 read as
  mm10, chr20 to chr22 as hg38). The same call could find another reference, or none, on another machine, and
  a GRCm39 store was given mm10 files. Now the reference is a GTF or FASTA passed by path, an
  annotation the store holds, or one copied from another store;
- the nearest gene was renamed to the dataset's symbol and restricted to the dataset's genes, so a peak moved
  to a farther gene when its nearest one was not measured (0.13 to 0.19% of peaks). Now it is the
  reference's gene, with its version-stripped Ensembl ID, and peak_gene_index joins peaks to genes by that ID.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.core.peak_annotation import annotate_peaks_tss, peak_gene_index, versionless_gene_id
from cytome.core.sequences import store_peak_sequences
from cytome.io.gtf_import import ensure_gtf_imported


def _store(path, genes=None):
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b"]}))
    ds.set_entity("genes", genes if genes is not None else pd.DataFrame({"gene_id": ["G1"], "symbol": ["Gene1"]}))
    ds.set_entity("peaks", pd.DataFrame({
        "peak_id": ["chr19:10000-10500", "chr19:50000-50500"], "chr": ["chr19", "chr19"],
        "start": [10000, 50000], "end_": [10500, 50500]}))
    ds.flush()
    ds.add_matrix("ATAC_counts", sp.csr_matrix(np.ones((2, 2), dtype=np.float32)))
    ds.flush()
    return ds


def _reference_dir(root):
    """Every file the old search looked for, for mm10 and hg38."""
    (root / "gtf").mkdir(parents=True)
    for name in ("gencode.vM25.basic.annotation.gtf.gz", "gencode.v44.basic.annotation.gtf.gz"):
        with gzip.open(root / "gtf" / name, "wt") as fh:
            fh.write('chr19\tx\tgene\t10001\t20000\t.\t+\t.\tgene_id "ENSMUSG00000000001.1"; gene_name "Found"; '
                     'gene_type "protein_coding";\n')
    fa = root / "refdata-cellranger-arc-mm10-2020-A-2.0.0" / "fasta"
    fa.mkdir(parents=True)
    (fa / "genome.fa").write_text(">chr19\n" + "ACGT" * 20000 + "\n")
    for g in ("mm10", "hg38"):
        (root / g).mkdir()
        (root / g / f"{g}_transcript_tss.bed").write_text("chr19\t10000\t10001\tFound\t0\t+\n")
    return root


def _annotated(ds):
    return ds._conn.execute(
        "SELECT COUNT(*) FROM peaks WHERE annotation IS NOT NULL OR nearest_gene IS NOT NULL").fetchone()[0]


def test_no_reference_is_found_on_disk(tmp_path, monkeypatch):
    ref = _reference_dir(tmp_path / "data")                    # also an ancestor 'data' dir of the store
    monkeypatch.setenv("CYTOME_REFERENCE_DIR", str(ref))
    monkeypatch.setenv("PIASO_DATA_DIR", str(ref))
    monkeypatch.chdir(tmp_path)
    ds = _store(tmp_path / "s.cytome")                          # chr19: the old guess said mm10
    assert ensure_gtf_imported(ds)["status"] == "empty"
    with pytest.raises(FileNotFoundError, match="holds no gene annotation; pass gtf_path="):
        annotate_peaks_tss(ds)
    assert _annotated(ds) == 0
    with pytest.raises(ValueError, match="fasta_path="):
        store_peak_sequences(ds)
    for retired in ({"genome": "mm10"}, {"restrict_to_dataset_genes": False}):
        with pytest.raises(TypeError):
            annotate_peaks_tss(ds, **retired)
    with pytest.raises(TypeError):
        store_peak_sequences(ds, genome="GRCm39")
    ds.close()


def test_the_gtf_the_store_records_is_named_not_read(tmp_path):
    ds = _store(tmp_path / "r.cytome")
    gtf = _reference_dir(tmp_path / "ref") / "gtf" / "gencode.vM25.basic.annotation.gtf.gz"
    ds.metadata["gene_annotation"] = {"gtf_path": str(gtf)}
    ds.flush()
    with pytest.raises(FileNotFoundError, match="records the GTF .*gencode.vM25"):
        annotate_peaks_tss(ds)
    out = annotate_peaks_tss(ds, gtf_path=gtf)                 # passed, it is imported and used
    assert out["reference"]["status"] == "imported" and _annotated(ds) == 2
    ds.close()


def test_genes_are_named_by_the_reference_and_joined_by_id(tmp_path):
    # Cell Ranger's vM23 calls ENSMUSG00000073730 '4933415F23Rik'; GENCODE vM25 calls it 'Ppp1r14bl'
    genes = pd.DataFrame({"gene_id": ["4933415F23Rik", "Sox2"], "symbol": ["4933415F23Rik", "Sox2"],
                          "gene_ids": ["ENSMUSG00000073730", "ENSMUSG00000074637"]})
    ds = _store(tmp_path / "g.cytome", genes=genes)
    gene_cols_before = [r[1] for r in ds._conn.execute("PRAGMA table_info(genes)")]
    ds._conn.executemany(
        "INSERT INTO _gene_annotation (gene_id, chrom, start, end, strand, gene_name, gene_type) VALUES (?,?,?,?,?,?,?)",
        [("ENSMUSG00000073730.3", "chr19", 10100, 15000, "+", "Ppp1r14bl", "protein_coding"),
         ("ENSMUSG00000074637.1", "chr19", 50800, 55000, "+", "Sox2", "protein_coding"),
         # not measured in this dataset, and nearer peak 1 than Sox2: still its nearest gene
         ("ENSMUSG00000999999.1", "chr19", 50240, 52000, "+", "Gm43517", "protein_coding")])
    ds._conn.commit()
    annotate_peaks_tss(ds)
    rows = ds._conn.execute(
        "SELECT nearest_gene, nearest_gene_id, nearest_pc_gene_id, nearest_any_gene FROM peaks ORDER BY peak_idx"
    ).fetchall()
    assert rows == [("Ppp1r14bl", "ENSMUSG00000073730", "ENSMUSG00000073730", "Ppp1r14bl"),
                    ("Gm43517", "ENSMUSG00000999999", "ENSMUSG00000999999", "Gm43517")]
    np.testing.assert_array_equal(peak_gene_index(ds), [0, -1])  # 4933415F23Rik by ID; Gm43517 not measured
    # the genes table is not touched (it gained biotype and strand columns before)
    assert [r[1] for r in ds._conn.execute("PRAGMA table_info(genes)")] == gene_cols_before
    ds.close()


def test_versionless_gene_id():
    assert versionless_gene_id("ENSG00000141510.18") == "ENSG00000141510"
    assert versionless_gene_id("ENSG00000182378.14_PAR_Y") == "ENSG00000182378"
    assert versionless_gene_id("ENSMUSG00000073730") == "ENSMUSG00000073730"
    assert versionless_gene_id("Sox2") == "Sox2"
    assert versionless_gene_id(None) is None and versionless_gene_id("") is None
