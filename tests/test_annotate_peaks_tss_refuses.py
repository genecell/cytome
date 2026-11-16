"""annotate_peaks_tss writes nothing when it has no reference, and replaces an annotation only when asked.

With no TSS reference it wrote 'Distal' and no nearest gene into every peak, over whatever the peaks held,
and returned normally (it happened on a real store on 2026-11-05: 351,065 peaks, all 'Distal'). PIASO's
quantifier calls it after every quantification. Audit 2026-11-07, Section 5.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.core.peak_annotation import annotate_peaks_tss


def _store(path):
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b"]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["G1"], "symbol": ["Gene1"]}))
    ds.set_entity("peaks", pd.DataFrame({
        "peak_id": ["chr1:900-1400", "chr1:50000-50500"], "chr": ["chr1", "chr1"],
        "start": [900, 50000], "end_": [1400, 50500]}))
    ds.flush()
    ds.add_matrix("ATAC_counts", sp.csr_matrix(np.ones((2, 2), dtype=np.float32)))
    ds.flush()
    return ds


def _annotated(ds):
    return ds._conn.execute(
        "SELECT COUNT(*) FROM peaks WHERE annotation IS NOT NULL OR nearest_gene IS NOT NULL").fetchone()[0]


def _add_reference(ds, chrom="chr1"):
    ds._conn.executemany(
        "INSERT INTO _gene_annotation (gene_id, chrom, start, end, strand, gene_name, gene_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [("ENSG00000000001.1", chrom, 1000, 5000, "+", "Gene1", "protein_coding")])
    ds._conn.commit()


def test_no_reference_raises_and_leaves_the_peaks_alone(tmp_path):
    ds = _store(tmp_path / "a.cytome")
    with pytest.raises(FileNotFoundError, match="holds no gene annotation"):
        annotate_peaks_tss(ds)
    assert _annotated(ds) == 0
    ds.close()


def test_a_reference_on_other_chromosomes_raises(tmp_path):
    ds = _store(tmp_path / "b.cytome")
    _add_reference(ds, chrom="chr2")
    with pytest.raises(ValueError, match="none of the peaks' chromosomes"):
        annotate_peaks_tss(ds)
    assert _annotated(ds) == 0
    ds.close()


def test_an_annotation_is_replaced_only_with_overwrite(tmp_path):
    ds = _store(tmp_path / "c.cytome")
    _add_reference(ds)
    annotate_peaks_tss(ds)
    assert _annotated(ds) == 2
    assert ds._conn.execute("SELECT annotation FROM peaks ORDER BY peak_idx").fetchall()[0][0] == "Promoter"
    with pytest.raises(ValueError, match="already carry an annotation"):
        annotate_peaks_tss(ds)
    out = annotate_peaks_tss(ds, overwrite=True)
    assert out["n_peaks"] == 2
    ds.close()


def _gtf(path, chrom):
    att = 'gene_id "ENSG00000000001"; gene_name "Gene1"; gene_biotype "protein_coding";'
    path.write_text(
        f'{chrom}\ttest\tgene\t1001\t5000\t.\t+\t.\t{att}\n'
        f'{chrom}\ttest\ttranscript\t1001\t5000\t.\t+\t.\t{att} transcript_id "T1";\n'
        f'{chrom}\ttest\texon\t1001\t1500\t.\t+\t.\t{att} transcript_id "T1"; exon_number "1";\n')
    return path


def _reference_rows(ds):
    return (ds._conn.execute("SELECT COUNT(*) FROM _gene_annotation").fetchone()[0],
            ds._conn.execute("SELECT COUNT(*) FROM _exon_annotation").fetchone()[0])


def test_a_refused_gtf_does_not_stay_in_the_store(tmp_path):
    # The GTF of another genome was imported before it was found to name none of the peaks' chromosomes,
    # and it stayed: the next call, with the right GTF, was then answered by the wrong one in the store.
    ds = _store(tmp_path / "d.cytome")
    wrong, right = _gtf(tmp_path / "wrong.gtf", "scaffold_9"), _gtf(tmp_path / "right.gtf", "chr1")
    with pytest.raises(ValueError, match="none of the peaks' chromosomes"):
        annotate_peaks_tss(ds, gtf_path=wrong)
    assert _reference_rows(ds) == (0, 0)
    assert ds.metadata.get("gene_annotation") is None
    assert _annotated(ds) == 0
    out = annotate_peaks_tss(ds, gtf_path=right)
    assert out["n_peaks"] == 2 and _annotated(ds) == 2
    assert _reference_rows(ds)[0] == 1
    assert ds.metadata.get("gene_annotation")["gtf"] == "right.gtf"
    ds.close()


def test_a_refused_reference_copied_from_another_store_does_not_stay(tmp_path):
    src = _store(tmp_path / "src.cytome")
    _add_reference(src, chrom="scaffold_9")
    ds = _store(tmp_path / "e.cytome")
    with pytest.raises(ValueError, match="none of the peaks' chromosomes"):
        annotate_peaks_tss(ds, copy_from_dataset=src)
    assert _reference_rows(ds) == (0, 0)
    src.close()
    ds.close()


def test_a_reference_the_store_already_held_is_left_in_it(tmp_path):
    ds = _store(tmp_path / "f.cytome")
    _add_reference(ds, chrom="chr2")
    with pytest.raises(ValueError, match="force=True"):       # the message says how to replace it
        annotate_peaks_tss(ds)
    assert _reference_rows(ds)[0] == 1
    ds.close()


def test_a_gtf_that_is_not_read_is_said_so(tmp_path):
    # A store that holds an annotation uses it. A gtf_path given beside it was ignored without a word.
    ds = _store(tmp_path / "g.cytome")
    first, other = _gtf(tmp_path / "first.gtf", "chr1"), _gtf(tmp_path / "other.gtf", "chr1")
    annotate_peaks_tss(ds, gtf_path=first)
    with pytest.warns(UserWarning, match="was not read"):
        annotate_peaks_tss(ds, gtf_path=other, overwrite=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # the same GTF again: nothing to say
        annotate_peaks_tss(ds, gtf_path=first, overwrite=True)
        annotate_peaks_tss(ds, overwrite=True)
    ds.close()
