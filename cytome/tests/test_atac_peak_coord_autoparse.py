"""Tests for ATAC ``from_anndata`` auto-parsing of chr/start/end_ from var_names.

When an ATAC AnnData has var_names formatted as ``"chr:start-end"`` and the
``chr``/``start``/``end_`` columns are missing from ``adata.var``,
``cytome.from_anndata(modality='ATAC', ...)`` derives them automatically and
emits a ``UserWarning``. When parsing fails (var_names don't all match the
pattern) and the columns are still missing, ``from_anndata`` raises a
``ValueError`` with an actionable hint instead of the generic SQLite
``IntegrityError: NOT NULL constraint failed: peaks.chr`` that would
otherwise surface at flush time.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _make_atac_adata(peak_strings, n_cells=5, has_explicit_coords=False):
    import anndata as ad
    n_peaks = len(peak_strings)
    X = sp.csr_matrix(np.eye(n_cells, n_peaks, dtype=np.float32))
    var = pd.DataFrame(index=list(peak_strings))
    if has_explicit_coords:
        chrs, starts, ends = [], [], []
        for s in peak_strings:
            chrom, coords = s.split(":")
            start, end = coords.split("-")
            chrs.append(chrom)
            starts.append(int(start))
            ends.append(int(end))
        var["chr"] = chrs
        var["start"] = starts
        var["end_"] = ends
    obs = pd.DataFrame(index=[f"AAA-{i}" for i in range(n_cells)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_autoparse_when_all_varnames_match(tmp_path):
    """All var_names match the canonical chr:start-end pattern → cytome derives
    the chr/start/end_ columns and emits a UserWarning."""
    import cytome
    peaks = [
        "chr1:3094641-3095334",
        "chr1:3102816-3103159",
        "chr2:5000-6000",
        "chrX:100-200",
    ]
    adata = _make_atac_adata(peaks)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = cytome.from_anndata(
            adata, modality="ATAC", output=str(tmp_path / "auto.cytome"),
        )
    derived_warnings = [
        str(w.message) for w in caught
        if "auto-derived chr/start/end_" in str(w.message)
    ]
    assert derived_warnings, (
        f"Expected an auto-derive UserWarning; got: "
        f"{[str(w.message) for w in caught]}"
    )
    assert "4 ATAC peaks" in derived_warnings[0]

    df = ds.peaks.to_pandas()
    assert list(df["chr"]) == ["chr1", "chr1", "chr2", "chrX"]
    assert list(df["start"]) == [3094641, 3102816, 5000, 100]
    assert list(df["end_"]) == [3095334, 3103159, 6000, 200]
    ds.close()


def test_no_warning_when_var_already_has_coords(tmp_path):
    """If adata.var already has chr/start/end_ columns, the auto-parse is a
    no-op and no warning fires."""
    import cytome
    peaks = ["chr1:100-200", "chr1:300-400", "chr1:500-600"]
    adata = _make_atac_adata(peaks, has_explicit_coords=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = cytome.from_anndata(
            adata, modality="ATAC", output=str(tmp_path / "explicit.cytome"),
        )
    derived = [w for w in caught if "auto-derived" in str(w.message)]
    assert not derived, (
        f"No auto-derive warning expected when var already has coords; got: "
        f"{[str(w.message) for w in derived]}"
    )
    df = ds.peaks.to_pandas()
    assert list(df["chr"]) == ["chr1", "chr1", "chr1"]
    ds.close()


def test_actionable_error_when_some_varnames_dont_match(tmp_path):
    """Mixed var_names → no partial parse, a clear ValueError with hint."""
    import cytome
    peaks = [
        "chr1:100-200",
        "GeneA",          # not a peak coord — partial-match guard kicks in
        "chr2:5000-6000",
    ]
    adata = _make_atac_adata(peaks)

    with pytest.raises(ValueError) as exc_info:
        cytome.from_anndata(
            adata, modality="ATAC", output=str(tmp_path / "mixed.cytome"),
        )
    msg = str(exc_info.value)
    assert "ATAC" in msg
    assert "chr" in msg and "start" in msg and "end_" in msg
    # The hint mentions both fix paths
    assert "var_names" in msg
    assert "Populate the columns explicitly" in msg


def test_actionable_error_when_varnames_use_unsupported_format(tmp_path):
    """Non-canonical formats (e.g. underscore-separated) trigger the
    actionable error rather than a cryptic SQLite IntegrityError."""
    import cytome
    peaks = ["chr1_100_200", "chr1_300_400"]  # Not chr:start-end
    adata = _make_atac_adata(peaks)
    with pytest.raises(ValueError) as exc_info:
        cytome.from_anndata(
            adata, modality="ATAC",
            output=str(tmp_path / "underscore.cytome"),
        )
    assert "chr:start-end" in str(exc_info.value)


def test_rna_modality_not_affected(tmp_path):
    """Auto-parse only triggers for ATAC. RNA imports must not be touched
    even if var_names happen to look like coordinates (unlikely but
    possible — a defensive test)."""
    import cytome
    rna_peak_lookalikes = ["chr1:100-200", "chr1:300-400", "chr1:500-600"]
    adata = _make_atac_adata(rna_peak_lookalikes)  # but used as RNA
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = cytome.from_anndata(
            adata, modality="RNA", output=str(tmp_path / "rna.cytome"),
        )
    derived = [w for w in caught if "auto-derived" in str(w.message)]
    assert not derived, "RNA modality must not trigger ATAC auto-parse."
    # RNA path stores into 'genes' (no chr/start/end_ NOT NULL constraints)
    df = ds.genes.to_pandas()
    assert list(df["gene_id"]) == rna_peak_lookalikes
    ds.close()


def test_var_names_with_chrM_and_chr_with_underscore(tmp_path):
    """Edge case: chromosome name with underscore (e.g. 'chr19_KI270938v1_alt'),
    the regex is anchored to '^[^:\\s]+' so anything that's not a colon or
    whitespace is fine."""
    import cytome
    peaks = [
        "chrM:1-100",
        "chr19_KI270938v1_alt:50-60",
        "chrUn_KI270519v1:200-300",
    ]
    adata = _make_atac_adata(peaks)
    ds = cytome.from_anndata(
        adata, modality="ATAC", output=str(tmp_path / "alt.cytome"),
    )
    df = ds.peaks.to_pandas()
    assert list(df["chr"]) == [
        "chrM", "chr19_KI270938v1_alt", "chrUn_KI270519v1",
    ]
    assert list(df["start"]) == [1, 50, 200]
    ds.close()
