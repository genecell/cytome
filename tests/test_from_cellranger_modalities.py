"""from_cellranger: modalities switch (rna/atac/both) + import_fragments deprecation + streaming."""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

import cytome


def _make_cellranger_mtx(folder: Path, with_frags=False):
    """Tiny multiome MTX dir: 3 genes + 2 peaks × 4 cells."""
    d = folder / "filtered_feature_bc_matrix"
    d.mkdir(parents=True)
    # features × cells (5 features, 4 cells)
    M = sp.csr_matrix(np.array([
        [3, 0, 5, 1],   # GENE1
        [0, 2, 0, 4],   # GENE2
        [1, 1, 1, 1],   # GENE3
        [0, 5, 2, 0],   # peak chr1
        [4, 0, 0, 3],   # peak chr2
    ], dtype=np.int64))
    scipy.io.mmwrite(str(d / "matrix.mtx"), M)
    with open(d / "matrix.mtx", "rb") as src, gzip.open(d / "matrix.mtx.gz", "wb") as dst:
        dst.write(src.read())
    (d / "matrix.mtx").unlink()
    with gzip.open(d / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(f"BC{i}-1" for i in range(4)) + "\n")
    feats = [
        ("ENSG1", "GENE1", "Gene Expression"),
        ("ENSG2", "GENE2", "Gene Expression"),
        ("ENSG3", "GENE3", "Gene Expression"),
        ("chr1:1000-1500", "chr1:1000-1500", "Peaks"),
        ("chr2:2000-2500", "chr2:2000-2500", "Peaks"),
    ]
    with gzip.open(d / "features.tsv.gz", "wt") as fh:
        for fid, nm, ft in feats:
            fh.write(f"{fid}\t{nm}\t{ft}\n")
    if with_frags:
        with gzip.open(folder / "atac_fragments.tsv.gz", "wt") as fh:
            fh.write("chr1\t1010\t1200\tBC0-1\t1\nchr2\t2010\t2200\tBC1-1\t1\n")


def test_modalities_both(tmp_path):
    _make_cellranger_mtx(tmp_path / "outs")
    ds = cytome.from_cellranger(tmp_path / "outs", tmp_path / "both.cytome", modalities="both")
    assert ds.n_genes == 3 and ds.n_peaks == 2
    ds.close()


def test_modalities_rna_only(tmp_path):
    _make_cellranger_mtx(tmp_path / "outs")
    ds = cytome.from_cellranger(tmp_path / "outs", tmp_path / "rna.cytome", modalities="rna")
    assert ds.n_genes == 3
    assert ds.n_peaks == 0          # ATAC peaks dropped entirely
    assert "ATAC" not in ds.modalities
    ds.close()


def test_modalities_atac_only(tmp_path):
    _make_cellranger_mtx(tmp_path / "outs")
    ds = cytome.from_cellranger(tmp_path / "outs", tmp_path / "atac.cytome", modalities="atac")
    assert ds.n_peaks == 2
    assert ds.n_genes == 0          # RNA genes dropped
    ds.close()


def test_bad_modalities(tmp_path):
    _make_cellranger_mtx(tmp_path / "outs")
    with pytest.raises(ValueError):
        cytome.from_cellranger(tmp_path / "outs", tmp_path / "x.cytome", modalities="rnaatac")


def test_import_fragments_deprecated_but_works(tmp_path):
    _make_cellranger_mtx(tmp_path / "outs", with_frags=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = cytome.from_cellranger(tmp_path / "outs", tmp_path / "frag.cytome",
                                    import_fragments=True)
    assert any(issubclass(w.category, DeprecationWarning)
               and "importCellRanger" in str(w.message) for w in caught)
    # fragments went into the streaming chunk store, not the legacy per-row tables
    tables = {r[0] for r in ds._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "fragment_chunks" in tables
    ds.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_import_fragments_default_imports_when_atac(tmp_path):
    """import_fragments=None (default) → fragments ARE imported when ATAC is present."""
    _make_cellranger_mtx(tmp_path / "outs", with_frags=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = cytome.from_cellranger(tmp_path / "outs", tmp_path / "d.cytome", modalities="both")
    assert "ATAC" in ds.modalities and ds.ATAC.fragments.n_fragments > 0
    ds.close()


def test_import_fragments_default_skipped_for_rna(tmp_path):
    """modalities='rna' → no fragments imported even if atac_fragments.tsv.gz exists, no warning."""
    _make_cellranger_mtx(tmp_path / "outs", with_frags=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)   # would raise if fragments imported
        ds = cytome.from_cellranger(tmp_path / "outs", tmp_path / "r.cytome", modalities="rna")
    assert "ATAC" not in ds.modalities and ds.n_peaks == 0
    ds.close()


def test_modalities_both_no_peaks_warns(tmp_path):
    """modalities='both' but a folder has only RNA features → warn 'imported RNA only'."""
    import scipy.io, scipy.sparse as sp, numpy as np
    d = (tmp_path / "outs") / "filtered_feature_bc_matrix"; d.mkdir(parents=True)
    M = sp.csr_matrix(np.array([[3, 0, 5, 1], [0, 2, 0, 4]], dtype=np.int64))   # 2 genes × 4 cells
    scipy.io.mmwrite(str(d / "matrix.mtx"), M)
    with open(d / "matrix.mtx", "rb") as s, gzip.open(d / "matrix.mtx.gz", "wb") as o:
        o.write(s.read())
    (d / "matrix.mtx").unlink()
    with gzip.open(d / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(f"BC{i}-1" for i in range(4)) + "\n")
    with gzip.open(d / "features.tsv.gz", "wt") as fh:
        fh.write("ENSG1\tGENE1\tGene Expression\nENSG2\tGENE2\tGene Expression\n")
    with pytest.warns(UserWarning, match="imported RNA only"):
        ds = cytome.from_cellranger(tmp_path / "outs", tmp_path / "ro.cytome", modalities="both")
    assert ds.n_genes == 2 and ds.n_peaks == 0
    ds.close()
