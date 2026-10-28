"""Converters build their store under a temporary name, and from_cellranger
streams its h5 matrix.

Audit 2026-10-27, part 4 improvement 1 (L1 improvement 6: a conversion killed
inside add_embedding left a store that opened and passed validate()) and L2
improvement 5 (from_cellranger read its .h5 matrix whole and copied its
transpose; from_10x_h5's batched writer is reused).
"""
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome

anndata = pytest.importorskip("anndata")
h5py = pytest.importorskip("h5py")


def _adata(n=6):
    a = anndata.AnnData(X=sp.csr_matrix(np.random.default_rng(0).poisson(1, (n, 4))
                                        .astype(np.float32)))
    a.obsm["X_umap"] = np.ones((n, 2))
    return a


def _leftovers(path):
    return [p for p in (str(path), str(path) + ".partial") if os.path.exists(p)]


def test_a_failed_conversion_leaves_no_store(tmp_path, monkeypatch):
    out = tmp_path / "a.cytome"

    def boom(self, *a, **k):
        raise RuntimeError("killed")
    monkeypatch.setattr(cytome.CytomeDataset, "add_embedding", boom)
    with pytest.raises(RuntimeError, match="killed"):
        cytome.from_anndata(_adata(), output=str(out))
    assert _leftovers(out) == []


def test_a_failed_conversion_keeps_the_store_it_would_replace(tmp_path, monkeypatch):
    out = tmp_path / "a.cytome"
    cytome.from_anndata(_adata(4), output=str(out)).close()

    def boom(self, *a, **k):
        raise RuntimeError("killed")
    monkeypatch.setattr(cytome.CytomeDataset, "add_embedding", boom)
    with pytest.raises(RuntimeError):
        cytome.from_anndata(_adata(6), output=str(out), force=True)
    ds = cytome.open(str(out))
    assert ds.n_cells == 4                                  # the old store, intact
    ds.close()
    assert not os.path.exists(str(out) + ".partial")


def _h5(path, M, ids, names, types, barcodes):
    M = sp.csc_matrix(M)                                    # features x cells
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = M.data, M.indices, M.indptr
        g["shape"] = np.array(M.shape)
        g["barcodes"] = np.array([b.encode() for b in barcodes])
        feat = g.create_group("features")
        feat["id"] = np.array([x.encode() for x in ids])
        feat["name"] = np.array([x.encode() for x in names])
        feat["feature_type"] = np.array([x.encode() for x in types])


def test_from_cellranger_streams_its_h5_and_drops_scaffold_peaks(tmp_path, monkeypatch):
    folder = tmp_path / "outs"
    folder.mkdir()
    rng = np.random.default_rng(1)
    M = rng.poisson(1, (5, 7)).astype(np.int32)              # 5 features x 7 cells
    ids = ["ENSG1", "ENSG2", "chr1:10-100", "chr2:5-50", "GL456233.1:1-90"]
    types = ["Gene Expression"] * 2 + ["Peaks"] * 3
    _h5(folder / "filtered_feature_bc_matrix.h5", M, ids, ["G1", "G2"] + ids[2:], types,
        [f"AAAC{i}" for i in range(7)])
    import cytome.io.convert_cellranger as cr
    calls = []
    real = cr._iter_10x_h5_cell_chunks

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(cr, "_iter_10x_h5_cell_chunks", counting)
    with pytest.warns(UserWarning, match="non-standard"):
        ds = cytome.from_cellranger(str(folder), str(tmp_path / "c.cytome"), sample_name="L1",
                                    import_fragments=False)
    assert calls                                               # the batched reader was used
    assert ds.RNA.counts.to_memory().toarray().tolist() == M[:2].T.tolist()
    assert ds.ATAC.counts.to_memory().toarray().tolist() == M[2:4].T.tolist()
    assert ds.peaks["peak_id"].tolist() == ["chr1:10-100", "chr2:5-50"]
    assert ds.cells["sample_id"].tolist() == ["L1"] * 7
    assert ds.validate(integrity=False).passed
    ds.close()


def test_from_10x_h5_and_from_mudata_are_atomic(tmp_path, monkeypatch):
    import cytome.io.convert_cellranger as cr
    p = tmp_path / "f.h5"
    _h5(p, np.ones((2, 3), dtype=np.int32), ["E1", "E2"], ["G1", "G2"],
        ["Gene Expression"] * 2, ["a", "b", "c"])

    def boom(*a, **k):
        raise RuntimeError("killed")
    monkeypatch.setattr(cr, "_iter_10x_h5_cell_chunks", boom)
    with pytest.raises(RuntimeError):
        cytome.from_10x_h5(str(p), str(tmp_path / "o.cytome"))
    assert _leftovers(tmp_path / "o.cytome") == []
    mudata = pytest.importorskip("mudata")
    md = mudata.MuData({"rna": _adata(3)})
    monkeypatch.setattr(cytome.CytomeDataset, "add_embedding",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("killed")))
    with pytest.raises(RuntimeError):
        cytome.from_mudata(md, output=str(tmp_path / "m.cytome"))
    assert _leftovers(tmp_path / "m.cytome") == []
