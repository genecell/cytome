"""2026-10-24: subset checks its own counts only, and the page-size request that never took effect is gone.

- ``subset`` (and so ``filter_cells``) ran SQLite's ``integrity_check`` on every new store, a read of
  every page in tree order; it now runs cytome's own checks only, and SQLite's when asked
  (``check_integrity=True`` on ``subset``, ``split``, ``downsample`` and ``filter_cells``).
  ``validate()`` still runs the full check by default, and ``integrity=False`` skips it.
- ``_configure_connection`` asked for 64 KB pages after switching a new file to WAL, which writes the
  header and fixes the default 4,096 bytes, so no store cytome created had 64 KB pages. The request is
  removed: new stores keep SQLite's default, as they always had, and a store with other pages (the
  Rust importer writes 64 KB) opens, reads and writes as before.
"""
import sqlite3

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import cytome
import cytome.utils.validation as validation_mod
from cytome.core.dataset import CytomeDataset
from cytome.io.sqlite_engine import create_database
from cytome.io.subset import subset


def _page_size(path):
    with open(path, "rb") as f:
        header = f.read(100)
    size = int.from_bytes(header[16:18], "big")
    return 65536 if size == 1 else size


def _store(path, n=40, g=25, seed=0):
    rng = np.random.default_rng(seed)
    ds = CytomeDataset(path, mode="w")
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"BC{i:03d}" for i in range(n)],
                                         "group": [str(i % 3) for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"g{j}" for j in range(g)]}))
    ds.flush()
    X = sp.csr_matrix(rng.poisson(0.8, size=(n, g)).astype(np.float32))
    ds.add_matrix("RNA_counts", X)
    ds.add_embedding("X_pca", rng.normal(size=(n, 5)).astype(np.float32))
    ds.flush()
    ds.close()
    return str(path), X


def _matrix(ds):
    """RNA_counts read back through iter_chunks, which yields (chunk, row indices)."""
    return sp.vstack([c[0] for c in ds.iter_chunks(modality="RNA", layer="counts", batch_size=16)]).tocsr()


def _with_pages(src, dst, size):
    """A copy of ``src`` rebuilt with ``size``-byte pages, in WAL mode like every store."""
    con = sqlite3.connect(src)
    con.execute(f"PRAGMA page_size={size}")
    con.execute(f"VACUUM INTO '{dst}'")
    con.close()
    out = sqlite3.connect(dst)
    out.execute("PRAGMA journal_mode=WAL")
    out.close()
    return str(dst)


def test_new_stores_have_sqlite_default_pages(tmp_path):
    conn = create_database(tmp_path / "a.cytome")
    assert conn.execute("PRAGMA page_size").fetchone()[0] == 4096
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()
    ds = cytome.create(tmp_path / "b.cytome")
    ds.close()
    assert _page_size(tmp_path / "b.cytome") == 4096
    adata = ad.AnnData(X=sp.random(30, 20, density=0.2, format="csr", dtype=np.float32, random_state=0),
                       obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(20)]))
    ds = cytome.from_anndata(adata, output=tmp_path / "c.cytome", main_layer_name="RNA_data")
    ds.close()
    assert _page_size(tmp_path / "c.cytome") == 4096


def test_a_64k_store_keeps_its_pages_and_works(tmp_path):
    p, X = _store(tmp_path / "s.cytome")
    p64 = _with_pages(p, tmp_path / "s64k.cytome", 65536)
    assert _page_size(p64) == 65536
    ds = cytome.open(p64, mode="r+")
    ds.cells["score"] = np.arange(X.shape[0], dtype=np.float64)
    ds.flush()
    assert (_matrix(ds) != X).nnz == 0
    ds.close()
    assert _page_size(p64) == 65536
    ds = cytome.open(p64, mode="r")
    assert np.array_equal(np.asarray(ds.cells["score"], dtype=np.float64), np.arange(X.shape[0]))
    assert ds.validate().passed
    ds.close()


def test_validate_runs_sqlite_check_unless_told_not_to(tmp_path):
    p, _ = _store(tmp_path / "s.cytome")
    ds = cytome.open(p, mode="r")
    full = ds.validate()
    quick = ds.validate(integrity=False)
    ds.close()
    assert full.passed and quick.passed
    assert "sqlite_integrity" in full.checks_passed
    assert "sqlite_integrity" not in quick.checks_passed
    assert set(quick.checks_passed) == set(full.checks_passed) - {"sqlite_integrity"}


def test_subset_checks_its_counts_only(tmp_path, monkeypatch):
    p, X = _store(tmp_path / "s.cytome")
    p64 = _with_pages(p, tmp_path / "s64k.cytome", 65536)
    calls = []
    real = validation_mod.validate

    def spy(ds, *args, **kwargs):
        calls.append(kwargs.get("integrity", args[0] if args else True))
        return real(ds, *args, **kwargs)

    monkeypatch.setattr(validation_mod, "validate", spy)
    keep = np.ones(X.shape[0], dtype=bool)
    keep[[1, 7, 30]] = False
    src = cytome.open(p64, mode="r")
    out = subset(src, keep, tmp_path / "sub.cytome")
    got = _matrix(out)
    out.close()
    src.close()
    assert calls == [False]
    assert (got != X[keep]).nnz == 0
    assert _page_size(tmp_path / "sub.cytome") == 4096           # a new file: SQLite's default
    monkeypatch.setattr(validation_mod, "validate", real)
    check = cytome.open(tmp_path / "sub.cytome", mode="r")
    assert check.validate().passed                               # the full check passes on it
    check.close()


def test_filter_cells_of_a_64k_store(tmp_path):
    p, X = _store(tmp_path / "s.cytome")
    p64 = _with_pages(p, tmp_path / "s64k.cytome", 65536)
    keep = np.ones(X.shape[0], dtype=bool)
    keep[::4] = False
    ds = cytome.open(p64, mode="r+")
    assert ds.filter_cells(keep) == int(keep.sum())
    assert (_matrix(ds) != X[keep]).nnz == 0
    assert ds.validate().passed
    ds.close()
    assert _page_size(p64) == 4096


def test_check_integrity_on_request(tmp_path, monkeypatch):
    """subset, split, downsample and filter_cells run SQLite's check only when asked (check_integrity),
    once per store they write; cytome's own checks run either way."""
    from cytome.io.subset import downsample, split
    p, X = _store(tmp_path / "s.cytome")
    calls = []
    real = validation_mod.validate

    def spy(ds, *args, **kwargs):
        calls.append(kwargs.get("integrity", args[0] if args else True))
        return real(ds, *args, **kwargs)

    monkeypatch.setattr(validation_mod, "validate", spy)
    keep = np.ones(X.shape[0], dtype=bool)
    keep[[2, 5]] = False
    labels = np.array(["a", "b"] * (X.shape[0] // 2))
    for flag in (False, True):
        calls.clear()
        src = cytome.open(p, mode="r")
        subset(src, keep, tmp_path / f"sub_{flag}.cytome", check_integrity=flag).close()
        outs = split(src, labels, {"a": tmp_path / f"a_{flag}.cytome", "b": tmp_path / f"b_{flag}.cytome"},
                     ["RNA_counts"], check_integrity=flag)
        for o in outs.values():
            o.close()
        downsample(src, n_cells=20, output=tmp_path / f"down_{flag}.cytome", check_integrity=flag).close()
        src.close()
        copy = _with_pages(p, tmp_path / f"f_{flag}.cytome", 4096)
        ds = cytome.open(copy, mode="r+")
        assert ds.filter_cells(keep, check_integrity=flag) == int(keep.sum())
        ds.close()
        assert calls == [flag] * 5, (flag, calls)        # subset, two split stores, downsample, filter_cells
    monkeypatch.setattr(validation_mod, "validate", real)
    check = cytome.open(tmp_path / "sub_True.cytome", mode="r")
    report = check.validate()
    check.close()
    assert report.passed and "sqlite_integrity" in report.checks_passed
