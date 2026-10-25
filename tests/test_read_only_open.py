"""Opening a store that cannot be written, and opening one without writing.

``cytome.open`` used to write on every open: it added the tables newer
versions define. A read-only store from an older version then failed with
SQLite's "attempt to write a readonly database", a store in a read-only
folder failed even at the same version, and a writable one was changed by
being opened, so a downloaded store stopped matching its checksum.

Now ``mode="r"`` never writes, and ``mode="r+"`` falls back to it with a
warning when the store cannot be written. One test per case of the review
(docs: 2026-09-28, section 5.2).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import warnings

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
import pandas as pd  # noqa: E402

import cytome  # noqa: E402
from cytome.io.sqlite_engine import ReadOnlyConnection  # noqa: E402

pytestmark = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores file permissions",
)


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _old_store(folder, name="s.cytome", checkpoint=True):
    """A small store that looks like one written by cytome 0.3.5: no
    ``graph_meta``/``graph_chunks`` and no ``matrix_meta.is_integer``, with a
    graph and an embedding to read."""
    path = folder / name
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(40, 12)).astype(np.float32)
    a = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"group": list("ab") * 20}, index=[f"c{i}" for i in range(40)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(12)]),
    )
    a.obsm["X_umap"] = rng.normal(size=(40, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cytome.from_anndata(a, output=str(path), force=True).close()
    con = sqlite3.connect(str(path))
    con.execute("DROP TABLE IF EXISTS graph_meta")
    con.execute("DROP TABLE IF EXISTS graph_chunks")
    cols = [r[1] for r in con.execute("PRAGMA table_info(matrix_meta)")]
    keep = [c for c in cols if c != "is_integer"]
    con.execute(f"CREATE TABLE mm_new AS SELECT {','.join(keep)} FROM matrix_meta")
    con.execute("DROP TABLE matrix_meta")
    con.execute("ALTER TABLE mm_new RENAME TO matrix_meta")
    # an old-style graph, in the per-edge table every version has
    con.execute(
        "INSERT INTO graph_edges(graph_name, axis, entity_table, row_idx, col_idx, value) "
        "VALUES ('knn_old', 'obs', 'cells', 0, 1, 1.0), ('knn_old', 'obs', 'cells', 1, 0, 1.0)")
    con.commit()
    if checkpoint:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    if checkpoint:
        for sfx in ("-wal", "-shm"):
            p = folder / f"{name}{sfx}"
            if p.exists():
                p.unlink()
    return path


def _read_everything(ds):
    """Every reader the review named: cells, matrix, chunks, chunk readers,
    embeddings, graphs, the columns an older store lacks."""
    out = {}
    out["cells"] = ds.cells.n
    out["group"] = list(ds.cells["group"][:2])
    names = [r[0] for r in ds._conn.execute("SELECT matrix_name FROM matrix_meta")]
    out["matrices"] = names
    layer = names[0].split("_", 1)[1]
    out["rows_read"] = sum(chunk.shape[0] for chunk, _ in ds.iter_chunks(
        modality="RNA", layer=layer, batch_size=16))
    assert out["rows_read"] == 40
    out["is_integer"] = ds._conn.execute(
        "SELECT is_integer FROM matrix_meta WHERE matrix_name = ?", (names[0],)).fetchone()[0]
    src = ds.chunk_source(names[0])
    ids = src.chunks("csr")[0]
    out["chunk_blobs"] = len(src.fetch("csr", ids))
    src.close()
    out["embeddings"] = sorted(ds.embeddings.keys())
    out["umap"] = ds.embeddings[out["embeddings"][0]].shape
    out["graphs"] = ds.graphs.keys()
    out["graph_nnz"] = ds.graphs["knn_old"].to_sparse().nnz
    return out


def _lock(*paths, mode=0o444):
    for p in paths:
        if os.path.exists(p):
            os.chmod(p, mode)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    yield d
    os.chmod(d, 0o755)  # so pytest can clean up
    for p in d.iterdir():
        os.chmod(p, 0o644)


def _open(path, mode="r+"):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = cytome.open(str(path), mode=mode)
    return ds, [x for x in w if issubclass(x.category, cytome.ReadOnlyWarning)]


# ---------------------------------------------------------------- the six cases

def test_file_read_only_folder_writable(folder):
    path = _old_store(folder)
    before = _md5(path)
    _lock(path)
    ds, warned = _open(path)
    try:
        assert ds.read_only and len(warned) == 1
        assert "mode=ro" in ds._conn.uri
        got = _read_everything(ds)
        assert got["cells"] == 40 and got["graphs"] == ["knn_old"] and got["graph_nnz"] == 2
        assert got["is_integer"] is None
    finally:
        ds.close()
    assert _md5(path) == before


def test_file_and_folder_read_only_no_sidecars(folder):
    path = _old_store(folder)
    before, listing = _md5(path), sorted(os.listdir(folder))
    _lock(path)
    _lock(folder, mode=0o555)
    ds, warned = _open(path)
    try:
        assert ds.read_only and warned
        assert "immutable=1" in ds._conn.uri
        _read_everything(ds)
    finally:
        ds.close()
    assert _md5(path) == before
    assert sorted(os.listdir(folder)) == listing, "nothing may be created in the folder"


def test_all_read_only_with_empty_sidecars(folder):
    path = _old_store(folder)
    before = _md5(path)
    # the sidecars a reader left behind: SQLite can use them read-only
    sqlite3.connect(f"file:{path}?mode=ro", uri=True).execute("SELECT 1 FROM cells").fetchone()
    assert (folder / "s.cytome-shm").exists()
    _lock(path, f"{path}-wal", f"{path}-shm")
    _lock(folder, mode=0o555)
    ds, _ = _open(path)
    try:
        assert ds.read_only and "mode=ro" in ds._conn.uri
        _read_everything(ds)
    finally:
        ds.close()
    assert _md5(path) == before


def test_committed_pages_in_wal_and_folder_read_only_is_refused(folder):
    path = _old_store(folder)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("UPDATE cells SET barcode = barcode || '-x'")
    con.commit()
    # keep the committed pages in the -wal: copy the three files aside while
    # the connection is open (closing would checkpoint), then put them back
    import shutil
    for sfx in ("", "-wal"):
        shutil.copy(f"{path}{sfx}", f"{path}{sfx}.keep")
    con.close()
    for sfx in ("", "-wal"):
        os.replace(f"{path}{sfx}.keep", f"{path}{sfx}")
    (folder / "s.cytome-shm").unlink(missing_ok=True)
    assert os.path.getsize(f"{path}-wal") > 0
    _lock(path, f"{path}-wal")
    _lock(folder, mode=0o555)
    with pytest.raises(cytome.ReadOnlyError, match="committed changes"):
        _open(path)


def test_rollback_journal_store_read_only(folder):
    path = _old_store(folder)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=DELETE")
    con.close()
    before = _md5(path)
    _lock(path)
    _lock(folder, mode=0o555)
    ds, _ = _open(path)
    try:
        assert ds.read_only and "mode=ro" in ds._conn.uri
        _read_everything(ds)
    finally:
        ds.close()
    assert _md5(path) == before


def test_writable_older_store_opened_r_is_not_changed(folder):
    path = _old_store(folder)
    before = _md5(path)
    ds, warned = _open(path, mode="r")
    try:
        assert ds.read_only and not warned, "asked for, so no warning"
        _read_everything(ds)
        with pytest.raises(cytome.ReadOnlyError):
            ds.cells["x"] = np.arange(40)
    finally:
        ds.close()
    assert _md5(path) == before, "opening with mode='r' must leave the checksum"
    # the first writable open is where the tables are added
    ds, warned = _open(path, mode="r+")
    try:
        assert not ds.read_only and not warned
        tables = {r[0] for r in ds._conn.execute(
            "SELECT name FROM main.sqlite_master WHERE type='table'")}
        assert {"graph_meta", "graph_chunks"} <= tables
    finally:
        ds.close()


# ---------------------------------------------------------------- the guards

def test_writes_are_refused_where_they_are_made(folder):
    path = _old_store(folder)
    ds, _ = _open(path, mode="r")
    try:
        # through the API: at the assignment, not at a later flush
        with pytest.raises(cytome.ReadOnlyError, match="opened read-only"):
            ds.cells["new"] = np.arange(40)
        assert not ds._pending_writes
        # raw SQL into the file
        with pytest.raises(cytome.ReadOnlyError):
            ds._conn.execute("UPDATE cells SET barcode = 'x'")
        # raw SQL into a shim: query_only, or the row would vanish at close
        assert "graph_meta" in ds._conn.shims
        with pytest.raises(cytome.ReadOnlyError):
            ds._conn.execute(
                "INSERT INTO graph_meta (graph_name, axis, entity_table, n_rows, n_cols, "
                "n_nonzero, dtype, n_chunks) VALUES ('g', 'obs', 'cells', 1, 1, 0, 'f', 0)")
        with pytest.raises(cytome.ReadOnlyError):
            ds.require_writable("clustering")
    finally:
        ds.close()


def test_read_only_error_is_sqlite_error_with_its_words(folder):
    """Code that already catches SQLite's read-only error keeps working."""
    path = _old_store(folder)
    ds, _ = _open(path, mode="r")
    try:
        with pytest.raises(sqlite3.OperationalError) as info:
            ds._conn.execute("DELETE FROM cells")
        assert isinstance(info.value, cytome.ReadOnlyError)
        assert "readonly" in str(info.value)
    finally:
        ds.close()


def test_shims_only_on_read_only_connections(folder):
    """On a writable connection a temp table would shadow the real one."""
    path = _old_store(folder)
    ds, _ = _open(path, mode="r+")
    try:
        assert not isinstance(ds._conn, ReadOnlyConnection)
        temp = ds._conn.execute("SELECT name FROM temp.sqlite_master").fetchall()
        assert temp == []
    finally:
        ds.close()


def test_reopen_keeps_read_only(folder):
    path = _old_store(folder)
    ds, _ = _open(path, mode="r")
    try:
        cells = ds.cells
        ds.reopen()
        assert ds.read_only
        assert cells.n == 40, "a kept accessor follows the reopen"
    finally:
        ds.close()


def test_immutable_store_changed_while_open_warns(folder):
    path = _old_store(folder)
    _lock(path)
    _lock(folder, mode=0o555)
    ds, _ = _open(path)
    assert "immutable=1" in ds._conn.uri
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.warns(cytome.ReadOnlyWarning, match="changed while it was open"):
        ds.close()


def test_mode_is_checked():
    with pytest.raises(ValueError, match="mode"):
        cytome.open("whatever.cytome", mode="w")


def test_filter_cells_on_read_only_store_fails_before_work(folder):
    path = _old_store(folder)
    ds, _ = _open(path, mode="r")
    try:
        with pytest.raises(cytome.ReadOnlyError, match="filter_cells"):
            ds.filter_cells(np.ones(40, dtype=bool))
    finally:
        ds.close()


# ------------------------------------------------- a shim view and ORDER BY rowid (2026-10-21)
def _store_missing_cell_column(folder, name="v.cytome"):
    """An older store whose cells table lacks a column of today's schema (n_fragments), so a read-only open
    shims it with a TEMP view."""
    path = _old_store(folder, name)
    con = sqlite3.connect(str(path))
    con.execute("ALTER TABLE cells DROP COLUMN n_fragments")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    return path


def test_entity_reads_on_a_shim_view_keep_rowid_order(folder):
    path = _store_missing_cell_column(folder)
    ds, _ = _open(path, mode="r")
    try:
        assert list(ds.cells["group"][:4]) == ["a", "b", "a", "b"]       # EntityTable: ORDER BY ROWID
        assert "cells" in ds._conn.shim_views
        assert list(ds.cells["n_fragments"][:2]) == [None, None]          # the shimmed column reads as NULL
        raw = [r[0] for r in ds._conn.execute('SELECT "group" FROM cells ORDER BY rowid')]
        key = [r[0] for r in ds._conn.execute('SELECT "group" FROM cells ORDER BY cell_idx')]
        assert raw == key and len(raw) == 40                                 # PIASO's raw readers
        mask = ds.cells.query_mask('"group" = \'a\'')
        assert mask.sum() == 20 and bool(mask[0]) and not bool(mask[1])
    finally:
        ds.close()


def test_the_exposed_rowid_is_hidden_from_listings(folder):
    path = _store_missing_cell_column(folder)
    ds, _ = _open(path, mode="r")
    try:
        assert "rowid" not in ds.cells.columns and "n_fragments" in ds.cells.columns
        assert "rowid" not in ds.cells.to_pandas().columns
        assert "rowid" not in ds.cells.query("1 = 1").columns
    finally:
        ds.close()


def test_order_by_rowid_on_a_shim_view_needs_no_sort(folder):
    path = _store_missing_cell_column(folder)
    ds, _ = _open(path, mode="r")
    try:
        plan = " ".join(str(r) for r in ds._conn.execute(
            'EXPLAIN QUERY PLAN SELECT "group" FROM cells ORDER BY rowid'))
        assert "TEMP B-TREE" not in plan.upper()
    finally:
        ds.close()


def test_a_current_store_gets_no_view_shim(folder):
    path = folder / "new.cytome"
    a = anndata.AnnData(X=np.ones((6, 3), dtype=np.float32),
                        obs=pd.DataFrame({"g": list("abcabc")}, index=[f"c{i}" for i in range(6)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cytome.from_anndata(a, output=str(path), force=True).close()
    ds, _ = _open(path, mode="r")
    try:
        assert ds._conn.shim_views == set()
        assert "rowid" not in ds.cells.columns
        assert list(ds.cells["g"]) == list("abcabc")
    finally:
        ds.close()


def test_subset_of_a_shimmed_read_only_store_copies_only_real_columns(folder):
    path = _store_missing_cell_column(folder)
    ds, _ = _open(path, mode="r")
    out = folder / "sub.cytome"
    try:
        from cytome.io.subset import subset
        mask = np.zeros(40, dtype=bool)
        mask[:10] = True
        sub = subset(ds, mask, str(out))
        sub.close()
    finally:
        ds.close()
    con = sqlite3.connect(str(out))
    cols = [r[1] for r in con.execute("PRAGMA table_info(cells)")]
    groups = [r[0] for r in con.execute('SELECT "group" FROM cells ORDER BY rowid')]
    con.close()
    assert "rowid" not in cols
    assert groups == list("ab") * 5
