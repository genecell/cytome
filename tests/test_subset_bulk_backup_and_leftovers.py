"""2026-10-15: subset written in bulk mode, the backup by hard link, and leftovers of a crashed run.

- ``subset`` builds ``<output>.partial`` without a journal or syncs, then renames it: the result
  equals the source rows, a failure leaves no partial file, and a free-space check runs first.
- The column-major subset (threads, raw-array row selection) gives the same bytes as scipy's
  fancy index followed by ``sort_indices``, including for a chunk whose indices are unsorted.
- ``create_database`` and ``filter_cells`` remove a crashed run's sidecars even when its main
  file is gone (SQLite would otherwise replay the old log into the new store).
- ``filter_cells(keep_original=...)`` keeps the unfiltered store by a hard link (a copy where
  links fail) and makes no backup when every cell passes.
"""
import errno
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
import cytome.io.subset as subset_mod
from cytome.core.dataset import CytomeDataset
from cytome.core.measurement import MeasurementLayer
from cytome.io.compression import compress_blob, decompress_blob
from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks
from cytome.io.sqlite_engine import create_database


def _store(path, n=60, g=30, t=500, seed=0):
    """Cells, an RNA matrix (row chunks), a column-major tile matrix and fragments on two
    chromosomes in several chunks. Returns the path and the source arrays."""
    rng = np.random.default_rng(seed)
    ds = CytomeDataset(path, mode="w")
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"BC{i:03d}" for i in range(n)],
                                         "group": [str(i % 3) for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"g{j}" for j in range(g)]}))
    ds.flush()
    X = sp.csr_matrix(rng.poisson(0.8, size=(n, g)).astype(np.float32))
    ds.add_matrix("RNA_counts", X)
    T = sp.random(n, t, density=0.08, format="csr", dtype=np.float32, random_state=rng)
    T.data = np.ceil(T.data * 4).astype(np.float32)
    ds.add_matrix("tiles_counts", T)
    ds.flush()
    ml = MeasurementLayer(ds._conn, "tiles_counts")
    ml.build_feature_index(chunk_size=64)                  # 8 column chunks
    ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name='tiles_counts'")
    ds._conn.execute("UPDATE matrix_meta SET n_chunks = 0 WHERE matrix_name='tiles_counts'")
    frags = {}
    for chrom, m in (("chr1", 900), ("chr2", 400)):
        s = np.sort(rng.integers(0, 10**6, size=m)).astype(np.int32)
        e = (s + rng.integers(50, 600, size=m)).astype(np.int32)
        c = rng.integers(0, n, size=m).astype(np.int32)
        _write_block_chunks(ds._conn, chrom, s, e, c, chunk_size=128, compression="lz4",
                            compression_level=1, chunk_idx_start=0, encoding=1)
        _update_fragment_meta(ds._conn, chrom, m)
        frags[chrom] = (s, e, c)
    ds._conn.commit()
    ds.close()
    return str(path), X, T, frags


def _fragments(ds, chrom):
    parts = list(ds.ATAC.fragments.iter_chromosome_chunks(chrom))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def _keep(n):
    keep = np.ones(n, dtype=bool)
    keep[[3, 10, 11, 40, 59]] = False
    return keep


def test_iter_chromosome_chunks_reads_every_chunk_in_order(tmp_path):
    p, _, _, frags = _store(tmp_path / "s.cytome")
    ds = cytome.open(p, mode="r")
    s, e, c = _fragments(ds, "chr1")
    assert len(list(ds.ATAC.fragments.iter_chromosome_chunks("chr1"))) == 8   # 900 / 128
    assert np.array_equal(s, frags["chr1"][0]) and np.array_equal(c, frags["chr1"][2])
    ds.close()


def test_subset_matches_the_source_rows_and_fragments(tmp_path):
    p, X, T, frags = _store(tmp_path / "s.cytome")
    keep = _keep(X.shape[0])
    ds = cytome.open(p)
    out = ds.subset(keep, output=str(tmp_path / "sub.cytome"))
    assert out.n_cells == int(keep.sum())
    assert (out.RNA.layer("counts").to_memory().tocsr() != X[keep]).nnz == 0
    assert (MeasurementLayer(out._conn, "tiles_counts").to_memory().tocsr() != T[keep]).nnz == 0
    remap = np.full(X.shape[0], -1); remap[keep] = np.arange(int(keep.sum()))
    for chrom, (s0, e0, c0) in frags.items():
        k = keep[c0]
        s, e, c = _fragments(out, chrom)
        assert np.array_equal(s, s0[k]) and np.array_equal(e, e0[k])
        assert np.array_equal(c, remap[c0[k]])
    assert out._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    out.close(); ds.close()
    assert not list(tmp_path.glob("sub.cytome.partial*"))


def test_csc_subset_bytes_equal_the_scipy_path_even_for_an_unsorted_chunk(tmp_path):
    p, X, T, _ = _store(tmp_path / "s.cytome")
    ds = cytome.open(p)
    # plant a chunk whose row indices are reversed inside one column
    cid, d_b, i_b, p_b, dt, comp = ds._conn.execute(
        "SELECT id, data_blob, indices_blob, indptr_blob, dtype, compression FROM matrix_csc_chunks "
        "WHERE matrix_name='tiles_counts' ORDER BY chunk_idx LIMIT 1").fetchone()
    d = np.frombuffer(decompress_blob(d_b, comp), np.dtype(dt)).copy()
    i = np.frombuffer(decompress_blob(i_b, comp), np.int32).copy()
    ptr = np.frombuffer(decompress_blob(p_b, comp), np.int32)
    j = int(np.argmax(np.diff(ptr)))                     # the fullest column
    a, b = ptr[j], ptr[j + 1]
    d[a:b], i[a:b] = d[a:b][::-1].copy(), i[a:b][::-1].copy()
    ds._conn.execute("UPDATE matrix_csc_chunks SET data_blob=?, indices_blob=? WHERE id=?",
                     (compress_blob(d.tobytes(), comp), compress_blob(i.tobytes(), comp), cid))
    ds._conn.commit()
    keep = _keep(X.shape[0])
    expected = []
    for col_start, col_end, chunk in MeasurementLayer(ds._conn, "tiles_counts").iter_columns():
        sub = sp.csc_matrix(chunk[np.flatnonzero(keep)]); sub.sort_indices()
        expected.append((compress_blob(np.ascontiguousarray(sub.data, np.float32).tobytes(), "lz4"),
                         compress_blob(sub.indices.astype(np.int32).tobytes(), "lz4"),
                         compress_blob(sub.indptr.astype(np.int32).tobytes(), "lz4")))
    out = ds.subset(keep, output=str(tmp_path / "sub.cytome"))
    got = out._conn.execute("SELECT data_blob, indices_blob, indptr_blob FROM matrix_csc_chunks "
                            "WHERE matrix_name='tiles_counts' ORDER BY chunk_idx").fetchall()
    assert [tuple(r) for r in got] == expected
    out.close(); ds.close()


def test_a_failed_subset_leaves_no_partial_file_and_the_source_intact(tmp_path, monkeypatch):
    p, X, _, _ = _store(tmp_path / "s.cytome")

    def boom(*a, **k):
        raise RuntimeError("planted failure")
    monkeypatch.setattr(subset_mod, "_subset_fragments", boom)
    ds = cytome.open(p)
    with pytest.raises(RuntimeError, match="planted"):
        ds.subset(_keep(X.shape[0]), output=str(tmp_path / "sub.cytome"))
    ds.close()
    assert not list(tmp_path.glob("sub.cytome*"))
    ds = cytome.open(p, mode="r")
    assert ds.n_cells == X.shape[0]
    ds.close()


def test_subset_checks_free_space_before_writing(tmp_path, monkeypatch):
    p, X, _, _ = _store(tmp_path / "s.cytome")

    class _Usage:
        total, used, free = 10**12, 10**12, 1
    monkeypatch.setattr(subset_mod.shutil, "disk_usage", lambda _: _Usage)
    ds = cytome.open(p)
    with pytest.raises(OSError) as err:
        ds.subset(_keep(X.shape[0]), output=str(tmp_path / "sub.cytome"))
    assert err.value.errno == errno.ENOSPC and "GB free" in str(err.value)
    assert not list(tmp_path.glob("sub.cytome*"))
    out = ds.subset(_keep(X.shape[0]), output=str(tmp_path / "sub.cytome"), check_free_space=False)
    assert out.n_cells == int(_keep(X.shape[0]).sum())
    out.close(); ds.close()


def test_create_database_removes_orphan_sidecars(tmp_path):
    p = tmp_path / "new.cytome"
    for force in (False, True):
        for e in ("-wal", "-shm"):
            Path(str(p) + e).write_bytes(b"left by a crashed writer" * 1000)
        conn = create_database(p, force=force)
        conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
        conn.commit()
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        conn.close()
        wal = Path(str(p) + "-wal")
        assert not wal.exists() or not wal.read_bytes().startswith(b"left by")
        if not force:
            p.unlink()                                 # second round: main file present again


def test_filter_cells_ignores_a_crashed_runs_leftovers(tmp_path):
    p, X, _, _ = _store(tmp_path / "s.cytome")
    for name in (".filter_tmp-wal", ".filter_tmp-shm", ".filter_tmp.partial"):
        Path(p + name).write_bytes(b"left by a crashed run" * 1000)
    ds = cytome.open(p)
    keep = _keep(X.shape[0])
    assert ds.filter_cells(keep) == int(keep.sum())
    ds.close()
    assert not list(tmp_path.glob("s.cytome.filter_tmp*"))
    ds = cytome.open(p, mode="r")
    assert ds.n_cells == int(keep.sum())
    assert ds._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    ds.close()


def test_filter_cells_keeps_the_original_by_hard_link(tmp_path):
    p, X, _, frags = _store(tmp_path / "s.cytome")
    raw = str(tmp_path / "s_raw.cytome")
    Path(raw).write_bytes(b"an older backup")                 # replaced, not appended to
    keep = _keep(X.shape[0])
    ds = cytome.open(p)
    assert ds.filter_cells(keep, keep_original=raw) == int(keep.sum())
    assert ds.n_cells == int(keep.sum())
    ds.close()
    assert os.stat(raw).st_ino != os.stat(p).st_ino and os.stat(raw).st_nlink == 1
    r = cytome.open(raw, mode="r")
    assert r.n_cells == X.shape[0]
    assert np.array_equal(_fragments(r, "chr2")[2], frags["chr2"][2])
    r.close()


def test_filter_cells_keep_original_copies_when_links_fail(tmp_path, monkeypatch):
    p, X, _, _ = _store(tmp_path / "s.cytome")
    raw = str(tmp_path / "s_raw.cytome")

    def no_link(a, b):
        raise OSError(errno.EXDEV, "cross-device link")
    monkeypatch.setattr(os, "link", no_link)
    ds = cytome.open(p)
    ds.filter_cells(_keep(X.shape[0]), keep_original=raw)
    ds.close()
    r = cytome.open(raw, mode="r")
    assert r.n_cells == X.shape[0]
    r.close()


def test_filter_cells_keep_original_makes_no_backup_when_every_cell_passes(tmp_path):
    p, X, _, _ = _store(tmp_path / "s.cytome")
    raw = tmp_path / "s_raw.cytome"
    ds = cytome.open(p)
    assert ds.filter_cells(np.ones(X.shape[0], dtype=bool), keep_original=str(raw)) == X.shape[0]
    ds.close()
    assert not raw.exists()
