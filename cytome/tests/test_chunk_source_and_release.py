"""ChunkSource hands out stored chunks as blobs; accessors survive a reopen;
``released`` frees every handle of this process on a file."""
from __future__ import annotations

import gc
import sqlite3

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cytome
from cytome.io.chunked_io import read_sparse_chunked
from cytome.io.compression import decompress_blob


def _make(tmp_path, n=900, g=300, name="s.cytome"):
    rng = np.random.default_rng(0)
    X = sp.random(n, g, density=0.05, format="csr", random_state=1, dtype=np.float32)
    X.data = rng.poisson(3, X.nnz).astype(np.float32) + 1
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = [f"g{j}" for j in range(g)]
    p = tmp_path / name
    cytome.from_anndata(A, modality="RNA", output=str(p)).close()
    return str(p), X


def _decode(blobs, dtype):
    d, i, p, comp = blobs
    f = lambda b, t: np.frombuffer(decompress_blob(b, comp), dtype=t)
    return f(d, dtype), f(i, np.int32), f(p, np.int32)


def test_blobs_decode_to_the_stored_matrix(tmp_path):
    path, X = _make(tmp_path)
    ds = cytome.open(path)
    ds.RNA.counts.build_feature_index()
    for layout in ("csr", "csc"):
        with ds.chunk_source("RNA_counts", n_readers=3) as src:
            n_rows, n_cols, dtype, has_csr, has_csc = src.meta()
            assert (n_rows, n_cols) == X.shape and has_csr and has_csc
            idx, start, end, nnz = src.chunks(layout)
            assert np.all(np.diff(idx) > 0) and nnz.sum() == X.nnz
            # out of order, in two batches, the second prefetched
            order = idx[::-1]
            half = len(order) // 2
            got = src.fetch(layout, order[:half], order[half:])
            got += src.fetch(layout, order[half:])
            parts = {}
            for k, b in zip(order, got):
                d, ind, ptr = _decode(b, np.dtype(dtype))
                parts[int(k)] = (d, ind, ptr)
            blocks = []
            for k, s, e in zip(idx, start, end):
                d, ind, ptr = parts[int(k)]
                shape = (e - s, n_cols) if layout == "csr" else (e - s, n_rows)
                blocks.append(sp.csr_matrix((d, ind, ptr), shape=shape))
            M = sp.vstack(blocks).tocsr()
            M = M if layout == "csr" else M.T.tocsr()
            assert (M != X).nnz == 0
    ds.close()


def test_pending_writes_are_visible_to_the_readers(tmp_path):
    path, X = _make(tmp_path)
    ds = cytome.open(path)
    Y = (X * 2).tocsr()
    ds.add_matrix("RNA_twice", Y)
    with ds.chunk_source("RNA_twice") as src:
        idx, *_ = src.chunks("csr")
        got = src.fetch("csr", idx[:1])
        d, _, _ = _decode(got[0], np.dtype(src.meta()[2]))
        assert d.sum() > 0
    ds.close()


def test_cache_serves_later_passes_and_respects_its_budget(tmp_path):
    path, X = _make(tmp_path)
    ds = cytome.open(path)
    with ds.chunk_source("RNA_counts", cache=True, cache_budget_bytes=1 << 30) as src:
        idx, *_ = src.chunks("csr")
        a = src.fetch("csr", idx)
        read = src.bytes_read
        b = src.fetch("csr", idx)
        assert src.bytes_read == read and a == b and src.cached_bytes == read
    with ds.chunk_source("RNA_counts", cache=True, cache_budget_bytes=10) as src:
        idx, *_ = src.chunks("csr")
        with pytest.warns(RuntimeWarning, match="budget"):
            src.fetch("csr", idx)
        assert src.cache is False and src.cached_bytes == 0
        read = src.bytes_read
        src.fetch("csr", idx)
        assert src.bytes_read == 2 * read
    ds.close()


def test_kept_accessors_follow_a_reopen(tmp_path):
    path, X = _make(tmp_path)
    ds = cytome.open(path)
    cells, layer, rna, emb, meta = ds.cells, ds.RNA.counts, ds.RNA, ds.embeddings, ds.metadata
    ds.reopen()
    assert len(cells.to_pandas()) == X.shape[0]
    assert layer.rows([0, 1]).shape == (2, X.shape[1])
    assert rna.counts.shape == X.shape
    assert list(emb.keys()) == list(ds.embeddings.keys())
    meta["k"] = 1
    ds.flush()
    assert ds.metadata["k"] == 1
    ds.close()


def test_a_running_iterator_does_not_survive(tmp_path):
    path, _ = _make(tmp_path)
    ds = cytome.open(path)
    it = ds.iter_chunks(modality="RNA", layer="counts", batch_size=100)
    next(it)
    ds.reopen()
    with pytest.raises(sqlite3.ProgrammingError):
        next(it)
    ds.close()


def test_accessors_of_a_closed_dataset_are_not_moved_to_another(tmp_path):
    p1, _ = _make(tmp_path, name="a.cytome")
    p2, X2 = _make(tmp_path, n=50, name="b.cytome")
    a = cytome.open(p1)
    kept = a.cells
    a.close()
    del a
    gc.collect()
    b = cytome.open(p2)
    b.reopen()
    with pytest.raises(sqlite3.ProgrammingError):
        kept.to_pandas()
    assert len(b.cells.to_pandas()) == X2.shape[0]
    b.close()


def test_released_closes_every_handle_on_the_file_and_reopens_them(tmp_path):
    path, X = _make(tmp_path)
    other_path, _ = _make(tmp_path, n=40, name="o.cytome")
    a, b, other = cytome.open(path), cytome.open(path), cytome.open(other_path)
    closed = cytome.open(path)
    closed.close()
    cells = a.cells
    a.metadata["pending"] = 7          # a queued write must reach the file first
    with cytome.released(path) as held:
        assert set(map(id, held)) == {id(a), id(b)}
        assert a.is_closed and b.is_closed and not other.is_closed
        # the file is ours: an exclusive writer can run
        c = sqlite3.connect(path)
        c.execute("BEGIN EXCLUSIVE")
        assert c.execute("SELECT value FROM _metadata WHERE key='pending'").fetchone() is not None
        c.execute("COMMIT")
        c.close()
    assert not a.is_closed and not b.is_closed and closed.is_closed
    assert len(cells.to_pandas()) == X.shape[0]
    for d in (a, b, other):
        d.close()


def test_released_accepts_a_dataset(tmp_path):
    path, _ = _make(tmp_path)
    a = cytome.open(path)
    with cytome.released(a) as held:
        assert held == [a] and a.is_closed
    assert not a.is_closed
    a.close()
