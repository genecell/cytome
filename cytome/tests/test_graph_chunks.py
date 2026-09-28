"""Graphs are stored as compressed row chunks; files written before keep reading."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.core.graph import GraphStore


def _ds(tmp_path, n=500):
    ds = cytome.create(str(tmp_path / "g.cytome"))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": [f"c{i}" for i in range(n)]}))
    ds.flush()
    return ds


def _graph(n=500, k=15, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n - 3), k)                  # the last three rows empty
    cols = rng.integers(0, n, rows.size)
    A = sp.csr_matrix((rng.random(rows.size).astype(dtype), (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr(); A.sum_duplicates(); A.sort_indices()
    return A.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("chunk_nnz", [1 << 20, 97, 1])
def test_round_trip_is_exact(tmp_path, monkeypatch, dtype, chunk_nnz):
    monkeypatch.setattr(GraphStore, "_CHUNK_NNZ", chunk_nnz)
    ds = _ds(tmp_path); A = _graph(dtype=dtype)
    ds.add_graph("connectivities", A); ds.flush()
    B = ds.graphs["connectivities"].to_sparse()
    assert B.shape == A.shape == (500, 500)                 # empty last rows kept
    assert B.dtype == dtype
    assert np.array_equal(B.indptr, A.indptr) and np.array_equal(B.indices, A.indices)
    assert np.array_equal(B.data, A.data)
    n_chunks = ds._conn.execute("SELECT n_chunks FROM graph_meta").fetchone()[0]
    assert (n_chunks == 1) if chunk_nnz > A.nnz else (n_chunks > 1)
    blocks = list(ds.graphs["connectivities"].iter_chunks())
    assert sp.vstack([b for _, _, b in blocks]).tocsr().__ne__(A).nnz == 0
    assert ds.graphs["connectivities"].shape == A.shape
    ds.close()


def test_rewrite_replaces_and_keys_list_it(tmp_path):
    ds = _ds(tmp_path)
    ds.add_graph("connectivities", _graph(seed=1)); ds.flush()
    ds.add_graph("connectivities", _graph(seed=2)); ds.add_graph("distances", _graph(seed=3)); ds.flush()
    assert (ds.graphs["connectivities"].to_sparse() != _graph(seed=2)).nnz == 0
    assert ds._conn.execute("SELECT COUNT(*) FROM graph_meta").fetchone()[0] == 2
    assert {"connectivities", "distances"} <= set(ds.graphs.keys())
    ds.close()


def test_a_graph_of_an_older_file_is_read_and_replaced(tmp_path):
    ds = _ds(tmp_path); A = _graph(seed=4)
    GraphStore(ds._conn, "connectivities")._write_rows(A); ds._conn.commit()   # as before 0.3.6
    assert ds._conn.execute("SELECT COUNT(*) FROM graph_meta").fetchone()[0] == 0
    assert (ds.graphs["connectivities"].to_sparse() != A).nnz == 0
    assert "connectivities" in ds.graphs.keys()
    ds.add_graph("connectivities", _graph(seed=5)); ds.flush()
    assert ds._conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 0
    assert (ds.graphs["connectivities"].to_sparse() != _graph(seed=5)).nnz == 0
    GraphStore(ds._conn, "knn")._write_rows(A); ds._conn.commit()
    assert (ds.graphs["knn"].to_sparse() != A).nnz == 0
    ds.close()


def test_to_anndata_carries_the_graph(tmp_path):
    import anndata as ad_
    X = sp.random(500, 20, density=0.1, format="csr", dtype=np.float32, random_state=0)
    ds = cytome.from_anndata(ad_.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(500)]),
                                         var=pd.DataFrame(index=[f"g{j}" for j in range(20)])),
                             output=str(tmp_path / "a.cytome"))
    A = _graph(seed=6)
    ds.add_graph("knn", A); ds.flush()          # the name to_anndata carries without a map
    ad = ds.to_anndata(modality="RNA")
    assert "connectivities" in ad.obsp and (sp.csr_matrix(ad.obsp["connectivities"]) != A).nnz == 0
    ds.close()


def test_the_store_is_small(tmp_path):
    import os
    ds = _ds(tmp_path, n=20000)
    A = _graph(n=20000, seed=7); ds.add_graph("connectivities", A); ds.flush()
    ds.close()
    raw = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
    assert os.path.getsize(tmp_path / "g.cytome") < 3 * raw + (2 << 20)
