"""Full-matrix reads preallocate, and embeddings can state their width.

read_sparse_chunked accumulated every chunk in a list and then
np.concatenate-d, so both the parts and the result were alive at once and a
full read peaked at about twice the matrix. matrix_meta records n_nonzero at
write time, so the destination is sizeable before the first blob is touched.

add_embedding had no dtype, so an embedding PIASO computed was float64 while
the same embedding converted from an h5ad was float32, in one file.
"""
import sqlite3

import numpy as np
import pytest
import scipy.sparse as sp

import cytome

anndata = pytest.importorskip("anndata")


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix((rng.random((400, 60)) < 0.3).astype(np.float32))
    a = anndata.AnnData(X=X)
    p = tmp_path / "t.cytome"
    d = cytome.from_anndata(a, output=str(p))
    return d, X, str(p)


def test_full_read_is_exact(ds):
    d, X, _ = ds
    got = d.RNA.counts.to_memory() if hasattr(d, "RNA") else None
    if got is None:
        from cytome.core.measurement import MeasurementLayer
        got = MeasurementLayer(d._conn, "RNA_counts").to_memory()
    assert got.shape == X.shape
    assert got.nnz == X.nnz
    assert np.allclose(got.toarray(), X.toarray())
    d.close()


def test_read_survives_a_wrong_n_nonzero(ds, recwarn):
    """If matrix_meta and the chunks disagree, trust the chunks and say so."""
    d, X, path = ds
    d.close()
    con = sqlite3.connect(path)
    con.execute("UPDATE matrix_meta SET n_nonzero = n_nonzero + 5 "
                "WHERE matrix_name = 'RNA_counts'")
    con.commit(); con.close()

    from cytome.core.measurement import MeasurementLayer
    d2 = cytome.open(path)
    try:
        with pytest.warns(UserWarning, match="matrix_meta says"):
            got = MeasurementLayer(d2._conn, "RNA_counts").to_memory()
        assert got.nnz == X.nnz
        assert np.allclose(got.toarray(), X.toarray())
    finally:
        d2.close()


def test_empty_matrix_reads_back(tmp_path):
    a = anndata.AnnData(X=sp.csr_matrix((5, 4), dtype=np.float32))
    p = tmp_path / "e.cytome"
    d = cytome.from_anndata(a, output=str(p))
    try:
        from cytome.core.measurement import MeasurementLayer
        got = MeasurementLayer(d._conn, "RNA_counts").to_memory()
        assert got.shape == (5, 4) and got.nnz == 0
    finally:
        d.close()


def test_add_embedding_dtype(ds):
    d, _, path = ds
    d.add_embedding("RNA_svd", np.random.rand(400, 3))                    # as given
    d.add_embedding("RNA_umap", np.random.rand(400, 2), dtype="float32")  # narrowed
    d.close()
    con = sqlite3.connect(path)
    got = dict(con.execute("SELECT array_name, dtype FROM embedding_meta"))
    con.close()
    assert got["RNA_svd"] == "float64"
    assert got["RNA_umap"] == "float32"


def test_add_embedding_dtype_roundtrips_values(ds):
    d, _, path = ds
    e = np.random.rand(400, 4)
    d.add_embedding("RNA_pca", e, dtype="float32")
    d.close()
    d2 = cytome.open(path)
    try:
        back = np.asarray(d2.embeddings["RNA_pca"])
        assert back.shape == e.shape
        assert np.allclose(back, e, atol=1e-6)
    finally:
        d2.close()
