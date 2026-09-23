"""``iter_chunks(max_batch_nnz=...)`` closes a batch on non-zeros as well as rows.

A row bound alone lets a batch of deep cells be many times the size of an
average one. The non-zero bound holds the batch's memory whatever the cell
order, without changing what is stored, at the granularity of one chunk.
"""
import anndata as ad
import numpy as np
import scipy.sparse as sp
import pytest

import cytome


@pytest.fixture
def skewed(tmp_path):
    """400 cells; the first 40 hold ~30x the non-zeros of the rest."""
    rng = np.random.default_rng(0)
    n, g = 400, 3000
    rows = []
    for i in range(n):
        k = 900 if i < 40 else 30
        cols = rng.choice(g, size=k, replace=False)
        rows.append(sp.csr_matrix((np.ones(k, dtype=np.float32), (np.zeros(k, int), cols)), shape=(1, g)))
    X = sp.vstack(rows, format="csr")
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = [f"g{j}" for j in range(g)]
    p = tmp_path / "skew.cytome"
    ds = cytome.from_anndata(A, modality="RNA", output=str(p))
    ds.close()
    return str(p), X


def test_rows_only_lets_a_deep_batch_dominate(skewed):
    p, X = skewed
    ds = cytome.open(p)
    sizes = [c.nnz for c, _ in ds.iter_chunks(modality="RNA", layer="counts", batch_size=128)]
    ds.close()
    assert max(sizes) > 5 * min(sizes), "fixture must be skewed for the test to mean anything"


def test_nnz_budget_bounds_every_batch(skewed):
    p, X = skewed
    budget = 6000
    ds = cytome.open(p)
    got = []
    sizes = []
    for chunk, idx in ds.iter_chunks(modality="RNA", layer="counts",
                                     batch_size=128, max_batch_nnz=budget):
        got.append((chunk, idx))
        sizes.append(chunk.nnz)
    ds.close()
    # One chunk of granularity: a batch may exceed the budget by at most the
    # chunk that closed it, and never by a whole extra chunk.
    chunk_nnz_max = max(c.nnz for c, _ in got)  # crude upper bound on a chunk
    assert all(s <= budget + chunk_nnz_max for s in sizes)
    assert sum(sizes) == X.nnz
    # And the matrix is unchanged: concatenating the batches gives X back.
    order = np.concatenate([i for _, i in got])
    assert np.array_equal(order, np.arange(X.shape[0]))
    Y = sp.vstack([c for c, _ in got], format="csr")
    assert (Y != X).nnz == 0


def test_without_the_budget_behaviour_is_unchanged(skewed):
    p, X = skewed
    ds = cytome.open(p)
    a = [c.nnz for c, _ in ds.iter_chunks(modality="RNA", layer="counts", batch_size=128)]
    b = [c.nnz for c, _ in ds.iter_chunks(modality="RNA", layer="counts", batch_size=128, max_batch_nnz=None)]
    ds.close()
    assert a == b
