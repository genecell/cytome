"""``from_10x_h5`` reads the matrix in batches, not all at once.

Until this change the converter built the whole matrix and then called
``.T.tocsr()``, holding roughly two full copies: a 73 MB multiome file peaked at
1,329 MB, 18x the file on disk. That made *creating* a cytome the memory ceiling
of a pipeline whose entire premise is that analysis streams.

Two properties are worth pinning, and they pull against each other:

* the result must be **identical** to the old whole-matrix construction, and
* memory must scale with the batch, not with the file.

A test that only checked the first would pass on a rewrite that quietly
re-materialised everything.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp

import cytome
from cytome.io.convert_cellranger import (
    _auto_batch_size,
    _iter_10x_h5_cell_chunks,
    _read_10x_h5_meta,
    _select_columns,
)


def test_top_level_wrapper_forwards_every_parameter():
    """``cytome.from_10x_h5`` restates the signature, so it can silently drop one.

    It has already happened twice: 0.2.3 shipped a ``modalities`` branch with no
    such parameter, and the streaming work added ``batch_size`` to the
    implementation while the wrapper kept calling the old signature -- so the
    argument was accepted at the top level and never reached the reader. Both
    were invisible until something called the function.
    """
    import inspect

    from cytome.io.convert_cellranger import from_10x_h5 as impl

    wrapper_params = inspect.signature(cytome.from_10x_h5).parameters
    impl_params = inspect.signature(impl).parameters

    missing = set(impl_params) - set(wrapper_params)
    assert not missing, (
        f"cytome.from_10x_h5 does not expose {sorted(missing)}; callers cannot "
        "reach it through the public entry point.")

    for name, p in impl_params.items():
        if p.default is not inspect.Parameter.empty:
            assert wrapper_params[name].default == p.default, (
                f"default for {name!r} differs: wrapper "
                f"{wrapper_params[name].default!r} vs impl {p.default!r}")


def _write_h5(path, n_genes, n_peaks, n_cells, density=0.3, seed=0):
    """A CellRanger v3 file: CSC over (features, cells), like the real thing."""
    import h5py

    rng = np.random.default_rng(seed)
    n_features = n_genes + n_peaks
    dense = (rng.random((n_features, n_cells)) < density) * rng.integers(
        1, 50, size=(n_features, n_cells))
    X = sp.csc_matrix(dense.astype(np.int32))

    ids = ([f"ENSG{i:06d}".encode() for i in range(n_genes)]
           + [f"chr1:{i*100}-{i*100+50}".encode() for i in range(n_peaks)])
    ftypes = [b"Gene Expression"] * n_genes + [b"Peaks"] * n_peaks

    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=X.data)
        g.create_dataset("indices", data=X.indices.astype(np.int64))
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array([n_features, n_cells], dtype=np.int64))
        g.create_dataset("barcodes", data=np.array(
            [f"C{i:05d}".encode() for i in range(n_cells)]))
        fg = g.create_group("features")
        fg.create_dataset("id", data=np.array(ids))
        fg.create_dataset("name", data=np.array(ids))
        fg.create_dataset("feature_type", data=np.array(ftypes))
        fg.create_dataset("genome", data=np.array([b"test"] * n_features))
    return X, n_features


def _reference(X, n_genes, n_features, kind):
    """What the old whole-matrix path produced: features x cells, masked, .T."""
    mask = np.zeros(n_features, dtype=bool)
    if kind == "rna":
        mask[:n_genes] = True
    else:
        mask[n_genes:] = True
    return X[mask, :].T.tocsr()


@pytest.mark.parametrize("batch_size", [1, 3, 7, 50, None])
def test_streamed_result_is_identical_to_whole_matrix(tmp_path, batch_size):
    """Every batch size must give the same matrix, including batch=1.

    Batch boundaries are where an off-by-one in the indptr rebase would show,
    so the small sizes matter more than the realistic ones.
    """
    h5 = tmp_path / "m.h5"
    n_genes, n_peaks, n_cells = 11, 9, 23
    X, n_features = _write_h5(h5, n_genes, n_peaks, n_cells)

    ds = cytome.from_10x_h5(h5, tmp_path / f"b{batch_size}.cytome",
                            force=True, batch_size=batch_size)
    try:
        for kind, modality in (("rna", "RNA"), ("atac", "ATAC")):
            expected = _reference(X, n_genes, n_features, kind)
            got = sp.vstack([c for c, _ in ds.iter_chunks(
                modality=modality, layer="counts", batch_size=1000)]).tocsr()
            expected.sort_indices()
            got.sort_indices()
            assert got.shape == expected.shape
            assert got.nnz == expected.nnz
            np.testing.assert_array_equal(got.indptr, expected.indptr)
            np.testing.assert_array_equal(got.indices, expected.indices)
            np.testing.assert_array_equal(
                got.data.astype(np.int64), expected.data.astype(np.int64))
    finally:
        ds.close()


def test_no_batch_holds_more_than_its_share_of_nonzeros(tmp_path):
    """The property that makes it streaming: bounded chunks.

    Asserted on the chunks themselves rather than on process RSS, which is too
    noisy to gate a test on.
    """
    h5 = tmp_path / "m.h5"
    n_genes, n_peaks, n_cells = 20, 20, 40
    _write_h5(h5, n_genes, n_peaks, n_cells, density=0.5)
    _, _, _, _, shape, group = _read_10x_h5_meta(h5)
    n_features, n_cells = shape

    total = 0
    for _, chunk in _iter_10x_h5_cell_chunks(h5, group, n_cells, n_features, 5):
        assert chunk.shape[0] <= 5, "a batch exceeded the requested cell count"
        total += chunk.nnz
    assert total > 0

    whole = next(iter(_iter_10x_h5_cell_chunks(h5, group, n_cells, n_features, n_cells)))
    assert total == whole[1].nnz, "batching lost or duplicated non-zeros"


def test_auto_batch_size_tracks_density():
    """A denser file must get a smaller batch, or the point is lost."""
    n_cells = 10_000
    sparse_ptr = np.arange(n_cells + 1, dtype=np.int64) * 100     # 100 nnz/cell
    dense_ptr = np.arange(n_cells + 1, dtype=np.int64) * 10_000   # 10k nnz/cell

    b_sparse = _auto_batch_size(sparse_ptr, n_cells)
    b_dense = _auto_batch_size(dense_ptr, n_cells)

    assert b_dense < b_sparse, (
        f"denser file got a larger batch ({b_dense} vs {b_sparse}); memory would "
        "scale with density instead of being bounded")
    assert 1 <= b_dense <= n_cells and 1 <= b_sparse <= n_cells
    # both should land near the non-zero target rather than at an arbitrary cap
    assert b_dense * 10_000 == pytest.approx(8_000_000, rel=0.5)


def test_select_columns_matches_fancy_indexing():
    """The hand-rolled column select must equal `chunk[:, mask]`."""
    rng = np.random.default_rng(1)
    dense = (rng.random((6, 12)) < 0.4) * rng.integers(1, 9, size=(6, 12))
    chunk = sp.csr_matrix(dense.astype(np.int32))

    mask = np.zeros(12, dtype=bool)
    mask[[1, 4, 5, 9, 11]] = True
    remap = np.full(12, -1, dtype=np.int64)
    remap[np.flatnonzero(mask)] = np.arange(mask.sum())

    got = _select_columns(chunk, remap, int(mask.sum()))
    expected = chunk[:, mask]
    got.sort_indices()
    expected = expected.tocsr()
    expected.sort_indices()
    np.testing.assert_array_equal(got.toarray(), expected.toarray())


def test_empty_batch_is_handled(tmp_path):
    """Cells with no counts at all must not break the indptr rebase."""
    import h5py

    h5 = tmp_path / "sparse.h5"
    n_features, n_cells = 5, 6
    # cells 2 and 3 are completely empty
    cols = {0: [(0, 3)], 1: [(1, 5)], 4: [(2, 7)], 5: [(0, 1), (4, 2)]}
    data, indices, indptr = [], [], [0]
    for c in range(n_cells):
        for r, v in cols.get(c, []):
            indices.append(r)
            data.append(v)
        indptr.append(len(data))

    with h5py.File(h5, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=np.array(data, dtype=np.int32))
        g.create_dataset("indices", data=np.array(indices, dtype=np.int64))
        g.create_dataset("indptr", data=np.array(indptr, dtype=np.int64))
        g.create_dataset("shape", data=np.array([n_features, n_cells], dtype=np.int64))
        g.create_dataset("barcodes", data=np.array(
            [f"C{i}".encode() for i in range(n_cells)]))
        fg = g.create_group("features")
        fg.create_dataset("id", data=np.array([f"G{i}".encode() for i in range(n_features)]))
        fg.create_dataset("name", data=np.array([f"G{i}".encode() for i in range(n_features)]))
        fg.create_dataset("feature_type", data=np.array([b"Gene Expression"] * n_features))
        fg.create_dataset("genome", data=np.array([b"t"] * n_features))

    ds = cytome.from_10x_h5(h5, tmp_path / "e.cytome", force=True, batch_size=2)
    try:
        got = sp.vstack([c for c, _ in ds.iter_chunks(
            modality="RNA", layer="counts", batch_size=100)]).tocsr()
        assert got.shape == (n_cells, n_features)
        assert got.nnz == len(data)
        assert got[2].nnz == 0 and got[3].nnz == 0
    finally:
        ds.close()
