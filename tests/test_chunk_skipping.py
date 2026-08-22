"""Masked reads must skip chunks instead of decompressing and discarding them.

A per-batch GDR over a 200k-cell cytome spent 93% of its wall clock here: the
mask was applied after ``iter_rows`` had decompressed every chunk, so each of
35 batches read the whole matrix on each of 17 SVD passes.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import cytome
import cytome.io.chunked_io as chunked_io

anndata = pytest.importorskip("anndata")

N_CELLS, N_GENES = 400, 30


def _chunk_spans(d):
    return d._conn.execute(
        "SELECT row_start, row_end FROM matrix_chunks WHERE matrix_name='RNA_counts' "
        "ORDER BY chunk_idx"
    ).fetchall()


def _n_chunks_touching(d, keep):
    """How many on-disk chunks a mask genuinely needs."""
    keep = np.sort(np.asarray(keep))
    return sum(1 for rs, re_ in _chunk_spans(d)
               if np.searchsorted(keep, rs, "left") < np.searchsorted(keep, re_, "left"))


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.3, size=(N_CELLS, N_GENES)).astype(np.float32))
    a = anndata.AnnData(X=X)
    d = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    yield d, X
    d.close()


def _read(d, keep=None):
    """Collect (rows, matrix) from iter_chunks and count decompressed chunks."""
    n_decompressed = 0
    real = chunked_io.decompress_blob

    def counting(blob, comp):
        nonlocal n_decompressed
        n_decompressed += 1
        return real(blob, comp)

    chunked_io.decompress_blob = counting
    try:
        parts, idx = [], []
        for chunk, rows in d.iter_chunks(modality="RNA", layer="counts", cell_mask=keep):
            parts.append(chunk)
            idx.append(np.asarray(rows))
    finally:
        chunked_io.decompress_blob = real
    # three blobs per chunk (data / indices / indptr)
    return np.concatenate(idx), sp.vstack(parts).toarray(), n_decompressed // 3


def test_no_mask_reads_everything(ds):
    d, X = ds
    rows, out, n_chunks = _read(d)
    np.testing.assert_array_equal(rows, np.arange(N_CELLS))
    np.testing.assert_allclose(out, X.toarray())
    assert n_chunks == len(_chunk_spans(d))


def test_contiguous_mask_skips_chunks(ds):
    """The GDR case: one batch occupies a contiguous span of rows."""
    d, X = ds
    keep = np.arange(N_CELLS // 4, N_CELLS // 4 + 32)
    rows, out, n_chunks = _read(d, keep)
    np.testing.assert_array_equal(rows, keep)
    np.testing.assert_allclose(out, X[keep].toarray())
    needed = _n_chunks_touching(d, keep)
    assert n_chunks == needed, f"read {n_chunks} chunks, only {needed} hold selected rows"
    assert n_chunks < len(_chunk_spans(d)), "a contiguous mask must skip something"


def test_shuffled_mask_degrades_gracefully(ds):
    """A scattered mask can't skip much, but must stay correct."""
    d, X = ds
    rng = np.random.default_rng(1)
    keep = np.sort(rng.choice(N_CELLS, size=120, replace=False))
    rows, out, n_chunks = _read(d, keep)
    np.testing.assert_array_equal(rows, keep)
    np.testing.assert_allclose(out, X[keep].toarray())
    assert n_chunks <= len(_chunk_spans(d))      # never worse than a full scan
    assert n_chunks == _n_chunks_touching(d, keep)


def test_single_row_mask_reads_one_chunk(ds):
    d, X = ds
    keep = np.array([137])
    rows, out, n_chunks = _read(d, keep)
    np.testing.assert_array_equal(rows, keep)
    np.testing.assert_allclose(out, X[keep].toarray())
    assert n_chunks == 1, f"one row should cost one chunk, read {n_chunks}"


def test_boolean_mask_matches_index_mask(ds):
    d, X = ds
    bool_mask = np.zeros(N_CELLS, dtype=bool)
    bool_mask[100:150] = True
    r1, o1, _ = _read(d, bool_mask)
    r2, o2, _ = _read(d, np.arange(100, 150))
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_allclose(o1, o2)


def test_empty_mask_reads_nothing(ds):
    d, _ = ds
    n = 0
    for _chunk, _rows in d.iter_chunks(modality="RNA", layer="counts",
                                       cell_mask=np.array([], dtype=np.int64)):
        n += 1
    assert n == 0
