"""One reader for every selection (2026-11-10).

``rows()``, ``columns()``, ``column()``, ``X[...]`` and ``to_memory()`` read through
:meth:`MeasurementLayer._select` and :func:`cytome.io.chunked_io.read_sparse_axis`:
along the axis a matrix is stored by, only the chunks that hold a wanted position;
along the other, every chunk once, keeping the wanted positions. Checked here against
NumPy on the same matrix stored three ways (row chunks only, column chunks only, both),
with chunk sizes that don't divide the axes and with empty chunks.

The defects this replaced (cytome 95e46ff): ``rows()`` and ``X[a:b]`` built the whole
matrix on a column-major store; ``columns()`` decoded every chunk before testing
whether it held a wanted column, read the whole matrix on a row-major store, and read
a boolean mask as the column numbers 0 and 1.
"""
from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import cytome
from cytome.core import measurement as meas
from cytome.core.measurement import FullPassWarning, MeasurementLayer
from cytome.io import chunked_io
from cytome.io.chunked_io import read_sparse_axis, read_sparse_rows, write_sparse_chunked

N_ROWS, N_COLS = 97, 61
ROW_CHUNK, COL_CHUNK = 13, 7
LAYOUTS = ("rows", "cols", "both")


def _matrix(n_rows=N_ROWS, n_cols=N_COLS, density=0.15, seed=7):
    """The test matrix, and its dense form for the small one (None for a large one)."""
    rng = np.random.default_rng(seed)
    X = sp.random(n_rows, n_cols, density=density, format="csr", random_state=seed, dtype=np.float32)
    X.data = rng.integers(1, 9, X.nnz).astype(np.float32)
    if (n_rows, n_cols) != (N_ROWS, N_COLS):
        return X, None
    D = X.toarray()
    D[:, 21:28] = 0              # column chunk 3 (columns 21-27) is empty
    D[39:52, :] = 0              # row chunk 3 (rows 39-51) is empty
    return sp.csr_matrix(D), D


def _store(tmp_path, layout, X, row_chunk=ROW_CHUNK, col_chunk=COL_CHUNK):
    ds = cytome.create(tmp_path / f"{layout}.cytome")
    ds.set_entity("cells", {"barcode": [f"c{i}" for i in range(X.shape[0])]})
    ds.set_entity("genes", {"gene_id": [f"g{i}" for i in range(X.shape[1])]})
    write_sparse_chunked(ds._conn, "RNA_counts", X, chunk_size=row_chunk, compression="zstd")
    ds._conn.commit()
    if layout in ("cols", "both"):
        MeasurementLayer(ds._conn, "RNA_counts").build_feature_index(chunk_size=col_chunk)
    if layout == "cols":                 # column-major only, as the quantifier writes peak matrices
        ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name='RNA_counts'")
        ds._conn.execute("UPDATE matrix_meta SET n_chunks=0 WHERE matrix_name='RNA_counts'")
        ds._conn.commit()
    return ds, MeasurementLayer(ds._conn, "RNA_counts")


def _np_positions(sel, n):
    """What NumPy indexing selects along an axis of ``n``."""
    if isinstance(sel, slice):
        return np.arange(*sel.indices(n))
    if isinstance(sel, (int, np.integer)):
        return np.array([sel + n if sel < 0 else sel])
    a = np.asarray(sel)
    if a.dtype == bool:
        return np.flatnonzero(a)
    a = a.astype(np.int64).ravel()
    return np.where(a < 0, a + n, a)


ROW_SELS = [
    [3, 7, 42, 96],                                  # sorted; 42 is in the empty row chunk
    [50, 3, 3, 96, 0, 41],                           # unsorted, repeated
    [-1, -97, 5],                                    # negative
    np.arange(N_ROWS) % 3 == 0,                      # a boolean mask
    [],                                              # nothing
    np.array([12, 13, 25, 26], dtype=np.int32),      # chunk edges
    np.arange(N_ROWS)[::-1],                         # every row, reversed
]
COL_SELS = [
    [0, 1, 30, 60],
    [33, 6, 6, 60, 0, 22],                           # 22 is in the empty column chunk
    [-1, -61, 9],
    np.arange(N_COLS) % 4 == 1,
    [],
    np.array([6, 7, 13, 14], dtype=np.uint16),
    np.arange(N_COLS)[::-1],
]


@pytest.fixture(params=LAYOUTS)
def stored(request, tmp_path):
    X, D = _matrix()
    ds, m = _store(tmp_path, request.param, X)
    meas._FULL_PASS_REPORTED.clear()
    yield request.param, m, X, D
    ds.close()


@pytest.mark.parametrize("k", range(len(ROW_SELS)))
def test_rows(stored, k):
    _layout, m, _X, D = stored
    sel = ROW_SELS[k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FullPassWarning)
        got = m.rows(sel)
    assert sp.isspmatrix_csr(got) and got.dtype == np.float32
    np.testing.assert_array_equal(got.toarray(), D[_np_positions(sel, N_ROWS)])


@pytest.mark.parametrize("k", range(len(COL_SELS)))
def test_columns(stored, k):
    _layout, m, _X, D = stored
    sel = COL_SELS[k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FullPassWarning)
        got = m.columns(sel)
    assert sp.isspmatrix_csr(got) and got.dtype == np.float32
    np.testing.assert_array_equal(got.toarray(), D[:, _np_positions(sel, N_COLS)])


def test_column_and_boolean_mask(stored):
    """columns() read a boolean mask as the column numbers 0 and 1 (95e46ff)."""
    _layout, m, _X, D = stored
    mask = np.zeros(N_COLS, dtype=bool)
    mask[[5, 40]] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FullPassWarning)
        np.testing.assert_array_equal(m.columns(mask).toarray(), D[:, [5, 40]])
        np.testing.assert_array_equal(m.column(-2).toarray(), D[:, [N_COLS - 2]])


@pytest.mark.parametrize("rsel", [slice(None), slice(5, 40), slice(None, None, 3), slice(80, 10, -4),
                                  slice(90, 200), 5, -2, [8, 2, 8], np.arange(N_ROWS) > 60])
@pytest.mark.parametrize("csel", [slice(None), slice(10, 30), slice(None, None, -5), [3, -1, 3],
                                  np.arange(N_COLS) < 9, 7])
def test_getitem(stored, rsel, csel):
    _layout, m, _X, D = stored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FullPassWarning)
        got = m[rsel, csel] if not (isinstance(csel, slice) and csel == slice(None)) else m[rsel]
    assert sp.isspmatrix_csr(got)
    exp = D[np.ix_(_np_positions(rsel, N_ROWS), _np_positions(csel, N_COLS))]
    np.testing.assert_array_equal(got.toarray(), exp)


def test_errors(stored):
    _layout, m, _X, _D = stored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FullPassWarning)
        for bad in ([N_ROWS], [-N_ROWS - 1]):
            with pytest.raises(IndexError):
                m.rows(bad)
        for bad in ([N_COLS], [-N_COLS - 1]):
            with pytest.raises(IndexError):
                m.columns(bad)
        with pytest.raises(IndexError):
            m.rows(np.ones(N_ROWS - 1, dtype=bool))          # a mask of the wrong length
        with pytest.raises(IndexError):
            m.columns(np.ones(N_COLS + 1, dtype=bool))
        with pytest.raises(TypeError):
            m.rows([1.5, 2.0])
        with pytest.raises(IndexError):
            m[N_ROWS]
        with pytest.raises(IndexError):
            m[0, 1, 2]


def test_to_memory_unchanged(stored):
    """The CSC path now places chunks by span and sizes the destination once; for a
    well-formed store its result is the old one, array for array."""
    layout, m, X, _D = stored
    got = m.to_memory()
    if layout == "cols":
        old = sp.hstack([c for _, _, c in m.iter_columns()], format="csc").tocsr()
    else:
        old = X
    assert sp.isspmatrix_csr(got) and got.dtype == old.dtype
    np.testing.assert_array_equal(got.indptr, old.indptr)
    np.testing.assert_array_equal(got.indices, old.indices)
    np.testing.assert_array_equal(got.data, old.data)


def _count_decodes(monkeypatch):
    seen = []
    real = chunked_io._decode_block

    def counting(*a, **k):
        seen.append(1)
        return real(*a, **k)
    monkeypatch.setattr(chunked_io, "_decode_block", counting)
    return seen


def test_chunks_read(tmp_path, monkeypatch):
    """Along the stored axis only the chunks holding a wanted position are decoded;
    along the other, each chunk once."""
    X, _D = _matrix()
    n_col_chunks = -(-N_COLS // COL_CHUNK)
    ds, m = _store(tmp_path, "cols", X)
    seen = _count_decodes(monkeypatch)
    m.columns([0, 1, 30, 30])                     # chunks 0 and 4
    assert len(seen) == 2
    seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FullPassWarning)
        m.rows([5, 70])
    assert len(seen) == n_col_chunks
    ds.close()
    (tmp_path / "r").mkdir()
    ds, m = _store(tmp_path / "r", "rows", X)
    seen.clear()
    m.rows([5, 6, 70])                            # row chunks 0 and 5
    assert len(seen) == 2
    ds.close()


def test_full_pass_warns_once(tmp_path, monkeypatch):
    X, _D = _matrix()
    ds, m = _store(tmp_path, "cols", X)
    meas._FULL_PASS_REPORTED.clear()
    with warnings.catch_warnings(record=True) as w:     # 9 chunks: under the threshold, no warning
        warnings.simplefilter("always")
        m.rows([1, 2])
    assert not w
    monkeypatch.setattr(meas, "_FULL_PASS_MIN_CHUNKS", 1)
    meas._FULL_PASS_REPORTED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.rows([1, 2])
        m.rows([3])
        m[4:9]
        m.columns([1])                            # the stored axis: no warning
    assert [x.category for x in w] == [FullPassWarning]
    ds.close()
    (tmp_path / "r").mkdir()
    ds, m = _store(tmp_path / "r", "rows", X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.rows([1, 2])                            # the stored axis
        m.columns([4, 5])
        m.columns([6])
    assert [x.category for x in w] == [FullPassWarning]
    ds.close()


def test_rows_of_column_major_memory(tmp_path):
    """Rows of a column-major matrix used to build the whole matrix first; now the
    result plus one chunk."""
    X, _D = _matrix(n_rows=4000, n_cols=30000, density=0.01, seed=3)
    ds, m = _store(tmp_path, "cols", X, row_chunk=1024, col_chunk=256)
    want = np.arange(0, 4000, 40)
    tracemalloc.start()
    full = m.to_memory()
    _cur, peak_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del full
    tracemalloc.start()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FullPassWarning)
        got = m.rows(want)
    _cur, peak_rows = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    np.testing.assert_array_equal(got.toarray(), X[want].toarray())
    assert peak_rows < 0.25 * peak_full, (peak_rows, peak_full)
    ds.close()


def test_storage_level_rules(tmp_path):
    """The chunk readers keep their own contract: no negative positions there."""
    X, D = _matrix()
    ds, _m = _store(tmp_path, "both", X)
    with pytest.raises(IndexError):
        read_sparse_rows(ds._conn, "RNA_counts", [-1])
    got = read_sparse_axis(ds._conn, "RNA_counts", "cols", major=[9, 2, 9], minor=[5, 1, 5])
    assert sp.isspmatrix_csc(got)
    np.testing.assert_array_equal(got.toarray(), D[np.ix_([5, 1, 5], [9, 2, 9])])
    got = read_sparse_axis(ds._conn, "RNA_counts", "rows", major=None, minor=[60, 0])
    np.testing.assert_array_equal(got.toarray(), D[:, [60, 0]])
    ds.close()
