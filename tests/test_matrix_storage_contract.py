"""The stored form of a sparse matrix, and readers that hold to it.

Audit 2026-10-27 (part 4 critical 2 and 4, major 15; L1 findings 2, 8, 14-18,
23): a full read wrapped its int32 index pointer past 2**31 entries; a
``csr_array`` with int64 indices was written as it was and could not be read
back; ``merge`` wrote int32 chunks under a float32 matrix, read back as float
bits; chunks written out of row order came back permuted; ``rows()`` needed
sorted indices; negative slices returned the wrong rows; a masked read put
every chunk in one SQL ``IN``; ``delete_matrix`` left a column-major matrix's
chunks behind; ``finalize`` waited out the busy timeout behind any reader.
"""
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
import cytome.io.chunked_io as cio
from cytome.core.measurement import MeasurementLayer
from cytome.io.compression import compress_blob, decompress_blob


def _store(tmp_path, n_cells=300, n_genes=15, name="s.cytome"):
    ds = cytome.create(str(tmp_path / name))
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(n_cells)]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"g{i}" for i in range(n_genes)]}))
    ds.flush()
    return ds


def _matrix(n=300, m=15, seed=0, dtype=np.float32):
    return sp.random(n, m, density=0.3, format="csr", random_state=seed, dtype=dtype)


def _equal(a, b):
    return a.shape == b.shape and (sp.csr_matrix(a) != sp.csr_matrix(b)).nnz == 0


def test_full_read_index_pointer_past_int32():
    """The decision and the arithmetic of a read past 2**31 - 1 entries, at a
    size a test can hold. The old statement wrapped: rows placed after
    2**31 - 7 entries came back with a negative index pointer."""
    assert cio._index_dtype(2**31 - 1) is np.int32
    assert cio._index_dtype(2**31) is np.int64
    at = 2**31 - 7
    old = np.zeros(4, dtype=np.int32)
    with np.errstate(over="ignore"):
        old[1:3] = np.array([0, 5, 10], dtype=np.int32)[1:] + at
    assert old[2] < 0                                   # what the int32 reader returned
    indptr = np.zeros(4, dtype=cio._index_dtype(at + 10))
    cio._place_rows(indptr, 0, 0, 2, np.array([0, 5, 10], dtype=np.int32), at)
    assert indptr[1:3].tolist() == [at + 5, at + 10]


def test_full_read_with_a_lowered_limit_is_exact(tmp_path, monkeypatch):
    ds = _store(tmp_path)
    A = _matrix()
    ds.add_matrix("RNA_counts", A)
    ds.flush()
    monkeypatch.setattr(cio, "_INT32_MAX", 100)        # A has about 1,350 entries
    assert _equal(cio.read_sparse_chunked(ds._conn, "RNA_counts"), A)
    ds.close()


def test_full_read_grows_when_recorded_sizes_are_short(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    ds.add_matrix("RNA_counts", A)
    ds.flush()
    ds._conn.execute("UPDATE matrix_chunks SET n_nonzero = 0 WHERE matrix_name = 'RNA_counts'")
    with pytest.warns(UserWarning, match="matrix_meta says"):
        got = cio.read_sparse_chunked(ds._conn, "RNA_counts")
    assert _equal(got, A)
    ds.close()


def test_csr_array_with_int64_indices_is_stored_as_int32(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    B = sp.csr_array(A)
    B.indices = B.indices.astype(np.int64)
    B.indptr = B.indptr.astype(np.int64)
    ds.add_matrix("RNA_counts", B)
    ds.flush()
    assert _equal(ds.RNA.counts.to_memory(), A)
    assert _equal(ds.RNA.counts[10:60], A[10:60])
    for nnz, blob, comp in ds._conn.execute(
            "SELECT n_nonzero, indices_blob, compression FROM matrix_chunks"):
        assert len(decompress_blob(blob, comp)) == 4 * nnz
    # the streaming writer too
    w = ds.create_layer_writer("RNA_arr", 300, 15, dtype=np.float32)
    w.write_chunk(B[:150], 0)
    w.write_chunk(B[150:], 150)
    w.finalize()
    assert _equal(ds.RNA.layer("arr").to_memory(), A)
    ds.close()


def test_old_store_with_int64_index_blobs_is_read(tmp_path):
    """What the writers stored for an int64-index csr_array before: still read."""
    ds = _store(tmp_path, n_cells=4, n_genes=3)
    A = sp.csr_matrix(np.array([[1, 0, 2], [0, 0, 0], [0, 3, 0], [4, 0, 5]], dtype=np.float32))
    c = ds._conn
    for ci, (r0, r1) in enumerate(((0, 2), (2, 4))):
        sub = A[r0:r1]
        c.execute(cio._INSERT_CHUNK_SQL, (
            "RNA_counts", ci, r0, r1, int(sub.nnz),
            compress_blob(sub.data.tobytes(), "zstd"),
            compress_blob(sub.indices.astype(np.int64).tobytes(), "zstd"),
            compress_blob(sub.indptr.astype(np.int64).tobytes(), "zstd"), "float32", "zstd"))
    c.execute("INSERT INTO matrix_meta(matrix_name,n_rows,n_cols,n_nonzero,dtype,row_entity,"
              "col_entity,chunk_size,n_chunks) VALUES ('RNA_counts',4,3,5,'float32','cells',"
              "'genes',2,2)")
    c.commit()
    assert _equal(ds.RNA.counts.to_memory(), A)
    assert _equal(ds.RNA.counts.rows([3, 0]), A[[3, 0]])
    ds.close()


def test_mixed_chunk_dtypes_are_decoded_by_their_own_dtype(tmp_path):
    """A matrix as merge wrote it from an int32 and a float32 input."""
    ds = _store(tmp_path, n_cells=4, n_genes=2)
    c = ds._conn
    C = sp.csr_matrix(np.array([[2, 0], [0, 3]], dtype=np.int32))
    for ci, (r0, block) in enumerate(((0, C), (2, C.astype(np.float32)))):
        c.execute(cio._INSERT_CHUNK_SQL, (
            "RNA_counts", ci, r0, r0 + 2, int(block.nnz),
            compress_blob(block.data.tobytes(), "zstd"),
            compress_blob(block.indices.astype(np.int32).tobytes(), "zstd"),
            compress_blob(block.indptr.astype(np.int32).tobytes(), "zstd"),
            str(block.data.dtype), "zstd"))
    c.execute("INSERT INTO matrix_meta(matrix_name,n_rows,n_cols,n_nonzero,dtype,row_entity,"
              "col_entity,chunk_size,n_chunks) VALUES ('RNA_counts',4,2,4,'float32','cells',"
              "'genes',2,2)")
    c.commit()
    with pytest.warns(UserWarning, match="decoded by their own dtype"):
        got = ds.RNA.counts.to_memory()
    assert got.dtype == np.float32
    assert got.toarray().tolist() == [[2, 0], [0, 3], [2, 0], [0, 3]]
    ds.close()


def test_big_endian_and_duplicate_input_is_stored_canonical(tmp_path):
    ds = _store(tmp_path, n_cells=3, n_genes=3)
    dense = np.array([[1, 0, 2], [0, 0, 0], [0, 3, 0]], dtype=">f4")
    ds.add_matrix("RNA_counts", dense)                  # scipy >= 1.18 refuses to build this itself
    big = sp.csr_matrix(dense.astype(np.float32))
    big.data = big.data.astype(">f4")                   # a big-endian sparse matrix, on any scipy
    ds.add_matrix("RNA_big", big)
    # duplicates built from (data, indices, indptr): scipy >= 1.18 sums them in the COO form
    dup = sp.csr_matrix((np.ones(3, dtype=np.float32), np.array([1, 1, 0]), np.array([0, 2, 2, 3])),
                        shape=(3, 3))
    assert dup.nnz == 3
    ds.add_matrix("RNA_dup", dup)
    ds.flush()
    for name in ("RNA_counts", "RNA_big"):
        assert ds._conn.execute("SELECT dtype FROM matrix_meta WHERE matrix_name=?",
                                (name,)).fetchone()[0] == "float32"
    assert ds.RNA.counts.to_memory().toarray().tolist() == dense.astype(np.float32).tolist()
    assert ds.RNA.layer("big").to_memory().toarray().tolist() == dense.astype(np.float32).tolist()
    m = ds.RNA.layer("dup").to_memory()
    assert m.nnz == 2 and m.toarray().tolist() == [[0, 2, 0], [0, 0, 0], [1, 0, 0]]
    ds.close()


def test_writer_refuses_a_lossy_float_into_an_integer_layer(tmp_path):
    ds = _store(tmp_path, n_cells=2, n_genes=2)
    w = ds.create_layer_writer("RNA_counts", 2, 2, dtype=np.int32)
    with pytest.raises(ValueError, match="do not fit"):
        w.write_chunk(sp.csr_matrix(np.array([[0.4, 0], [0, 1]], dtype=np.float32)), 0)
    w.write_chunk(sp.csr_matrix(np.array([[2.0, 0], [0, 1]], dtype=np.float32)), 0)  # integral
    w.finalize()
    assert ds.RNA.counts.to_memory().toarray().tolist() == [[2, 0], [0, 1]]
    ds.close()


def test_writer_refuses_chunks_out_of_order_or_past_the_end(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    w = ds.create_layer_writer("RNA_ooo", 300, 15, dtype=np.float32)
    w.write_chunk(A[150:], 150)
    with pytest.raises(ValueError, match="row order"):
        w.write_chunk(A[:150], 0)
    with pytest.raises(ValueError, match="pass the layer"):
        w.write_chunk(A[:10], 300)
    with pytest.raises(ValueError, match="columns"):
        w.write_chunk(A[:, :3], 300)
    ds.close()


def test_rows_a_writer_skips_read_as_zero_rows(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    w = ds.create_layer_writer("RNA_gap", 300, 15, dtype=np.float32)
    w.write_chunk(A[:100], 0)
    w.write_chunk(A[200:], 200)          # rows 100..199 are never written
    w.finalize()
    expect = A.tolil()
    expect[100:200] = 0
    expect = expect.tocsr()
    with pytest.warns(UserWarning, match="rows no chunk holds are zero"):
        assert _equal(ds.RNA.layer("gap").to_memory(), expect)
    assert _equal(ds.RNA.layer("gap")[90:210], expect[90:210])
    assert _equal(ds.RNA.layer("gap").rows([150, 5, 250]), expect[[150, 5, 250]])
    ds.close()


def test_chunks_stored_out_of_order_are_read_in_row_order(tmp_path):
    """A store whose chunk order is not its row order (an older writer): rows
    are placed by row_start, by every reader."""
    ds = _store(tmp_path)
    A = _matrix()
    c = ds._conn
    for ci, (r0, r1) in enumerate(((150, 300), (0, 150))):
        sub = A[r0:r1]
        c.execute(cio._INSERT_CHUNK_SQL, (
            "RNA_ooo", ci, r0, r1, int(sub.nnz),
            compress_blob(sub.data.tobytes(), "zstd"),
            compress_blob(sub.indices.astype(np.int32).tobytes(), "zstd"),
            compress_blob(sub.indptr.astype(np.int32).tobytes(), "zstd"), "float32", "zstd"))
    c.execute("INSERT INTO matrix_meta(matrix_name,n_rows,n_cols,n_nonzero,dtype,row_entity,"
              "col_entity,chunk_size,n_chunks) VALUES ('RNA_ooo',300,15,?,'float32','cells',"
              "'genes',150,2)", (int(A.nnz),))
    c.commit()
    ml = ds.RNA.layer("ooo")
    assert _equal(ml.to_memory(), A)
    assert _equal(ml[140:160], A[140:160])
    got = list(ds.iter_chunks("RNA", "ooo"))
    assert np.array_equal(np.concatenate([r for _, r in got]), np.arange(300))
    assert _equal(sp.vstack([m for m, _ in got]), A)
    ds.close()


def test_rows_in_any_order_with_repeats(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    ds.add_matrix("RNA_counts", A)
    ds.add_embedding("RNA_pca", np.arange(300 * 4, dtype=np.float64).reshape(300, 4))
    ds.flush()
    idx = np.array([250, 3, 3, 120, 0, 299])
    assert _equal(ds.RNA.counts.rows(idx), A[idx])
    E = np.arange(300 * 4, dtype=np.float64).reshape(300, 4)
    assert np.array_equal(cio.read_dense_rows(ds._conn, "RNA_pca", idx), E[idx])
    with pytest.raises(IndexError):
        ds.RNA.counts.rows([0, 300])
    with pytest.raises(IndexError):
        cio.read_dense_rows(ds._conn, "RNA_pca", [-1])
    ds.close()


def test_slices_count_negative_positions_and_steps(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    ds.add_matrix("RNA_counts", A)
    E = np.random.default_rng(1).random((300, 4))
    ds.add_embedding("RNA_pca", E)
    ds.flush()
    ml = ds.RNA.counts
    for sl in (slice(-10, None), slice(None, -1), slice(5, 50, 7), slice(None, None, -1),
               slice(40, 10, -3), slice(10, 5)):
        assert _equal(ml[sl], A[sl]), sl
    from cytome.core.embedding import EmbeddingArray
    ea = EmbeddingArray(ds._conn, "RNA_pca")
    for sl in (slice(-10, None), slice(None, -1), slice(5, 50, 7), slice(None, None, -1)):
        assert np.array_equal(ea[sl], E[sl]), sl
    assert _equal(ml[np.int64(7)], A[7])
    assert np.array_equal(ea[np.int64(-1)], E[-1:])
    ds.close()


def test_iter_chunks_refuses_a_mask_that_does_not_fit(tmp_path):
    ds = _store(tmp_path)
    ds.add_matrix("RNA_counts", _matrix())
    ds.flush()
    with pytest.raises(ValueError, match="boolean cell_mask"):
        list(ds.iter_chunks("RNA", "counts", cell_mask=np.ones(299, dtype=bool)))
    with pytest.raises(IndexError):
        list(ds.iter_chunks("RNA", "counts", cell_mask=np.array([0, 300])))
    ds.close()


@pytest.mark.skipif(not hasattr(sqlite3.Connection, "setlimit"), reason="Python < 3.11")
def test_masked_read_of_many_chunks_stays_under_the_variable_limit(tmp_path):
    ds = _store(tmp_path, n_cells=4000, n_genes=5)
    A = _matrix(4000, 5)
    w = ds.create_layer_writer("RNA_counts", 4000, 5, dtype=np.float32, storage_chunk_size=4)
    w.write_chunk(A, 0)
    w.finalize()
    ds._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 200)   # 1,000 chunks wanted
    keep = np.arange(0, 4000, 4)
    got = sp.vstack([m for m, _ in ds.iter_chunks("RNA", "counts", cell_mask=keep)])
    assert _equal(got, A[keep])
    ds.close()


def test_delete_and_overwrite_remove_column_chunks(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    ds.add_matrix("RNA_counts", A)
    ds.flush()
    ds.RNA.counts.build_feature_index(chunk_size=4)
    n = lambda: ds._conn.execute("SELECT COUNT(*) FROM matrix_csc_chunks").fetchone()[0]
    assert n() > 0
    w = ds.create_layer_writer("RNA_counts", 300, 15, dtype=np.float32)   # overwrite
    w.write_chunk(A, 0)
    w.finalize()
    assert n() == 0
    ds.RNA.counts.build_feature_index(chunk_size=4)
    assert ds.delete_matrix("RNA_counts") == ["RNA_counts"]
    assert n() == 0
    ds.close()


def test_feature_index_columns_in_order_with_repeats(tmp_path):
    ds = _store(tmp_path)
    A = _matrix()
    ds.add_matrix("RNA_counts", A)
    ds.flush()
    ml = ds.RNA.counts
    ml.build_feature_index(chunk_size=4)
    idx = [14, 2, 2, 7, -1]
    assert _equal(ml.columns(idx), A[:, [14, 2, 2, 7, 14]])
    other = MeasurementLayer(ds._conn, "RNA_counts")          # committed: visible again
    assert other.has_feature_index
    ds.close()


def test_finalize_does_not_wait_for_another_reader(tmp_path):
    ds = _store(tmp_path)
    ds._conn.execute("PRAGMA busy_timeout=3000")
    reader = sqlite3.connect(str(ds.path))
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM cells").fetchone()     # holds a read snapshot
    w = ds.create_layer_writer("RNA_counts", 300, 15, dtype=np.float32)
    w.write_chunk(_matrix(), 0)
    t = time.perf_counter()
    w.finalize()
    assert time.perf_counter() - t < 2.0
    reader.rollback()
    reader.close()
    ds.close()


def test_chunk_range_queries_use_the_covering_index(tmp_path):
    ds = _store(tmp_path)
    ds.add_matrix("RNA_counts", _matrix())
    ds.flush()
    plan = " ".join(str(r) for r in ds._conn.execute(
        "EXPLAIN QUERY PLAN SELECT chunk_idx, row_start, row_end, COALESCE(n_nonzero, 0) "
        "FROM matrix_chunks WHERE matrix_name = ? ORDER BY chunk_idx", ("RNA_counts",)))
    assert "COVERING INDEX idx_matrix_chunks_span" in plan
    ds.close()
