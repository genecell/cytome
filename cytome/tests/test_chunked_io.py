from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cytome.io.chunk_tuning import compute_chunk_size
from cytome.io.chunked_io import (
    read_dense_chunked,
    read_dense_slice,
    read_sparse_chunked,
    read_sparse_rows_iter,
    read_sparse_slice,
    write_dense_chunked,
    write_sparse_chunked,
)
from cytome.io.sqlite_engine import create_database, close_database


class TestChunkedIO:
    def test_sparse_roundtrip(self, tmp_cytome, small_rna_matrix):
        conn = create_database(tmp_cytome)
        write_sparse_chunked(conn, "RNA_counts", small_rna_matrix, 64, "zstd")
        out = read_sparse_chunked(conn, "RNA_counts")
        assert out.shape == small_rna_matrix.shape
        assert (out != small_rna_matrix).nnz == 0
        close_database(conn)

    def test_sparse_slice(self, tmp_cytome, small_rna_matrix):
        conn = create_database(tmp_cytome)
        write_sparse_chunked(conn, "RNA_counts", small_rna_matrix, 64, "zstd")
        out = read_sparse_slice(conn, "RNA_counts", 10, 50)
        expected = small_rna_matrix[10:50]
        assert (out != expected).nnz == 0
        close_database(conn)

    def test_sparse_iter(self, tmp_cytome, small_rna_matrix):
        conn = create_database(tmp_cytome)
        write_sparse_chunked(conn, "RNA_counts", small_rna_matrix, 64, "zstd")
        seen = 0
        for start, end, chunk in read_sparse_rows_iter(conn, "RNA_counts"):
            seen += end - start
            assert chunk.shape[0] == end - start
        assert seen == small_rna_matrix.shape[0]
        close_database(conn)

    def test_dense_roundtrip(self, tmp_cytome):
        conn = create_database(tmp_cytome)
        arr = np.random.randn(200, 10).astype(np.float32)
        write_dense_chunked(conn, "RNA_pca", arr, 32, "zstd")
        out = read_dense_chunked(conn, "RNA_pca")
        assert np.allclose(arr, out)
        sli = read_dense_slice(conn, "RNA_pca", 5, 20)
        assert np.allclose(sli, arr[5:20])
        close_database(conn)

    def test_chunk_tuning_bounds(self):
        c = compute_chunk_size(1000, 100, 10000)
        assert 16 <= c <= 10000
