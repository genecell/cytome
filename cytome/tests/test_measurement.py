from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cytome


class TestMeasurementLayer:
    def test_write_read_sparse(self, tmp_cytome, small_rna_matrix):
        ds = cytome.create(tmp_cytome)
        ds.add_matrix("RNA_counts", small_rna_matrix)
        ds.flush()
        out = ds.RNA.counts.to_memory()
        assert out.shape == small_rna_matrix.shape
        assert (out != small_rna_matrix).nnz == 0
        ds.close()

    def test_slice_rows(self, tmp_cytome, small_rna_matrix):
        ds = cytome.create(tmp_cytome)
        ds.add_matrix("RNA_counts", small_rna_matrix)
        ds.flush()
        out = ds.RNA.counts[:100, :]
        assert (out != small_rna_matrix[:100, :]).nnz == 0
        ds.close()

    def test_slice_cols(self, tmp_cytome, small_rna_matrix):
        ds = cytome.create(tmp_cytome)
        ds.add_matrix("RNA_counts", small_rna_matrix)
        ds.flush()
        out = ds.RNA.counts[:, :50]
        assert (out != small_rna_matrix[:, :50]).nnz == 0
        ds.close()

    def test_iter_rows(self, tmp_cytome, small_rna_matrix):
        ds = cytome.create(tmp_cytome)
        ds.add_matrix("RNA_counts", small_rna_matrix)
        ds.flush()
        total_rows = 0
        for start, end, chunk in ds.RNA.counts.iter_rows():
            assert chunk.shape[0] == end - start
            total_rows += end - start
        assert total_rows == small_rna_matrix.shape[0]
        ds.close()

    def test_build_feature_index(self, tmp_cytome, small_rna_matrix):
        ds = cytome.create(tmp_cytome)
        ds.add_matrix("RNA_counts", small_rna_matrix)
        ds.flush()
        ds.RNA.counts.build_feature_index()
        assert ds.RNA.counts.has_feature_index
        cols = list(ds.RNA.counts.iter_columns())
        assert len(cols) > 0
        ds.close()

    def test_auto_chunk_size(self):
        from cytome.io.chunk_tuning import compute_chunk_size

        c = compute_chunk_size(1000, 500, 10000)
        assert 16 <= c <= 10000

    def test_dtype_preservation(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        mat = sp.identity(50, format="csr", dtype=np.float32)
        ds.add_matrix("RNA_counts", mat)
        ds.flush()
        assert ds.RNA.counts.dtype == np.float32
        ds.close()
