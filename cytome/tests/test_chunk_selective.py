"""Tests for chunk-selective row reading and subsetting."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

import cytome
from cytome.core.measurement import MeasurementLayer
from cytome.io.chunked_io import read_sparse_rows, read_dense_rows


def _make_ds(path, n_cells=200, n_genes=50, chunk_size=None):
    """Create a test dataset with known data."""
    ds = cytome.create(path)
    rng = np.random.default_rng(42)

    ds.set_entity("cells", {
        "barcode": [f"c{i}" for i in range(n_cells)],
        "cell_type": rng.choice(["A", "B", "C"], size=n_cells),
    })
    ds.set_entity("genes", {
        "gene_id": [f"G{i}" for i in range(n_genes)],
    })

    mat = sp.random(n_cells, n_genes, density=0.3, format="csr",
                    dtype=np.float32, random_state=42)
    ds.add_matrix("RNA_counts", mat)

    emb = rng.standard_normal((n_cells, 10)).astype(np.float32)
    ds.add_embedding("RNA_obsm_X_pca", emb)
    ds.flush()
    return ds, mat.toarray(), emb


# ═══════════════════════════════════════════════════════════════════════════
#  read_sparse_rows tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReadSparseRows:

    def test_all_rows_matches_full(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        full = MeasurementLayer(ds._conn, "RNA_counts").to_memory().toarray()
        all_idx = np.arange(ds.n_cells)
        selective = read_sparse_rows(ds._conn, "RNA_counts", all_idx).toarray()
        np.testing.assert_array_equal(selective, full)
        ds.close()

    def test_subset_correct_values(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        idx = np.array([0, 5, 10, 50, 99, 150, 199])
        result = read_sparse_rows(ds._conn, "RNA_counts", idx).toarray()
        np.testing.assert_array_equal(result, expected[idx])
        ds.close()

    def test_contiguous_range(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        idx = np.arange(20, 80)
        result = read_sparse_rows(ds._conn, "RNA_counts", idx).toarray()
        np.testing.assert_array_equal(result, expected[idx])
        ds.close()

    def test_single_row(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        idx = np.array([42])
        result = read_sparse_rows(ds._conn, "RNA_counts", idx).toarray()
        np.testing.assert_array_equal(result, expected[42:43])
        ds.close()

    def test_empty_indices(self, tmp_path):
        ds, _, _ = _make_ds(tmp_path / "a.cytome")
        result = read_sparse_rows(ds._conn, "RNA_counts", np.array([], dtype=np.int64))
        assert result.shape == (0, 50)
        ds.close()

    def test_preserves_sparse_format_and_dtype(self, tmp_path):
        ds, _, _ = _make_ds(tmp_path / "a.cytome")
        idx = np.array([0, 10, 20])
        result = read_sparse_rows(ds._conn, "RNA_counts", idx)
        assert sp.issparse(result)
        assert result.format == "csr"
        assert result.dtype == np.float32
        ds.close()

    def test_random_5pct_subset(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        rng = np.random.default_rng(123)
        idx = np.sort(rng.choice(200, size=10, replace=False))
        result = read_sparse_rows(ds._conn, "RNA_counts", idx).toarray()
        np.testing.assert_array_equal(result, expected[idx])
        ds.close()


# ═══════════════════════════════════════════════════════════════════════════
#  read_dense_rows tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReadDenseRows:

    def test_all_rows(self, tmp_path):
        ds, _, expected_emb = _make_ds(tmp_path / "a.cytome")
        all_idx = np.arange(200)
        result = read_dense_rows(ds._conn, "RNA_obsm_X_pca", all_idx)
        np.testing.assert_array_almost_equal(result, expected_emb)
        ds.close()

    def test_subset_rows(self, tmp_path):
        ds, _, expected_emb = _make_ds(tmp_path / "a.cytome")
        idx = np.array([0, 10, 50, 199])
        result = read_dense_rows(ds._conn, "RNA_obsm_X_pca", idx)
        np.testing.assert_array_almost_equal(result, expected_emb[idx])
        ds.close()

    def test_empty(self, tmp_path):
        ds, _, _ = _make_ds(tmp_path / "a.cytome")
        result = read_dense_rows(ds._conn, "RNA_obsm_X_pca", np.array([], dtype=np.int64))
        assert result.shape == (0, 10)
        ds.close()


# ═══════════════════════════════════════════════════════════════════════════
#  MeasurementLayer.rows() tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMeasurementLayerRows:

    def test_rows_matches_to_memory_slice(self, tmp_path):
        ds, _, _ = _make_ds(tmp_path / "a.cytome")
        layer = MeasurementLayer(ds._conn, "RNA_counts")
        idx = np.array([3, 7, 42, 100, 198])
        selective = layer.rows(idx).toarray()
        full_slice = layer.to_memory().toarray()[idx]
        np.testing.assert_array_equal(selective, full_slice)
        ds.close()


# ═══════════════════════════════════════════════════════════════════════════
#  to_anndata(cell_mask=...) tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectiveToAnndata:

    def test_mask_subset_matches_full_then_slice(self, tmp_path):
        ds, _, _ = _make_ds(tmp_path / "a.cytome")
        full = ds.to_anndata("RNA")
        idx = np.array([0, 5, 10, 50, 100])
        sub = ds.to_anndata("RNA", cell_mask=idx)
        np.testing.assert_array_equal(
            sub.X.toarray() if sp.issparse(sub.X) else sub.X,
            full.X.toarray()[idx] if sp.issparse(full.X) else full.X[idx],
        )
        assert sub.shape[0] == len(idx)
        assert sub.shape[1] == full.shape[1]
        ds.close()

    def test_boolean_mask(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        mask = np.zeros(200, dtype=bool)
        mask[[0, 10, 20]] = True
        sub = ds.to_anndata("RNA", cell_mask=mask)
        assert sub.shape[0] == 3
        X = sub.X.toarray() if sp.issparse(sub.X) else sub.X
        np.testing.assert_array_equal(X, expected[[0, 10, 20]])
        ds.close()

    def test_obs_correctly_subsetted(self, tmp_path):
        ds, _, _ = _make_ds(tmp_path / "a.cytome")
        full_obs = ds.to_anndata("RNA").obs
        idx = np.array([5, 15, 25])
        sub = ds.to_anndata("RNA", cell_mask=idx)
        assert sub.shape[0] == 3
        assert list(sub.obs["barcode"]) == [f"c{i}" for i in idx]
        ds.close()

    def test_obsm_correctly_subsetted(self, tmp_path):
        ds, _, expected_emb = _make_ds(tmp_path / "a.cytome")
        idx = np.array([10, 30, 50])
        sub = ds.to_anndata("RNA", cell_mask=idx)
        # Key is X_{name_without_RNA_prefix} = X_obsm_X_pca
        obsm_key = list(sub.obsm.keys())[0]
        np.testing.assert_array_almost_equal(sub.obsm[obsm_key], expected_emb[idx])
        ds.close()

    def test_var_unchanged(self, tmp_path):
        ds, _, _ = _make_ds(tmp_path / "a.cytome")
        full = ds.to_anndata("RNA")
        idx = np.array([0, 1, 2])
        sub = ds.to_anndata("RNA", cell_mask=idx)
        assert sub.shape[1] == full.shape[1]
        ds.close()

    def test_none_mask_returns_full(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        full = ds.to_anndata("RNA", cell_mask=None)
        assert full.shape[0] == 200
        ds.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Updated subset() tests
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkSelectiveSubset:

    def test_subset_produces_correct_result(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        idx = np.array([0, 5, 10, 50, 100, 199])
        sub = ds.subset(idx, output=tmp_path / "sub.cytome")
        sub_mat = sub.RNA.layer("counts").to_memory().toarray()
        np.testing.assert_array_equal(sub_mat, expected[idx])
        assert sub.n_cells == len(idx)
        sub.close()
        ds.close()

    def test_subset_by_mask(self, tmp_path):
        ds, expected, _ = _make_ds(tmp_path / "a.cytome")
        mask = np.zeros(200, dtype=bool)
        mask[[0, 1, 2]] = True
        sub = ds.subset(mask, output=tmp_path / "sub.cytome")
        assert sub.n_cells == 3
        sub.close()
        ds.close()

    def test_subset_preserves_embeddings(self, tmp_path):
        ds, _, expected_emb = _make_ds(tmp_path / "a.cytome")
        idx = np.array([10, 20, 30])
        sub = ds.subset(idx, output=tmp_path / "sub.cytome")
        sub_emb = sub.embeddings["RNA_obsm_X_pca"]
        np.testing.assert_array_almost_equal(sub_emb, expected_emb[idx])
        sub.close()
        ds.close()
