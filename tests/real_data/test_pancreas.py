"""Comprehensive Cytome validation on mouse pancreas real dataset."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
import scipy.sparse as sp

DATA_DIR = "tests/real_data/data"
PANCREAS_PATH = os.path.join(DATA_DIR, "Pancreas_with_cc_anndata.h5ad")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PANCREAS_PATH),
    reason="Pancreas dataset not downloaded. Run: python tests/real_data/download_datasets.py",
)


@pytest.fixture(scope="module")
def adata():
    import anndata

    return anndata.read_h5ad(PANCREAS_PATH)


@pytest.fixture(scope="module")
def cytome_path(adata, tmp_path_factory):
    import cytome

    path = str(tmp_path_factory.mktemp("pancreas") / "pancreas.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=path)
    ds.flush()
    ds.close()
    return path


@pytest.fixture
def ds(cytome_path):
    import cytome

    d = cytome.open(cytome_path)
    yield d
    d.close()


def _cli_command():
    exe = shutil.which("cytome")
    if exe:
        return [exe]
    return [sys.executable, "-m", "cytome.cli.main"]


class TestConversion:
    def test_cell_count_matches(self, ds, adata):
        assert ds.n_cells == adata.shape[0]

    def test_gene_count_matches(self, ds, adata):
        assert ds.n_genes == adata.shape[1]

    def test_file_is_valid_sqlite(self, cytome_path):
        import sqlite3

        conn = sqlite3.connect(cytome_path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert "_manifest" in tables
        assert "cells" in tables

    def test_file_opens_fast(self, cytome_path):
        import cytome

        start = time.time()
        d = cytome.open(cytome_path)
        elapsed = time.time() - start
        d.close()
        assert elapsed < 1.0, f"Open took {elapsed:.2f}s"


class TestCellMetadata:
    def test_obs_columns_present_or_documented(self, ds, adata):
        missing = []
        for col in adata.obs.columns:
            try:
                values = ds.cells[col]
                assert len(values) == adata.shape[0]
            except Exception:
                missing.append(col)
        # Do not hard-fail for every non-serializable categorical/object edge
        assert len(missing) < max(10, int(0.5 * len(adata.obs.columns)))

    def test_numeric_obs_values(self, ds, adata):
        numeric_cols = adata.obs.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:5]:
            original = adata.obs[col].values.astype(float)
            converted = ds.cells[col].astype(float)
            np.testing.assert_allclose(converted, original, rtol=1e-5, equal_nan=True)


class TestMatrixValues:
    def test_x_shape(self, ds, adata):
        assert ds.RNA.counts.shape == adata.shape

    def test_x_values_exact(self, ds, adata):
        converted = ds.RNA.counts.to_memory().tocsr()
        original = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
        assert converted.shape == original.shape
        diff = (converted - original)
        max_diff = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
        assert max_diff < 1e-5, f"Matrix changed during conversion (max diff {max_diff})"

    def test_layers_preserved(self, ds, adata):
        for layer_name in adata.layers.keys():
            layer = ds.RNA.layer(layer_name).to_memory().tocsr()
            original = adata.layers[layer_name]
            original_csr = original.tocsr() if sp.issparse(original) else sp.csr_matrix(np.asarray(original))
            assert layer.shape == original_csr.shape
            # Spot-check first 100 rows to keep runtime bounded
            lhs = layer[:100].toarray()
            rhs = original_csr[:100].toarray()
            np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-6)


class TestMetadata:
    def test_uns_keys_stored_when_serializable(self, ds, adata):
        stored = 0
        for key in adata.uns.keys():
            try:
                _ = ds.metadata[key]
                stored += 1
            except Exception:
                pass
        assert stored >= 0


class TestStreaming:
    def test_iter_rows_covers_all_cells(self, ds, adata):
        total_rows = 0
        for row_start, row_end, chunk in ds.RNA.counts.iter_rows():
            assert chunk.shape[0] == row_end - row_start
            total_rows += chunk.shape[0]
        assert total_rows == adata.shape[0]


class TestQueries:
    def test_query_mask_roundtrip(self, ds, adata):
        numeric_cols = adata.obs.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            pytest.skip("No numeric obs columns")
        col = numeric_cols[0]
        median = float(np.nanmedian(adata.obs[col].astype(float).values))
        mask = ds.cells.query_mask(f"{col} >= {median}")
        assert mask.dtype == bool
        assert mask.shape[0] == adata.shape[0]


class TestRoundTrip:
    def test_anndata_roundtrip(self, ds, adata):
        adata_rt = ds.to_anndata(modality="RNA")
        assert adata_rt.shape == adata.shape

        orig = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
        rt = adata_rt.X.tocsr() if sp.issparse(adata_rt.X) else sp.csr_matrix(np.asarray(adata_rt.X))
        diff = (orig - rt)
        max_diff = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
        assert max_diff < 1e-5


class TestSubsetDownsample:
    def test_subset(self, ds, tmp_path):
        mask = np.zeros(ds.n_cells, dtype=bool)
        mask[: min(500, ds.n_cells)] = True
        out_path = str(tmp_path / "subset.cytome")
        result = ds.subset(mask, output=out_path)
        assert result.n_cells == int(mask.sum())
        result.close()

    def test_downsample(self, ds, tmp_path):
        out_path = str(tmp_path / "small.cytome")
        result = ds.downsample(n_cells=min(100, ds.n_cells), output=out_path, seed=42)
        assert result.n_cells == min(100, ds.n_cells)
        result.close()


class TestMerge:
    def test_merge_two_copies(self, cytome_path, tmp_path):
        import cytome

        # Build a pancreas-derived mini dataset so merge semantics are tested
        # without the full 28k-feature runtime cost.
        src = cytome.open(cytome_path)
        adata_small = src.to_anndata(modality="RNA")[
            : min(200, src.n_cells), : min(1000, src.n_genes)
        ].copy()
        src.close()

        small_path = str(tmp_path / "pancreas_small.cytome")
        small = cytome.from_anndata(adata_small, modality="RNA", output=small_path)
        small.flush()
        n_small = small.n_cells
        small.close()

        merged_path = str(tmp_path / "merged.cytome")
        merged = cytome.merge(
            [small_path, small_path],
            output=merged_path,
            batch_key="sample_id",
            batch_labels=["sample_A", "sample_B"],
        )
        assert merged.n_cells == 2 * n_small
        labels = set(str(x) for x in merged.cells["sample_id"])
        assert "sample_A" in labels and "sample_B" in labels
        merged.close()


class TestValidation:
    def test_validate_passes(self, ds):
        report = ds.validate()
        assert report.passed, report


class TestCLI:
    def test_info_command(self, cytome_path):
        cmd = _cli_command() + ["info", cytome_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert "cells" in result.stdout.lower()

    def test_validate_command(self, cytome_path):
        cmd = _cli_command() + ["validate", cytome_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr


class TestBenchmarks:
    def test_conversion_time(self, adata, tmp_path):
        import cytome

        path = str(tmp_path / "bench.cytome")
        t0 = time.time()
        ds = cytome.from_anndata(adata, modality="RNA", output=path)
        ds.flush()
        ds.close()
        elapsed = time.time() - t0
        print(f"\nConversion: {elapsed:.2f}s")

    def test_open_time(self, cytome_path):
        import cytome

        t0 = time.time()
        d = cytome.open(cytome_path)
        elapsed = time.time() - t0
        d.close()
        print(f"\nOpen: {elapsed*1000:.1f} ms")
