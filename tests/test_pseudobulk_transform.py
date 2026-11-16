"""A cube can hold the sums of a layer normalized on the fly, in its one pass.

COSG scores the sums of the normalized values it reads (log1p, INFOG, TF-IDF), so a cube of raw counts cannot
give COSG's markers. So aggregate_pseudobulk applies a per-chunk transform, which COSG supplies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.core.pseudobulk import aggregate_pseudobulk


def _log_cpm(chunk, rows):
    tot = np.asarray(chunk.sum(axis=1)).ravel()
    scale = np.where(tot > 0, 1e4 / np.maximum(tot, 1e-12), 0.0)
    out = sp.diags(scale) @ chunk
    out.data = np.log1p(out.data)
    return out


def _store(path, n=150, m=20, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.8, (n, m)).astype(np.float32)
    labels = rng.choice(["a", "b", "c"], n)
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(n)], "cell_type": labels}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"G{j}" for j in range(m)]}))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds, X, labels


def test_a_transformed_cube_sums_the_transformed_values(tmp_path):
    ds, X, labels = _store(tmp_path / "a.cytome")
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=_log_cpm, transform_name="log_cpm")
    T = _log_cpm(sp.csr_matrix(X), None).toarray()
    for gi, g in enumerate(cube.group_names):
        m = labels == g
        np.testing.assert_allclose(cube.sum[gi], T[m].sum(axis=0), rtol=1e-5)
        np.testing.assert_allclose(cube.sum_sq[gi], (T[m] ** 2).sum(axis=0), rtol=1e-5)
        np.testing.assert_array_equal(cube.nnz[gi], (T[m] > 0).sum(axis=0))
        assert cube.total_counts[gi] == pytest.approx(X[m].sum())          # the stored counts' totals
    # cached under its own key, apart from the plain cube
    plain = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")
    assert not np.allclose(plain.sum, cube.sum)
    again = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=_log_cpm, transform_name="log_cpm")
    np.testing.assert_allclose(again.sum, cube.sum)
    keys = {r[0] for r in ds._conn.execute("SELECT cube_key FROM pseudobulk_cubes")}
    assert keys == {"RNA_counts::cell_type", "RNA_counts::cell_type::log_cpm"}
    ds.close()


def test_an_unnamed_transform_is_not_cached(tmp_path):
    ds, X, labels = _store(tmp_path / "b.cytome")
    aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=_log_cpm)
    assert ds._conn.execute("SELECT COUNT(*) FROM pseudobulk_cubes").fetchone()[0] == 0
    ds.close()


def _depth_log_cpm(depth):
    """log-CPM with every cell's depth given, so it also normalizes a column chunk (part of each row)."""
    def f(chunk, rows, col_start=None):
        out = sp.diags(1e4 / np.maximum(depth[rows], 1e-12)) @ chunk
        out.data = np.log1p(out.data)
        return out
    return f


def test_a_column_major_matrix_gives_the_cube_of_its_row_major_copy(tmp_path):
    ds, X, labels = _store(tmp_path / "c.cytome")
    f = _depth_log_cpm(X.sum(axis=1).astype(np.float64))
    rows = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=f, cache=False)
    ds.RNA.counts.build_feature_index(chunk_size=6)          # 20 features: chunks of 6, 6, 6 and 2 columns
    ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name = 'RNA_counts'")
    ds._conn.commit()
    cols = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=f, cache=False)
    assert cols.group_names == rows.group_names
    np.testing.assert_allclose(cols.sum, rows.sum, rtol=1e-6)
    np.testing.assert_allclose(cols.sum_sq, rows.sum_sq, rtol=1e-6)
    np.testing.assert_array_equal(cols.nnz, rows.nnz)
    np.testing.assert_allclose(cols.total_counts, rows.total_counts)
    np.testing.assert_allclose(cols.total_counts_sq, rows.total_counts_sq)
    # a transform written for row chunks only is told what a column chunk needs
    with pytest.raises(TypeError, match="col_start"):
        aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=_log_cpm, cache=False)
    ds.close()


def test_a_new_transform_version_recomputes_and_replaces_the_cached_cube(tmp_path):
    ds, X, labels = _store(tmp_path / "d.cytome")
    depth = X.sum(axis=1).astype(np.float64)
    v1 = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=_depth_log_cpm(depth),
                              transform_name="lc", transform_version="depth-1")
    # the statistics change (here, every depth doubles), and so does the fingerprint
    v2 = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=_depth_log_cpm(2 * depth),
                              transform_name="lc", transform_version="depth-2")
    assert not np.allclose(v1.sum, v2.sum)
    rows = ds._conn.execute("SELECT cube_key, digest_json FROM pseudobulk_cubes").fetchall()
    assert [r[0] for r in rows] == ["RNA_counts::cell_type::lc"]                 # one cube, replaced
    assert '"transform_version": "depth-2"' in rows[0][1]
    # the same fingerprint reads the cached cube, even through a transform that would give other sums
    again = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", transform=_depth_log_cpm(depth),
                                 transform_name="lc", transform_version="depth-2")
    np.testing.assert_allclose(again.sum, v2.sum)
    ds.close()
