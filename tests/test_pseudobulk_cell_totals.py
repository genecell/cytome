"""A cube's depth is its own matrix's cell totals, summed and squared by group, for row- and column-major stores.

e1cb92f chose the depth from a cells-table column by name (total_counts, n_counts, n_umi, then n_fragments for
RNA), so an RNA cube on a store without RNA totals was given ATAC fragments. The depth is now each cell's total in the aggregated
matrix, accumulated in the same pass, and its square.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import cytome
from cytome.core.pseudobulk import aggregate_pseudobulk


def _store(path, n=200, m=30, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.gamma(1.0, 1.0, (n, 1)) * 0.6, (n, m)).astype(np.float32)
    cells = pd.DataFrame({"barcode": [f"c{i}" for i in range(n)],
                          "cell_type": rng.choice(["a", "b", "c"], n),
                          "batch": rng.choice(["p", "q"], n),
                          "n_fragments": rng.integers(1000, 5000, n).astype(float)})
    ds = cytome.create(path)
    ds.set_entity("cells", cells)
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"G{j}" for j in range(m)]}))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds, X, cells


def _expected(X, labels, names):
    d = X.sum(axis=1).astype(np.float64)
    return (np.array([d[labels == g].sum() for g in names]), np.array([(d[labels == g] ** 2).sum() for g in names]))


def test_cell_totals_from_row_and_column_chunks(tmp_path):
    ds, X, cells = _store(tmp_path / "a.cytome")
    labels = cells["cell_type"].to_numpy()
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", cache=False)
    tot, tot_sq = _expected(X, labels, cube.group_names)
    np.testing.assert_allclose(cube.total_counts, tot)
    np.testing.assert_allclose(cube.total_counts_sq, tot_sq)
    frags = cells["n_fragments"].to_numpy()
    np.testing.assert_allclose(cube.total_fragments, [frags[labels == g].sum() for g in cube.group_names])

    # the same from column chunks only
    ds.RNA.counts.build_feature_index(chunk_size=7)
    ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name = 'RNA_counts'")
    ds._conn.commit()
    cube_csc = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", cache=False, force=True, threads=3)
    np.testing.assert_allclose(cube_csc.total_counts, tot)
    np.testing.assert_allclose(cube_csc.total_counts_sq, tot_sq)
    np.testing.assert_allclose(cube_csc.sum, cube.sum, rtol=1e-6)
    ds.close()


def test_cell_totals_add_over_marginalization(tmp_path):
    ds, X, cells = _store(tmp_path / "b.cytome")
    joint = aggregate_pseudobulk(ds, modality="RNA", by=["cell_type", "batch"], cache=False)
    direct = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", cache=False)
    marg = joint.marginalize("cell_type")
    assert list(marg.group_names) == list(direct.group_names)
    np.testing.assert_allclose(marg.total_counts, direct.total_counts)
    np.testing.assert_allclose(marg.total_counts_sq, direct.total_counts_sq)
    ds.close()


def test_a_cube_cached_in_the_old_format_is_recomputed(tmp_path):
    import json
    ds, X, cells = _store(tmp_path / "c.cytome")
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")
    # make the cached cube look older (no cube_version) and wrong (its sums zeroed)
    from cytome.io.compression import compress_blob
    digest = json.loads(ds._conn.execute("SELECT digest_json FROM pseudobulk_cubes").fetchone()[0])
    digest.pop("cube_version", None)
    zero = compress_blob(np.zeros_like(cube.sum, dtype=np.float32).tobytes(), "zstd")
    ds._conn.execute("UPDATE pseudobulk_cubes SET digest_json = ?, sum_blob = ?", (json.dumps(digest), zero))
    ds._conn.commit()
    again = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")
    np.testing.assert_allclose(again.sum, cube.sum)
    ds.close()
