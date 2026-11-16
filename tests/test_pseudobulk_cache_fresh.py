"""A cached pseudobulk cube is reused only when it still describes the cells, with today's depths.

Three ways a stale cube was served (audit 2026-11-07, Section 4.5):
- the label hash looked at every 2,048th cell only, so relabelling cells between the sampled ones kept it;
- the cache key ignored min_cells, so a cube cached with min_cells=1 came back for min_cells=3;
- a cube marginalized from a cached superset got the squared depths of a fresh grouping by position, though
  the two list their groups in different orders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import cytome
from cytome.core.pseudobulk import aggregate_pseudobulk


def _store(path, cells: pd.DataFrame, n_genes=3, X=None):
    ds = cytome.create(path)
    ds.set_entity("cells", cells)
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"G{i}" for i in range(n_genes)]}))
    ds.flush()
    if X is None:
        X = np.random.default_rng(0).poisson(2.0, (len(cells), n_genes))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.asarray(X, dtype=np.float32)))
    ds.flush()
    return ds


def test_relabelling_cells_between_the_old_hash_samples_refreshes_the_cube(tmp_path):
    n = 5000                                            # the old hash read every 2nd cell
    ds = _store(tmp_path / "a.cytome", pd.DataFrame({"barcode": [f"c{i}" for i in range(n)],
                                                     "cell_type": ["a"] * n}))
    assert aggregate_pseudobulk(ds, modality="RNA", by="cell_type").n_cells.tolist() == [n]
    labels = np.array(["a"] * n, dtype=object)
    labels[[1, 3, 5, 7]] = "b"                          # cells the old hash never read
    ds.cells["cell_type"] = labels
    ds.flush()
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")
    assert dict(zip(cube.group_names, cube.n_cells.tolist())) == {"a": n - 4, "b": 4}
    ds.close()


def test_a_cube_cached_with_another_min_cells_is_not_reused(tmp_path):
    ds = _store(tmp_path / "b.cytome", pd.DataFrame({"barcode": [f"c{i}" for i in range(7)],
                                                     "cell_type": list("aaaaabb")}))
    assert aggregate_pseudobulk(ds, modality="RNA", by="cell_type").group_names == ["a", "b"]
    assert aggregate_pseudobulk(ds, modality="RNA", by="cell_type", min_cells=3).group_names == ["a"]
    ds.close()


def test_a_marginalized_superset_gets_each_groups_own_depths(tmp_path):
    # Superset groups, sorted: (A, y), (B, x). Marginalized to cell_type they come out as [y, x]; a fresh
    # grouping by cell_type sorts them [x, y].
    cells = pd.DataFrame({
        "barcode": [f"c{i}" for i in range(4)],
        "region": ["A", "A", "B", "B"],
        "cell_type": ["y", "y", "x", "x"],
        "n_fragments": [100.0, 300.0, 1000.0, 3000.0],
    })
    X = np.array([[1, 0, 0], [1, 1, 0], [3, 0, 0], [2, 2, 0]])            # cells' totals 1, 2, 3, 4
    ds = _store(tmp_path / "c.cytome", cells, X=X)
    aggregate_pseudobulk(ds, modality="RNA", by=["region", "cell_type"])        # cached superset
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")             # marginalized from it
    frags = dict(zip(cube.group_names, cube.total_fragments.tolist()))
    tot = dict(zip(cube.group_names, cube.total_counts.tolist()))
    tot_sq = dict(zip(cube.group_names, cube.total_counts_sq.tolist()))
    assert frags == {"y": 400.0, "x": 4000.0}
    assert tot == {"y": 3.0, "x": 7.0}
    assert tot_sq == {"y": 1.0 + 4.0, "x": 9.0 + 16.0}
    ds.close()


def test_a_superset_cube_that_lacks_cells_is_not_marginalized(tmp_path):
    # A cube over (cell_type, region) leaves out the cells with no region. Marginalized to cell_type it
    # would count 3 "a" cells where there are 4: the grouping asked for is computed afresh instead.
    cells = pd.DataFrame({
        "barcode": [f"c{i}" for i in range(6)],
        "cell_type": ["a", "a", "a", "a", "b", "b"],
        "region": ["A", "A", "B", None, "A", "B"],
    })
    X = np.array([[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]])
    ds = _store(tmp_path / "d.cytome", cells, X=X)
    sup = aggregate_pseudobulk(ds, modality="RNA", by=["cell_type", "region"])  # cached; 5 of the 6 cells
    assert int(sup.n_cells.sum()) == 5
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")
    fresh = aggregate_pseudobulk(ds, modality="RNA", by="cell_type", cache=False)
    assert dict(zip(cube.group_names, cube.n_cells.tolist())) == {"a": 4, "b": 2}
    assert dict(zip(cube.group_names, cube.sum[:, 0].tolist())) == {"a": 15.0, "b": 48.0}
    assert np.array_equal(cube.sum, fresh.sum) and np.array_equal(cube.n_cells, fresh.n_cells)
    ds.close()


def test_a_superset_cube_missing_a_whole_group_is_not_marginalized(tmp_path):
    cells = pd.DataFrame({"barcode": [f"c{i}" for i in range(4)], "cell_type": ["a", "a", "b", "b"],
                          "region": ["A", "B", None, None]})
    ds = _store(tmp_path / "e.cytome", cells, X=np.array([[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]]))
    aggregate_pseudobulk(ds, modality="RNA", by=["cell_type", "region"])        # holds no "b" cell
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")
    assert dict(zip(cube.group_names, cube.sum[:, 0].tolist())) == {"a": 3.0, "b": 12.0}
    ds.close()


def test_a_complete_superset_cube_is_still_marginalized(tmp_path, monkeypatch):
    import cytome.core.pseudobulk as P
    cells = pd.DataFrame({"barcode": [f"c{i}" for i in range(4)], "cell_type": ["a", "a", "b", "b"],
                          "region": ["A", "B", "A", "B"]})
    ds = _store(tmp_path / "f.cytome", cells, X=np.array([[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]]))
    aggregate_pseudobulk(ds, modality="RNA", by=["cell_type", "region"])
    calls = []
    real = P.PseudobulkCube.marginalize
    monkeypatch.setattr(P.PseudobulkCube, "marginalize", lambda self, cols: calls.append(cols) or real(self, cols))
    cube = aggregate_pseudobulk(ds, modality="RNA", by="cell_type")
    assert calls == [["cell_type"]]                      # served from the cached superset
    assert dict(zip(cube.group_names, cube.sum[:, 0].tolist())) == {"a": 3.0, "b": 12.0}
    assert ds._conn.execute("SELECT COUNT(*) FROM pseudobulk_cubes").fetchone()[0] == 1
    ds.close()


def test_a_matrix_written_again_is_not_served_from_its_old_cube(tmp_path):
    # the same shape and the same non-zero pattern, other values: a normalized layer computed again
    cells = pd.DataFrame({"barcode": [f"c{i}" for i in range(4)], "cell_type": ["a", "a", "b", "b"]})
    X = np.array([[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], dtype=np.float32)
    ds = _store(tmp_path / "g.cytome", cells, X=X)
    assert aggregate_pseudobulk(ds, modality="RNA", by="cell_type").sum[:, 0].tolist() == [3.0, 12.0]
    ds.add_matrix("RNA_counts", sp.csr_matrix(X * 2))
    ds.flush()
    assert aggregate_pseudobulk(ds, modality="RNA", by="cell_type").sum[:, 0].tolist() == [6.0, 24.0]
    # and the new cube is the one kept
    assert aggregate_pseudobulk(ds, modality="RNA", by="cell_type").sum[:, 0].tolist() == [6.0, 24.0]
    ds.close()
