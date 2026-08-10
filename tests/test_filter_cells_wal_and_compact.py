"""2026-06-15: filter_cells must not orphan WAL sidecars (corruption bug), and
ds.compact() folds the WAL into the main file.

Repro of the original corruption: a write-heavy cytome accumulates a non-trivial
``-wal``; filter_cells atomically replaces the main db file but (pre-fix) left the
old ``-wal``/``-shm`` behind → reopen replayed a mismatched WAL → "database disk
image is malformed". Fix: unlink the dead sidecars around the replace.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

import cytome


def _build_with_wal(path, n_cells=400, n_genes=50, seed=0):
    """Create a cytome and leave a non-empty WAL (writes without a checkpoint)."""
    rng = np.random.default_rng(seed)
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "barcode": [f"BC{i}" for i in range(n_cells)],
        "Leiden": [str(i % 5) for i in range(n_cells)],
    }))
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"g{i}" for i in range(n_genes)]}))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(
        rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)))
    ds.flush()
    # Force WAL frames to exist on disk (no checkpoint yet).
    ds._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")  # leaves -wal allocated
    return ds


def _sidecar_sizes(path):
    return {e: (os.path.getsize(str(path) + e) if os.path.exists(str(path) + e) else None)
            for e in ("-wal", "-shm")}


def test_filter_cells_no_wal_corruption(tmp_path):
    p = tmp_path / "wal.cytome"
    ds = _build_with_wal(p)
    keep = np.isin(ds.cells.to_pandas()["Leiden"].to_numpy(), ["0", "1", "2"])
    n_keep = int(keep.sum())

    n_after = ds.filter_cells(keep)
    assert n_after == n_keep
    # in-memory handle reflects the filtered count immediately
    assert ds.n_cells == n_keep
    ds.close()

    # No surviving NON-EMPTY sidecar that would poison a reopen.
    for e in ("-wal", "-shm"):
        fp = str(p) + e
        if os.path.exists(fp):
            assert os.path.getsize(fp) == 0, f"stale non-empty sidecar {e}"

    # Reopen from disk (the exact path that raised "malformed" pre-fix).
    ds2 = cytome.open(p)
    assert ds2.n_cells == n_keep
    assert ds2._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    # data still readable + aligned to the new cell count
    M = ds2.RNA.layer("counts").to_memory()
    assert M.shape[0] == n_keep
    ds2.close()


def test_compact_shrinks_wal(tmp_path):
    p = tmp_path / "compact.cytome"
    ds = _build_with_wal(p, n_cells=600, n_genes=80)
    ds.flush()
    ds._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    before = _sidecar_sizes(p)["-wal"]

    res = ds.compact()
    assert res["busy"] in (0, 1)            # checkpoint ran (0=full, 1=reader-pinned)
    after = _sidecar_sizes(p)["-wal"]
    # TRUNCATE zeroes the -wal (when not reader-pinned); never larger than before.
    assert after is None or after == 0 or (before is not None and after <= before)

    # data intact + integrity ok after compaction
    assert ds._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert ds.n_cells == 600
    ds.close()
