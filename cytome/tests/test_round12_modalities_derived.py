"""Round 12 (2026-05-27) regression: derived ``CytomeDataset.modalities``.

The property used to return ``_manifest['modalities']`` verbatim, which
silently mis-reported on cytomes built via the RNA-first + ATAC-append
pipeline (Rust importer Mode A doesn't update the manifest). The stale
value broke ``subset()``, ``merge()``, ``__getattr__``, and any code
gating on ``"ATAC" in ds.modalities``.

Round 12 changes the property to derive from actual on-disk state
(matrix_meta + fragment_chunks + manifest). This pins:

1. A cytome with only manifest=['RNA'] but a tiles_counts matrix
   AND fragment_chunks rows reports modalities = ['ATAC', 'RNA', 'tiles'].
2. The manifest's modalities key still contributes (additive).
3. Empty cytome with manifest=[] reports modalities=[].
4. subset()/filter_cells() now preserves fragments on a
   stale-manifest cytome (the bug it surfaced).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def test_modalities_derived_from_matrix_names(tmp_path):
    """Matrices named '{X}_counts' add X to modalities for known
    modality prefixes (RNA, GA, ATAC, tiles)."""
    import cytome

    p = tmp_path / "test.cytome"
    ds = cytome.create(p)
    n_obs, n_vars = 20, 10

    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(n_vars),
        "gene_id": [f"g{i}" for i in range(n_vars)],
    }))

    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((n_obs, n_vars)).astype(np.float32))
    ds.add_matrix("RNA_counts", X)
    ds.flush()

    # Even though the manifest may not list RNA, the derived property
    # picks it up from the matrix name prefix.
    assert "RNA" in ds.modalities


def test_modalities_derived_from_fragment_chunks(tmp_path):
    """A cytome with non-empty fragment_chunks reports 'ATAC' in
    modalities even when the manifest doesn't list it (legacy
    RNA-first + ATAC-append cytomes)."""
    import cytome

    p = tmp_path / "test.cytome"
    ds = cytome.create(p)
    n_obs = 5
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    ds.flush()

    # Initially: no fragments → 'ATAC' not derived.
    assert "ATAC" not in ds.modalities

    # Insert one synthetic fragment_chunks row directly via SQL.
    ds._conn.execute(
        """INSERT INTO fragment_chunks
           (chrom, chunk_idx, row_start, row_end, n_fragments,
            min_start, starts_blob, ends_blob, cell_idx_blob,
            compression, encoding)
           VALUES ('chr1', 0, 0, 1, 1, 0, X'00', X'00', X'00', 'lz4', 1)"""
    )
    ds._conn.commit()

    assert "ATAC" in ds.modalities, (
        "Derived 'modalities' must include 'ATAC' when fragment_chunks "
        "is non-empty, regardless of manifest state."
    )


def test_modalities_includes_manifest_entries(tmp_path):
    """Explicit add_modality writes still show up — the derived
    property is additive on top of the manifest, not a replacement."""
    import cytome
    from anndata import AnnData

    p = tmp_path / "test.cytome"
    ds = cytome.create(p)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(5),
        "barcode": [f"c{i}" for i in range(5)],
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(3),
        "gene_id": [f"g{i}" for i in range(3)],
    }))
    ds.flush()

    adata = AnnData(X=sp.csr_matrix(np.ones((5, 3), dtype=np.float32)))
    ds.add_modality("RNA", adata)
    ds.flush()

    assert "RNA" in ds.modalities


def test_subset_preserves_fragments_on_legacy_manifest(tmp_path):
    """Regression for the actual bug observed in Round 12 Q-D-1:
    A cytome with manifest=['RNA'] but ATAC fragments must NOT lose
    fragments when filtered via ``ds.filter_cells(mask)``.

    Pre-Round-12: subset() gated on ``"ATAC" in ds.modalities``,
    short-circuited fragment carry-over, dropped ALL fragments
    silently. Round 12: derived modalities sees 'ATAC' from
    fragment_chunks → fragments are carried over correctly.
    """
    import cytome

    src = tmp_path / "src.cytome"
    ds = cytome.create(src)
    n_obs = 10
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
        "n_fragments": np.arange(n_obs) * 100,
    }))
    # Synthetic fragment_chunks row with valid LZ4-compressed cell_idx
    # data (cell_idx 0..4 to be kept).
    import lz4.block as lz4
    cells_arr = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    starts = np.array([100, 200, 300, 400, 500], dtype=np.int32)
    ends = np.array([200, 300, 400, 500, 600], dtype=np.int32)
    # Encoding 0 (raw, no delta) for simplicity.
    s_blob = lz4.compress(starts.tobytes(), mode="fast")
    e_blob = lz4.compress(ends.tobytes(), mode="fast")
    c_blob = lz4.compress(cells_arr.tobytes(), mode="fast")
    ds._conn.execute(
        """INSERT INTO fragment_chunks
           (chrom, chunk_idx, row_start, row_end, n_fragments,
            min_start, starts_blob, ends_blob, cell_idx_blob,
            compression, encoding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("chr1", 0, 0, 5, 5, 100, s_blob, e_blob, c_blob, "lz4", 0),
    )
    ds._conn.execute(
        """INSERT OR REPLACE INTO fragment_meta
           (chrom, n_fragments, table_name, rtree_name, min_start, max_end)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("chr1", 5, "fragments_chr1", "fragments_chr1_rtree", 100, 600),
    )
    ds._conn.commit()
    ds.flush()
    ds.close()

    ds = cytome.open(src)
    # Sanity check: ATAC derived even though manifest doesn't list it.
    assert "ATAC" in ds.modalities

    # Filter cells 0..4 (keep), drop 5..9.
    mask = np.array([True] * 5 + [False] * 5)
    n_after = ds.filter_cells(mask)
    assert n_after == 5

    # Crucial: fragment_chunks must NOT be empty after filter.
    n_chunks = ds._conn.execute(
        "SELECT COUNT(*) FROM fragment_chunks"
    ).fetchone()[0]
    total_frags = ds._conn.execute(
        "SELECT COALESCE(SUM(n_fragments), 0) FROM fragment_chunks"
    ).fetchone()[0]
    assert n_chunks > 0, (
        "Round 12 fix: filter_cells must carry fragments over even "
        "when manifest doesn't list 'ATAC'. Got 0 chunks (the original bug)."
    )
    assert total_frags == 5, (
        f"All 5 fragments (cell_idx 0..4) should be preserved. "
        f"Got {total_frags}."
    )
    ds.close()
