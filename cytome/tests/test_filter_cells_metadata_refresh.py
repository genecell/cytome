"""Regression tests for the stale-cache fix in
``CytomeDataset.filter_cells``.

Before v0.2.1, ``filter_cells`` closed and reopened ``self._conn``
but did not reset the cached ``_metadata_obj`` (a ``MetadataStore``
that holds the connection at construction time). Any subsequent
``ds.metadata.get(...)`` raised
``ProgrammingError: cannot operate on a closed database``.

These tests pin the fix (the ``_refresh_after_reopen`` helper and
its wiring at both reopen sites) and the contract that callers
can use ``ds.metadata`` interchangeably before and after
``filter_cells``.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import cytome


def _make_ds(path, n_cells: int = 20, n_genes: int = 8):
    ds = cytome.create(path)
    ds.set_entity("cells", {
        "barcode": [f"c{i}" for i in range(n_cells)],
        "group": np.where(np.arange(n_cells) % 2 == 0, "A", "B"),
    })
    ds.set_entity("genes", {
        "gene_id": [f"g{i}" for i in range(n_genes)],
    })
    ds.add_matrix(
        "RNA_counts",
        sp.random(n_cells, n_genes, density=0.4, format="csr",
                  dtype=np.float32, random_state=1),
    )
    ds.flush()
    return ds


# ---------------------------------------------------------------------------
# 1. The primary regression: metadata.get must not raise after filter_cells
# ---------------------------------------------------------------------------

def test_metadata_get_after_filter_cells_does_not_raise(tmp_path):
    """The exact failure mode from the PIASO inferGA / filter_cells
    audit: write a metadata key, call ``filter_cells``, then read.

    Pre-fix this raised ``ProgrammingError``.
    """
    ds = _make_ds(tmp_path / "x.cytome")
    ds.metadata["probe_key"] = {"foo": "bar"}
    ds.flush()

    mask = np.ones(ds.n_cells, dtype=bool)
    mask[0] = False
    ds.filter_cells(mask)

    # Must not raise.
    val = ds.metadata.get("probe_key")
    assert val == {"foo": "bar"}
    ds.close()


# ---------------------------------------------------------------------------
# 2. Identity check — the accessor must be a fresh instance after reopen
# ---------------------------------------------------------------------------

def test_metadata_accessor_rebound_after_filter_cells(tmp_path):
    """``ds.metadata`` should return a different ``MetadataStore``
    object after ``filter_cells`` — proving the cache was invalidated."""
    ds = _make_ds(tmp_path / "x.cytome")
    # Force construction of the cached accessor.
    before = ds.metadata
    assert before is ds.metadata, (
        "metadata accessor must be cached on first access (sanity)"
    )

    mask = np.ones(ds.n_cells, dtype=bool)
    mask[0] = False
    ds.filter_cells(mask)

    after = ds.metadata
    assert after is not before, (
        "metadata accessor must be a new instance after filter_cells "
        "(stale-cache fix)"
    )
    ds.close()


# ---------------------------------------------------------------------------
# 3. Metadata written before filter_cells survives the atomic replace
# ---------------------------------------------------------------------------

def test_metadata_persists_across_filter_cells(tmp_path):
    """``ds.subset`` copies metadata into the tmp file; the atomic
    replace then promotes it. After filter_cells, the original
    metadata must still read back."""
    ds = _make_ds(tmp_path / "x.cytome")
    ds.metadata["persists"] = {"value": 42}
    ds.metadata["also_persists"] = [1, 2, 3]
    ds.flush()

    mask = np.ones(ds.n_cells, dtype=bool)
    mask[:3] = False
    ds.filter_cells(mask)

    assert ds.metadata.get("persists") == {"value": 42}
    assert ds.metadata.get("also_persists") == [1, 2, 3]
    ds.close()


# ---------------------------------------------------------------------------
# 4. Writes that happen AFTER filter_cells must land
# ---------------------------------------------------------------------------

def test_metadata_writable_after_filter_cells(tmp_path):
    """Set a key only AFTER filter_cells. The fresh MetadataStore
    must be bound to the live connection so the write succeeds."""
    ds = _make_ds(tmp_path / "x.cytome")
    mask = np.ones(ds.n_cells, dtype=bool)
    mask[0] = False
    ds.filter_cells(mask)

    ds.metadata["written_after"] = {"ok": True}
    ds.flush()

    # Re-read from a fresh open to be sure it landed on disk too.
    ds.close()
    ds2 = cytome.open(tmp_path / "x.cytome")
    assert ds2.metadata.get("written_after") == {"ok": True}
    ds2.close()


# ---------------------------------------------------------------------------
# 5. Error rollback path also refreshes the cache
# ---------------------------------------------------------------------------

def test_filter_cells_error_path_also_refreshes(tmp_path, monkeypatch):
    """Monkey-patch ``Path.replace`` so the atomic move raises.
    ``filter_cells`` rolls back by reopening the original connection;
    the rollback path must also reset the cached metadata accessor.
    Otherwise the user's ``ds`` object is left with a dead-conn cache
    after the exception bubbles up.
    """
    from pathlib import Path as _Path
    ds = _make_ds(tmp_path / "x.cytome")
    ds.metadata["before_fail"] = {"value": "still_here"}
    ds.flush()

    # Capture the cached accessor before the failed call.
    before = ds.metadata

    real_replace = _Path.replace

    def _raising_replace(self, target):
        # Only trip on the filter_cells tmp → original swap.
        if str(self).endswith(".filter_tmp"):
            raise OSError("simulated atomic-replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(_Path, "replace", _raising_replace)

    mask = np.ones(ds.n_cells, dtype=bool)
    mask[0] = False
    with pytest.raises(OSError, match="simulated"):
        ds.filter_cells(mask)

    # The rollback reopen happened — accessor must be fresh.
    after = ds.metadata
    assert after is not before, (
        "rollback reopen path must also reset the cached MetadataStore"
    )
    # And the live accessor must work (no ProgrammingError).
    assert ds.metadata.get("before_fail") == {"value": "still_here"}
    ds.close()


# ---------------------------------------------------------------------------
# 6. Manifest sanity — refresh helper also re-reads the manifest
# ---------------------------------------------------------------------------

def test_manifest_refreshed_after_filter_cells(tmp_path):
    """``_refresh_after_reopen`` owns the manifest re-read.
    ``ds.n_cells`` must reflect the post-filter count immediately."""
    ds = _make_ds(tmp_path / "x.cytome", n_cells=20)
    mask = np.ones(20, dtype=bool)
    mask[:5] = False
    n_after = ds.filter_cells(mask)
    assert n_after == 15
    assert ds.n_cells == 15
    ds.close()


# ---------------------------------------------------------------------------
# 7. Other accessors keep working — sanity check the refactor
# ---------------------------------------------------------------------------

def test_other_accessors_unaffected(tmp_path):
    """Non-cached accessors (``cells``, ``provenance``, ``embeddings``)
    were always constructed fresh per-access, so the refactor must
    not have broken them. Defensive smoke test."""
    ds = _make_ds(tmp_path / "x.cytome")
    ds.add_embedding("RNA_pca", np.random.randn(ds.n_cells, 5).astype(np.float32))
    ds.flush()

    mask = np.ones(ds.n_cells, dtype=bool)
    mask[0] = False
    ds.filter_cells(mask)

    # cells: fresh EntityTable per access
    assert ds.cells.to_pandas().shape[0] == ds.n_cells
    # provenance: fresh ProvenanceLog per access
    _ = ds.provenance.show()
    # embeddings: fresh _EmbeddingAccessor per access
    assert "RNA_pca" in ds.embeddings
    arr = ds.embeddings["RNA_pca"]
    assert arr.shape[0] == ds.n_cells
    ds.close()
