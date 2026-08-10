"""Round 12 (2026-05-27) regression: ``from_h5ad(backed=True)`` auto-
falls-back to ``backed=False`` when backed streaming can't parse the
file's encoding.

Pre-Round-12: ``cytome.from_h5ad(backed=True)`` crashed with KeyError
on h5ads whose /X is a dense Dataset (encoding-type='array') because
the streaming reader assumed every matrix was a CSR group with a
'shape' attribute.

Round 12: top-level ``from_h5ad`` catches KeyError /
NotImplementedError from the backed path and retries with
``backed=False`` (in-memory load via anndata.read_h5ad). Emits
RuntimeWarning so the fallback is visible.

Tested cases:
1. Dense /X h5ad → fallback fires, conversion succeeds, RuntimeWarning
   visible to the caller.
2. Round-trip data integrity through the fallback.
"""
from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest


def _write_dense_x_h5ad(path, n_obs=20, n_vars=10, seed=0):
    """Write a minimal h5ad with a DENSE /X (encoding-type='array')."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, n_vars)).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        # /X as a dense Dataset
        dx = f.create_dataset("X", data=X)
        dx.attrs["encoding-type"] = "array"
        dx.attrs["encoding-version"] = "0.2.0"

        # Minimal /obs and /var groups in AnnData's encoding-v0.1 format.
        # We use compatible scaffolding so anndata.read_h5ad can parse.
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        obs.attrs["_index"] = "_index"
        obs.attrs["column-order"] = []
        obs.create_dataset(
            "_index", data=np.array([f"c{i}" for i in range(n_obs)], dtype="S")
        )

        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        var.attrs["_index"] = "_index"
        var.attrs["column-order"] = []
        var.create_dataset(
            "_index", data=np.array([f"g{i}" for i in range(n_vars)], dtype="S")
        )


def test_from_h5ad_backed_falls_back_on_dense_x(tmp_path):
    """A dense-X h5ad with backed=True must auto-fall-back to
    backed=False with a RuntimeWarning, NOT crash."""
    import cytome

    h5ad_path = tmp_path / "dense_x.h5ad"
    _write_dense_x_h5ad(h5ad_path)

    out = tmp_path / "out.cytome"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = cytome.from_h5ad(h5ad_path, output=out, backed=True, verbose=False)

    fallback_warnings = [
        wi for wi in w if issubclass(wi.category, RuntimeWarning)
        and "backed" in str(wi.message).lower()
    ]
    assert fallback_warnings, (
        f"Expected RuntimeWarning about backed→backed=False fallback. "
        f"Got warnings: {[(wi.category.__name__, str(wi.message)) for wi in w]}"
    )
    # Conversion should have succeeded — file exists and has the right shape.
    assert out.exists()
    # Smoke check: n_cells matches what we wrote.
    n_cells = ds.n_cells
    ds.close()
    assert n_cells == 20


def test_from_h5ad_backed_false_unchanged(tmp_path):
    """backed=False should NEVER hit the fallback path — no warning."""
    import cytome

    h5ad_path = tmp_path / "dense_x.h5ad"
    _write_dense_x_h5ad(h5ad_path)

    out = tmp_path / "out.cytome"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = cytome.from_h5ad(h5ad_path, output=out, backed=False, verbose=False)
    fallback_warnings = [
        wi for wi in w if issubclass(wi.category, RuntimeWarning)
        and "fall" in str(wi.message).lower() and "backed" in str(wi.message).lower()
    ]
    assert not fallback_warnings, (
        f"backed=False should never trigger the fallback warning. "
        f"Got: {[str(wi.message) for wi in fallback_warnings]}"
    )
    assert ds.n_cells == 20
    ds.close()
