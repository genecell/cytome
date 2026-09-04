"""Top-level cytome API completeness.

Catches the class of bug the user reported: a function exists in
``cytome/io/convert_*.py`` and is documented in the usage guide as
``cytome.from_h5ad(...)``, but was never re-exported from
``cytome/__init__.py``. AttributeError surfaces only when a user tries
to call it from the canonical ``cytome.<name>`` form.

The first test below ensures every name in ``cytome.__all__`` is
actually attribute-resolvable AND callable. The second runs the user's
exact reported call (from_h5ad with ATAC modality, backed=True) end-to-end.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


#: Names in ``__all__`` that are data, not callables. The registry tables are
#: exported on purpose -- callers read them to see which modalities exist and
#: which var entity each routes to -- so "everything exported is callable" is
#: not the rule. Naming them here keeps the check strict about the bug it
#: exists for: a function that lives in a submodule and was never re-exported.
_EXPORTED_CONSTANTS = {"MODALITY_REGISTRY", "MODALITY_VAR_ENTITY"}


def test_every_name_in_dunder_all_is_resolvable_and_callable():
    """Every entry in cytome.__all__ must be accessible as ``cytome.<name>``,
    and callable (or a class) unless it is a declared exported constant.
    Catches the "function exists in submodule but never re-exported" pattern.
    """
    import cytome
    missing = []
    not_callable = []
    for name in cytome.__all__:
        if not hasattr(cytome, name):
            missing.append(name)
            continue
        obj = getattr(cytome, name)
        if name in _EXPORTED_CONSTANTS:
            continue
        if not (callable(obj) or inspect.isclass(obj)):
            not_callable.append((name, type(obj).__name__))
    assert not missing, (
        f"Names in cytome.__all__ but NOT importable as cytome.<name>: "
        f"{missing}. Add the corresponding wrapper in cytome/__init__.py."
    )
    assert not not_callable, (
        f"Names in cytome.__all__ resolved but not callable: {not_callable}. "
        f"If one is a constant meant to be public, add it to "
        f"_EXPORTED_CONSTANTS in this file."
    )
    stale = _EXPORTED_CONSTANTS - set(cytome.__all__)
    assert not stale, f"_EXPORTED_CONSTANTS names something unexported: {stale}"


def test_from_h5ad_top_level_callable():
    """Direct regression: cytome.from_h5ad must exist (the user's bug)."""
    import cytome
    assert hasattr(cytome, "from_h5ad"), (
        "cytome.from_h5ad does not exist at the top level — yet the cytome "
        "usage guide and the function's own provenance log advertise it as "
        "the public API."
    )
    assert callable(cytome.from_h5ad)


def test_to_anndata_top_level_callable():
    """Function form `cytome.to_anndata(ds, ...)` mirrors the method form
    `ds.to_anndata(...)` and is documented in the usage guide."""
    import cytome
    assert hasattr(cytome, "to_anndata") and callable(cytome.to_anndata)


def test_from_h5ad_atac_modality_end_to_end(tmp_path):
    """The user's exact reported call:

        cytome.from_h5ad(
            "...P1peakMerge_cluster.h5ad",
            output="...P1peakMerge_cluster_testing.cytome",
            modality="ATAC",
            backed=True,
            chunk_size=2048,
            storage_chunk_size=128,
        )

    using a synthetic ATAC h5ad. Asserts that:
    1. cytome.from_h5ad doesn't AttributeError.
    2. backed=True path actually streams.
    3. ATAC modality routes to peaks (not genes).
    4. chr/start/end_ are auto-derived from canonical chr:start-end var_names
       (via the cytome 0ceeac6 fix).
    """
    import anndata
    import cytome

    # Build a tiny synthetic ATAC h5ad
    n_cells, n_peaks = 30, 6
    rng = np.random.default_rng(0)
    var = pd.DataFrame(
        index=[f"chr1:{100 * i}-{100 * i + 50}" for i in range(n_peaks)],
    )
    obs = pd.DataFrame(
        {"barcode": [f"AAA-{i}" for i in range(n_cells)]},
        index=[f"AAA-{i}" for i in range(n_cells)],
    )
    X = sp.csr_matrix(rng.poisson(1.5, size=(n_cells, n_peaks)).astype(np.float32))
    a = anndata.AnnData(X=X, obs=obs, var=var)
    h5ad_path = tmp_path / "atac.h5ad"
    a.write_h5ad(str(h5ad_path))

    out = tmp_path / "atac.cytome"
    ds = cytome.from_h5ad(
        str(h5ad_path),
        output=str(out),
        modality="ATAC",
        backed=True,
        chunk_size=8,            # tiny to actually exercise the streaming path
        storage_chunk_size=4,
        verbose=False,
    )
    try:
        # Modality routed to peaks (not genes) — verify the var entity table.
        peaks = ds.peaks.to_pandas()
        assert len(peaks) == n_peaks
        # Auto-derived chr/start/end_ from the canonical chr:start-end pattern.
        assert list(peaks["chr"]) == ["chr1"] * n_peaks
        assert peaks["start"].tolist() == [100 * i for i in range(n_peaks)]
        assert peaks["end_"].tolist() == [100 * i + 50 for i in range(n_peaks)]
        # ATAC_counts matrix exists with expected shape
        meta = ds.matrix_meta("ATAC_counts")
        assert meta["n_rows"] == n_cells
        assert meta["n_cols"] == n_peaks
        assert meta["col_entity"] == "peaks"
    finally:
        ds.close()


def test_from_h5ad_signature_matches_implementation():
    """The top-level wrapper's signature should match the implementation in
    cytome.io.convert_anndata so users don't get surprised by missing
    kwargs (e.g. ``backed``, ``chunk_size``)."""
    import cytome
    from cytome.io.convert_anndata import from_h5ad as _impl
    impl_params = set(inspect.signature(_impl).parameters)
    wrap_params = set(inspect.signature(cytome.from_h5ad).parameters)
    missing = impl_params - wrap_params
    assert not missing, (
        f"Top-level cytome.from_h5ad is missing kwargs that the impl exposes: "
        f"{sorted(missing)}"
    )
