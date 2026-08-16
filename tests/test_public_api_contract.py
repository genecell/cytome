"""The public API surface downstream packages are allowed to depend on.

PIASO requires cytome and reads features through the modality routing here.
Before 0.2.3 it reached into ``cytome.utils.modality`` -- an internal path on
which cytome promised nothing -- so reshaping ``MODALITY_REGISTRY`` would have
broken PIASO silently, at a distance, in a release nobody connected to the
change.

This file is the contract, and it is deliberately the *primary* protection
rather than a version ceiling in PIASO. A ceiling in the downstream package
only survives a break: it pins users to an old cytome after the damage is done,
and it constrains environments we do not control. A test here fails in CI
*before* the release goes out, which is the protection worth having when the
same people maintain both packages.

If a change makes something below fail, that is the signal to bump the major
version and coordinate with PIASO -- not to edit the expectation.
"""
from __future__ import annotations

import inspect

import pytest

import cytome


# --------------------------------------------------------------------------
# Names promoted to the top level in 0.2.3. Removing or renaming any of these
# breaks `import piaso`'s feature resolver.
# --------------------------------------------------------------------------
PUBLIC_MODALITY_API = [
    "MODALITY_REGISTRY",
    "MODALITY_VAR_ENTITY",
    "modality_var_entity",
    "modality_feature_table_info",
    "modality_has_feature",
    "read_feature_column",
    "read_feature_columns",
    "modality_cell_depth",
]


@pytest.mark.parametrize("name", PUBLIC_MODALITY_API)
def test_modality_api_is_importable_from_the_top_level(name):
    assert hasattr(cytome, name), (
        f"cytome.{name} is gone. PIASO imports it from the top level; "
        "removing it is a major-version change."
    )
    assert name in cytome.__all__, (
        f"cytome.{name} exists but is missing from __all__, so it reads as "
        "private. Public API must be declared."
    )


def test_registry_entry_shape_is_four_tuples():
    """``(modality, entity_table, idx_col, id_columns)``.

    PIASO's plot side projects these to 3-tuples. If a fifth field is ever
    appended, that projection silently starts unpacking the wrong thing.
    """
    assert cytome.MODALITY_REGISTRY, "registry is empty"
    for entry in cytome.MODALITY_REGISTRY:
        assert isinstance(entry, tuple) and len(entry) == 4, (
            f"registry entry {entry!r} is not a 4-tuple. Downstream unpacking "
            "is positional; changing the arity is a major-version change."
        )
        modality, entity, idx_col, id_cols = entry
        assert isinstance(modality, str) and modality
        assert isinstance(entity, str) and entity
        assert isinstance(idx_col, str) and idx_col.endswith("_idx")
        assert isinstance(id_cols, tuple) and id_cols


def test_rna_is_first_for_auto_detect():
    """Order is semantics, not style.

    Callers that auto-detect which modality holds a feature iterate the
    registry in order. RNA first means a bare gene name resolves to expression
    rather than to gene activity.
    """
    assert cytome.MODALITY_REGISTRY[0][0] == "RNA"
    assert [e[0] for e in cytome.MODALITY_REGISTRY][:4] == [
        "RNA", "GA", "ATAC", "tiles",
    ]


@pytest.mark.parametrize("name,required", [
    ("modality_has_feature", ["ds", "modality", "feature"]),
    ("read_feature_column", ["ds", "modality"]),
    ("read_feature_columns", ["ds", "modality"]),
    ("modality_cell_depth", ["ds", "modality"]),
    ("modality_feature_table_info", ["ds", "modality"]),
])
def test_helper_signatures_keep_their_leading_parameters(name, required):
    """Callers pass these positionally; reordering breaks them silently."""
    params = list(inspect.signature(getattr(cytome, name)).parameters)
    assert params[:len(required)] == required, (
        f"cytome.{name} leading parameters changed from {required} to "
        f"{params[:len(required)]}. These are passed positionally downstream."
    )


# --------------------------------------------------------------------------
# Shared internals with a stable import path. Not user API -- not in __all__ --
# but PIASO's fragment and peak quantification imports them, so the path and
# the names are a contract too. They moved here from PIASO in 0.2.3.
# --------------------------------------------------------------------------
CHUNKING_SYMBOLS = [
    "MAX_CHUNK_FILES",
    "DEFAULT_CHUNK_SIZE",
    "_compute_chunk_params",
    "ChunkBucketWriter",
    "_process_chunk",
    "_assemble_csr",
    "_parse_peak_metadata",
    "_parse_feature_metadata",
    "_write_chunks_to_cytome",
]


@pytest.mark.parametrize("name", CHUNKING_SYMBOLS)
def test_chunking_symbols_stay_importable(name):
    from cytome.io import chunking

    assert hasattr(chunking, name), (
        f"cytome.io.chunking.{name} is gone. PIASO imports it "
        "(piaso.preprocessing._streaming_io re-exports these); moving or "
        "renaming it needs a coordinated PIASO change."
    )


def test_chunking_does_not_import_piaso():
    """The reverse dependency this move existed to delete.

    cytome importing PIASO made `pip install cytome` alone produce a package
    whose fragment-tiling path raised ImportError. If it ever comes back, the
    two packages are mutually dependent again.
    """
    import pathlib

    import cytome as _c

    root = pathlib.Path(_c.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{i}"
        for path in root.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if line.strip().startswith(("import piaso", "from piaso"))
    ]
    assert not offenders, (
        "cytome imports piaso at: " + ", ".join(offenders)
        + ". The dependency points cytome <- piaso; it must not point back."
    )


def test_compute_chunk_params_bounds_the_file_count():
    """The one behavioural invariant callers rely on: bounded open files."""
    from cytome.io.chunking import MAX_CHUNK_FILES, _compute_chunk_params

    for n_cells in (1, 1_000, 50_000, 4_100_000):
        chunk_size, n_chunks = _compute_chunk_params(n_cells)
        assert n_chunks <= MAX_CHUNK_FILES, (
            f"{n_cells} cells produced {n_chunks} chunks, above the "
            f"{MAX_CHUNK_FILES} cap -- this is what exhausts file descriptors "
            "on atlas-scale runs."
        )
        assert chunk_size * n_chunks >= n_cells, "chunks do not cover all cells"


def test_from_10x_h5_runs(tmp_path):
    """It referenced an undefined name, so every call raised NameError.

    0.2.3 shipped `from_10x_h5` with a `modalities` branch copied from
    `from_cellranger` but no such parameter, so the function was unusable for
    every input. Nothing caught it because no test called it -- checking that a
    function *exists* is not checking that it runs.
    """
    import numpy as np
    from scipy import sparse as sp

    import cytome

    h5 = tmp_path / "tiny.h5"
    X = sp.csc_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], dtype=np.int32))
    import h5py

    with h5py.File(h5, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=X.data)
        g.create_dataset("indices", data=X.indices.astype(np.int64))
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(X.shape, dtype=np.int64))
        g.create_dataset("barcodes", data=np.array([b"C1", b"C2", b"C3"]))
        fg = g.create_group("features")
        fg.create_dataset("id", data=np.array([b"G1", b"G2", b"chr1:1-2", b"chr1:5-9"]))
        fg.create_dataset("name", data=np.array([b"G1", b"G2", b"chr1:1-2", b"chr1:5-9"]))
        fg.create_dataset("feature_type", data=np.array(
            [b"Gene Expression", b"Gene Expression", b"Peaks", b"Peaks"]))
        fg.create_dataset("genome", data=np.array([b"test"] * 4))

    for mods in ("both", "rna", "atac"):
        ds = cytome.from_10x_h5(h5, tmp_path / f"{mods}.cytome",
                                modalities=mods, force=True)
        mods_on_file = set(ds.modalities)
        ds.close()
        if mods == "rna":
            assert "ATAC" not in mods_on_file
        elif mods == "atac":
            assert "RNA" not in mods_on_file

    with pytest.raises(ValueError, match="modalities must be"):
        cytome.from_10x_h5(h5, tmp_path / "bad.cytome", modalities="nope", force=True)
