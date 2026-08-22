"""The `write_*` / `skip_*` flags must mean the same thing on both from_h5ad paths.

`from_h5ad(backed=False)` delegates to `from_anndata`. Before this test,
that delegation passed only `modality`, `output` and `force`, so a caller
asking for a counts-only conversion got every layer, `.raw.X`, obsm, obsp
and uns written anyway -- with no error and no warning, because the
argument was accepted by the signature it never reached. The cost is not
theoretical: a release conversion doubled in size and shipped a
`RNA_raw_X` measurement holding log1p values under a name that reads as
raw counts.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import cytome

anndata = pytest.importorskip("anndata")


def _adata(seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix((rng.random((40, 12)) < 0.3).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.layers["log1p"] = X.copy()
    ad.layers["scaled"] = X.copy()
    ad.obsm["X_pca"] = rng.random((40, 5))
    ad.varm["loadings"] = rng.random((12, 3))
    ad.obsp["connectivities"] = sp.csr_matrix(rng.random((40, 40)) < 0.1)
    ad.varp["cor"] = sp.csr_matrix(rng.random((12, 12)) < 0.1)
    ad.uns["note"] = "keepme"
    ad.raw = anndata.AnnData(X=X.copy())
    return ad


def _matrices(ds):
    return {r[0] for r in ds._conn.execute(
        "SELECT matrix_name FROM matrix_meta").fetchall()}


@pytest.mark.parametrize("backed", [False, True])
def test_write_flags_are_honoured_on_both_paths(tmp_path, backed):
    src = tmp_path / "src.h5ad"
    _adata().write_h5ad(src)

    out = tmp_path / f"counts_only_{backed}.cytome"
    ds = cytome.from_h5ad(
        str(src), output=str(out), modality="RNA", backed=backed,
        write_raw=False, skip_layers=["log1p", "scaled"],
        write_obsm=False, write_obsp=False,
        write_varm=False, write_varp=False, write_uns=False,
        force=True, verbose=False,
    )
    try:
        names = _matrices(ds)
        assert "RNA_counts" in names
        assert "RNA_raw_X" not in names, "write_raw=False still wrote raw.X"
        assert not any(n.endswith(("_log1p", "_scaled")) for n in names), (
            f"skip_layers ignored: {sorted(names)}")
        assert "note" not in ds.metadata, "write_uns=False still wrote uns"
        assert not list(ds.list_embeddings()), "write_obsm=False still wrote obsm"
    finally:
        ds.close()


@pytest.mark.parametrize("backed", [False, True])
def test_defaults_still_write_everything(tmp_path, backed):
    """The flags default to True; turning nothing off must change nothing."""
    src = tmp_path / "src.h5ad"
    _adata().write_h5ad(src)

    out = tmp_path / f"full_{backed}.cytome"
    ds = cytome.from_h5ad(str(src), output=str(out), modality="RNA",
                          backed=backed, force=True, verbose=False)
    try:
        names = _matrices(ds)
        assert "RNA_raw_X" in names
        assert any(n.endswith("_log1p") for n in names)
        assert ds.metadata.get("note") == "keepme"
    finally:
        ds.close()
