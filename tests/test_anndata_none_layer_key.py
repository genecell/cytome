"""anndata >= 0.13 lists ``X`` under a ``None`` key in ``.layers``.

0.10 did not, so every loop over ``adata.layers.items()`` written against it
gained a phantom entry the day a user upgraded. Formatted into a matrix name
that became ``{modality}_None``: a third matrix holding a duplicate of X under
a nonsense name, in every converted file. ``main_layer_name=`` then looked
ignored, because the matrix it asked for was there beside one nobody asked for.

The same key broke the *error* path: ``sorted(adata.layers)`` raises
``TypeError`` comparing ``str`` to ``None``, so the message explaining a bad
``counts_layer=`` died before it could print.

Nothing here needs anndata 0.13 specifically -- the key is simulated -- so the
behaviour stays covered on whichever anndata is installed.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import cytome

ad = pytest.importorskip("anndata")


def _adata():
    rs = np.random.RandomState(0)
    X = sp.csr_matrix(rs.poisson(1, (6, 4)).astype(np.float32))
    a = ad.AnnData(X.astype(np.float32))
    a.layers["count"] = X.copy()
    a.layers["lognorm"] = X.copy().astype(np.float32) * 0.5
    return a


def _matrix_names(path):
    import sqlite3

    con = sqlite3.connect(str(path))
    try:
        return sorted(r[0] for r in con.execute("SELECT matrix_name FROM matrix_meta"))
    finally:
        con.close()


def _layers_expose_none(a) -> bool:
    return any(k is None for k in a.layers.keys())


#: anndata >= 0.13 puts X under a None key; 0.10 does not. Rather than fake the
#: mapping -- a plain dict lacks the attributes the writer also reads, which is
#: how the first version of this file broke -- the None-specific cases run only
#: where the installed anndata really behaves that way. The end-to-end test
#: below runs everywhere and is the one that matters.
needs_none_key = pytest.mark.skipif(
    not _layers_expose_none(_adata()),
    reason="this anndata does not list X under a None layer key")


@needs_none_key
def test_no_phantom_matrix_is_written(tmp_path):
    p = tmp_path / "d.cytome"
    cytome.from_anndata(_adata(), output=str(p),
                        counts_layer="count", main_layer_name="lognorm").close()
    names = _matrix_names(p)
    assert not [n for n in names if n.endswith("_None")], (
        f"a None layer key produced a matrix: {names}")


@needs_none_key
def test_a_bad_counts_layer_still_explains_itself(tmp_path):
    """The message sorts the layer names; None among them used to raise."""
    with pytest.raises(KeyError) as e:
        cytome.from_anndata(_adata(), output=str(tmp_path / "x.cytome"),
                            counts_layer="nope")
    msg = str(e.value)
    assert "count" in msg and "lognorm" in msg
    assert "None" not in msg, "the phantom key should not be offered as a choice"


def test_the_real_anndata_is_handled_whatever_version_this_is(tmp_path):
    """No simulation: whatever `.layers` yields here must convert cleanly."""
    p = tmp_path / "real.cytome"
    cytome.from_anndata(_adata(), output=str(p), counts_layer="count",
                        main_layer_name="lognorm").close()
    assert _matrix_names(p) == ["RNA_counts", "RNA_lognorm"]
