"""`{modality}_counts` holds raw integer counts, or it does not exist.

Before 0.3.0, `from_anndata` wrote `adata.X` to `{modality}_counts` whatever
it contained. A file converted from a log-normalised AnnData therefore stored
normalised values under the one name every downstream default reads as raw
counts -- which is how `run_cosg_cytome(layer='auto')` came to log-normalise
an already log-normalised matrix while being documented as equivalent to
`cosg.cosg(adata)`.

A shipped tutorial file ended up with `RNA_count` (raw, from a layer) beside
`RNA_counts` (normalised X): one letter apart, opposite meanings.
"""
import sqlite3

import numpy as np
import pytest
import scipy.sparse as sp


def _matrices(path):
    con = sqlite3.connect(path)
    try:
        return dict(con.execute("SELECT matrix_name, is_integer FROM matrix_meta"))
    finally:
        con.close()


@pytest.fixture
def adatas():
    ad = pytest.importorskip("anndata")
    rs = np.random.RandomState(0)

    def build(normalized_X):
        a = ad.AnnData(X=sp.csr_matrix(rs.poisson(3.0, (60, 12)).astype(np.float32)))
        a.var_names = [f"g{i}" for i in range(12)]
        a.layers["count"] = a.X.copy()
        if normalized_X:
            a.X = sp.csr_matrix(np.log1p(a.X.toarray()))
        return a

    return build


def test_integer_x_still_becomes_counts(tmp_path, adatas):
    cytome = pytest.importorskip("cytome")
    p = str(tmp_path / "a.cytome")
    cytome.from_anndata(adatas(False), output=p).close()
    m = _matrices(p)
    assert "RNA_counts" in m and m["RNA_counts"] == 1


def test_normalized_x_is_not_called_counts(tmp_path, adatas):
    """The invariant. This is the whole release."""
    cytome = pytest.importorskip("cytome")
    p = str(tmp_path / "b.cytome")
    with pytest.warns(UserWarning, match="integer counts"):
        cytome.from_anndata(adatas(True), output=p).close()
    m = _matrices(p)
    assert "RNA_data" in m and m["RNA_data"] == 0
    # the raw layer keeps its own name; nothing claims to be counts
    assert m.get("RNA_counts") is None


def test_counts_layer_puts_the_raw_matrix_under_counts(tmp_path, adatas):
    cytome = pytest.importorskip("cytome")
    p = str(tmp_path / "c.cytome")
    cytome.from_anndata(adatas(True), output=p, counts_layer="count").close()
    m = _matrices(p)
    assert m["RNA_counts"] == 1          # raw, integer
    assert m["RNA_data"] == 0            # X, honestly named
    # and the counts are stored once, not under two near-identical names
    assert "RNA_count" not in m


def test_main_layer_name_is_honoured(tmp_path, adatas):
    cytome = pytest.importorskip("cytome")
    p = str(tmp_path / "d.cytome")
    cytome.from_anndata(adatas(True), output=p, counts_layer="count",
                        main_layer_name="lognorm").close()
    m = _matrices(p)
    assert sorted(m) == ["RNA_counts", "RNA_lognorm"]
    assert m["RNA_counts"] == 1 and m["RNA_lognorm"] == 0


def test_a_false_counts_layer_is_an_error(tmp_path, adatas):
    """Refusing a mislabel is not the same as refusing data.

    The caller asserted "this layer is raw counts". Storing a normalised
    matrix under that name is precisely the bug this release closes, so a
    wrong assertion has to fail loudly rather than be written down.
    """
    cytome = pytest.importorskip("cytome")
    a = adatas(True)
    a.layers["count"] = sp.csr_matrix(np.log1p(a.layers["count"].toarray()))
    with pytest.raises(ValueError, match="does not hold integer values"):
        cytome.from_anndata(a, output=str(tmp_path / "e.cytome"),
                            counts_layer="count")


def test_missing_counts_layer_names_what_exists(tmp_path, adatas):
    cytome = pytest.importorskip("cytome")
    with pytest.raises(KeyError, match="available"):
        cytome.from_anndata(adatas(True), output=str(tmp_path / "f.cytome"),
                            counts_layer="nope")


def test_is_integer_is_recorded_for_every_matrix(tmp_path, adatas):
    """Consumers must not have to re-derive this.

    COSG's own probe was dead code that always answered "cannot tell" while
    looking correct; recording the fact once, at the only moment it is cheap
    and certain, is what stops that repeating.
    """
    cytome = pytest.importorskip("cytome")
    p = str(tmp_path / "g.cytome")
    cytome.from_anndata(adatas(True), output=p, counts_layer="count").close()
    for name, flag in _matrices(p).items():
        assert flag in (0, 1), f"{name} has is_integer={flag!r}"


def test_round_trip_restores_X(tmp_path, adatas):
    """Renaming the matrix must not lose which one was `adata.X`."""
    cytome = pytest.importorskip("cytome")
    a = adatas(True)
    p = str(tmp_path / "h.cytome")
    cytome.from_anndata(a, output=p, counts_layer="count",
                        main_layer_name="lognorm").close()
    ds = cytome.open(p)
    try:
        back = ds.to_anndata()
    finally:
        ds.close()
    got = back.X.toarray() if sp.issparse(back.X) else np.asarray(back.X)
    assert np.allclose(got, a.X.toarray(), atol=1e-5), "X did not round-trip"
