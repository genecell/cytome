"""A value outside a categorical column's recorded categories is kept (audit decision 6a).

The categories recorded in ``_column_meta`` can be stale: a later write of the column's
values does not update them. Such a value came back missing from ``to_anndata`` -- lost, and
not to be told from a missing label -- while the R package kept it. Both now keep it, as a
category after the recorded ones, with a warning that names the column and the values. A
colour list stored for the column gets a neutral grey for each kept category, so the
recorded categories keep their colours.
"""
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

import cytome
from cytome.io.convert_anndata import KEPT_CATEGORY_COLOR

anndata = pytest.importorskip("anndata")


def _store(tmp_path, obs, uns=None):
    a = anndata.AnnData(X=np.ones((len(obs), 2), dtype=np.float32), obs=obs, uns=uns or {})
    path = str(tmp_path / "t.cytome")
    cytome.from_anndata(a, output=path).close()
    return path


def _set(path, column, cells, value):
    """Write ``value`` into ``column`` for ``cells`` with plain SQL, as another tool would."""
    con = sqlite3.connect(path)
    con.executemany(f'UPDATE cells SET "{column}" = ? WHERE cell_idx = ?',
                    [(value, int(c)) for c in cells])
    con.commit()
    con.close()


def _labels(series):
    return [None if pd.isna(v) else v for v in series.astype(object)]


def test_a_value_outside_the_recorded_categories_is_kept_after_them(tmp_path):
    obs = pd.DataFrame({"cell_type": pd.Categorical(
        ["Astro", "Excit", "Inhib", "Excit", None, "Astro"], categories=["Excit", "Inhib", "Astro"])},
        index=[f"c{i}" for i in range(6)])
    path = _store(tmp_path, obs, uns={"cell_type_colors": ["#111111", "#222222", "#333333"]})
    _set(path, "cell_type", [1, 3], "Zeta")
    ds = cytome.open(path, mode="r")
    with pytest.warns(UserWarning, match=r"2 values of cells\.cell_type are not among its "
                                         r"recorded categories \(Zeta\)"):
        a = ds.to_anndata()
    col = a.obs["cell_type"]
    assert list(col.cat.categories) == ["Excit", "Inhib", "Astro", "Zeta"]
    assert _labels(col) == ["Astro", "Zeta", "Inhib", "Zeta", None, "Astro"]
    # the recorded categories keep their colours; the kept one is grey
    assert list(a.uns["cell_type_colors"]) == ["#111111", "#222222", "#333333", KEPT_CATEGORY_COLOR]
    with pytest.warns(UserWarning, match=r"cells\.cell_type"):
        df = ds.cells.to_pandas(restore_dtypes=True)
    assert list(df["cell_type"].cat.categories) == ["Excit", "Inhib", "Astro", "Zeta"]
    assert _labels(df["cell_type"]) == _labels(col)
    ds.close()


def test_integer_and_ordered_categories_keep_an_outside_value_in_its_type(tmp_path):
    obs = pd.DataFrame({
        "cluster": pd.Categorical([0, 1, 2, 1]),
        "stage": pd.Categorical(["lo", "hi", "lo", "mid"], categories=["lo", "mid", "hi"], ordered=True),
    }, index=[f"c{i}" for i in range(4)])
    path = _store(tmp_path, obs)
    _set(path, "cluster", [3], 7)
    _set(path, "stage", [0, 2], "top")
    ds = cytome.open(path, mode="r")
    with pytest.warns(UserWarning) as rec:
        a = ds.to_anndata()
    msgs = [str(w.message) for w in rec]
    assert any("1 values of cells.cluster" in m and "(7)" in m for m in msgs), msgs
    assert any("2 values of cells.stage" in m and "(top)" in m for m in msgs), msgs
    assert list(a.obs["cluster"].cat.categories) == [0, 1, 2, 7]
    assert _labels(a.obs["cluster"]) == [0, 1, 2, 7]
    stage = a.obs["stage"]
    assert stage.cat.ordered and list(stage.cat.categories) == ["lo", "mid", "hi", "top"]
    assert _labels(stage) == ["top", "hi", "top", "mid"]
    ds.close()


def test_a_column_within_its_categories_reads_without_a_warning(tmp_path):
    obs = pd.DataFrame({"cell_type": pd.Categorical(["a", "b", None])}, index=["x", "y", "z"])
    path = _store(tmp_path, obs, uns={"cell_type_colors": ["#111111", "#222222"]})
    ds = cytome.open(path, mode="r")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a = ds.to_anndata()
    assert not [w for w in rec if "recorded categories" in str(w.message)]
    assert _labels(a.obs["cell_type"]) == ["a", "b", None]
    assert list(a.uns["cell_type_colors"]) == ["#111111", "#222222"]
    ds.close()
