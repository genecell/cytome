"""SQLite column names are case-insensitive; the API now agrees with it.

The bug this pins: ``ds.cells["leiden"] = labels`` against a table holding
``Leiden`` passed a case-sensitive membership check as "new", the ALTER's
duplicate-column error was swallowed, and the UPDATE resolved onto ``Leiden``
— a silent overwrite. ``set_categories("leiden")`` then raised on its own
case-sensitive guard, so the fresh order was never stored and the stale one
survived to plot time. Meanwhile ``ds.cells["leiden"]`` raised KeyError on
the very column just written.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import cytome


def _build(path, n_cells=6):
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"BC{i}" for i in range(n_cells)],
        "Leiden": [str(i % 2) for i in range(n_cells)],
    }))
    ds.flush()
    return ds


def test_case_variant_write_warns_and_lands_on_the_stored_column(tmp_path):
    ds = _build(tmp_path / "t.cytome")
    new = [str(i % 3) for i in range(6)]
    with pytest.warns(UserWarning, match="resolves to existing column 'Leiden'"):
        ds.cells["leiden"] = new
    ds.flush()
    # One column, not two: SQLite would have merged them anyway — now the API
    # says so instead of letting the reader discover it.
    cols = [c for c in ds.cells.columns if c.lower() == "leiden"]
    assert cols == ["Leiden"]
    assert list(ds.cells["Leiden"]) == new


def test_readback_under_the_written_casing_works(tmp_path):
    """You can read what you just wrote, under the name you wrote it to."""
    ds = _build(tmp_path / "t.cytome")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.cells["leiden"] = [str(i % 3) for i in range(6)]
        ds.flush()
        assert list(ds.cells["leiden"]) == list(ds.cells["Leiden"])


def test_membership_is_case_insensitive_like_the_storage(tmp_path):
    ds = _build(tmp_path / "t.cytome")
    assert "Leiden" in ds.cells
    assert "leiden" in ds.cells
    assert "LEIDEN" in ds.cells
    assert "not_there" not in ds.cells


def test_set_categories_resolves_a_case_variant(tmp_path):
    ds = _build(tmp_path / "t.cytome")
    with pytest.warns(UserWarning, match="resolves to existing column"):
        ds.set_categories("leiden", order=["0", "1"])
    # Stored under the canonical casing, where every reader looks.
    assert ds.get_categories("Leiden")["order"] == ["0", "1"]


def test_full_overwrite_drops_categories_even_for_a_label_subset(tmp_path):
    """The subset-silent hazard: re-clustering that yields a subset of the old
    labels used to keep the old colours — mapped onto different cells."""
    ds = _build(tmp_path / "t.cytome")
    ds.set_categories("Leiden", order=["0", "1"],
                      colors=["#4E79A7", "#E69F00"])
    with pytest.warns(UserWarning, match="stored category order/colors were dropped"):
        ds.cells["Leiden"] = ["0"] * 6          # subset of the stored order
    ds.flush()
    assert ds.get_categories("Leiden") in (None, {})


def test_overwrite_then_set_categories_is_clean(tmp_path):
    """The writer sequence piaso.tl.leiden uses: overwrite, then re-persist a
    fresh order. Ends with the fresh order and no stale survivors."""
    ds = _build(tmp_path / "t.cytome")
    ds.set_categories("Leiden", order=["0", "1"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.cells["leiden"] = [str(i % 3) for i in range(6)]
        ds.set_categories("leiden", order=["0", "1", "2"])
    assert ds.get_categories("Leiden")["order"] == ["0", "1", "2"]


def test_fresh_column_write_stays_silent_and_creates_it(tmp_path):
    ds = _build(tmp_path / "t.cytome")
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        ds.cells["cluster_v2"] = [str(i) for i in range(6)]
        ds.flush()
    assert list(ds.cells["cluster_v2"]) == [str(i) for i in range(6)]
