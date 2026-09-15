"""A column's type has to survive the round trip through SQLite.

``_write_entity_table`` declares a SQLite type per column. It used to default
to INTEGER and reach TEXT only through ``is_object_dtype`` -- fine until pandas
3, which turns on ``future.infer_string`` so a string column is ``StringDtype``
rather than ``object``. It then matched no branch, was declared INTEGER, and
cluster labels written as ``"0"``, ``"1"``, ``"2"`` were read back as ``0``,
``1``, ``2``.

Nothing raised. SQLite stores what it is handed and only the read is typed, so
the damage showed up much later, wherever a label was compared to a string or
looked up in a stored category order.

These run under ``future.infer_string`` explicitly, so the pandas 3 behaviour
is covered on pandas 2 as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cytome
from cytome.core.dataset import _sql_column_type


@pytest.fixture(params=[False, True], ids=["object-strings", "infer-string"])
def string_mode(request):
    try:
        with pd.option_context("future.infer_string", request.param):
            yield request.param
    except (KeyError, ValueError):
        if request.param:
            pytest.skip("pandas has no future.infer_string option")
        yield request.param


def test_a_string_column_is_declared_text(string_mode):
    assert _sql_column_type(pd.Series(["a", "b"])) == "TEXT"


def test_digit_strings_are_text_not_integers(string_mode):
    """The exact failure: labels that *look* numeric are still labels."""
    assert _sql_column_type(pd.Series(["0", "1", "2"])) == "TEXT"


@pytest.mark.parametrize("values,want", [
    ([1, 2, 3], "INTEGER"),
    ([1.5, 2.5], "REAL"),
    ([True, False], "INTEGER"),
    (pd.Categorical(["a", "b"]), "TEXT"),
])
def test_the_other_dtypes_are_unchanged(values, want):
    assert _sql_column_type(pd.Series(values)) == want


def test_an_unknown_dtype_falls_back_to_text():
    """TEXT loses the type; INTEGER lost the value. Prefer the recoverable one."""
    assert _sql_column_type(pd.Series(pd.to_datetime(["2026-01-01"]))) == "TEXT"


def test_labels_round_trip_through_a_real_dataset(string_mode, tmp_path):
    ds = cytome.create(str(tmp_path / "t.cytome"))
    try:
        n = 6
        ds.set_entity("cells", pd.DataFrame({
            "cell_idx": np.arange(n),
            "barcode": [f"BC{i}" for i in range(n)],
            "cluster": [str(i % 3) for i in range(n)],
            "score": np.linspace(0, 1, n),
            "count": np.arange(n),
        }))
        ds.flush()

        got = list(ds.cells["cluster"])
        assert all(isinstance(v, str) for v in got), (
            f"cluster labels came back as {type(got[0]).__name__}")
        assert got == [str(i % 3) for i in range(n)]
        assert isinstance(list(ds.cells["score"])[0], float)
        assert isinstance(list(ds.cells["count"])[0], (int, np.integer))
    finally:
        ds.close()


def test_a_column_added_after_the_table_exists(string_mode, tmp_path):
    """The ALTER TABLE path, which is where piaso.tl.leiden writes."""
    ds = cytome.create(str(tmp_path / "t.cytome"))
    try:
        n = 6
        ds.set_entity("cells", pd.DataFrame({
            "cell_idx": np.arange(n),
            "barcode": [f"BC{i}" for i in range(n)],
        }))
        ds.flush()
        labels = np.array([str(i % 3) for i in range(n)])
        ds.cells["leiden"] = labels
        ds.flush()
        assert list(ds.cells["leiden"]) == list(labels)
    finally:
        ds.close()
