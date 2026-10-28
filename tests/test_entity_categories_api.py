"""A public reader for the dtypes recorded beside an entity table.

Requested 2026-10-28 by the PIASO plotting work: a store converted from AnnData
keeps each categorical column's category order only in ``_column_meta``, and
PIASO read it through cytome's private ``_restore_column_dtypes``.
``EntityTable.categories(column)`` returns the stored order and
``to_pandas(restore_dtypes=True)`` the restored table; the default ``to_pandas()``
is unchanged.
"""
import numpy as np
import pandas as pd
import pytest

import cytome

anndata = pytest.importorskip("anndata")


@pytest.fixture
def ds(tmp_path):
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(["B", "T", "NK", "T"], categories=["T", "NK", "B"]),
        "stage": pd.Categorical(["lo", "hi", "lo", "mid"], categories=["lo", "mid", "hi"],
                                ordered=True),
        "flag": pd.array([True, pd.NA, False, True], dtype="boolean"),
        "plain": ["a", "b", "c", "d"],
    }, index=[f"c{i}" for i in range(4)])
    d = cytome.from_anndata(anndata.AnnData(X=np.ones((4, 2), dtype=np.float32), obs=obs),
                            output=str(tmp_path / "c.cytome"))
    yield d
    d.close()


def test_categories_in_their_stored_order(ds):
    assert ds.cells.categories("cell_type") == ["T", "NK", "B"]
    assert ds.cells.categories("Cell_Type") == ["T", "NK", "B"]      # SQLite names ignore case
    assert ds.cells.categories("stage") == ["lo", "mid", "hi"]
    assert ds.cells.categories("plain") is None
    with pytest.raises(KeyError):
        ds.cells.categories("nope")


def test_to_pandas_restores_the_recorded_dtypes_on_request(ds):
    plain = ds.cells.to_pandas()
    # unchanged default: the stored text (object before pandas 3, its str dtype since)
    assert not isinstance(plain["cell_type"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_string_dtype(plain["cell_type"].dtype)
    restored = ds.cells.to_pandas(restore_dtypes=True)
    assert list(restored["cell_type"].cat.categories) == ["T", "NK", "B"]
    assert restored["stage"].cat.ordered
    assert restored["flag"].dtype == "boolean" and restored["flag"].isna().tolist() == \
        [False, True, False, False]
