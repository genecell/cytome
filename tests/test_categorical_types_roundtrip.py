"""Categoricals of any category type, and missing labels, survive the store.

Audit 2026-10-27, part 4 critical 6 and major 14 (L1 finding 7, L2 finding 5):
a categorical column was stored as TEXT and restored against its numeric
categories, so integer, boolean and float categoricals came back all NaN in
every ``to_anndata``; a nullable boolean NA came back True; a missing label of
an object or categorical column was stored as the text ``'nan'`` (a fake group
in COSG and in plots), and ``Int64``, ``string`` and ``boolean`` columns with
``pd.NA`` failed the flush.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

import cytome

anndata = pytest.importorskip("anndata")


def _obs():
    return pd.DataFrame({
        "intcat": pd.Categorical([1, 2, 1, None]),
        "boolcat": pd.Categorical([True, False, True, True]),
        "floatcat": pd.Categorical([0.5, 1.5, 0.1, 0.30000000000000004]),
        "strcat": pd.Categorical(["1", "2", "1", "10"]),
        "ordered": pd.Categorical(["lo", "hi", "mid", "lo"], categories=["lo", "mid", "hi"],
                                  ordered=True),
        "nullbool": pd.array([True, pd.NA, False, True], dtype="boolean"),
        "nullint": pd.array([1, pd.NA, 3, 4], dtype="Int64"),
    }, index=[f"c{i}" for i in range(4)])


def _roundtrip(tmp_path, obs):
    a = anndata.AnnData(X=np.ones((4, 2), dtype=np.float32), obs=obs)
    ds = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    back = ds.to_anndata()
    return ds, back


def test_categoricals_of_every_category_type_roundtrip(tmp_path):
    obs = _obs()
    ds, back = _roundtrip(tmp_path, obs)
    for col in ("intcat", "boolcat", "floatcat", "strcat", "ordered"):
        pd.testing.assert_series_equal(back.obs[col].reset_index(drop=True),
                                       obs[col].reset_index(drop=True), check_names=False)
    assert back.obs["ordered"].cat.ordered
    assert back.obs["nullbool"].dtype == "boolean"
    assert back.obs["nullbool"].isna().tolist() == [False, True, False, False]
    assert back.obs["nullint"].dtype == "Int64" and back.obs["nullint"].isna().sum() == 1
    ds.close()


def test_category_values_are_stored_in_their_own_type(tmp_path):
    """Raw SQL and the R package read the labels, in the categories' type."""
    ds, _ = _roundtrip(tmp_path, _obs())
    decl = {r[1]: r[2] for r in ds._conn.execute("PRAGMA table_info(cells)")}
    assert decl["intcat"] == "INTEGER" and decl["boolcat"] == "INTEGER"
    assert decl["floatcat"] == "REAL" and decl["strcat"] == "TEXT"
    assert ds.cells["intcat"].tolist() == [1, 2, 1, None]
    ds.close()


def test_a_store_written_before_restores_its_categories(tmp_path):
    """The old writer kept every categorical as TEXT: what SQLite made of 1,
    True and 0.5 in a TEXT column. Those stores come back exactly too."""
    ds = cytome.create(str(tmp_path / "old.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": list("abcd")}))
    ds.flush()
    c = ds._conn
    for col, vals, cats in (
            ("intcat", [1, 2, 1, None], [1, 2]),
            ("boolcat", [True, False, True, True], [False, True]),
            ("floatcat", [0.5, 1.5, 0.1, 0.5], [0.1, 0.5, 1.5]),
            ("nullbool", [1, None, 0, 1], None)):
        c.execute(f'ALTER TABLE cells ADD COLUMN "{col}" TEXT' if cats is not None else
                  f'ALTER TABLE cells ADD COLUMN "{col}" INTEGER')
        c.executemany(f'UPDATE cells SET "{col}" = ? WHERE cell_idx = ?',
                      [(v, i) for i, v in enumerate(vals)])
        import json
        c.execute("INSERT INTO _column_meta VALUES ('cells', ?, ?, ?)",
                  (col, "categorical" if cats is not None else "bool",
                   json.dumps(cats) if cats is not None else None))
    c.execute("INSERT INTO genes(gene_id) VALUES ('g0')")
    c.commit()
    assert ds.cells["intcat"].tolist() == ["1", "2", "1", None]     # TEXT on disk
    import scipy.sparse as sp
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((4, 1), dtype=np.float32)))
    ds.flush()
    back = ds.to_anndata()
    assert back.obs["intcat"].tolist()[:3] == [1, 2, 1] and pd.isna(back.obs["intcat"].iloc[3])
    assert back.obs["boolcat"].tolist() == [True, False, True, True]
    assert back.obs["floatcat"].tolist() == [0.5, 1.5, 0.1, 0.5]
    assert back.obs["nullbool"].isna().tolist() == [False, True, False, False]
    assert back.obs["nullbool"].dtype == "boolean"
    ds.close()


@pytest.mark.parametrize("values", [
    pd.Series(["x", np.nan, "y", "x"]),
    pd.Series(pd.Categorical(["x", None, "y", "x"])),
    np.array(["x", None, "y", "x"], dtype=object),
    ["x", None, "y", "x"],
])
def test_missing_labels_are_null_not_nan_text(tmp_path, values):
    ds = cytome.create(str(tmp_path / "m.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": list("abcd")}))
    ds.flush()
    ds.cells["label"] = values
    ds.flush()
    assert ds.cells["label"].tolist() == ["x", None, "y", "x"]
    assert ds._conn.execute("SELECT COUNT(*) FROM cells WHERE label = 'nan'").fetchone()[0] == 0
    ds.close()


@pytest.mark.parametrize("values,decl,expect", [
    (pd.Series([1, pd.NA, 3, 4], dtype="Int64"), "INTEGER", [1, None, 3, 4]),
    (pd.Series(["a", pd.NA, "b", "c"], dtype="string"), "TEXT", ["a", None, "b", "c"]),
    (pd.Series([True, pd.NA, False, True], dtype="boolean"), "INTEGER", [1, None, 0, 1]),
    (pd.Series([1.5, np.nan, 2.0, 3.0], dtype="Float64"), "REAL", [1.5, None, 2.0, 3.0]),
])
def test_nullable_columns_with_na_are_written(tmp_path, values, decl, expect):
    ds = cytome.create(str(tmp_path / "n.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": list("abcd")}))
    ds.flush()
    ds.cells["col"] = values
    ds.flush()
    assert ds.cells["col"].tolist() == expect
    assert {r[1]: r[2] for r in ds._conn.execute("PRAGMA table_info(cells)")}["col"] == decl
    ds.close()


def test_to_anndata_shows_a_missing_label_as_missing(tmp_path):
    obs = pd.DataFrame({"cell_type": pd.Categorical(["T", None, "B", "T"])},
                       index=[f"c{i}" for i in range(4)])
    ds, back = _roundtrip(tmp_path, obs)
    assert "nan" not in [str(c) for c in back.obs["cell_type"].cat.categories]
    assert back.obs["cell_type"].isna().tolist() == [False, True, False, False]
    ds.close()
