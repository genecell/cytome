"""Tables, Series and record arrays in metadata come back as they went in.

Audit 2026-10-27, part 4 majors 19 and 20 (L1 finding 13, L2 finding 8):
metadata DataFrames were read with dtype inference ("01" came back as 1, so a
table keyed by cluster labels stopped matching them), categoricals came back
as object, a Series' string index came back as integers; record arrays (scanpy's
``uns['rank_genes_groups']``) were written with ``str(dtype)``, which numpy
cannot read, and validate() passed on such a store.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import cytome

anndata = pytest.importorskip("anndata")


def _store(tmp_path):
    ds = cytome.create(str(tmp_path / "m.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b"]}))
    ds.flush()
    return ds


def _reopened(ds):
    path = ds.path
    ds.close()
    return cytome.open(str(path))


def test_a_table_comes_back_with_its_types(tmp_path):
    df = pd.DataFrame({
        "code": ["01", "02", "10"],
        "cluster": pd.Categorical(["3", "1", "3"], categories=["1", "3"]),
        "n": pd.array([1, None, 3], dtype="Int64"),
        "flag": [True, False, True],
        "when": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
        5: [0.5, 1.5, 2.5],
    }, index=["0", "1", "10"])
    ds = _store(tmp_path)
    ds.metadata["df"] = df
    assert ds.metadata["df"]["code"].tolist() == ["01", "02", "10"]       # pending, too
    ds = _reopened(ds)
    back = ds.metadata["df"]
    pd.testing.assert_frame_equal(back, df, check_index_type=True)
    ds.close()


def test_a_series_keeps_its_index_type(tmp_path):
    ds = _store(tmp_path)
    ds.metadata["s_str"] = pd.Series([5, 6], index=["0", "1"])
    ds.metadata["s_int"] = pd.Series([5.5, 6.5], index=[0, 1])
    ds.metadata["s_cat"] = pd.Series(pd.Categorical(["a", "b"]), index=["x", "y"])
    ds = _reopened(ds)
    assert ds.metadata["s_str"].index.tolist() == ["0", "1"]
    assert ds.metadata["s_int"].index.tolist() == [0, 1]
    assert ds.metadata["s_cat"].dtype == "category"
    ds.close()


def test_a_table_written_before_reads_its_values_as_written(tmp_path):
    """No records beside it (an older store): read without guessing types."""
    ds = _store(tmp_path)
    old = pd.DataFrame({"code": ["01", "02"]}).to_json(orient="split")
    ds._conn.execute("INSERT INTO _metadata VALUES ('old', ?, 'dataframe')", (old,))
    ds._conn.commit()
    assert ds.metadata["old"]["code"].tolist() == ["01", "02"]
    ds.close()


def test_records_go_with_their_key(tmp_path):
    ds = _store(tmp_path)
    ds.metadata["k"] = pd.DataFrame({"c": pd.Categorical(["a"])})
    ds.flush()
    n = lambda: ds._conn.execute("SELECT COUNT(*) FROM _column_meta WHERE table_name = "
                                 "'_metadata:k'").fetchone()[0]
    assert n() == 2                                   # the column and the index
    ds.metadata["k"] = [1, 2]
    ds.flush()
    assert n() == 0 and ds.metadata["k"] == [1, 2]
    ds.metadata["k"] = pd.DataFrame({"c": pd.Categorical(["a"])})
    ds.flush()
    del ds.metadata["k"]
    ds.flush()
    assert n() == 0
    ds.close()


def _rank_genes_groups():
    names = np.array([("A", "B"), ("C", "D")], dtype=[("0", "<U5"), ("1", "<U5")])
    scores = np.array([(1.5, 2.5), (0.5, 0.25)], dtype=[("0", "<f4"), ("1", "<f4")])
    return {"params": {"groupby": "leiden"}, "names": names, "scores": scores}


def test_record_arrays_roundtrip(tmp_path):
    ds = _store(tmp_path)
    rgg = _rank_genes_groups()
    ds.metadata["rank_genes_groups"] = rgg
    ds.metadata["names"] = rgg["names"]
    ds = _reopened(ds)
    back = ds.metadata["rank_genes_groups"]
    assert back["names"].dtype == rgg["names"].dtype
    assert back["names"].tolist() == rgg["names"].tolist()
    assert back["scores"].tolist() == rgg["scores"].tolist()
    assert ds.metadata["names"].dtype.names == ("0", "1")
    assert ds.validate(integrity=True).passed
    ds.close()


def test_record_arrays_an_older_version_wrote_are_read(tmp_path):
    ds = _store(tmp_path)
    rec = np.array([("A", 1.5), ("B", 2.5)], dtype=[("names", "<U5"), ("scores", "<f4")])
    old = json.dumps({"rgg": {"__ndarray__": rec.tolist(), "dtype": str(rec.dtype)}})
    ds._conn.execute("INSERT INTO _metadata VALUES ('old', ?, 'dict')", (old,))
    ds._conn.commit()
    back = ds.metadata["old"]["rgg"]
    assert back.dtype == rec.dtype and back.tolist() == rec.tolist()
    ds.close()


def test_rank_genes_groups_survives_an_anndata_roundtrip(tmp_path):
    a = anndata.AnnData(X=np.ones((2, 2), dtype=np.float32))
    a.uns["rank_genes_groups"] = _rank_genes_groups()
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    b = ds.to_anndata()
    assert b.uns["rank_genes_groups"]["names"].tolist() == [("A", "B"), ("C", "D")]
    b.write_h5ad(tmp_path / "b.h5ad")
    ds.close()


def test_validate_finds_unreadable_metadata(tmp_path):
    ds = _store(tmp_path)
    ds._conn.execute("INSERT INTO _metadata VALUES ('bad', ?, 'ndarray')",
                     (json.dumps({"data": [1], "dtype": "no-such-dtype"}),))
    ds._conn.commit()
    assert "metadata_unreadable:bad (TypeError)" in ds.validate(integrity=True).checks_failed
    ds.close()


def test_a_mixed_obsm_table_is_not_stored_as_pointers(tmp_path):
    a = anndata.AnnData(X=np.ones((2, 2), dtype=np.float32))
    a.obsm["covs"] = pd.DataFrame({"batch": pd.Categorical(["x", "y"]), "n": [1, 2]},
                                  index=a.obs_names)
    a.obsm["num"] = pd.DataFrame({"u": [0.5, 1.5], "v": [1, 2]}, index=a.obs_names)
    with pytest.warns(UserWarning, match="non-numeric"):
        ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    assert "RNA_covs" not in ds.embeddings and ds.embeddings["RNA_num"].tolist() == [[0.5, 1], [1.5, 2]]
    with pytest.raises(TypeError, match="numbers"):
        ds.add_embedding("obj", np.array([[1, "a"]], dtype=object))
    ds.close()
