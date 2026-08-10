from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cytome


def test_store_dict(tmp_cytome):
    ds = cytome.create(tmp_cytome)
    ds.metadata["colors"] = {"A": "#f00", "B": "#00f"}
    ds.flush()
    assert ds.metadata["colors"]["A"] == "#f00"
    ds.close()


def test_store_list_of_dicts(tmp_cytome):
    ds = cytome.create(tmp_cytome)
    value = [{"k": 1}, {"k": 2}]
    ds.metadata["lod"] = value
    ds.flush()
    assert ds.metadata["lod"] == value
    ds.close()


def test_store_ndarray_dataframe_series(tmp_cytome):
    ds = cytome.create(tmp_cytome)
    arr = np.array([1, 2, 3])
    df = pd.DataFrame({"a": [1, 2]})
    ser = pd.Series([3, 4], index=["x", "y"])
    ds.metadata["arr"] = arr
    ds.metadata["df"] = df
    ds.metadata["ser"] = ser
    ds.flush()
    assert np.array_equal(ds.metadata["arr"], arr)
    assert list(ds.metadata["df"]["a"]) == [1, 2]
    assert int(ds.metadata["ser"]["x"]) == 3
    ds.close()


def test_store_ndarray_dtype_and_string_array(tmp_cytome):
    ds = cytome.create(tmp_cytome)
    arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arr_s = np.array(["a", "b", "c"], dtype=str)
    ds.metadata["arr_f32"] = arr_f32
    ds.metadata["arr_s"] = arr_s
    ds.flush()
    back_f32 = ds.metadata["arr_f32"]
    back_s = ds.metadata["arr_s"]
    assert isinstance(back_f32, np.ndarray)
    assert back_f32.dtype == np.float32
    assert isinstance(back_s, np.ndarray)
    assert back_s.dtype.kind in {"U", "O", "S"}
    assert list(back_s) == ["a", "b", "c"]
    ds.close()


def test_numpy_types_and_delete_keys_items(tmp_cytome):
    ds = cytome.create(tmp_cytome)
    ds.metadata["x"] = {"a": np.int64(5), "b": np.float32(1.5)}
    ds.flush()
    assert ds.metadata["x"]["a"] == 5
    keys = ds.metadata.keys()
    assert "x" in keys
    items = dict(ds.metadata.items())
    assert "x" in items
    del ds.metadata["x"]
    ds.flush()
    with pytest.raises(KeyError):
        _ = ds.metadata["x"]
    ds.close()


def test_unsupported_type_error_message(tmp_cytome):
    ds = cytome.create(tmp_cytome)

    class NoJson:
        pass

    with pytest.raises(TypeError) as exc:
        ds.metadata["bad"] = NoJson()
    assert "Cannot serialize value of type" in str(exc.value)
    ds.close()
