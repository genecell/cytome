"""Columns written together are written in one pass.

SQLite keeps a row's columns together, so changing one value rewrites the
whole row: every column write copied the whole cells table, twice in WAL
mode. flush() now writes adjacent column writes of one table with one
UPDATE. These tests hold it to what the old path stored, byte for byte, and
to the order of writes.
"""
from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
import pandas as pd  # noqa: E402

import cytome  # noqa: E402

N = 30


def _store(path):
    rng = np.random.default_rng(1)
    a = anndata.AnnData(
        X=rng.poisson(1.0, size=(N, 5)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(N)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cytome.from_anndata(a, output=str(path), force=True).close()
    return path


def _columns():
    rng = np.random.default_rng(2)
    floats = rng.normal(size=N)
    floats[3] = np.nan
    obj = np.array([None if i % 7 == 0 else f"s{i}" for i in range(N)], dtype=object)
    obj_ints = np.array([np.int64(i) for i in range(N)], dtype=object)
    return {
        "ints": np.arange(N, dtype=np.int64),
        "int32": np.arange(N, dtype=np.int32) * 3,
        "floats": floats,
        "float32": floats.astype(np.float32),
        "bools": np.arange(N) % 3 == 0,
        "strings": np.array([f"cluster_{i % 4}" for i in range(N)]),
        "objects": obj,
        "numpy_ints_in_objects": obj_ints,
        "categorical": pd.Categorical([f"t{i % 3}" for i in range(N)]),
        "series": pd.Series(np.arange(N) * 0.5),
    }


def _dump(path, names):
    con = sqlite3.connect(str(path))
    types = {r[1]: r[2] for r in con.execute("PRAGMA table_info(cells)")}
    sel = ", ".join(f'typeof("{n}"), "{n}"' for n in names)
    rows = con.execute(f"SELECT {sel} FROM cells ORDER BY rowid").fetchall()
    con.close()
    return [types[n] for n in names], rows


def test_one_pass_stores_what_one_at_a_time_stored(tmp_path):
    cols = _columns()
    # the old way: a flush per column, one UPDATE pass each
    one = _store(tmp_path / "one.cytome")
    ds = cytome.open(str(one))
    for name, values in cols.items():
        ds.cells[name] = values
        ds.flush()
    ds.close()
    # the new way: all columns, one flush
    many = _store(tmp_path / "many.cytome")
    ds = cytome.open(str(many))
    for name, values in cols.items():
        ds.cells[name] = values
    ds.flush()
    ds.close()
    assert _dump(one, list(cols)) == _dump(many, list(cols))


def test_one_update_statement_for_adjacent_columns(tmp_path):
    path = _store(tmp_path / "s.cytome")
    ds = cytome.open(str(path))
    seen = []
    ds._conn.set_trace_callback(seen.append)
    for k in range(4):
        ds.cells[f"res_{k}"] = np.arange(N) % (k + 2)
    ds.flush()
    ds._conn.set_trace_callback(None)
    ds.close()
    # the trace shows each execution with its values filled in
    updates = [s for s in seen if s.lstrip().upper().startswith("UPDATE CELLS")]
    assert len(updates) == N, "one row update per cell, not one per cell and column"
    assert all(all(f'"res_{k}"' in s for k in range(4)) for s in updates)


def test_order_is_kept_across_a_table_replacement(tmp_path):
    """A column written before a whole-table write must not move after it."""
    path = _store(tmp_path / "s.cytome")
    ds = cytome.open(str(path))
    ds.cells["a"] = np.arange(N)
    frame = ds.cells.to_pandas()
    frame["b"] = 1
    ds.set_entity("cells", frame)          # replaces the table: has 'a', not 'z'
    ds.cells["z"] = np.arange(N) * 2
    ds.flush()
    assert list(ds.cells["a"]) == list(range(N))
    assert list(ds.cells["z"]) == [2 * i for i in range(N)]
    ds.close()


def test_pending_column_can_be_read_before_flush(tmp_path):
    path = _store(tmp_path / "s.cytome")
    ds = cytome.open(str(path))
    ds.cells["lab"] = np.array([f"x{i}" for i in range(N)])
    assert "lab" in ds.cells
    assert "lab" in ds.cells.columns
    assert list(ds.cells["lab"][:2]) == ["x0", "x1"]
    ds.cells["lab2"] = np.arange(N)
    assert "lab2" in ds.cells.to_pandas().columns
    ds.close()


def test_same_column_twice_keeps_the_last_values(tmp_path):
    path = _store(tmp_path / "s.cytome")
    ds = cytome.open(str(path))
    ds.cells["lab"] = np.zeros(N, dtype=int)
    ds.cells["other"] = np.ones(N, dtype=int)
    ds.cells["lab"] = np.arange(N)
    with pytest.warns(UserWarning, match="case-insensitive"):
        ds.cells["LAB"] = np.arange(N) + 100   # the same SQLite column
    ds.flush()
    assert list(ds.cells["lab"]) == [i + 100 for i in range(N)]
    assert [c for c in ds.cells.columns if c.lower() == "lab"] == ["lab"]
    ds.close()


def test_caller_array_changed_after_write_does_not_reach_the_store(tmp_path):
    path = _store(tmp_path / "s.cytome")
    ds = cytome.open(str(path))
    values = np.arange(N)
    ds.cells["v"] = values
    values[:] = -1
    ds.flush()
    assert list(ds.cells["v"]) == list(range(N))
    ds.close()
