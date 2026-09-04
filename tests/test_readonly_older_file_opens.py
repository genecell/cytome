"""Opening a file must not require permission to modify it.

The 0.3.0 schema upgrade adds ``matrix_meta.is_integer`` on open. On a 0.2.x
file that is read-only -- which is how the released datasets ship, mode 444 --
the ALTER raised ``OperationalError: attempt to write a readonly database``
from inside ``_create_schema``, a line nobody reading that traceback would
connect to a schema upgrade. The file was fine; the upgrade was optional.

Readers treat a missing column and a NULL the same way ("unknown, probe if you
care"), so skipping it costs nothing.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest


def _make_pre_030(path):
    """A cytome whose matrix_meta has no is_integer column."""
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    import cytome

    a = anndata.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(4, 3),
        obs=pd.DataFrame(index=list("abcd")),
        var=pd.DataFrame(index=list("xyz")),
    )
    cytome.from_anndata(a, output=str(path), force=True).close()

    con = sqlite3.connect(str(path))
    cols = [r[1] for r in con.execute("PRAGMA table_info(matrix_meta)")]
    if "is_integer" in cols:
        keep = [c for c in cols if c != "is_integer"]
        con.execute(f"CREATE TABLE mm_new AS SELECT {','.join(keep)} FROM matrix_meta")
        con.execute("DROP TABLE matrix_meta")
        con.execute("ALTER TABLE mm_new RENAME TO matrix_meta")
    con.commit()
    con.close()


def test_readonly_pre_030_file_still_opens(tmp_path):
    import cytome

    path = tmp_path / "ro.cytome"
    _make_pre_030(path)
    for suffix in ("", "-wal", "-shm"):
        p = tmp_path / f"ro.cytome{suffix}"
        if p.exists():
            os.chmod(p, 0o444)

    ds = cytome.open(str(path))
    try:
        assert ds._conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0] == 4
        cols = [r[1] for r in ds._conn.execute("PRAGMA table_info(matrix_meta)")]
        assert "is_integer" not in cols, (
            "the column cannot have been added to a read-only file")
    finally:
        ds.close()


def test_writable_pre_030_file_is_still_upgraded(tmp_path):
    """The skip must apply only when the file cannot take the column."""
    import cytome

    path = tmp_path / "rw.cytome"
    _make_pre_030(path)

    ds = cytome.open(str(path))
    try:
        cols = [r[1] for r in ds._conn.execute("PRAGMA table_info(matrix_meta)")]
        assert "is_integer" in cols, (
            "a writable file should still get the 0.3.0 column")
    finally:
        ds.close()
