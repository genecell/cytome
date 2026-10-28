"""The flush is one transaction that cannot poison the dataset, and writes
outside it are committed.

Audit 2026-10-27, part 4 majors 10-13; L1 findings 3-6, 19, 20: one failed
flush kept its payload first in the queue, so every later flush and close()
failed and the connection stayed open; rewriting ``cells`` failed on every
spatial store (the coordinates' foreign key); a buffered column was lost by
``add_entity_column``, which read the table without flushing, and by a key
written twice (a dict keeps its first slot); spatial images, coordinates and
feature-index writes were never committed; a new column's ALTER TABLE
committed on its own before a flush failed; metadata reads ignored pending
writes.
"""
import os
import sqlite3
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome


def _store(tmp_path, name="s.cytome", n=4):
    ds = cytome.create(str(tmp_path / name))
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(n)]}))
    ds.flush()
    return ds


def _columns(path, table="cells"):
    con = sqlite3.connect(str(path))
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


def test_a_failed_flush_sets_its_write_aside(tmp_path):
    ds = _store(tmp_path)
    ds.metadata["kept"] = 1                                   # a valid write, queued first
    ds.set_entity("genes", pd.DataFrame({"symbol": ["A"]}))   # gene_id is NOT NULL
    with pytest.warns(RuntimeWarning, match="set aside"):
        with pytest.raises(sqlite3.IntegrityError):
            ds.flush()
    assert "entity_table:genes" in ds._failed_writes
    ds.add_embedding("ok", np.ones((4, 2)))                    # the next flush works
    ds.close()                                                 # and so does close()
    assert ds.is_closed
    back = cytome.open(str(tmp_path / "s.cytome"))
    assert back.metadata["kept"] == 1 and "ok" in back.embeddings
    back.close()


def test_close_writes_the_rest_and_closes_when_a_write_fails(tmp_path):
    ds = _store(tmp_path)
    ds.metadata["kept"] = 1
    ds.set_entity("genes", pd.DataFrame({"symbol": ["A"]}))
    with pytest.warns(RuntimeWarning):
        with pytest.raises(sqlite3.IntegrityError):
            ds.close()
    assert ds.is_closed
    ds.close()                                                 # a second close: nothing
    back = cytome.open(str(tmp_path / "s.cytome"))
    assert back.metadata["kept"] == 1
    back.close()


def test_a_failed_flush_leaves_no_new_column(tmp_path):
    ds = _store(tmp_path)
    ds.cells["newcol"] = np.arange(4)
    ds.set_entity("genes", pd.DataFrame({"symbol": ["A"]}))
    with pytest.warns(RuntimeWarning):
        with pytest.raises(sqlite3.IntegrityError):
            ds.flush()
    assert "newcol" not in _columns(tmp_path / "s.cytome")
    ds.flush()                                                  # the column, written now
    assert "newcol" in _columns(tmp_path / "s.cytome")
    ds.close()


def test_cells_of_a_spatial_store_can_be_rewritten(tmp_path):
    ds = _store(tmp_path)
    ds.set_spatial_coords(np.arange(8.0).reshape(4, 2))
    ds.add_cells_column("x", np.arange(4))
    df = ds.cells.to_pandas()
    df["y"] = list("abcd")
    ds.set_entity("cells", df)
    ds.flush()
    assert ds.cells["y"].tolist() == list("abcd")
    # dropping a cell the coordinates refer to is still refused, at the commit
    ds.set_entity("cells", df.iloc[:3])
    with pytest.warns(RuntimeWarning):
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            ds.flush()
    assert ds.n_cells == 4
    ds.close()


def test_buffered_column_survives_add_cells_column(tmp_path):
    ds = _store(tmp_path)
    ds.cells["cell_type"] = np.array(list("abab"))           # buffered
    ds.add_cells_column("leiden", np.arange(4))
    assert ds.cells["cell_type"].tolist() == list("abab")
    ds.close()
    back = cytome.open(str(tmp_path / "s.cytome"))
    assert back.cells["cell_type"].tolist() == list("abab")
    assert back.cells["leiden"].tolist() == [0, 1, 2, 3]
    back.close()


def test_a_key_written_again_goes_behind_the_writes_between(tmp_path):
    ds = _store(tmp_path)
    df = ds.cells.to_pandas()                                 # taken before x exists
    ds.cells["x"] = np.arange(4)
    ds.set_entity("cells", df)
    ds.cells["x"] = np.arange(4) + 10
    ds.flush()
    assert ds.cells["x"].tolist() == [10, 11, 12, 13]
    ds.close()


def test_the_key_column_is_not_written_as_a_column(tmp_path):
    ds = _store(tmp_path)
    with pytest.raises(ValueError, match="key"):
        ds.cells["cell_idx"] = np.arange(4)[::-1]
    with pytest.raises(ValueError, match="key"):
        ds.add_cells_column("cell_idx", np.arange(4))
    ds.close()


def test_metadata_reads_see_pending_writes(tmp_path):
    ds = _store(tmp_path)
    ds.metadata["k"] = 1
    ds.flush()
    ds.metadata["k"] = 2
    assert ds.metadata["k"] == 2 and ds.metadata.get("k") == 2
    ds.metadata["new"] = [1, 2]
    assert "new" in ds.metadata and "new" in ds.metadata.keys()
    assert dict(ds.metadata.items())["new"] == [1, 2]
    del ds.metadata["k"]
    assert "k" not in ds.metadata and ds.metadata.get("k") is None
    ds.close()


_OTHER_PROCESS = textwrap.dedent("""
    import sqlite3, sys
    con = sqlite3.connect(sys.argv[1], timeout=2)
    print(con.execute(sys.argv[2]).fetchone()[0])
    con.execute("BEGIN IMMEDIATE"); con.rollback()      # can it write?
    print("writable")
""")


def _from_other_process(path, sql):
    out = subprocess.run([sys.executable, "-c", _OTHER_PROCESS, str(path), sql],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.split(), out.stderr


def test_spatial_writes_are_committed(tmp_path):
    ds = _store(tmp_path)
    ds.set_spatial_coords(np.arange(8.0).reshape(4, 2))
    ds.add_spatial_image("lib", "hires", np.zeros((3, 3), dtype=np.uint8))
    got, err = _from_other_process(tmp_path / "s.cytome", "SELECT COUNT(*) FROM spatial_coords")
    assert got == ["4", "writable"], err
    ds.delete_spatial_image("lib", "hires")
    got, err = _from_other_process(tmp_path / "s.cytome", "SELECT COUNT(*) FROM spatial_images")
    assert got == ["0", "writable"], err
    ds.close()


def test_a_layer_writer_never_finalized_is_rolled_back_at_close(tmp_path):
    ds = _store(tmp_path)
    A = sp.csr_matrix(np.eye(4, 3, dtype=np.float32))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1", "g2"]}))
    ds.add_matrix("RNA_counts", A)
    ds.flush()
    w = ds.create_layer_writer("RNA_counts", 4, 3, dtype=np.float32)   # replaces it...
    w.write_chunk(A[:2] * 7, 0)                                        # ...half way
    ds.metadata["note"] = "x"                      # queued: written after the rollback
    with pytest.warns(RuntimeWarning, match="rolled back"):
        ds.close()
    back = cytome.open(str(tmp_path / "s.cytome"))
    assert (back.RNA.counts.to_memory() != A).nnz == 0                 # the old matrix
    assert back.metadata["note"] == "x"
    back.close()


@pytest.mark.parametrize("how", ["close", "reopen", "released", "flush"])
def test_a_layer_writer_dropped_by_an_error_is_rolled_back(tmp_path, how):
    # The writer is a local of a step that fails half way: the caller has no
    # handle to it, and its rows (the old layer already deleted) were committed
    # by whatever let go of the connection next.
    ds = _store(tmp_path)
    A = sp.csr_matrix(np.eye(4, 3, dtype=np.float32))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1", "g2"]}))
    ds.add_matrix("RNA_counts", A)
    ds.flush()

    def step():
        w = ds.create_layer_writer("RNA_counts", 4, 3, dtype=np.float32)
        w.write_chunk(A[:2] * 7, 0)
        raise MemoryError("the loop failed")
    with pytest.raises(MemoryError):
        step()
    ds.metadata["note"] = "x"
    flushed = how == "flush"
    if flushed:
        ds.flush()                                 # joins the writer's transaction
        how = "close"
    with pytest.warns(RuntimeWarning, match="rolled back" + (".*metadata:note" if flushed else "")):
        if how == "close":
            ds.close()
        elif how == "reopen":
            ds.reopen()
            ds.close()
        else:
            with cytome.released(ds):
                pass
            ds.close()
    back = cytome.open(str(tmp_path / "s.cytome"))
    assert (back.RNA.counts.to_memory() != A).nnz == 0                 # the old matrix
    # a write queued is written after the rollback; one flushed into the
    # writer's transaction went with it
    assert ("note" in back.metadata) is (not flushed)
    assert back.validate(integrity=False).passed
    back.close()


def _other_reader(path, sql):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute(sql)]
    finally:
        con.close()


def test_a_flush_while_a_writer_is_open_is_committed_with_the_layer(tmp_path):
    # cytome's own quantifier writes its chunks, flushes the feature table,
    # then finalizes: the flush must neither fail nor commit the layer half
    # written, and the three must land together.
    path = tmp_path / "s.cytome"
    ds = _store(tmp_path)
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1", "g2"]}))
    A = sp.csr_matrix(np.eye(4, 3, dtype=np.float32))
    ds.add_matrix("RNA_counts", A)
    ds.flush()
    w = ds.create_layer_writer("RNA_counts", 4, 3, dtype=np.float32)
    w.write_chunk(A * 5, 0)
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["h0", "h1", "h2"]}))
    ds.flush()
    assert ds.genes["gene_id"].tolist() == ["h0", "h1", "h2"]          # this connection
    assert _other_reader(path, "SELECT gene_id FROM genes") == ["g0", "g1", "g2"]
    assert _other_reader(path, "SELECT n_nonzero FROM matrix_meta") == [3]
    w.finalize()
    assert _other_reader(path, "SELECT gene_id FROM genes") == ["h0", "h1", "h2"]
    assert ds.RNA.counts.to_memory().toarray().tolist() == (A * 5).toarray().tolist()
    assert ds.validate(integrity=False).passed
    ds.close()


def test_a_failing_write_flushed_into_a_writer_leaves_its_rows(tmp_path):
    ds = _store(tmp_path)
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1", "g2"]}))
    ds.flush()
    w = ds.create_layer_writer("RNA_counts", 4, 3, dtype=np.float32)
    w.write_chunk(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), 0)
    ds.metadata["kept"] = 1
    ds.set_entity("genes", pd.DataFrame({"symbol": ["A"]}))            # gene_id is NOT NULL
    with pytest.warns(RuntimeWarning, match="set aside"):
        with pytest.raises(sqlite3.IntegrityError):
            ds.flush()
    assert "entity_table:genes" in ds._failed_writes
    ds.flush()                                     # the rest, into the transaction
    w.write_chunk(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), 2)
    w.finalize()
    ds.close()
    back = cytome.open(str(tmp_path / "s.cytome"))
    assert back.RNA.counts.to_memory().sum() == 12                     # every row kept
    assert back.metadata["kept"] == 1
    assert back.genes["gene_id"].tolist() == ["g0", "g1", "g2"]
    back.close()


def test_a_flush_before_a_writer_has_written_is_committed_at_once(tmp_path):
    # overwrite=False and no chunk yet: the writer holds nothing to protect,
    # so the flush is an ordinary one, and a rollback of the writer later does
    # not take (nor claim to take) it
    ds = _store(tmp_path)
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1", "g2"]}))
    ds.flush()
    w = ds.create_layer_writer("X", 4, 3, dtype=np.float32, overwrite=False)
    ds.metadata["note"] = "x"
    ds.flush()
    assert _other_reader(tmp_path / "s.cytome",
                         "SELECT COUNT(*) FROM _metadata WHERE key = 'note'") == [1]
    w.write_chunk(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), 0)
    with pytest.warns(RuntimeWarning, match="rolled back") as rec:
        ds.close()
    assert "note" not in str(rec[0].message)
    back = cytome.open(str(tmp_path / "s.cytome"))
    assert back.metadata["note"] == "x"
    back.close()
