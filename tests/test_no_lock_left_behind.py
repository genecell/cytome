"""No write transaction is left open behind the caller's back.

Found by the PIASO import work on 2026-10-28 (piaso-tools-a): cytome.from_cellranger
with several folders discarded the dataset cytome.merge returned, whose last
write (the provenance record) was never committed, and a reference cycle kept
its connection alive until a garbage-collector pass. A following write in the
same process (PIASO's importCellRanger importing fragments) waited out the 60 s
busy timeout and failed with "database is locked". ProvenanceLog.log left the
same open transaction behind for any caller outside a flush.

(A connection that is not closed is freed only by a gc pass in any case: CPython's
sqlite3 keeps a statement cache that refers back to its connection.)
"""
import gc
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome

h5py = pytest.importorskip("h5py")


def _can_write(path, timeout=1.0):
    con = sqlite3.connect(str(path), timeout=timeout)
    t = time.perf_counter()
    try:
        con.execute("BEGIN IMMEDIATE")
        con.rollback()
        return time.perf_counter() - t < timeout
    except sqlite3.OperationalError:
        return False
    finally:
        con.close()


def _folder(root, name, n):
    f = root / name
    f.mkdir()
    M = sp.csc_matrix(np.ones((2, n), dtype=np.int32))
    with h5py.File(f / "filtered_feature_bc_matrix.h5", "w") as h:
        g = h.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = M.data, M.indices, M.indptr
        g["shape"] = np.array(M.shape)
        g["barcodes"] = np.array([f"AAA{i}".encode() for i in range(n)])
        ft = g.create_group("features")
        ft["id"] = np.array([b"E1", b"E2"])
        ft["name"] = np.array([b"G1", b"G2"])
        ft["feature_type"] = np.array([b"Gene Expression"] * 2)
    return str(f)


def test_from_cellranger_of_several_folders_leaves_the_store_writable(tmp_path):
    out = tmp_path / "m.cytome"
    gc.disable()                                   # what a gc pass would hide
    try:
        ds = cytome.from_cellranger([_folder(tmp_path, "A", 3), _folder(tmp_path, "B", 2)],
                                    str(out), sample_name=["a", "b"], import_fragments=False)
        ds.close()
        assert _can_write(out)
    finally:
        gc.enable()


def test_a_provenance_record_is_committed(tmp_path):
    ds = cytome.create(str(tmp_path / "p.cytome"))
    ds.provenance.log(operation="test", function_name="f", parameters={}, package_name="x",
                      package_version="0", input_objects=[], output_objects=[])
    assert not ds._conn.in_transaction
    assert _can_write(tmp_path / "p.cytome")
    other = sqlite3.connect(str(tmp_path / "p.cytome"))
    assert other.execute("SELECT COUNT(*) FROM _provenance WHERE operation = 'test'").fetchone()[0] == 1
    other.close()
    ds.close()
