"""A frozen store cannot be written by anything until it is unfrozen, and every in-place edit is recorded.

Why: nine stores had been edited in place in one day, without backups, including a benchmark substrate; nothing
broke, and nothing would have stopped it.
"""
from __future__ import annotations

import os
import sqlite3
import stat
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.io.sqlite_engine import ReadOnlyError, ReadOnlyWarning


def _store(path):
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b", "c"], "cell_type": ["x", "y", "x"]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["G1", "G2"]}))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.arange(6, dtype=np.float32).reshape(3, 2)))
    ds.flush()
    ds.close()
    return path


def _provenance(path):
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return [(r[0], r[1]) for r in con.execute("SELECT operation, function_name FROM _provenance ORDER BY id")]
    finally:
        con.close()


def _tables(path):
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))
    finally:
        con.close()


def test_a_frozen_store_refuses_every_writer_until_it_is_unfrozen(tmp_path):
    path = _store(tmp_path / "f.cytome")
    con = sqlite3.connect(path)                     # an older store: it lacks a table this version defines
    con.execute("DROP TABLE pseudobulk_cubes")
    con.commit()
    con.close()
    before = _tables(path)
    with pytest.raises(ValueError, match="reason"):
        cytome.freeze(path, reason="")
    info = cytome.freeze(path, reason="GRN benchmark substrate")
    assert _tables(path) == before                   # the freeze writes its record, and no schema
    assert info["reason"] == "GRN benchmark substrate" and cytome.frozen_info(path)["at"] == info["at"]
    assert not stat.S_IMODE(os.stat(path).st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    with pytest.raises(ValueError, match="frozen already"):
        cytome.freeze(path, reason="again")

    # cytome opens it for reading, says why, and refuses a write
    with pytest.warns(ReadOnlyWarning, match="frozen"):
        ds = cytome.open(path)
    assert ds.read_only and ds.frozen["reason"] == "GRN benchmark substrate"
    assert ds.cells["cell_type"].tolist() == ["x", "y", "x"]
    with pytest.raises(ReadOnlyError, match="frozen"):
        ds.require_writable("a test")
    ds.close()
    with pytest.raises(ReadOnlyError, match="frozen"):
        cytome.require_writable(path, "a test")
    # nor can another writer (PIASO's Rust tools, the sqlite3 shell) open it for writing
    con = sqlite3.connect(path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("CREATE TABLE intruder (a)")
    con.close()
    # and it is not replaced by a store written over it
    with pytest.raises(ReadOnlyError, match="not replaced"):
        cytome.create(path, force=True)
    assert cytome.frozen_info(path) is not None

    lifted = cytome.unfreeze(path, reason="rerun of the substrate")
    assert lifted["reason"] == "GRN benchmark substrate" and cytome.frozen_info(path) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReadOnlyWarning)
        ds = cytome.open(path)
    assert not ds.read_only
    ds.cells["note"] = ["1", "2", "3"]
    ds.flush()
    ds.close()
    assert [op for op, _ in _provenance(path)][-2:] == ["freeze", "unfreeze"]


def test_an_open_dataset_freezes_itself_and_stays_readable(tmp_path):
    path = _store(tmp_path / "d.cytome")
    ds = cytome.open(path)
    ds.freeze("figure 6 source")
    assert ds.read_only and ds.frozen["reason"] == "figure 6 source"
    assert ds.RNA.counts.to_memory().toarray().sum() == 15
    with pytest.raises(ReadOnlyError):
        ds.cells["note"] = ["1", "2", "3"]
        ds.flush()
    ds.unfreeze("figure revised")
    assert not ds.read_only and ds.frozen is None
    ds.cells["note"] = ["1", "2", "3"]
    ds.flush()
    ds.close()


def test_in_place_edits_leave_a_provenance_record(tmp_path):
    path = _store(tmp_path / "e.cytome")
    ds = cytome.open(path)
    ds.set_entity("peaks", pd.DataFrame({"peak_id": ["chr1:900-1400"], "chr": ["chr1"], "start": [900],
                                         "end_": [1400]}))
    ds.flush()
    gtf = tmp_path / "g.gtf"
    gtf.write_text('chr1\tt\tgene\t1001\t5000\t.\t+\t.\tgene_id "ENSG00000000001.1"; gene_name "G1"; '
                   'gene_type "protein_coding";\n')
    cytome.annotate_peaks_tss(ds, gtf_path=gtf)
    cytome.sync_gene_metadata(ds)
    ds.close()
    names = [f for op, f in _provenance(path) if op == "edit"]
    assert "cytome.import_gtf" in names and "cytome.annotate_peaks_tss" in names
    assert "cytome.sync_gene_metadata" in names                 # G1 got its coordinates
