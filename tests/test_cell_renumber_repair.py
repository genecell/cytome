"""Renumbering cells carries every table that names a cell by its cell_idx, and only those.

Matrices, embeddings, graphs and fragments name a cell by its position; spatial_coords and cell_sets by its
cell_idx value. repair() of 0.4.0's first version read cell sets from a column that does not exist
(cell_ids_blob; the column is cell_indices, JSON) and fragments from another (cell_id_blob), so both remaps
were skipped without a word, and a store with cell sets kept them pointing at the wrong cells. Audit
2026-11-07, Section 3.4.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

import cytome
from cytome.io.compression import decompress_blob
from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks
from cytome.utils.validation import repair, validate

OLD_IDS = [10, 25, 50, 100]


def _store(path, *, fragments=None, n_fragments=None):
    """Four cells whose cell_idx are OLD_IDS (gaps, as an older subset left them)."""
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"bc{i}" for i in range(4)], "sample_id": ["s"] * 4}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["G0", "G1"], "symbol": ["A", "B"]}))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.arange(8, dtype=np.float32).reshape(4, 2)))
    ds.flush()
    conn = ds._conn
    conn.execute("PRAGMA defer_foreign_keys = ON")
    for pos in (3, 2, 1, 0):
        conn.execute("UPDATE cells SET cell_idx = ? WHERE cell_idx = ?", (OLD_IDS[pos], pos))
    if fragments is not None:
        cells = np.asarray(fragments, dtype=np.int32)
        starts = np.arange(len(cells), dtype=np.int32) * 1000 + 100
        _write_block_chunks(conn, "chr1", starts, starts + 150, cells, chunk_size=3, compression="lz4",
                            compression_level=1, chunk_idx_start=0, encoding=1)
        _update_fragment_meta(conn, "chr1", len(cells))
    if n_fragments is not None:
        for old, nf in zip(OLD_IDS, n_fragments):
            conn.execute("UPDATE cells SET n_fragments = ? WHERE cell_idx = ?", (int(nf), old))
    conn.commit()
    return ds


def _fragment_cells(ds):
    out = []
    for blob, comp in ds._conn.execute("SELECT cell_idx_blob, compression FROM fragment_chunks ORDER BY chrom, chunk_idx"):
        out.extend(np.frombuffer(decompress_blob(blob, comp), dtype=np.int32).tolist())
    return out


def test_repair_carries_cell_sets_and_spatial_coords_by_value(tmp_path):
    ds = _store(tmp_path / "a.cytome")
    ds._conn.executemany("INSERT INTO cell_sets (set_name, cell_indices) VALUES (?, ?)",
                         [("odd", json.dumps([25, 100])), ("first", json.dumps([10]))])
    ds._conn.executemany("INSERT INTO spatial_coords (cell_idx, x, y, z) VALUES (?, ?, ?, ?)",
                         [(25, 1.0, 2.0, 0.0), (100, 3.0, 4.0, 0.0)])
    ds._conn.commit()
    assert not validate(ds, integrity=False).passed

    actions = repair(ds)
    assert any(a.startswith("cells: compacted cell_idx") for a in actions)
    assert [r[0] for r in ds._conn.execute("SELECT cell_idx FROM cells ORDER BY rowid")] == [0, 1, 2, 3]
    sets = dict(ds._conn.execute("SELECT set_name, cell_indices FROM cell_sets"))
    assert json.loads(sets["odd"]) == [1, 3] and json.loads(sets["first"]) == [0]
    assert ds._conn.execute("SELECT cell_idx, x FROM spatial_coords ORDER BY cell_idx").fetchall() == [(1, 1.0), (3, 3.0)]
    assert validate(ds, integrity=False).passed
    ds.close()


def test_repair_leaves_positional_fragments_alone(tmp_path):
    frags = [0, 1, 1, 2, 3, 3, 3]                       # by position: what every writer produces
    ds = _store(tmp_path / "b.cytome", fragments=frags, n_fragments=[1, 2, 1, 3])
    actions = repair(ds)
    assert any(a.startswith("cells: compacted cell_idx") and "rewritten" not in a for a in actions)
    assert _fragment_cells(ds) == frags
    ds.close()


def test_repair_rewrites_fragments_written_by_cell_idx(tmp_path):
    frags = [10, 25, 25, 50, 100, 100, 100]             # by cell_idx value
    ds = _store(tmp_path / "c.cytome", fragments=frags, n_fragments=[1, 2, 1, 3])
    actions = repair(ds)
    assert any("rewritten from cell_idx values to positions" in a for a in actions)
    assert _fragment_cells(ds) == [0, 1, 1, 2, 3, 3, 3]
    ds.close()


def test_repair_leaves_the_cells_when_fragments_fit_neither_reading(tmp_path):
    frags = [0, 1, 1, 2, 3, 3, 3]
    ds = _store(tmp_path / "d.cytome", fragments=frags, n_fragments=[5, 5, 5, 5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actions = repair(ds)
    assert any(a.startswith("cells: left as is") for a in actions)
    assert any("neither reading" in str(w.message) for w in caught)
    assert [r[0] for r in ds._conn.execute("SELECT cell_idx FROM cells ORDER BY rowid")] == OLD_IDS
    assert _fragment_cells(ds) == frags
    ds.close()


def test_rewriting_cells_carries_cell_sets_and_refuses_a_different_row_count(tmp_path):
    ds = _store(tmp_path / "e.cytome")
    ds._conn.execute("INSERT INTO cell_sets (set_name, cell_indices) VALUES (?, ?)", ("last", json.dumps([100])))
    ds._conn.commit()
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"bc{i}" for i in range(4)], "sample_id": ["t"] * 4}))
    ds.flush()
    assert json.loads(ds._conn.execute("SELECT cell_indices FROM cell_sets").fetchone()[0]) == [3]

    ds.set_entity("cells", pd.DataFrame({"barcode": ["x", "y"], "sample_id": ["t", "t"]}))
    try:
        ds.flush()
    except ValueError as exc:
        assert "cell sets" in str(exc)
    else:
        raise AssertionError("a cells rewrite to another row count must not orphan the cell sets")
    ds.close()


def test_repair_rebuilds_the_peak_rtree_after_compacting_peaks(tmp_path):
    ds = cytome.create(tmp_path / "f.cytome")
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b"]}))
    ds.set_entity("peaks", pd.DataFrame({"peak_id": ["chr1:100-200", "chr1:300-400"], "chr": ["chr1", "chr1"],
                                         "start": [100, 300], "end_": [200, 400]}))
    ds.flush()
    from cytome.index.builder import build_peak_index
    conn = ds._conn
    for new, old in ((7, 1), (5, 0)):
        conn.execute("UPDATE peaks SET peak_idx = ? WHERE peak_idx = ?", (new, old))
    build_peak_index(conn)
    conn.commit()
    assert sorted(r[0] for r in conn.execute("SELECT id FROM peaks_rtree")) == [5, 7]

    actions = repair(ds)
    assert any(a.startswith("peaks: compacted peak_idx") and "R-tree rebuilt" in a for a in actions)
    assert sorted(r[0] for r in conn.execute("SELECT id FROM peaks_rtree")) == [0, 1]
    ds.close()
