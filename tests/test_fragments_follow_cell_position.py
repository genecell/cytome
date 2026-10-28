"""Fragments follow their cell by position, never by barcode.

A merged store of 10x libraries holds the same barcode in several libraries
(the identity of a cell is its library and its barcode). ``subset`` remapped
fragments to the kept cells by barcode string, so after ``filter_cells`` a
kept cell received the fragments of every cell with its barcode, and
``export_by_group`` exported other libraries' cells of the same barcodes.
Audit 2026-10-27, part 4 critical 1; L2 minor (export_by_group).
"""
import gzip

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cytome.core.dataset import CytomeDataset
from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks
from cytome.io.merge import merge

N = 10
BARCODES = [f"AAAC{i:02d}-1" for i in range(N)]


def _library(path, per_cell, label):
    """A store of N cells, each with ``per_cell`` fragments on chr1."""
    ds = CytomeDataset(path, mode="w")
    ds.set_entity("cells", pd.DataFrame({"barcode": BARCODES, "lib": label}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g1", "g2"]}))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((N, 2), dtype=np.float32)))
    ds.flush()
    cells = np.repeat(np.arange(N, dtype=np.int32), per_cell)
    starts = np.arange(cells.size, dtype=np.int32) * 100 + 1000
    _write_block_chunks(ds._conn, "chr1", starts, starts + 50, cells, chunk_size=500_000,
                        compression="lz4", compression_level=1, chunk_idx_start=0, encoding=1)
    _update_fragment_meta(ds._conn, "chr1", int(cells.size))
    ds._conn.commit()
    ds.close()
    return path


def _per_cell(ds):
    counts = np.zeros(ds.n_cells, dtype=np.int64)
    for _s, _e, c in ds.fragments.iter_chromosome_chunks("chr1"):
        counts += np.bincount(c, minlength=ds.n_cells)
    return counts.tolist()


def _merged(tmp_path):
    a = _library(str(tmp_path / "A.cytome"), 5, "A")
    b = _library(str(tmp_path / "B.cytome"), 20, "B")
    return merge([a, b], output=str(tmp_path / "AB.cytome"), batch_labels=["A", "B"])


def test_subset_keeps_each_cells_own_fragments(tmp_path):
    ab = _merged(tmp_path)
    assert _per_cell(ab) == [5] * N + [20] * N
    keep = np.r_[0:5, N + 5:N + 10]            # five of A, the other five of B
    out = ab.subset(keep, output=str(tmp_path / "s.cytome"))
    assert _per_cell(out) == [5] * 5 + [20] * 5
    out.close()
    only_a = ab.subset(np.arange(5), output=str(tmp_path / "a5.cytome"))
    assert _per_cell(only_a) == [5] * 5            # was [25] * 5: B's cells' fragments too
    only_a.close()
    ab.close()


def test_filter_cells_keeps_each_cells_own_fragments(tmp_path):
    ab = _merged(tmp_path)
    mask = np.zeros(2 * N, dtype=bool)
    mask[N:N + 3] = True                           # three cells of B
    assert ab.filter_cells(mask) == 3
    assert _per_cell(ab) == [20, 20, 20]
    ab.close()


def test_export_by_group_exports_the_groups_cells_only(tmp_path):
    ab = _merged(tmp_path)
    paths = ab.fragments.export_by_group("lib", str(tmp_path / "exp"))
    n_lines = {}
    for p in paths:
        with gzip.open(p, "rt") as fh:
            n_lines[p.name] = sum(1 for _ in fh)
    assert n_lines == {"lib_A.fragments.tsv.gz": 5 * N, "lib_B.fragments.tsv.gz": 20 * N}
    ab.close()
