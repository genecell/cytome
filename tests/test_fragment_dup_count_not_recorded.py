"""``dup_count`` of fragments stored in compressed chunks is missing, not 1 (audit decision 6b).

Compressed chunks (``fragment_chunks``: PIASO's importer, the streaming import, every
``subset``) keep each fragment once and record no duplicate count. The readers gave 1, which
looked like a count; the R package gives NA. Both give a missing value now. The older
per-chromosome tables give the counts they hold, and an exported fragments file has 1 in its
fifth column for chunked fragments, as the 10x format needs an integer there.
"""
import gzip
import warnings

import numpy as np
import pandas as pd

from cytome.core.dataset import CytomeDataset
from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks, import_fragments

BARCODES = ["AAAC-1", "AAAG-1", "AAAT-1"]


def _store(path):
    ds = CytomeDataset(path, mode="w")
    ds.set_entity("cells", pd.DataFrame({"barcode": BARCODES}))
    ds.flush()
    return ds


def _chunked(path):
    ds = _store(path)
    cells = np.array([0, 1, 2, 0], dtype=np.int32)
    starts = np.array([1000, 1100, 1200, 1300], dtype=np.int32)
    _write_block_chunks(ds._conn, "chr1", starts, starts + 50, cells, chunk_size=500_000,
                        compression="lz4", compression_level=1, chunk_idx_start=0, encoding=1)
    _update_fragment_meta(ds._conn, "chr1", int(cells.size))
    ds._conn.commit()
    return ds


def _missing(counts, n):
    assert str(counts.dtype) == "Int64", counts.dtype
    assert len(counts) == n and bool(pd.isna(counts).all()), list(counts)


def test_chunked_fragments_have_no_duplicate_count(tmp_path):
    ds = _chunked(str(tmp_path / "c.cytome"))
    frags = ds.fragments
    _missing(frags.query_region("chr1", 0, 10_000)["dup_count"], 4)
    _missing(frags.query_cells([0])["dup_count"], 2)
    _missing(frags.query_cells_on_chrom("chr1", [1, 2])["dup_count"], 2)
    (chrom, block), = list(frags.iter_chromosomes())
    _missing(block["dup_count"], 4)
    _missing(frags.query_region("chr1", 50_000, 60_000)["dup_count"], 0)      # no fragment there
    ds.close()


def test_export_writes_one_where_no_count_is_recorded(tmp_path):
    ds = _chunked(str(tmp_path / "c.cytome"))
    whole = ds.fragments.export(str(tmp_path / "all.tsv.gz"))
    region = ds.fragments.export(str(tmp_path / "region.tsv.gz"), region=("chr1", 0, 10_000))
    ds.close()
    for path in (whole, region):
        with gzip.open(path, "rt") as fh:
            rows = [line.rstrip("\n").split("\t") for line in fh]
        assert len(rows) == 4 and {r[4] for r in rows} == {"1"}


def test_per_row_tables_give_the_counts_they_hold(tmp_path):
    frag = tmp_path / "frags.tsv"
    frag.write_text("chr1\t1000\t1050\tAAAC-1\t3\nchr1\t1100\t1150\tAAAG-1\t1\n")
    ds = _store(str(tmp_path / "p.cytome"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import_fragments(ds._conn, str(frag), {b: i for i, b in enumerate(BARCODES)})
    for got in (ds.fragments.query_region("chr1", 0, 10_000)["dup_count"],
                ds.fragments.query_cells([0, 1])["dup_count"]):
        assert str(got.dtype) == "Int64" and got.tolist() == [3, 1]
    ds.close()
