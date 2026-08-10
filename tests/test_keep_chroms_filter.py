"""keep_chroms='standard' drops non-standard scaffolds at import (peaks + fragments).

Covers:
- populate_peak_rtree skips scaffold peaks with a warning instead of crashing
  (chrom_to_int would raise on e.g. GL456233.1).
- import_fragments(keep_chroms='standard') drops scaffold fragments.
"""

import gzip
import sqlite3
import warnings

import pandas as pd
import pytest

import cytome
from cytome.index.rtree import populate_peak_rtree, create_peak_rtree
from cytome.io.convert_fragments import import_fragments


def _make_peaks_table(conn):
    conn.execute(
        "CREATE TABLE peaks (peak_idx INTEGER PRIMARY KEY, chr TEXT, start INTEGER, end_ INTEGER)"
    )
    rows = [
        (0, "chr1", 100, 600),
        (1, "chr2", 200, 700),
        (2, "GL456233.1", 50, 300),   # scaffold — must be skipped, not crash
        (3, "chrX", 400, 900),
    ]
    conn.executemany("INSERT INTO peaks VALUES (?, ?, ?, ?)", rows)
    conn.commit()


def test_populate_peak_rtree_skips_scaffold():
    conn = sqlite3.connect(":memory:")
    _make_peaks_table(conn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        populate_peak_rtree(conn)   # must NOT raise on GL456233.1
    assert any("non-standard chromosomes" in str(w.message) for w in caught)
    n = conn.execute("SELECT COUNT(*) FROM peaks_rtree").fetchone()[0]
    assert n == 3   # 3 standard peaks indexed, scaffold dropped


def test_import_fragments_keep_chroms_standard(tmp_path):
    frag = tmp_path / "frags.tsv.gz"
    lines = [
        "chr1\t100\t200\tBC1\t1",
        "chr2\t300\t400\tBC1\t1",
        "GL456233.1\t10\t50\tBC2\t1",   # scaffold fragment — must be dropped
        "chrX\t500\t600\tBC2\t1",
    ]
    with gzip.open(frag, "wt") as fh:
        fh.write("\n".join(lines) + "\n")

    ds = cytome.create(str(tmp_path / "frag.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": ["BC1", "BC2"]}))
    ds.flush()
    conn = ds._conn
    mapping = {"BC1": 0, "BC2": 1}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import_fragments(conn, frag, mapping, build_index=True, keep_chroms="standard")
    msgs = [str(w.message) for w in caught]
    assert any("non-standard chromosome" in m for m in msgs)

    # No fragments_GL456233.1 table should exist; standard chroms present
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert not any("GL456233" in t for t in tables)
    assert "fragments_chr1" in tables


def test_import_fragments_keep_all_scaffold_is_the_footgun(tmp_path):
    # The legacy per-row importer names tables ``fragments_<chrom>``; a scaffold
    # like ``GL456233.1`` contains a dot that breaks the SQL identifier — which is
    # precisely why keep_chroms='standard' is the default. Document that here.
    frag = tmp_path / "frags2.tsv.gz"
    with gzip.open(frag, "wt") as fh:
        fh.write("chr1\t100\t200\tBC1\t1\nGL456233.1\t10\t50\tBC1\t1\n")
    ds = cytome.create(str(tmp_path / "frag2.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": ["BC1"]}))
    ds.flush()
    with pytest.raises(sqlite3.OperationalError):
        import_fragments(ds._conn, frag, {"BC1": 0}, build_index=False, keep_chroms="all")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
