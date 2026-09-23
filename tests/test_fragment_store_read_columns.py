"""FragmentStore.chunks() and FragmentStore.read(): decode only what is asked.

With ``encoding == 1`` the importer stores ``end - start`` in the ends blob,
so the length column is one blob away and needs no starts. ``read`` fetches
only the blobs the requested columns depend on, and ``chunks`` is the
metadata a caller looks at before deciding what to read. Both encodings are
covered, because a cytome can hold chunks written by two importer versions.
"""
import numpy as np
import pandas as pd
import pytest

import cytome
from cytome.io.compression import compress_blob


def _cytome_with_two_encodings(path):
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(4),
                                         "barcode": list("abcd")}))
    starts = np.array([100, 150, 400, 1000], np.int32)
    ends = np.array([250, 300, 470, 1120], np.int32)
    cells = np.array([0, 1, 1, 3], np.int32)
    # chunk 1: encoding 0 -- absolute starts and ends
    ds._conn.execute(
        "INSERT INTO fragment_chunks (chrom, chunk_idx, row_start, row_end, n_fragments, "
        "min_start, starts_blob, ends_blob, cell_idx_blob, compression, encoding) "
        "VALUES ('chr1', 0, 0, 4, 4, 100, ?, ?, ?, 'lz4', 0)",
        (compress_blob(starts.tobytes(), "lz4"), compress_blob(ends.tobytes(), "lz4"),
         compress_blob(cells.tobytes(), "lz4")))
    # chunk 2: encoding 1 -- delta starts, lengths in the ends blob
    deltas = np.diff(starts, prepend=0).astype(np.int32)
    lengths = (ends - starts).astype(np.int32)
    ds._conn.execute(
        "INSERT INTO fragment_chunks (chrom, chunk_idx, row_start, row_end, n_fragments, "
        "min_start, starts_blob, ends_blob, cell_idx_blob, compression, encoding) "
        "VALUES ('chr2', 0, 0, 4, 4, 100, ?, ?, ?, 'lz4', 1)",
        (compress_blob(deltas.tobytes(), "lz4"), compress_blob(lengths.tobytes(), "lz4"),
         compress_blob(cells.tobytes(), "lz4")))
    ds._conn.commit()
    return ds, starts, ends, cells


def test_chunks_lists_metadata_in_storage_order(tmp_path):
    ds, *_ = _cytome_with_two_encodings(tmp_path / "a.cytome")
    rows = ds.fragments.chunks()
    assert [(r[1], r[2], r[3]) for r in rows] == [("chr1", 0, 4), ("chr2", 0, 4)]
    assert ds.fragments.chunks(chrom="chr2")[0][1] == "chr2"
    ds.close()


@pytest.mark.parametrize("chunk", [0, 1])
def test_read_gives_the_same_arrays_under_both_encodings(tmp_path, chunk):
    ds, starts, ends, cells = _cytome_with_two_encodings(tmp_path / "b.cytome")
    cid = ds.fragments.chunks()[chunk][0]
    out = ds.fragments.read(cid, columns=("starts", "ends", "lengths", "cells"))
    np.testing.assert_array_equal(out["starts"], starts)
    np.testing.assert_array_equal(out["ends"], ends)
    np.testing.assert_array_equal(out["lengths"], ends - starts)
    np.testing.assert_array_equal(out["cells"], cells)
    ds.close()


def test_lengths_alone_touches_only_the_ends_blob_when_encoded(tmp_path, monkeypatch):
    """The point of the column: one blob, no cumsum."""
    ds, starts, ends, cells = _cytome_with_two_encodings(tmp_path / "c.cytome")
    import cytome.core.fragments as fr
    calls = []
    real = fr.decode_starts
    monkeypatch.setattr(fr, "decode_starts", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    enc1 = ds.fragments.chunks()[1][0]
    out = ds.fragments.read(enc1, columns=("lengths",))
    np.testing.assert_array_equal(out["lengths"], ends - starts)
    assert calls == [], "encoding 1 must not decode the starts for lengths"
    assert set(out) == {"lengths"}
    # encoding 0 has no choice
    enc0 = ds.fragments.chunks()[0][0]
    out0 = ds.fragments.read(enc0, columns=("lengths",))
    np.testing.assert_array_equal(out0["lengths"], ends - starts)
    assert len(calls) == 1
    ds.close()


def test_unknown_column_and_missing_chunk_are_named(tmp_path):
    ds, *_ = _cytome_with_two_encodings(tmp_path / "d.cytome")
    with pytest.raises(ValueError, match="unknown column"):
        ds.fragments.read(1, columns=("length",))
    with pytest.raises(KeyError, match="no chunk with id 999"):
        ds.fragments.read(999, columns=("lengths",))
    ds.close()
