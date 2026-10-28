"""The chunk assembly used by the fragment and peak quantifiers stores one
entry per (cell, feature), not one per hit.

scipy's ``csr_matrix((data, (rows, cols)))`` keeps duplicate entries (scipy
1.13 does not sum them at construction), so a cell with three fragments in one
peak was stored as three entries of 1: dense values were right, but ``nnz``
and ``matrix_meta.n_nonzero`` were inflated, ``binary=True`` clipped each
duplicate to 1 and left the sum at 3, and any transform applied per stored
entry (log1p in TF-IDF) was wrong. Audit 2026-10-27, H1 finding 4.
"""
import os

import numpy as np
import scipy.sparse as sp

from cytome.io.chunking import _assemble_csr, _process_chunk


def _chunk_file(tmp_path, pairs):
    path = os.path.join(str(tmp_path), "chunk_0.bin")
    np.asarray(pairs, dtype=np.int32).tofile(path)
    return path


def test_process_chunk_sums_duplicate_hits(tmp_path):
    # cell 0: three hits in feature 1; cell 1: one hit in feature 0
    path = _chunk_file(tmp_path, [0, 1, 0, 1, 0, 1, 1, 0])
    m = _process_chunk(path, 2, 3, binary=False)
    assert m.nnz == 2
    assert m.has_canonical_format
    assert m.toarray().tolist() == [[0, 3, 0], [1, 0, 0]]


def test_process_chunk_binary_clips_the_sum(tmp_path):
    path = _chunk_file(tmp_path, [0, 1, 0, 1, 0, 1, 1, 0])
    m = _process_chunk(path, 2, 3, binary=True)
    assert m.toarray().tolist() == [[0, 1, 0], [1, 0, 0]]
    assert m.nnz == 2


def test_assemble_csr_canonicalises_its_chunks():
    # a chunk with duplicates, as a caller may build it: from (data, indices, indptr), since
    # scipy >= 1.18 sums duplicates when it builds a CSR from coordinates
    dup = sp.csr_matrix((np.ones(3, dtype=np.float32), np.array([2, 2, 0]), np.array([0, 2, 3])),
                        shape=(2, 3))
    assert dup.nnz == 3
    other = sp.csr_matrix(np.array([[0, 1, 0]], dtype=np.float32))
    out = _assemble_csr([dup, other], 3, 3)
    assert out.nnz == 3
    assert out.toarray().tolist() == [[0, 0, 2], [1, 0, 0], [0, 1, 0]]
    # each stored entry is one (cell, feature): a per-entry transform is right
    log = out.copy()
    log.data = np.log1p(log.data)
    assert np.allclose(log.toarray(), np.log1p(out.toarray()))



def _store_with_a_layer(tmp_path):
    """A store with an ATAC layer over two peaks."""
    import pandas as pd
    import cytome
    ds = cytome.create(str(tmp_path / "q.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": ["c0", "c1"]}))
    ds.set_entity("peaks", pd.DataFrame({"peak_id": ["chr1:1-50", "chr1:60-90"], "chr": ["chr1"] * 2,
                                         "start": [1, 60], "end_": [50, 90]}))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(np.ones((2, 2), dtype=np.float32)),
                  col_entity="peaks")
    ds.flush()
    return ds


def _quantify_again(ds, tmp_path):
    """Quantify the store again, over three peaks."""
    from cytome.io.chunking import _write_chunks_to_cytome
    d = tmp_path / "chunks"
    d.mkdir()
    np.asarray([0, 2, 0, 2, 1, 0], dtype=np.int32).tofile(str(d / "chunk_0.bin"))
    _write_chunks_to_cytome(ds, str(d), 1, 2, 2, 3, False, ["c0", "c1"],
                            var_names=["chr1:1-50", "chr1:60-90", "chr2:5-40"])


def test_quantifier_writes_layer_and_feature_table_together(tmp_path):
    # cytome's quantifier flushes the feature table while its layer writer is
    # open; the flush goes into the writer's transaction (it was refused, and
    # before that it committed the layer half written)
    ds = _store_with_a_layer(tmp_path)
    _quantify_again(ds, tmp_path)
    assert ds.ATAC.counts.to_memory().toarray().tolist() == [[0, 0, 2], [1, 0, 0]]
    assert ds.peaks["peak_id"].tolist() == ["chr1:1-50", "chr1:60-90", "chr2:5-40"]
    assert ds.validate(integrity=False).passed
    ds.close()


def test_a_quantification_stopped_before_finalize_leaves_the_old_layer(tmp_path, monkeypatch):
    import cytome
    import pytest
    from cytome.io.chunked_io import ChunkedLayerWriter

    def stopped(self, *a, **k):
        raise KeyboardInterrupt
    ds = _store_with_a_layer(tmp_path)
    monkeypatch.setattr(ChunkedLayerWriter, "finalize", stopped)
    with pytest.raises(KeyboardInterrupt):
        _quantify_again(ds, tmp_path)
    monkeypatch.undo()
    with pytest.warns(RuntimeWarning, match="rolled back.*entity_table:peaks"):
        ds.close()
    back = cytome.open(str(tmp_path / "q.cytome"))
    assert back.ATAC.counts.to_memory().toarray().tolist() == [[1, 1], [1, 1]]
    assert back.peaks["peak_id"].tolist() == ["chr1:1-50", "chr1:60-90"]
    assert back.validate(integrity=False).passed
    back.close()
