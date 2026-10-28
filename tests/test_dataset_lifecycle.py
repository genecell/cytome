"""A dataset's lifecycle: backups, reopens, copies, closed datasets, reads.

Audit 2026-10-27, part 4 majors 16 and 17; L1 findings 10, 11, 24-26:
``filter_cells(keep_original=<the store itself>)`` deleted the store; a kept
layer kept the shape it had before ``filter_cells`` and close()-then-reopen()
dropped kept accessors; ``copy.copy(ds)`` recursed, and ``repr``/``hasattr``
of a closed dataset raised; ``get_categories`` wrote in read mode;
``matrix_meta()`` reported a ``rowid`` column on an older store opened
read-only; ``released()`` left ChunkSource readers and image reconnections open.
"""
import copy
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome


def _store(tmp_path, name="s.cytome", n=30):
    ds = cytome.create(str(tmp_path / name))
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(n)],
                                         "ct": list("ab") * (n // 2)}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"g{i}" for i in range(5)]}))
    A = sp.random(n, 5, density=0.5, format="csr", random_state=0, dtype=np.float32)
    ds.add_matrix("RNA_counts", A)
    ds.add_embedding("RNA_pca", np.arange(n * 2, dtype=np.float64).reshape(n, 2), flush=False)
    ds.flush()
    return ds, A


def _fds_on(path):
    target = os.path.realpath(str(path))
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(os.readlink(f"/proc/self/fd/{fd}")).startswith(target):
                n += 1
        except OSError:
            pass
    return n


@pytest.mark.parametrize("how", ["same path", "hard link"])
def test_keep_original_cannot_be_the_store_itself(tmp_path, how):
    ds, _ = _store(tmp_path)
    target = ds.path
    if how == "hard link":
        target = tmp_path / "link.cytome"
        os.link(ds.path, target)
    with pytest.raises(ValueError, match="store itself"):
        ds.filter_cells(np.arange(30) < 10, keep_original=target)
    assert ds.path.exists() and ds.n_cells == 30
    ds.close()


def test_kept_accessors_follow_filter_cells_and_close_reopen(tmp_path):
    ds, A = _store(tmp_path)
    rows = ds.RNA.counts                            # row-major
    ds.RNA.layer("counts").build_feature_index(chunk_size=2)
    cells = ds.cells
    from cytome.core.embedding import EmbeddingArray
    emb = EmbeddingArray(ds._conn, "RNA_pca")
    ds.filter_cells(np.arange(30) < 10)
    assert rows.shape == (10, 5) and rows.to_memory().shape == (10, 5)
    assert (rows.to_memory() != A[:10]).nnz == 0
    assert (rows.columns([4, 1]) != A[:10][:, [4, 1]]).nnz == 0
    assert emb.shape == (10, 2)
    ds.close()
    ds.reopen()
    assert cells.n == 10 and cells["ct"].tolist()[:2] == ["a", "b"]
    assert rows.shape == (10, 5)
    ds.close()


def test_a_closed_dataset_and_copies_fail_soft(tmp_path):
    ds, _ = _store(tmp_path)
    shallow = copy.copy(ds)                          # recursed in __getattr__
    assert shallow.path == ds.path
    ds.close()
    assert "closed" in repr(ds)
    assert not hasattr(ds, "RNA")
    ds.close()                                       # a second close does nothing


def test_get_categories_never_writes_a_read_only_store(tmp_path):
    ds, _ = _store(tmp_path)
    ds.set_categories("ct", order=["a", "b"])
    ds._conn.execute("UPDATE cells SET ct = 'z' WHERE cell_idx = 0")   # outside the tracked paths
    ds._conn.commit()
    ds.close()
    r = cytome.open(str(tmp_path / "s.cytome"), mode="r")
    assert r.get_categories("ct") is None            # raised ReadOnlyError
    r.close()


def test_matrix_meta_of_an_older_store_opened_read_only(tmp_path):
    ds, _ = _store(tmp_path)
    ds.close()
    con = sqlite3.connect(str(tmp_path / "s.cytome"))
    con.executescript(                                               # as before 0.3.0
        "CREATE TABLE mm AS SELECT matrix_name, n_rows, n_cols, n_nonzero, dtype, row_entity, "
        "col_entity, chunk_size, n_chunks, has_csc, csc_chunk_size, csc_n_chunks, created_at, "
        "provenance_id FROM matrix_meta; DROP TABLE matrix_meta; "
        "ALTER TABLE mm RENAME TO matrix_meta;")
    con.commit()
    con.close()
    r = cytome.open(str(tmp_path / "s.cytome"), mode="r")
    meta = r.matrix_meta("RNA_counts")
    assert "rowid" not in meta and meta["is_integer"] is None
    r.close()


def test_released_closes_chunk_source_readers(tmp_path):
    ds, _ = _store(tmp_path)
    src = ds.chunk_source("RNA_counts", n_readers=2)
    ids = src.chunks("csr")[0]
    first = src.fetch("csr", ids)
    assert src._conns
    with cytome.released(ds):
        assert not src._conns                        # no SQLite handle of this process
        assert ds.is_closed
    again = src.fetch("csr", ids)                    # readers open again
    assert [a[0] for a in again] == [f[0] for f in first]
    src.close()
    ds.close()


def test_an_image_read_after_its_dataset_closed_keeps_no_handle(tmp_path):
    ds, _ = _store(tmp_path)
    ds.add_spatial_image("lib", "hires", np.arange(12, dtype=np.uint8).reshape(3, 4))
    img = ds.spatial_images.as_uns()["lib"]["images"]["hires"]
    ds.close()
    before = _fds_on(tmp_path / "s.cytome")
    assert img.shape == (3, 4)
    assert np.asarray(img).tolist()[0] == [0, 1, 2, 3]
    assert _fds_on(tmp_path / "s.cytome") == before
