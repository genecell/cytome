"""validate() checks each matrix against its own feature table, and repair()
never makes a matrix unreadable.

Audit 2026-10-27, part 4 critical 8 and improvement 1 (L1 findings 23, 28):
a raw matrix wider than X (scanpy's variable-gene layout) and a sparse obsm of
more than 500 columns were recorded against ``genes``, so the store failed
validation as soon as it was converted; ``filter_cells`` then advised
``repair()``, which set their column count to the gene count and left them
unreadable. validate() now also checks column-chunk counts, row coverage,
graph and fragment chunk counts, chunk dtypes and cell identity; repair()
removes orphan column chunks, gives missing rows a real empty chunk, and
rewrites a matrix whose chunks hold several dtypes.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
import cytome.io.chunked_io as cio
from cytome.io.compression import compress_blob

anndata = pytest.importorskip("anndata")


def _raw_store(tmp_path, old_style=False):
    rng = np.random.default_rng(0)
    raw = sp.csr_matrix(rng.poisson(1, (30, 10)).astype(np.float32))
    full = anndata.AnnData(X=raw, var=pd.DataFrame(index=[f"g{i}" for i in range(10)]))
    a = full[:, :4].copy()
    a.raw = full
    a.obsm["X_ga"] = sp.random(30, 600, density=0.01, format="csr", random_state=1)
    ds = cytome.from_anndata(a, output=str(tmp_path / "a.cytome"))
    if old_style:   # what converters before 0.3.9 recorded
        ds._conn.execute("UPDATE matrix_meta SET col_entity = 'genes' "
                         "WHERE matrix_name IN ('RNA_raw_X', 'RNA_ga')")
        ds._conn.commit()
    return ds, raw


@pytest.mark.parametrize("old_style", [False, True])
def test_raw_and_wide_obsm_validate_and_survive_repair(tmp_path, old_style):
    ds, raw = _raw_store(tmp_path, old_style)
    assert ds.validate(integrity=False).passed
    ds.repair()
    assert ds.matrix_meta("RNA_raw_X")["n_cols"] == 10
    assert ds.matrix_meta("RNA_ga")["n_cols"] == 600
    assert (ds.RNA.layer("raw_X").to_memory() != raw).nnz == 0
    with _no_validation_warning():
        ds.filter_cells(np.arange(30) < 20)
    ds.close()


class _no_validation_warning:
    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings(record=True)
        self._w = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        bad = [str(w.message) for w in self._w if "failed validation" in str(w.message)]
        assert not bad, bad


def test_repair_never_cuts_columns_below_the_entries(tmp_path):
    ds = cytome.create(str(tmp_path / "s.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": list("abc")}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1"]}))
    ds.add_matrix("RNA_wide", sp.csr_matrix(np.array([[0, 0, 0, 5], [1, 0, 0, 0], [0, 0, 0, 0]],
                                                     dtype=np.float32)))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((3, 2), dtype=np.float32)))
    ds.flush()
    assert "matrix_cols:RNA_wide" in ds.validate(integrity=False).checks_failed
    with pytest.warns(RuntimeWarning, match="left at 4"):
        ds.repair()
    assert ds.matrix_meta("RNA_wide")["n_cols"] == 4
    assert ds.RNA.layer("wide").to_memory().toarray()[0, 3] == 5
    # a column count that is safe to fix is fixed
    ds._conn.execute("UPDATE matrix_meta SET n_cols = 5 WHERE matrix_name = 'RNA_counts'")
    ds._conn.commit()
    ds.repair()
    assert ds.matrix_meta("RNA_counts")["n_cols"] == 2
    ds.close()


def test_missing_rows_get_an_empty_chunk(tmp_path):
    ds = cytome.create(str(tmp_path / "s.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(5)]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1"]}))
    ds.flush()
    w = ds.create_layer_writer("RNA_counts", 3, 2, dtype=np.float32)   # three rows of five
    w.write_chunk(sp.csr_matrix(np.ones((3, 2), dtype=np.float32)), 0)
    w.finalize()
    assert "matrix_rows:RNA_counts" in ds.validate(integrity=False).checks_failed
    ds.repair()
    assert ds.validate(integrity=False).passed
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # no "chunks hold" warning
        m = ds.RNA.counts.to_memory()
    assert m.shape == (5, 2) and m[3:].nnz == 0
    ds.close()


def test_chunks_of_several_dtypes_are_found_and_rewritten(tmp_path):
    ds = cytome.create(str(tmp_path / "s.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": list("abcd")}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1"]}))
    ds.flush()
    C = sp.csr_matrix(np.array([[2, 0], [0, 3]], dtype=np.int32))
    for ci, (r0, block) in enumerate(((0, C), (2, C.astype(np.float32)))):
        ds._conn.execute(cio._INSERT_CHUNK_SQL, (
            "RNA_counts", ci, r0, r0 + 2, 2, compress_blob(block.data.tobytes(), "zstd"),
            compress_blob(block.indices.astype(np.int32).tobytes(), "zstd"),
            compress_blob(block.indptr.astype(np.int32).tobytes(), "zstd"),
            str(block.data.dtype), "zstd"))
    ds._conn.execute("INSERT INTO matrix_meta(matrix_name,n_rows,n_cols,n_nonzero,dtype,"
                     "row_entity,col_entity,chunk_size,n_chunks) VALUES "
                     "('RNA_counts',4,2,4,'float32','cells','genes',2,2)")
    ds._conn.commit()
    assert "matrix_chunk_dtype:RNA_counts" in ds.validate(integrity=True).checks_failed
    ds.repair()
    assert ds.validate(integrity=True).passed
    dtypes = {r[0] for r in ds._conn.execute("SELECT DISTINCT dtype FROM matrix_chunks")}
    assert len(dtypes) == 1
    assert ds.RNA.counts.to_memory().toarray().tolist() == [[2, 0], [0, 3], [2, 0], [0, 3]]
    ds.close()


def test_validate_checks_column_chunks_graphs_fragments_and_identity(tmp_path):
    from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks
    ds = cytome.create(str(tmp_path / "s.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b", "a"]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0", "g1"]}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((3, 2), dtype=np.float32)))
    ds.add_graph("RNA_obsp_knn", sp.csr_matrix(np.eye(3, dtype=np.float32)))
    ds.flush()
    ds.RNA.counts.build_feature_index(chunk_size=1)
    s = np.array([10, 20], dtype=np.int32)
    _write_block_chunks(ds._conn, "chr1", s, s + 5, np.array([0, 1], dtype=np.int32),
                        chunk_size=100, compression="lz4", compression_level=1,
                        chunk_idx_start=0, encoding=1)
    _update_fragment_meta(ds._conn, "chr1", 2)
    ds._conn.commit()
    failed = ds.validate(integrity=False).checks_failed
    assert any(f.startswith("cells_duplicate_barcodes") for f in failed)
    ds._conn.execute("UPDATE cells SET sample_id = 'L1' WHERE cell_idx = 2")
    ds._conn.commit()
    assert ds.validate(integrity=False).passed           # (library, barcode) names one cell
    c = ds._conn
    c.execute("DELETE FROM matrix_csc_chunks WHERE chunk_idx = 1")
    c.execute("DELETE FROM graph_chunks")
    c.execute("UPDATE fragment_meta SET n_fragments = 3")
    c.commit()
    failed = ds.validate(integrity=False).checks_failed
    for expect in ("matrix_csc_chunk_count:RNA_counts", "matrix_col_coverage:RNA_counts",
                   "graph_chunk_count:RNA_obsp_knn", "fragment_count:chr1"):
        assert expect in failed, (expect, failed)
    ds.close()


def test_repair_removes_orphan_column_chunks(tmp_path):
    ds = cytome.create(str(tmp_path / "s.cytome"))
    ds.set_entity("cells", pd.DataFrame({"barcode": list("ab")}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g0"]}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((2, 1), dtype=np.float32)))
    ds.flush()
    ds.RNA.counts.build_feature_index()
    ds._conn.execute("DELETE FROM matrix_meta")
    ds._conn.commit()
    assert "matrix_orphan_chunks" in ds.validate(integrity=False).checks_failed
    ds.repair()
    assert ds._conn.execute("SELECT COUNT(*) FROM matrix_csc_chunks").fetchone()[0] == 0
    ds.close()
