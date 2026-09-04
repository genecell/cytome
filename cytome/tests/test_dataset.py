from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cytome


class TestCytomeDataset:
    def test_create_empty(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        assert ds.n_cells == 0
        ds.close()

    def test_open_close(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        ds.close()
        ds2 = cytome.open(tmp_cytome)
        ds2.close()

    def test_context_manager(self, tmp_cytome):
        with cytome.create(tmp_cytome) as ds:
            ds.set_entity("cells", {"barcode": ["c1", "c2"], "sample_id": ["S1", "S1"]})
        ds2 = cytome.open(tmp_cytome)
        assert ds2.cells.n == 2
        ds2.close()

    def test_repr(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        text = repr(ds)
        assert "CytomeDataset" in text
        ds.close()

    def test_flush(self, tmp_cytome):
        """add_embedding persists immediately; flush=False batches it.

        The default is flush=True -- writing an embedding and not seeing it on
        disk is the surprising behaviour, not the other way round. flush=False
        is for callers adding several at once.
        """
        def n_rows(ds):
            return ds._conn.execute(
                "SELECT COUNT(*) FROM embedding_meta").fetchone()[0]

        ds = cytome.create(tmp_cytome)
        ds.add_embedding("e", np.zeros((10, 2), dtype=np.float32))
        assert n_rows(ds) == 1                       # persisted by default
        ds.add_embedding("f", np.zeros((10, 2), dtype=np.float32), flush=False)
        assert n_rows(ds) == 1                       # held back
        ds.flush()
        assert n_rows(ds) == 2
        ds.close()

    def test_validate_valid(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        report = ds.validate()
        assert report.passed
        ds.close()

    def test_write_behind_matrix(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        m = sp.identity(5, format="csr", dtype=np.float32)
        ds.add_matrix("RNA_counts", m)
        assert ds._conn.execute("SELECT COUNT(*) FROM matrix_meta").fetchone()[0] == 0
        ds.flush()
        assert ds._conn.execute("SELECT COUNT(*) FROM matrix_meta").fetchone()[0] == 1
        ds.close()
