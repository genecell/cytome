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
        ds = cytome.create(tmp_cytome)
        ds.add_embedding("e", np.zeros((10, 2), dtype=np.float32))
        assert ds._conn.execute("SELECT COUNT(*) FROM embedding_meta").fetchone()[0] == 0
        ds.flush()
        assert ds._conn.execute("SELECT COUNT(*) FROM embedding_meta").fetchone()[0] == 1
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
