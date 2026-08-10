from __future__ import annotations

import cytome


class TestEntityTable:
    def test_create_cells(self, tmp_cytome, sample_cell_metadata):
        ds = cytome.create(tmp_cytome)
        ds.set_entity("cells", sample_cell_metadata)
        ds.flush()
        assert ds.cells.n == 500
        ds.close()

    def test_query(self, tmp_cytome, sample_cell_metadata):
        ds = cytome.create(tmp_cytome)
        ds.set_entity("cells", sample_cell_metadata)
        ds.flush()
        out = ds.cells.query("n_genes > 1000")
        assert len(out) > 0
        ds.close()

    def test_add_column(self, tmp_cytome, sample_cell_metadata):
        ds = cytome.create(tmp_cytome)
        ds.set_entity("cells", sample_cell_metadata)
        ds.flush()
        ds.cells["new_col"] = [1] * ds.cells.n
        ds.flush()
        assert "new_col" in ds.cells.columns
        ds.close()

    def test_getitem_column(self, tmp_cytome, sample_cell_metadata):
        ds = cytome.create(tmp_cytome)
        ds.set_entity("cells", sample_cell_metadata)
        ds.flush()
        vals = ds.cells["barcode"]
        assert vals.shape[0] == 500
        ds.close()

    def test_to_pandas(self, tmp_cytome, sample_cell_metadata):
        ds = cytome.create(tmp_cytome)
        ds.set_entity("cells", sample_cell_metadata)
        ds.flush()
        df = ds.cells.to_pandas()
        assert len(df) == 500
        ds.close()
