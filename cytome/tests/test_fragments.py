from __future__ import annotations

import scipy.sparse as sp

import cytome


class TestFragmentStore:
    def _prepare_ds(self, tmp_cytome, sample_cell_metadata):
        ds = cytome.create(tmp_cytome)
        ds.set_entity("cells", sample_cell_metadata)
        ds.add_matrix("ATAC_counts", sp.csr_matrix((500, 1), dtype="int32"))
        ds.flush()
        return ds

    def test_import_synthetic_fragments(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file):
        ds = self._prepare_ds(tmp_cytome, sample_cell_metadata)
        ds.ATAC.import_fragments(synthetic_fragments_file)
        assert ds.ATAC.fragments.n_fragments > 0
        ds.close()

    def test_query_region(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file):
        ds = self._prepare_ds(tmp_cytome, sample_cell_metadata)
        ds.ATAC.import_fragments(synthetic_fragments_file)
        out = ds.ATAC.fragments.query_region("chr1", 0, 50_000_000)
        assert out["start"].size >= 0
        ds.close()

    def test_query_regions_batch(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file):
        ds = self._prepare_ds(tmp_cytome, sample_cell_metadata)
        ds.ATAC.import_fragments(synthetic_fragments_file)
        out = ds.ATAC.fragments.query_regions([("chr1", 0, 1000), ("chr2", 0, 1000)])
        assert len(out) == 2
        ds.close()

    def test_query_cells(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file):
        ds = self._prepare_ds(tmp_cytome, sample_cell_metadata)
        ds.ATAC.import_fragments(synthetic_fragments_file)
        out = ds.ATAC.fragments.query_cells([0, 1, 2])
        assert out["cell_idx"].size >= 0
        ds.close()

    def test_export_10x(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file, tmp_path):
        ds = self._prepare_ds(tmp_cytome, sample_cell_metadata)
        ds.ATAC.import_fragments(synthetic_fragments_file)
        out = tmp_path / "frag.tsv.gz"
        ds.ATAC.fragments.export(out)
        assert out.exists()
        ds.close()

    def test_export_by_group(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file, tmp_path):
        ds = self._prepare_ds(tmp_cytome, sample_cell_metadata)
        ds.ATAC.import_fragments(synthetic_fragments_file)
        outs = ds.ATAC.fragments.export_by_group("cell_type", tmp_path)
        assert len(outs) > 0
        ds.close()

    def test_export_region(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file, tmp_path):
        ds = self._prepare_ds(tmp_cytome, sample_cell_metadata)
        ds.ATAC.import_fragments(synthetic_fragments_file)
        out = tmp_path / "frag_region.tsv.gz"
        ds.ATAC.fragments.export(out, region=("chr1", 0, 1_000_000))
        assert out.exists()
        ds.close()
