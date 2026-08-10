from __future__ import annotations

import scipy.sparse as sp

import cytome
from cytome.io.export_bigwig import cache_coverage, export_coverage, export_coverage_region, get_cached_coverage


class TestExportBigWig:
    def _prepare(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file):
        ds = cytome.create(tmp_cytome)
        ds.set_entity("cells", sample_cell_metadata)
        ds.add_matrix("ATAC_counts", sp.csr_matrix((500, 1), dtype="int32"))
        ds.flush()
        ds.ATAC.import_fragments(synthetic_fragments_file)
        return ds

    def test_export_coverage(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file, tmp_path):
        pytest = __import__("pytest")
        pytest.importorskip("pyBigWig")
        ds = self._prepare(tmp_cytome, sample_cell_metadata, synthetic_fragments_file)
        outs = export_coverage(ds, groupby="cell_type", output_dir=tmp_path)
        assert len(outs) > 0
        for p in outs:
            assert p.exists()
        ds.close()

    def test_export_region(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file, tmp_path):
        pytest = __import__("pytest")
        pytest.importorskip("pyBigWig")
        ds = self._prepare(tmp_cytome, sample_cell_metadata, synthetic_fragments_file)
        outs = export_coverage_region(ds, "cell_type", ("chr1", 0, 1_000_000), tmp_path)
        assert len(outs) > 0
        ds.close()

    def test_cache_coverage(self, tmp_cytome, sample_cell_metadata, synthetic_fragments_file):
        ds = self._prepare(tmp_cytome, sample_cell_metadata, synthetic_fragments_file)
        cache_coverage(ds, groupby="cell_type", bin_size=1000, normalize="cpm")
        vals = get_cached_coverage(ds, group_name="CD8 T", chrom="chr1", bin_size=1000, normalize="cpm")
        assert vals.ndim == 1
        ds.close()
