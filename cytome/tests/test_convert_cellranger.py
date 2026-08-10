from __future__ import annotations

import gzip

import cytome
from cytome.io.convert_cellranger import from_cellranger, from_cellranger_arc


class TestConvertCellranger:
    def test_from_cellranger(self, synthetic_cellranger_dir, tmp_path):
        out = tmp_path / "cr.cytome"
        ds = from_cellranger(synthetic_cellranger_dir, out)
        assert ds.n_cells == 500
        assert ds.RNA.counts.shape[1] == 200
        ds.close()

    def test_from_cellranger_arc(self, synthetic_cellranger_dir, synthetic_fragments_file, tmp_path):
        outs = synthetic_cellranger_dir
        with open(synthetic_fragments_file, "rb") as f_in, open(outs / "atac_fragments.tsv.gz", "wb") as f_out:
            f_out.write(f_in.read())

        peaks = outs / "atac_peaks.bed"
        with open(peaks, "wt") as f:
            f.write("chr1\t100\t200\n")
            f.write("chr1\t300\t400\n")

        out = tmp_path / "arc.cytome"
        ds = from_cellranger_arc(outs, out, import_fragments=True, build_index=True)
        assert ds.n_cells == 500
        assert ds.ATAC.fragments.n_fragments > 0
        ds.close()
