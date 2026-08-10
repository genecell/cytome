from __future__ import annotations

import numpy as np

import cytome
from cytome.io.convert_anndata import from_anndata, to_anndata, update_from_anndata


class TestAnnDataConversion:
    def test_from_anndata(self, sample_anndata, tmp_cytome):
        ds = from_anndata(sample_anndata, modality="RNA", output=tmp_cytome)
        assert ds.RNA.counts.shape == sample_anndata.shape
        ds.close()

    def test_to_anndata(self, sample_anndata, tmp_cytome):
        ds = from_anndata(sample_anndata, modality="RNA", output=tmp_cytome)
        out = to_anndata(ds, modality="RNA")
        assert out.shape == sample_anndata.shape
        ds.close()

    def test_roundtrip_preserves_data(self, sample_anndata, tmp_cytome):
        ds = cytome.from_anndata(sample_anndata, modality="RNA", output=tmp_cytome)
        back = ds.to_anndata("RNA")
        assert back.shape == sample_anndata.shape
        assert (back.X != sample_anndata.X).nnz == 0
        ds.close()

    def test_update_from_anndata(self, sample_anndata, tmp_cytome):
        ds = from_anndata(sample_anndata, modality="RNA", output=tmp_cytome)
        sample_anndata.obs["new_col"] = np.arange(sample_anndata.n_obs)
        update_from_anndata(ds, sample_anndata)
        assert "new_col" in ds.cells.columns
        ds.close()
