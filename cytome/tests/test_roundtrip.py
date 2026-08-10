from __future__ import annotations

import numpy as np

import cytome


class TestEndToEnd:
    def test_full_workflow(self, sample_anndata, tmp_cytome):
        ds = cytome.from_anndata(sample_anndata, modality="RNA", output=tmp_cytome)
        ds.cells["score"] = np.random.randn(ds.cells.n)
        ds.add_embedding("RNA_umap", np.random.randn(ds.cells.n, 2).astype(np.float32))
        ds.flush()
        ds.close()

        ds2 = cytome.open(tmp_cytome)
        assert "score" in ds2.cells.columns
        assert ds2.embeddings["RNA_umap"].shape[1] == 2
        ds2.close()
