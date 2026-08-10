from __future__ import annotations

import numpy as np

import cytome


class TestEmbeddingArray:
    def test_write_read(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        arr = np.random.randn(100, 3).astype(np.float32)
        ds.add_embedding("RNA_pca", arr)
        ds.flush()
        out = ds.embeddings["RNA_pca"]
        assert out.shape == arr.shape
        assert np.allclose(out, arr)
        ds.close()

    def test_slice(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        arr = np.arange(200, dtype=np.float32).reshape(100, 2)
        ds.add_embedding("RNA_pca", arr)
        ds.flush()
        emb = ds._conn
        from cytome.core.embedding import EmbeddingArray

        e = EmbeddingArray(emb, "RNA_pca")
        sub = e[:10, :]
        assert sub.shape == (10, 2)
        ds.close()
