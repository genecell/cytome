from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

import cytome


torch = pytest.importorskip("torch")


def test_dataloader_basic(tmp_path):
    ds = cytome.create(tmp_path / "dl.cytome")
    ds.set_entity("cells", {"barcode": [f"c{i}" for i in range(40)], "batch": ["b1"] * 40})
    ds.set_entity("genes", {"gene_id": [f"g{i}" for i in range(12)], "symbol": [f"g{i}" for i in range(12)]})
    mat = sp.random(40, 12, density=0.3, format="csr", dtype=np.float32, random_state=3)
    ds.add_matrix("RNA_counts", mat)
    ds.flush()

    loader = ds.to_pytorch(modalities=["RNA"], layers={"RNA": "counts"}, batch_size=8, shuffle=False, num_workers=0)
    total = 0
    for batch in loader:
        assert "RNA" in batch
        total += batch["RNA"].shape[0]
    assert total == 40
    ds.close()
