from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

import cytome


def test_build_iter_drop_csc(tmp_path):
    ds = cytome.create(tmp_path / "x.cytome")
    ds.set_entity("cells", {"barcode": [f"c{i}" for i in range(30)]})
    ds.set_entity("genes", {"gene_id": [f"g{i}" for i in range(20)], "symbol": [f"g{i}" for i in range(20)]})
    mat = sp.random(30, 20, density=0.2, format="csr", dtype=np.float32, random_state=2)
    ds.add_matrix("RNA_counts", mat)
    ds.flush()

    m = ds.RNA.counts
    assert not m.has_feature_index
    with pytest.raises(RuntimeError):
        list(m.iter_columns())

    m.build_feature_index()
    assert m.has_feature_index
    cols = list(m.iter_columns())
    assert len(cols) > 0

    c = m.column(3)
    assert c.shape == (30, 1)
    many = m.columns([1, 3, 5])
    assert many.shape == (30, 3)

    m.drop_feature_index()
    assert not m.has_feature_index
    # fallback still works
    assert m.column(1).shape == (30, 1)
    ds.close()
