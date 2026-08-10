from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cytome


def _make_ds(path, prefix, n=30, genes=None):
    ds = cytome.create(path)
    genes = genes or [f"G{i}" for i in range(20)]
    ds.set_entity("cells", {"barcode": [f"{prefix}_{i}" for i in range(n)], "cell_type": ["A"] * n})
    ds.set_entity("genes", {"gene_id": genes, "symbol": genes})
    mat = sp.random(n, len(genes), density=0.2, format="csr", dtype=np.float32, random_state=0)
    ds.add_matrix("RNA_counts", mat)
    ds.metadata["meta"] = {"prefix": prefix}
    ds.flush()
    ds.close()


def test_merge_two_datasets_intersection(tmp_path):
    p1, p2, out = tmp_path / "s1.cytome", tmp_path / "s2.cytome", tmp_path / "m.cytome"
    _make_ds(p1, "s1", genes=[f"G{i}" for i in range(20)])
    _make_ds(p2, "s2", genes=[f"G{i}" for i in range(10, 30)])
    ds = cytome.merge([p1, p2], out, gene_strategy="intersection")
    assert ds.n_cells == 60
    assert ds.n_genes == 10
    ds.close()


def test_merge_union_and_batch_key(tmp_path):
    p1, p2, out = tmp_path / "s1.cytome", tmp_path / "s2.cytome", tmp_path / "u.cytome"
    _make_ds(p1, "s1")
    _make_ds(p2, "s2")
    ds = cytome.merge([p1, p2], out, batch_key="sample_id", gene_strategy="union")
    cells = ds.cells.to_pandas()
    assert "sample_id" in cells.columns
    assert ds.n_genes == 20
    assert any(k.startswith("s1:") for k in ds.metadata.keys())
    ds.close()
