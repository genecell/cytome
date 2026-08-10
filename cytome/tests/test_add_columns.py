"""Round 25: add_cells_column / add_genes_column / add_entity_column + add_embedding flush default."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _tiny(path, n=20, g=5, with_symbol=False):
    import cytome
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)]}))
    genes = {"gene_idx": np.arange(g), "gene_id": [f"ENSG{i}" for i in range(g)]}
    if with_symbol:
        genes["symbol"] = [f"Gene{i}" for i in range(g)]
    ds.set_entity("genes", pd.DataFrame(genes))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((n, g), np.float32)))
    ds.flush()
    return ds


def test_add_cells_column_persists_without_manual_flush(tmp_path):
    import cytome
    p = tmp_path / "a.cytome"
    ds = _tiny(p); n = ds.n_cells
    ds.add_cells_column("leiden", [f"c{i % 3}" for i in range(n)])   # flush=True default
    ds.close()
    ds = cytome.open(str(p))
    assert "leiden" in ds.cells.columns
    assert list(ds.cells["leiden"])[:3] == ["c0", "c1", "c2"]
    ds.close()


def test_add_genes_column_backfills_symbol_for_resolver(tmp_path):
    from cytome.utils.modality import modality_feature_table_info
    ds = _tiny(tmp_path / "b.cytome", with_symbol=False)
    assert modality_feature_table_info(ds, "RNA")[2] == "gene_id"   # before
    ds.add_genes_column("symbol", [f"Gene{i}" for i in range(5)])
    assert modality_feature_table_info(ds, "RNA")[2] == "symbol"    # after backfill
    ds.close()


def test_add_embedding_flush_default(tmp_path):
    import cytome
    p = tmp_path / "c.cytome"
    ds = _tiny(p); n = ds.n_cells
    ds.add_embedding("X_umap", np.random.RandomState(0).randn(n, 2).astype(np.float32))
    ds.close()
    ds = cytome.open(str(p))
    assert "X_umap" in ds.list_embeddings()
    ds.close()


def test_add_column_length_validation(tmp_path):
    ds = _tiny(tmp_path / "d.cytome")
    with pytest.raises(ValueError):
        ds.add_cells_column("bad", [1, 2, 3])
    ds.close()
