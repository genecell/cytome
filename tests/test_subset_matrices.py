"""subset(matrices=...) carries only the matrices named."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome


def _two_matrix_store(path):
    ds = cytome.create(str(path))
    n = 30
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)]}))
    ds.set_entity("peaks", pd.DataFrame({"peak_idx": np.arange(50), "peak_id": [f"chr1:{i}-{i+1}" for i in range(50)],
                                         "chr": ["chr1"] * 50, "start": np.arange(50), "end_": np.arange(50) + 1}))
    ds.set_entity("tiles", pd.DataFrame({"tile_idx": np.arange(80), "tile_id": [f"chr1:{i}-{i+1}" for i in range(80)],
                                         "chr": ["chr1"] * 80, "start": np.arange(80), "end_": np.arange(80) + 1}))
    rng = np.random.default_rng(0)
    ds.add_matrix("ATAC_counts", sp.csr_matrix((rng.random((n, 50)) < 0.3).astype(np.float32)))
    ds.add_matrix("tiles_counts", sp.csr_matrix((rng.random((n, 80)) < 0.3).astype(np.float32)))
    ds.flush()
    return ds


def _matrices(ds):
    return sorted(r[0] for r in ds._conn.execute("SELECT matrix_name FROM matrix_meta"))


def test_only_the_named_matrix_is_carried(tmp_path):
    ds = _two_matrix_store(tmp_path / "two.cytome")
    mask = np.zeros(30, bool); mask[5:20] = True
    sub = ds.subset(mask, output=str(tmp_path / "sub.cytome"), matrices=["ATAC_counts"],
                    include_fragments=False, include_embeddings=False)
    assert _matrices(sub) == ["ATAC_counts"]
    assert sub.n_cells == 15
    assert "ATAC" in sub.modalities and "tiles" not in sub.modalities
    sub.close(); ds.close()


def test_default_carries_everything(tmp_path):
    ds = _two_matrix_store(tmp_path / "two.cytome")
    sub = ds.subset(np.ones(30, bool), output=str(tmp_path / "all.cytome"),
                    include_fragments=False, include_embeddings=False)
    assert _matrices(sub) == ["ATAC_counts", "tiles_counts"]
    sub.close(); ds.close()


def test_naming_a_matrix_the_store_lacks_is_an_error(tmp_path):
    ds = _two_matrix_store(tmp_path / "two.cytome")
    with pytest.raises(KeyError, match="not in the store"):
        ds.subset(np.ones(30, bool), output=str(tmp_path / "x.cytome"), matrices=["RNA_counts"])
    ds.close()
