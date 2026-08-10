"""Streaming matrix subset: the subset must be bit-identical to source[keep_idx]
without materialising whole matrices (the ADVIS/HEA filter_cells OOM fix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import cytome
from cytome.io.subset import subset


def _build_multimatrix_cytome(path, n_cells=5000, n_genes=40, n_tiles=120, seed=0):
    """A cytome with two matrices (RNA_counts + ATAC_counts) and >ROW_CAP rows
    so the streaming flush (ROW_CAP=4096) fires mid-loop."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    X = sp.random(n_cells, n_genes, density=0.2, format="csr",
                  random_state=seed, data_rvs=lambda n: rng.integers(1, 9, n)).astype("float32")
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"ct": rng.integers(0, 5, n_cells).astype(str)},
                         index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    ds = cytome.from_anndata(a, modality="RNA", output=str(path))
    # add a second (ATAC) matrix with its own col entity
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": np.arange(n_tiles),
        "peak_id": [f"chr1:{i*100}-{i*100+50}" for i in range(n_tiles)],
        "chr": ["chr1"] * n_tiles,
        "start": [i * 100 for i in range(n_tiles)],
        "end_": [i * 100 + 50 for i in range(n_tiles)],
    }))
    atac = sp.random(n_cells, n_tiles, density=0.1, format="csr",
                     random_state=seed + 1,
                     data_rvs=lambda n: rng.integers(1, 5, n)).astype("float32")
    ds.add_matrix("ATAC_counts", atac)
    ds.flush()
    return ds, X.tocsr(), atac.tocsr()


def test_streaming_matrix_subset_identical_to_source(tmp_path):
    ds, rna_src, atac_src = _build_multimatrix_cytome(tmp_path / "src.cytome")
    n = ds.n_cells
    rng = np.random.default_rng(7)
    mask = rng.random(n) < 0.6           # keep ~60%
    keep_idx = np.where(mask)[0]

    out = subset(ds, mask, output=str(tmp_path / "sub.cytome"),
                 include_fragments=False, include_embeddings=False)
    assert out.n_cells == len(keep_idx)

    # modalities registered (RNA + ATAC)
    assert "RNA" in out.modalities and "ATAC" in out.modalities

    from cytome.core.measurement import MeasurementLayer
    rna_out = MeasurementLayer(out._conn, "RNA_counts").to_memory().tocsr()
    atac_out = MeasurementLayer(out._conn, "ATAC_counts").to_memory().tocsr()

    # bit-identical to source[keep_idx], same row order
    exp_rna = rna_src[keep_idx]
    exp_atac = atac_src[keep_idx]
    assert rna_out.shape == exp_rna.shape
    assert atac_out.shape == exp_atac.shape
    assert (rna_out != exp_rna).nnz == 0, "RNA subset differs from source[keep_idx]"
    assert (atac_out != exp_atac).nnz == 0, "ATAC subset differs from source[keep_idx]"

    # barcodes also aligned to keep_idx
    out_bc = list(out.cells["barcode"])
    src_bc = [f"c{i}" for i in keep_idx]
    assert out_bc == src_bc
    ds.close(); out.close()


def test_streaming_matrix_subset_boolean_and_index_mask_agree(tmp_path):
    """Boolean mask and the equivalent index array produce the same matrices."""
    ds, rna_src, _ = _build_multimatrix_cytome(tmp_path / "src2.cytome", n_cells=3000)
    mask = np.zeros(ds.n_cells, dtype=bool)
    mask[::3] = True
    idx = np.where(mask)[0]

    out_b = subset(ds, mask, output=str(tmp_path / "b.cytome"),
                   include_fragments=False, include_embeddings=False)
    out_i = subset(ds, idx, output=str(tmp_path / "i.cytome"),
                   include_fragments=False, include_embeddings=False)
    from cytome.core.measurement import MeasurementLayer
    mb = MeasurementLayer(out_b._conn, "RNA_counts").to_memory().tocsr()
    mi = MeasurementLayer(out_i._conn, "RNA_counts").to_memory().tocsr()
    assert (mb != mi).nnz == 0
    assert (mb != rna_src[idx]).nnz == 0
    ds.close(); out_b.close(); out_i.close()


def test_filter_cells_all_true_mask_short_circuits(tmp_path):
    """F1: an all-pass mask is a no-op — filter_cells must NOT rewrite the file
    (skips the expensive subset+replace; ADVIS wasted ~2.3h on this at 71 GB)."""
    import os, time
    ds, _, _ = _build_multimatrix_cytome(tmp_path / "f1.cytome", n_cells=300)
    path = str(ds.path)
    n = ds.n_cells
    mtime0 = os.path.getmtime(path)
    time.sleep(1.05)
    n_after = ds.filter_cells(np.ones(n, dtype=bool))
    assert n_after == n
    assert os.path.getmtime(path) == mtime0, "all-True filter must not rewrite the cytome"
    assert not os.path.exists(path + ".filter_tmp"), "no temp file left behind"
    # integer indices covering every cell also short-circuit
    assert ds.filter_cells(np.arange(n)) == n
    # a real (partial) filter still rewrites + drops cells
    m = np.ones(n, dtype=bool); m[:40] = False
    assert ds.filter_cells(m) == n - 40
    assert ds.n_cells == n - 40
    ds.close()
