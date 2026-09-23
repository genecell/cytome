"""Column-major (CSC-only) matrices through subset and to_anndata.

A matrix written with ``--layout csc`` has column chunks and no row chunks.
Subsetting cells must keep it column-major and select the same rows; loading
it must assemble the same matrix the row-major store gives.
"""
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pytest

import cytome
from cytome.core.measurement import MeasurementLayer


def _make(tmp_path, n=300, g=700):
    rng = np.random.default_rng(1)
    X = sp.random(n, g, density=0.05, format="csr", dtype=np.float32, random_state=rng)
    X.data = np.ceil(X.data * 5).astype(np.float32)
    A = ad.AnnData(X=X); A.obs_names = [f"c{i}" for i in range(n)]; A.var_names = [f"g{j}" for j in range(g)]
    p = tmp_path / "m.cytome"
    ds = cytome.from_anndata(A, modality="RNA", output=str(p))
    ml = MeasurementLayer(ds._conn, "RNA_counts")
    ml.build_feature_index(chunk_size=128)             # CSC chunks beside the rows
    # make it column-major only: drop the row chunks
    ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name='RNA_counts'")
    ds._conn.execute("UPDATE matrix_meta SET n_chunks = 0 WHERE matrix_name='RNA_counts'")
    ds._conn.commit(); ds.close()
    return str(p), X


def test_to_memory_and_to_anndata_assemble_from_columns(tmp_path):
    p, X = _make(tmp_path)
    ds = cytome.open(p)
    ml = MeasurementLayer(ds._conn, "RNA_counts")
    assert ml.is_column_major
    Y = ml.to_memory()
    assert (Y != X).nnz == 0
    A = ds.to_anndata(modality="RNA")
    assert (sp.csr_matrix(A.X) != X).nnz == 0
    with pytest.raises(ValueError):
        next(iter(ml.iter_rows()))
    ds.close()


def test_subset_keeps_the_matrix_column_major_and_selects_rows(tmp_path):
    p, X = _make(tmp_path)
    ds = cytome.open(p)
    keep = np.zeros(X.shape[0], dtype=bool); keep[::3] = True
    out = ds.subset(keep, output=str(tmp_path / "sub.cytome"))
    ml = MeasurementLayer(out._conn, "RNA_counts")
    assert ml.is_column_major, "subset must not silently drop the column layout"
    Y = ml.to_memory()
    assert (Y != X[keep]).nnz == 0
    out.close(); ds.close()
