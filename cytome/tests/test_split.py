"""split(): one store per group, each matrix read once, and every store exactly what subset() writes."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.io.subset import split, subset
from cytome.core.measurement import MeasurementLayer


def _store(tmp_path, csc=True, n=700, g=400):
    import anndata as ad
    rng = np.random.default_rng(3)
    X = sp.random(n, g, density=0.05, format="csr", random_state=4, dtype=np.float32)
    X.data = rng.poisson(2, X.nnz).astype(np.float32) + 1
    grp = rng.choice(["a", "b", "c", "dd"], n, p=[0.5, 0.3, 0.15, 0.05])
    A = ad.AnnData(X=X, obs=pd.DataFrame({"grp": grp}, index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(g)]))
    p = tmp_path / "src.cytome"
    ds = cytome.from_anndata(A, modality="ATAC", output=str(p))
    ds.metadata["note"] = "kept"
    if csc:
        csc_m = X.tocsc()
        ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name='ATAC_counts'")
        k = 0
        for k, s in enumerate(range(0, g, 37)):
            e = min(s + 37, g); sub = csc_m[:, s:e].tocsc(); sub.sort_indices()
            ds._conn.execute(
                "INSERT INTO matrix_csc_chunks (matrix_name, chunk_idx, col_start, col_end, n_nonzero,"
                " data_blob, indices_blob, indptr_blob, dtype, compression) VALUES (?,?,?,?,?,?,?,?,?,'none')",
                ("ATAC_counts", k, s, e, int(sub.nnz), sub.data.astype(np.float32).tobytes(),
                 sub.indices.astype(np.int32).tobytes(), sub.indptr.astype(np.int32).tobytes(), "float32"))
        ds._conn.execute("UPDATE matrix_meta SET n_chunks=0, has_csc=1, csc_chunk_size=37, csc_n_chunks=? "
                         "WHERE matrix_name='ATAC_counts'", (k + 1,))
        ds._conn.commit()
    return ds, X, grp


def _matrix(ds):
    ml = MeasurementLayer(ds._conn, "ATAC_counts")
    if ds._conn.execute("SELECT n_chunks FROM matrix_meta WHERE matrix_name='ATAC_counts'").fetchone()[0] == 0:
        return sp.hstack([c for _, _, c in ml.iter_columns()]).tocsr()
    return ml.to_memory().tocsr()


def _table(ds, t):
    return ds._conn.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall()


@pytest.mark.parametrize("csc", [True, False])
def test_every_store_is_what_subset_writes(tmp_path, csc):
    ds, X, grp = _store(tmp_path, csc=csc)
    groups = ["a", "c", "dd"]                      # "b" is not split off
    outs = {g: str(tmp_path / f"split_{g}.cytome") for g in groups}
    stores = split(ds, grp, outs, matrices=["ATAC_counts"])
    assert set(stores) == set(groups)
    for g in groups:
        ref = subset(ds, grp == g, str(tmp_path / f"sub_{g}.cytome"), matrices=["ATAC_counts"],
                     include_fragments=False, include_embeddings=False)
        got = stores[g]
        A, B = _matrix(got), _matrix(ref)
        assert A.shape == B.shape == (int((grp == g).sum()), X.shape[1])
        assert (A != B).nnz == 0 and (A != X[np.flatnonzero(grp == g)]).nnz == 0
        for t in ("cells", "peaks", "matrix_csc_chunks" if csc else "matrix_chunks"):
            assert _table(got, t) == _table(ref, t), t
        meta = "SELECT n_rows, n_cols, n_nonzero, dtype, n_chunks, has_csc, csc_n_chunks FROM matrix_meta"
        assert got._conn.execute(meta).fetchall() == ref._conn.execute(meta).fetchall()
        assert got.modalities == ref.modalities == ["ATAC"]
        assert got.metadata["note"] == "kept"
        got.close(); ref.close()
    ds.close()


def test_bad_labels_are_refused(tmp_path):
    ds, X, grp = _store(tmp_path)
    with pytest.raises(ValueError, match="labels for"):
        split(ds, grp[:-1], {"a": str(tmp_path / "x.cytome")}, matrices=["ATAC_counts"])
    with pytest.raises(ValueError, match="no cell has the label"):
        split(ds, grp, {"zz": str(tmp_path / "y.cytome")}, matrices=["ATAC_counts"])
    with pytest.raises(KeyError, match="not in the store"):
        split(ds, grp, {"a": str(tmp_path / "z.cytome")}, matrices=["RNA_counts"])
    ds.close()


def test_dataset_method(tmp_path):
    ds, X, grp = _store(tmp_path)
    stores = ds.split(grp, {"a": str(tmp_path / "m.cytome")}, ["ATAC_counts"])
    assert stores["a"].n_cells == int((grp == "a").sum())
    stores["a"].close(); ds.close()
