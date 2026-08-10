"""Force-guard: cytome creators must not silently truncate an existing file.

`create_database` (and therefore every creator funnelling through
`CytomeDataset(mode='w')`) raises FileExistsError when the output exists unless
`force=True`. Tempfile fallbacks and internal scratch creations pass force=True
so only the user-facing output path is guarded.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import cytome
from cytome.io.sqlite_engine import create_database


def _seed(path):
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"barcode": ["a", "b"]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": ["g1", "g2"], "symbol": ["g1", "g2"]}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((2, 2), dtype=np.float32)))
    ds.flush(); ds.close()
    return path


def test_create_database_guards(tmp_path):
    p = str(tmp_path / "x.cytome")
    create_database(p).close()
    with pytest.raises(FileExistsError, match="already exists"):
        create_database(p)
    # force=True overwrites cleanly
    create_database(p, force=True).close()


def test_create_guards_and_force(tmp_path):
    p = _seed(str(tmp_path / "a.cytome"))
    with pytest.raises(FileExistsError):
        cytome.create(p)
    ds = cytome.create(p, force=True)  # overwrite → fresh empty schema
    assert ds.n_cells == 0
    ds.close()


def test_merge_guards_and_force(tmp_path):
    a = _seed(str(tmp_path / "m1.cytome"))
    b = _seed(str(tmp_path / "m2.cytome"))
    out = str(tmp_path / "merged.cytome")
    m = cytome.merge([a, b], output=out)
    m.close()
    with pytest.raises(FileExistsError, match="already exists"):
        cytome.merge([a, b], output=out)
    m2 = cytome.merge([a, b], output=out, force=True)
    assert m2.n_cells == 4
    m2.close()


def test_from_anndata_tempfile_unaffected_but_output_guarded(tmp_path):
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(
        X=sp.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    # output=None → tempfile path: must NOT raise (force=True applied internally)
    ds = cytome.from_anndata(adata)
    ds.close()
    # explicit output: first write ok, second raises, force overwrites
    out = str(tmp_path / "fa.cytome")
    cytome.from_anndata(adata, output=out).close()
    with pytest.raises(FileExistsError):
        cytome.from_anndata(adata, output=out)
    cytome.from_anndata(adata, output=out, force=True).close()
