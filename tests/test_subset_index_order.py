"""subset says when it sorts or drops the indices it is given.

Audit 2026-10-27, L2 minor: subset silently sorted and de-duplicated integer
indices (a subset holds each cell once, in the store's order).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import cytome


@pytest.fixture
def ds(tmp_path):
    d = cytome.create(str(tmp_path / "s.cytome"))
    d.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(5)]}))
    d.flush()
    yield d
    d.close()


def test_repeated_and_unsorted_indices_are_reported(ds, tmp_path):
    with pytest.warns(UserWarning, match="repeated"):
        out = ds.subset([1, 1, 3], output=str(tmp_path / "a.cytome"))
    assert out.cells["barcode"].tolist() == ["c1", "c3"]
    out.close()
    with pytest.warns(UserWarning, match="store's order"):
        out = ds.subset([3, 1], output=str(tmp_path / "b.cytome"))
    assert out.cells["barcode"].tolist() == ["c1", "c3"]
    out.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds.subset(np.array([True, False, True, False, False]),
                  output=str(tmp_path / "c.cytome")).close()
        ds.subset([0, 4], output=str(tmp_path / "d.cytome")).close()
