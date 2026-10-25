"""Metadata keyed by labels with "/" survives an h5ad file.

``to_anndata`` copied every metadata entry into ``uns`` as it was. An h5ad file stores each key of a mapping and
each column of a table as a name in the file, where "/" separates groups: anndata 0.13 refuses such a name by
default, and earlier versions split it into nested groups, so a key "L2/3 IT" (a cell type, in the marker tables
PIASO records) came back as "L2". Such names now go into ``uns`` with "/" written as "%2F", and ``from_anndata``
and ``from_h5ad`` turn "%2F" back into "/".
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import cytome

MARKERS = {"L2/3 IT": ["G1", "G2"], "Astro": ["G3"]}
NESTED = {"params": {"floors": {"L2/3 IT": 0.05, "Astro": 0.1}, "mu": 1.0}}


def _store(tmp_path):
    a = ad.AnnData(np.arange(12, dtype=np.float32).reshape(4, 3))
    a.obs_names = [f"c{i}" for i in range(4)]
    a.var_names = ["G1", "G2", "G3"]
    a.obs["ct"] = pd.Categorical(["L2/3 IT", "L2/3 IT", "Astro", "Astro"])
    ds = cytome.from_anndata(a, output=str(tmp_path / "s.cytome"))
    ds.metadata["runGDR_marker_genes"] = MARKERS
    ds.metadata["cosg_run"] = NESTED
    ds.metadata["plain"] = {"a": 1, "b": [1, 2]}
    ds.flush()
    return ds


def _plain(obj):
    """Lists for arrays, so values read back from an h5ad file compare with the originals."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_plain(x) for x in obj]
    return obj.item() if isinstance(obj, np.generic) else obj


def _slash_names(obj, path="uns"):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and "/" in k:
                bad.append(f"{path}/{k}")
            bad += _slash_names(v, f"{path}/{k}")
    elif isinstance(obj, pd.DataFrame):
        bad += [f"{path}:{c}" for c in obj.columns if isinstance(c, str) and "/" in c]
    return bad


def test_to_anndata_writes_no_slash_into_uns(tmp_path):
    ds = _store(tmp_path)
    try:
        a = ds.to_anndata()
    finally:
        ds.close()
    assert _slash_names(dict(a.uns)) == []
    assert a.uns["runGDR_marker_genes"]["L2%2F3 IT"] == ["G1", "G2"]
    assert a.uns["cosg_run"]["params"]["floors"]["L2%2F3 IT"] == 0.05
    assert a.uns["plain"] == {"a": 1, "b": [1, 2]}


@pytest.mark.parametrize("backed", [False, True])
def test_a_round_trip_through_h5ad_restores_the_keys(tmp_path, backed):
    ds = _store(tmp_path)
    try:
        a = ds.to_anndata()
    finally:
        ds.close()
    h5 = tmp_path / "s.h5ad"
    a.write_h5ad(h5)
    back = cytome.from_h5ad(h5, output=str(tmp_path / f"back_{backed}.cytome"), backed=backed, verbose=False)
    try:
        assert _plain(back.metadata["runGDR_marker_genes"]) == MARKERS
        assert _plain(back.metadata["cosg_run"]["params"]["floors"]) == NESTED["params"]["floors"]
        assert _plain(back.metadata["plain"]) == {"a": 1, "b": [1, 2]}
    finally:
        back.close()


def test_an_in_memory_round_trip_restores_the_keys(tmp_path):
    ds = _store(tmp_path)
    try:
        a = ds.to_anndata()
    finally:
        ds.close()
    back = cytome.from_anndata(a, output=str(tmp_path / "mem.cytome"))
    try:
        assert _plain(back.metadata["runGDR_marker_genes"]) == MARKERS
    finally:
        back.close()


def test_writes_where_anndata_refuses_slashes(tmp_path):
    settings = getattr(ad, "settings", None)
    if settings is None or not hasattr(settings, "disallow_forward_slash_in_h5ad"):
        pytest.skip("this anndata has no switch for slashes in h5ad keys")
    ds = _store(tmp_path)
    try:
        a = ds.to_anndata()
    finally:
        ds.close()
    old = settings.disallow_forward_slash_in_h5ad
    settings.disallow_forward_slash_in_h5ad = True
    try:
        a.write_h5ad(tmp_path / "strict.h5ad")
    finally:
        settings.disallow_forward_slash_in_h5ad = old


def test_tables_and_values():
    from cytome.io.convert_anndata import _metadata_value, _uns_value
    df = pd.DataFrame({"L2/3 IT": [1.0], "Astro": [2.0]})
    out = _uns_value({"t": df})
    assert list(out["t"].columns) == ["L2%2F3 IT", "Astro"]
    assert list(_metadata_value(out)["t"].columns) == ["L2/3 IT", "Astro"]
    assert _uns_value([1, "a/b"]) == [1, "a/b"]                   # values are not names
    assert _uns_value("a/b") == "a/b"


def test_two_keys_made_one_are_refused():
    from cytome.io.convert_anndata import _uns_value
    with pytest.raises(ValueError, match="rename one of them"):
        _uns_value({"a/b": 1, "a%2Fb": 2})
