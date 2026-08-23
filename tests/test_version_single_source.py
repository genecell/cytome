"""The package version must have exactly one source of truth.

0.2.6 was prepared by bumping `pyproject.toml` alone while `cytome/__init__.py`
kept `__version__ = "0.2.5"`. Nothing failed locally, because an editable
install's stale metadata made both reads return 0.2.5 -- both wrong, therefore
equal. A clean CI install read 0.2.6 from the metadata and 0.2.5 from the
literal, and the manifest test caught it one push before release.

The fix is structural rather than a second literal kept in step by hand:
`cytome/__init__.py` holds the only literal, pyproject declares the version
dynamic and reads that attribute, and `writer_version` records the running
code's attribute rather than the installed metadata.
"""
import json
import pathlib
import re
import sqlite3

import numpy as np
import pytest
import scipy.sparse as sp

import cytome
from cytome.io import sqlite_engine


def _pyproject() -> str:
    root = pathlib.Path(cytome.__file__).resolve().parent.parent
    pp = root / "pyproject.toml"
    if not pp.exists():                    # installed wheel, no source tree
        pytest.skip("no pyproject.toml beside the package")
    return pp.read_text()


def test_pyproject_declares_no_version_literal_of_its_own():
    """A second literal is a second thing to forget; there must not be one."""
    text = _pyproject()
    assert re.search(r'^version\s*=\s*"', text, re.M) is None, (
        "pyproject.toml carries its own version literal -- it drifts from "
        "cytome/__init__.py the moment a release is prepared")
    assert re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', text, re.M), (
        "pyproject.toml must declare dynamic = [\"version\"]")
    assert re.search(r'version\s*=\s*\{\s*attr\s*=\s*"cytome\.__version__"',
                     text), (
        "the dynamic version must read cytome.__version__")


def test_the_single_literal_is_a_plain_module_level_string():
    """setuptools reads the attribute statically; a computed value breaks the
    build in a way no test here would see."""
    src = pathlib.Path(cytome.__file__).read_text()
    m = re.search(r'^__version__ = "([^"]+)"$', src, re.M)
    assert m, "cytome/__init__.py needs one plain __version__ = \"...\" literal"
    assert m.group(1) == cytome.__version__
    assert len(re.findall(r'^__version__\s*=', src, re.M)) == 1


def test_writer_version_follows_the_code_not_the_metadata(tmp_path,
                                                          monkeypatch):
    """Simulate the stale editable install: metadata says one thing, the code
    another. The manifest must name the code."""
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "0.0.0-stale")
    assert sqlite_engine._package_version() == cytome.__version__

    anndata = pytest.importorskip("anndata")
    a = anndata.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.float32)))
    out = tmp_path / "t.cytome"
    ds = cytome.from_anndata(a, output=str(out))
    ds.close()
    con = sqlite3.connect(str(out))
    try:
        man = {k: json.loads(v) for k, v in
               con.execute("SELECT key, value FROM _manifest")}
    finally:
        con.close()
    assert man["writer_version"] == f"cytome {cytome.__version__}"
    assert "stale" not in man["writer_version"]
