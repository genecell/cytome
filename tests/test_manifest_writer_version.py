"""The manifest must name the version that actually wrote the file.

`writer_version` was the literal "cytome 0.1.0" for every file the package
ever produced, including files written by 0.2.4. The manifest exists to
answer "which version made this", and it was answering wrong: five files
staged for a public release all claimed 0.1.0.
"""
import sqlite3
import json

import numpy as np
import pytest
import scipy.sparse as sp

import cytome

anndata = pytest.importorskip("anndata")


def _manifest(path):
    con = sqlite3.connect(str(path))
    try:
        return {k: json.loads(v) for k, v in
                con.execute("SELECT key, value FROM _manifest")}
    finally:
        con.close()


def test_writer_version_is_the_installed_version(tmp_path):
    a = anndata.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.float32)))
    out = tmp_path / "t.cytome"
    ds = cytome.from_anndata(a, output=str(out))
    ds.close()

    man = _manifest(out)
    assert man["writer_version"] == f"cytome {cytome.__version__}"
    assert man["writer_version"] != "cytome 0.1.0" or cytome.__version__ == "0.1.0"


def test_format_version_is_not_the_package_version(tmp_path):
    """They are different numbers and must not be conflated: the format is a
    compatibility contract, the package is a release."""
    a = anndata.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.float32)))
    out = tmp_path / "t.cytome"
    ds = cytome.from_anndata(a, output=str(out))
    ds.close()

    man = _manifest(out)
    assert man["format_version"] == "1.0.0"
    assert man["min_reader_version"] == "1.0.0"
