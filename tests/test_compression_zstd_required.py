"""zstandard is a REQUIRED dependency since 0.3.7, as lz4 is.

The writers record the method they ask for beside each blob. Up to 0.3.6, a
``zstd`` request without zstandard compressed with zlib and the blob was still
labelled ``zstd``. cytome's own reader decodes a ``zstd`` label by the blob's
bytes, so nothing in Python noticed; cytome for R before 0.1.2 trusted the label
and failed on every such store ("zstd frame content size unknown"), and any other
reader would too. A store must not record a compression it did not use, so the
codec is required and a missing one raises at the write.
"""
from __future__ import annotations

import os
import zlib

import pytest

from cytome.io import compression as C


def test_zstandard_is_a_declared_dependency():
    """If this fails, zstandard has fallen out of [project] dependencies."""
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("zstandard") for d in deps), deps


def test_zstandard_is_importable_in_a_plain_install():
    assert C._ZSTD_AVAILABLE, "zstandard is required but not importable: check [project] dependencies"


def test_a_zstd_blob_is_a_zstd_frame():
    payload = b"y" * 500
    blob = C.compress_blob(payload, "zstd")
    assert blob[:4] == C._ZSTD_MAGIC
    assert C.decompress_blob(blob, "zstd") == payload


def test_missing_zstandard_raises_instead_of_writing_zlib_under_a_zstd_label(monkeypatch):
    monkeypatch.setattr(C, "_ZSTD_AVAILABLE", False)
    with pytest.raises(ImportError, match="pip install zstandard"):
        C.compress_blob(b"x" * 100, "zstd")
    # zlib asked for by name still works without zstandard
    assert C.decompress_blob(C.compress_blob(b"x" * 100, "zlib"), "zlib") == b"x" * 100


def test_stores_written_by_0_3_6_without_zstandard_still_read():
    """Those blobs are zlib streams under the label zstd; the reader decides by the bytes."""
    payload = bytes(range(256)) * 40
    legacy = zlib.compress(payload, 6)
    assert legacy[:2] in C._ZLIB_HEADERS
    assert C.decompress_blob(legacy, "zstd") == payload
