"""lz4 is a REQUIRED dependency, not an optional one.

Fragment import defaults to ``compression="lz4"`` (convert_fragments) and the
merge fragment path hardcodes it, so a `pip install cytome` without lz4 cannot
import or merge fragments at all. It was previously undeclared in pyproject and
the failure surfaced only in a clean CI environment — every developer machine
happened to have lz4 already.
"""
from __future__ import annotations

import pytest

from cytome.io import compression as C


def test_lz4_is_importable_in_a_plain_install():
    """If this fails, lz4 has fallen out of the declared dependencies."""
    assert C._LZ4_AVAILABLE, (
        "lz4 is required (fragment import defaults to it and merge hardcodes "
        "it) but is not importable — check [project] dependencies"
    )


def test_lz4_roundtrip():
    payload = b"chr1\t100\t200\tBC1\t1\n" * 500
    blob = C.compress_blob(payload, "lz4")
    assert blob != payload
    assert C.decompress_blob(blob, "lz4") == payload


def test_missing_lz4_raises_something_actionable(monkeypatch):
    """Not a silent fallback to zlib: the method name is stored next to the
    blob, so falling back would label zlib bytes 'lz4' and break decompression
    on read. Failing loudly is the only safe behaviour.
    """
    monkeypatch.setattr(C, "_LZ4_AVAILABLE", False)
    with pytest.raises(ImportError, match="pip install lz4"):
        C.compress_blob(b"x" * 100, "lz4")


def test_zstd_still_degrades_to_zlib(monkeypatch):
    """zstd is genuinely optional and must keep degrading, unlike lz4.

    This asymmetry is the whole point: zstd falls through to zlib, lz4 cannot,
    and that is why one is an extra and the other is a core dependency.
    """
    monkeypatch.setattr(C, "_ZSTD_AVAILABLE", False)
    payload = b"y" * 500
    blob = C.compress_blob(payload, "zstd")
    assert C.decompress_blob(blob, "zstd") == payload
