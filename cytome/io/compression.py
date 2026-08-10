"""Compression helpers for Cytome BLOB storage."""

from __future__ import annotations

import zlib


_ZSTD_AVAILABLE = True
try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    _ZSTD_AVAILABLE = False
    zstd = None

_LZ4_AVAILABLE = True
try:
    import lz4.block as _lz4_block
except ImportError:  # pragma: no cover - optional dependency
    _LZ4_AVAILABLE = False
    _lz4_block = None


def compress_blob(data: bytes, method: str = "zstd", level: int | None = None) -> bytes:
    """Compress bytes using zstd (preferred), lz4, or zlib.

    Parameters
    ----------
    data
        Raw bytes to compress.
    method
        Compression method name. Supported: ``zstd``, ``zlib``, ``lz4``.
    level
        Compression level override. If None, uses default (zstd=3, zlib=6).

    Returns
    -------
    bytes
        Compressed data.
    """
    chosen = _resolve_method(method)
    if chosen == "zstd" and _ZSTD_AVAILABLE:
        return zstd.ZstdCompressor(level=level if level is not None else 3).compress(data)
    if chosen == "lz4":
        if not _LZ4_AVAILABLE:
            # Deliberately NOT a silent fallback to zlib: the caller records
            # the method name alongside the blob, so falling back would label
            # zlib bytes as lz4 and break decompression later.
            raise ImportError(
                "lz4 compression was requested but lz4 is not installed. "
                "It is a required dependency of cytome: pip install lz4"
            )
        return _lz4_block.compress(data, store_size=True)
    return zlib.compress(data, level=level if level is not None else 6)


def decompress_blob(data: bytes, method: str = "zstd") -> bytes:
    """Decompress bytes using zstd, lz4, or zlib.

    When *method* is ``"lz4"`` or ``"zlib"`` it is trusted directly because
    lz4's 4-byte prepended-size header can collide with zlib magic bytes
    (``\\x78\\x9c``).  Auto-detection is only used when *method* is the
    legacy default ``"zstd"`` to handle old data that was really zlib.

    Parameters
    ----------
    data
        Compressed bytes.
    method
        Compression method used for encoding.

    Returns
    -------
    bytes
        Decompressed data.
    """
    if method in ("lz4", "zlib"):
        actual = method
    else:
        actual = _detect_method(data) if len(data) >= 4 else method
    if actual == "zstd":
        if not _ZSTD_AVAILABLE:
            raise ImportError(
                "Data is zstd-compressed but zstandard is not installed. "
                "pip install zstandard"
            )
        return zstd.ZstdDecompressor().decompress(data)
    if actual == "lz4":
        if not _LZ4_AVAILABLE:
            raise ImportError(
                "Data is lz4-compressed but lz4 is not installed. "
                "pip install lz4"
            )
        return _lz4_block.decompress(data)
    return zlib.decompress(data)


def decode_starts(blob: bytes, method: str, encoding: int = 0):
    """Decompress and decode a starts blob.

    encoding 0: raw int32.  encoding 1: delta-encoded int32 (cumsum).
    """
    import numpy as np
    raw = decompress_blob(blob, method)
    arr = np.frombuffer(raw, dtype=np.int32).copy()
    if encoding == 1:
        arr = np.cumsum(arr, dtype=np.int64).astype(np.int32)
    return arr


def decode_ends(blob: bytes, method: str, starts, encoding: int = 0):
    """Decompress and decode an ends blob.

    encoding 0: raw int32.  encoding 1: lengths (ends = starts + lengths).
    """
    import numpy as np
    raw = decompress_blob(blob, method)
    arr = np.frombuffer(raw, dtype=np.int32).copy()
    if encoding == 1:
        arr = starts + arr
    return arr


def _resolve_method(method: str) -> str:
    if method not in {"zstd", "zlib", "lz4"}:
        raise ValueError(f"Unsupported compression method: {method}")
    if method == "zstd" and not _ZSTD_AVAILABLE:
        return "zlib"
    if method == "lz4" and not _LZ4_AVAILABLE:
        raise ImportError("lz4 not installed. pip install lz4")
    return method


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZLIB_HEADERS = {b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda"}


def _detect_method(data: bytes) -> str:
    """Detect compression method from magic bytes."""
    if data[:4] == _ZSTD_MAGIC:
        return "zstd"
    if data[:2] in _ZLIB_HEADERS:
        return "zlib"
    return "lz4"
