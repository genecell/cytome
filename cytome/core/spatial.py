"""Spatial tissue images and their scale factors, inside the cytome file.

A spatial dataset's image is a per-dataset artifact and belongs in the single
file — the format's whole selling point. Storage rule (one sentence): **arrays
are canonical; files pass through verbatim.**

- An ``np.ndarray`` is stored as its raw bytes compressed with the codec the
  rest of the file already uses (``format='raw-zstd'``). Exact for every dtype
  — float32 registrations included — and dependency-free in both directions.
  PNG-encoding floats to uint8 would silently break round-trip on exactly the
  images that are not screenshots of 8-bit files.
- A ``.png`` / ``.jpg`` / ``.tif`` **path** is stored as the file's bytes,
  verbatim (``format='png' | 'jpeg' | 'tiff'``). No decode–re-encode, so no
  quality loss and byte-exact provenance; width/height come from a small
  stdlib header parse, so *storing* needs no imaging dependency. *Decoding*
  such a row into an array needs Pillow (or tifffile for TIFF) — optional,
  with an actionable error. Pyramidal/OME multiscale stays out of scope: the
  first plane is what is stored and decoded; that world belongs to OME-Zarr
  tooling, and the interoperation path is conversion.

Scale factors are rows, not a JSON blob: the scanpy convention is a per-library
dict of floats whose keys we must not anticipate (``tissue_hires_scalef``,
``spot_diameter_fullres``, custom entries). Rows survive keys we never heard
of, which is what round-trip fidelity means.

Spot coordinates are NOT stored here — they are the ``spatial`` embedding
(``obsm['spatial']``), in full-resolution pixel units. ``image pixels =
coords * tissue_<img_key>_scalef``.
"""
from __future__ import annotations

import sqlite3
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..io.compression import compress_blob, decompress_blob

_SPATIAL_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS spatial_images (
    library_id   TEXT NOT NULL,
    img_key      TEXT NOT NULL,
    image        BLOB NOT NULL,
    format       TEXT NOT NULL,      -- 'raw-zstd' | 'png' | 'jpeg' | 'tiff'
    height       INTEGER NOT NULL,
    width        INTEGER NOT NULL,
    channels     INTEGER NOT NULL,   -- 1, 3 or 4; 0 = unknown (file passthrough)
    dtype        TEXT NOT NULL,      -- numpy dtype str; '' for file passthrough
    PRIMARY KEY (library_id, img_key)
);
CREATE TABLE IF NOT EXISTS spatial_scalefactors (
    library_id   TEXT NOT NULL,
    key          TEXT NOT NULL,
    value        REAL NOT NULL,
    PRIMARY KEY (library_id, key)
);
"""


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(_SPATIAL_TABLES_SQL)


# --------------------------------------------------------------------------
# Header parsers — dimensions without an imaging dependency.
# --------------------------------------------------------------------------

def _png_dims(data: bytes) -> Tuple[int, int]:
    """(height, width) from a PNG's IHDR. PNG stores width first."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG file")
    width, height = struct.unpack(">II", data[16:24])
    return height, width


def _jpeg_dims(data: bytes) -> Tuple[int, int]:
    """(height, width) from the first SOF marker of a JPEG."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0..SOF15 except DHT(C4)/DAC(CC)/RST/etc.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return height, width
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seg_len
    raise ValueError("no SOF marker found in JPEG")


def _tiff_dims(data: bytes) -> Tuple[int, int]:
    """(height, width) from the FIRST IFD of a (Big)TIFF.

    For a pyramidal/OME file this is the full-resolution plane, which is the
    one decoders return by default.
    """
    if data[:2] == b"II":
        end = "<"
    elif data[:2] == b"MM":
        end = ">"
    else:
        raise ValueError("not a TIFF file")
    magic = struct.unpack(end + "H", data[2:4])[0]
    if magic == 42:                                   # classic TIFF
        ifd_off = struct.unpack(end + "I", data[4:8])[0]
        n_entries = struct.unpack(end + "H", data[ifd_off:ifd_off + 2])[0]
        entry_size, entry_start = 12, ifd_off + 2
        count_fmt, tag_val_off = "I", 8
    elif magic == 43:                                 # BigTIFF
        ifd_off = struct.unpack(end + "Q", data[8:16])[0]
        n_entries = struct.unpack(end + "Q", data[ifd_off:ifd_off + 8])[0]
        entry_size, entry_start = 20, ifd_off + 8
        count_fmt, tag_val_off = "Q", 12
    else:
        raise ValueError("unrecognised TIFF magic")
    width = height = None
    for k in range(int(n_entries)):
        e = entry_start + k * entry_size
        tag, typ = struct.unpack(end + "HH", data[e:e + 4])
        if tag not in (256, 257):
            continue
        # value fits inline for SHORT(3)/LONG(4)/LONG8(16) with count 1
        raw = data[e + tag_val_off:e + tag_val_off + 8]
        if typ == 3:
            val = struct.unpack(end + "H", raw[:2])[0]
        elif typ == 4:
            val = struct.unpack(end + "I", raw[:4])[0]
        else:
            val = struct.unpack(end + "Q", raw[:8])[0]
        if tag == 256:
            width = val
        else:
            height = val
    if width is None or height is None:
        raise ValueError("TIFF first IFD lacks ImageWidth/ImageLength")
    return int(height), int(width)


_FILE_FORMATS = {
    ".png": ("png", _png_dims),
    ".jpg": ("jpeg", _jpeg_dims),
    ".jpeg": ("jpeg", _jpeg_dims),
    ".tif": ("tiff", _tiff_dims),
    ".tiff": ("tiff", _tiff_dims),
}


def _decode_file_bytes(data: bytes, fmt: str) -> np.ndarray:
    """Decode passthrough bytes to an array via optional dependencies."""
    if fmt == "tiff":
        try:
            import tifffile
            return np.asarray(tifffile.imread(__import__("io").BytesIO(data)))
        except ImportError:
            pass
    try:
        from PIL import Image
        import io as _io
        return np.asarray(Image.open(_io.BytesIO(data)))
    except ImportError:
        raise ImportError(
            f"decoding a stored '{fmt}' image needs Pillow "
            f"(`pip install pillow`"
            + (" or tifffile" if fmt == "tiff" else "")
            + "). The image bytes are stored verbatim and are unaffected."
        )


# --------------------------------------------------------------------------
# The accessor
# --------------------------------------------------------------------------

class _SpatialImageAccessor:
    """Lazy access to stored tissue images; mirrors ``_EmbeddingAccessor``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _has_tables(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='spatial_images'"
        ).fetchone()
        return row is not None

    def keys(self) -> List[Tuple[str, str]]:
        if not self._has_tables():
            return []
        rows = self._conn.execute(
            "SELECT library_id, img_key FROM spatial_images "
            "ORDER BY library_id, img_key"
        )
        return [(r[0], r[1]) for r in rows]

    def libraries(self) -> List[str]:
        if not self._has_tables():
            return []
        rows = self._conn.execute(
            "SELECT DISTINCT library_id FROM spatial_images ORDER BY library_id"
        )
        return [r[0] for r in rows]

    def __contains__(self, lk: Tuple[str, str]) -> bool:
        return tuple(lk) in self.keys()

    def __len__(self) -> int:
        return len(self.keys())

    def info(self, library_id: str, img_key: str) -> Dict:
        """Row metadata without decoding: format, height, width, channels, dtype."""
        row = self._conn.execute(
            "SELECT format, height, width, channels, dtype FROM spatial_images "
            "WHERE library_id=? AND img_key=?", (library_id, img_key)
        ).fetchone() if self._has_tables() else None
        if row is None:
            raise KeyError(f"no spatial image ({library_id!r}, {img_key!r})")
        return {"format": row[0], "height": row[1], "width": row[2],
                "channels": row[3], "dtype": row[4]}

    def raw_bytes(self, library_id: str, img_key: str) -> Tuple[bytes, str]:
        """The stored blob and its format — verbatim for file-backed rows."""
        row = self._conn.execute(
            "SELECT image, format FROM spatial_images "
            "WHERE library_id=? AND img_key=?", (library_id, img_key)
        ).fetchone() if self._has_tables() else None
        if row is None:
            raise KeyError(f"no spatial image ({library_id!r}, {img_key!r})")
        return bytes(row[0]), row[1]

    def __getitem__(self, lk: Tuple[str, str]) -> np.ndarray:
        library_id, img_key = lk
        blob, fmt = self.raw_bytes(library_id, img_key)
        meta = self.info(library_id, img_key)
        if fmt == "raw-zstd":
            arr = np.frombuffer(
                decompress_blob(blob, "zstd"), dtype=np.dtype(meta["dtype"])
            )
            shape = (meta["height"], meta["width"])
            if meta["channels"] > 1:
                shape += (meta["channels"],)
            return arr.reshape(shape).copy()
        return _decode_file_bytes(blob, fmt)

    def scalefactors(self, library_id: str) -> Dict[str, float]:
        if not self._has_tables():
            return {}
        rows = self._conn.execute(
            "SELECT key, value FROM spatial_scalefactors WHERE library_id=? "
            "ORDER BY key", (library_id,)
        )
        return {r[0]: r[1] for r in rows}

    def as_uns(self) -> Dict:
        """The scanpy ``uns['spatial']`` shape, exactly.

        ``{lib: {'images': {key: ndarray}, 'scalefactors': {k: float}}}`` —
        consumers of the Visium convention need no cytome-specific logic
        beyond calling this.
        """
        out: Dict = {}
        for lib, key in self.keys():
            entry = out.setdefault(lib, {"images": {}, "scalefactors": {}})
            entry["images"][key] = self[lib, key]
        for lib in list(out):
            out[lib]["scalefactors"] = self.scalefactors(lib)
        # libraries that have scalefactors but no image rows still round-trip
        if self._has_tables():
            for (lib,) in self._conn.execute(
                "SELECT DISTINCT library_id FROM spatial_scalefactors"
            ):
                if lib not in out:
                    out[lib] = {"images": {},
                                "scalefactors": self.scalefactors(lib)}
        return out

    # ---------------------------------------------------------------- ROI
    def crop(self, library_id: str, img_key: str,
             x: Tuple[float, float], y: Tuple[float, float],
             units: str = "fullres",
             pad: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """A rectangular region of the image, addressed in coordinate units.

        ``x``/``y`` are (min, max) ranges. ``units='fullres'`` (default) means
        the same units as the ``spatial`` embedding — full-resolution pixels —
        and the ranges are scaled by ``tissue_<img_key>_scalef`` internally;
        ``units='pixels'`` addresses the stored image directly. ``pad`` (same
        units as the ranges) grows the window, e.g. by one spot diameter.

        Returns ``(sub_image, info)`` where ``info`` carries what a plot needs
        to place the crop: ``x_offset``/``y_offset`` (pixels into the stored
        image), ``scalef``, and ``extent`` — the matplotlib
        ``imshow(extent=...)`` 4-tuple **in the input units**, y already
        ordered for ``origin='upper'``.
        """
        img = self[library_id, img_key]
        sf = 1.0
        if units == "fullres":
            sf = self.scalefactors(library_id).get(
                f"tissue_{img_key}_scalef", 1.0)
        elif units != "pixels":
            raise ValueError("units must be 'fullres' or 'pixels'")
        x0, x1 = sorted(float(v) for v in x)
        y0, y1 = sorted(float(v) for v in y)
        x0, x1 = x0 - pad, x1 + pad
        y0, y1 = y0 - pad, y1 + pad
        h, w = img.shape[0], img.shape[1]
        c0 = max(0, int(np.floor(x0 * sf)))
        c1 = min(w, int(np.ceil(x1 * sf)))
        r0 = max(0, int(np.floor(y0 * sf)))
        r1 = min(h, int(np.ceil(y1 * sf)))
        if c1 <= c0 or r1 <= r0:
            raise ValueError(
                f"ROI x={x}, y={y} ({units}) falls outside the "
                f"{h}x{w} image at scalef={sf:g}")
        sub = img[r0:r1, c0:c1].copy()
        info = {
            "x_offset": c0, "y_offset": r0, "scalef": sf,
            # extent in the INPUT units so scatter coords overlay directly:
            # (left, right, bottom, top) with top < bottom for origin='upper'.
            "extent": (c0 / sf, c1 / sf, r1 / sf, r0 / sf),
        }
        return sub, info


# --------------------------------------------------------------------------
# Write side (called from Dataset methods)
# --------------------------------------------------------------------------

def add_spatial_image(conn: sqlite3.Connection, library_id: str, img_key: str,
                      image, scalefactors: Optional[Dict[str, float]] = None,
                      replace: bool = False) -> None:
    _ensure_tables(conn)
    exists = conn.execute(
        "SELECT 1 FROM spatial_images WHERE library_id=? AND img_key=?",
        (library_id, img_key)).fetchone() is not None
    if exists and not replace:
        raise ValueError(
            f"spatial image ({library_id!r}, {img_key!r}) exists; "
            f"pass replace=True to overwrite")

    if isinstance(image, (str, bytes)) and not isinstance(image, np.ndarray):
        if isinstance(image, bytes):
            raise TypeError("pass an ndarray or a file path, not raw bytes")
        import os
        ext = os.path.splitext(image)[1].lower()
        if ext not in _FILE_FORMATS:
            raise ValueError(
                f"unsupported image file '{ext}'. Supported passthrough "
                f"formats: png, jpg/jpeg, tif/tiff. For anything else, load "
                f"it as an array (tifffile/PIL) and pass the array — arrays "
                f"are stored exactly.")
        fmt, dims = _FILE_FORMATS[ext]
        with open(image, "rb") as fh:
            blob = fh.read()
        height, width = dims(blob)
        channels, dtype = 0, ""            # unknown until decoded
    else:
        arr = np.ascontiguousarray(image)
        if arr.ndim == 2:
            channels = 1
        elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            channels = int(arr.shape[2])
        else:
            raise ValueError(
                f"image must be HxW or HxWxC with C in (1,3,4); "
                f"got shape {arr.shape}")
        height, width = int(arr.shape[0]), int(arr.shape[1])
        dtype = str(arr.dtype)
        fmt = "raw-zstd"
        blob = compress_blob(arr.tobytes(), "zstd")

    conn.execute(
        "INSERT OR REPLACE INTO spatial_images "
        "(library_id, img_key, image, format, height, width, channels, dtype) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (library_id, img_key, blob, fmt, height, width, channels, dtype))

    for k, v in (scalefactors or {}).items():
        conn.execute(
            "INSERT OR REPLACE INTO spatial_scalefactors "
            "(library_id, key, value) VALUES (?,?,?)",
            (library_id, k, float(v)))


def delete_spatial_image(conn: sqlite3.Connection, library_id: str,
                         img_key: str) -> None:
    cur = conn.execute(
        "DELETE FROM spatial_images WHERE library_id=? AND img_key=?",
        (library_id, img_key))
    if cur.rowcount == 0:
        raise KeyError(f"no spatial image ({library_id!r}, {img_key!r})")


# --------------------------------------------------------------------------
# Coordinates: the schema's spatial_coords table + R*-tree
# --------------------------------------------------------------------------
# The base schema has always carried `spatial_coords(cell_idx, x, y, z)` and a
# `spatial_rtree` index (subset.py preserves them) — but nothing wrote them.
# They are the queryable side of spatial data: the `spatial` embedding is what
# analysis and plotting consume, and this table is what makes "which cells are
# in this rectangle?" an indexed lookup instead of a scan. set_spatial_coords
# keeps both in sync from one call.

def set_spatial_coords(conn: sqlite3.Connection, coords,
                       cell_idx=None) -> None:
    """Write per-cell x/y(/z) into ``spatial_coords`` and rebuild the R*-tree.

    ``coords``: (n, 2) or (n, 3) array in full-resolution pixel units (the
    same units as the ``spatial`` embedding). ``cell_idx``: optional explicit
    cell indices (defaults to 0..n-1).
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"coords must be (n, 2) or (n, 3); got {arr.shape}")
    idx = (np.arange(arr.shape[0]) if cell_idx is None
           else np.asarray(cell_idx, dtype=np.int64))
    if idx.shape[0] != arr.shape[0]:
        raise ValueError("cell_idx length != coords length")
    z = arr[:, 2] if arr.shape[1] == 3 else np.full(arr.shape[0], np.nan)
    conn.execute("DELETE FROM spatial_coords")
    conn.execute("DELETE FROM spatial_rtree")
    conn.executemany(
        "INSERT INTO spatial_coords (cell_idx, x, y, z) VALUES (?,?,?,?)",
        [(int(i), float(x), float(y), None if np.isnan(zz) else float(zz))
         for i, x, y, zz in zip(idx, arr[:, 0], arr[:, 1], z)])
    conn.executemany(
        "INSERT INTO spatial_rtree (id, min_x, max_x, min_y, max_y) "
        "VALUES (?,?,?,?,?)",
        [(int(i), float(x), float(x), float(y), float(y))
         for i, x, y in zip(idx, arr[:, 0], arr[:, 1])])


def cells_in_region(conn: sqlite3.Connection, x, y) -> np.ndarray:
    """Cell indices inside the rectangle ``x=(x0,x1), y=(y0,y1)`` — an R*-tree
    range query over ``spatial_coords``, in the same full-resolution units as
    the coordinates and :meth:`_SpatialImageAccessor.crop`. Sorted ascending.
    """
    x0, x1 = sorted(float(v) for v in x)
    y0, y1 = sorted(float(v) for v in y)
    try:
        rows = conn.execute(
            "SELECT id FROM spatial_rtree "
            "WHERE min_x >= ? AND max_x <= ? AND min_y >= ? AND max_y <= ?",
            (x0, x1, y0, y1)).fetchall()
    except sqlite3.OperationalError:
        return np.empty(0, dtype=np.int64)
    return np.array(sorted(r[0] for r in rows), dtype=np.int64)
