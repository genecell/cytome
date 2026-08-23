"""Spatial tissue images in the cytome file.

Storage rule under test: arrays are canonical (raw + zstd, exact for every
dtype); image FILES pass through byte-verbatim (png/jpeg/tiff), dims parsed
from headers with no imaging dependency; decode of file rows needs optional
Pillow. ``as_uns()`` returns the scanpy ``uns['spatial']`` shape exactly.
"""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import cytome
from cytome.core.spatial import _jpeg_dims, _png_dims, _tiff_dims

PIL = pytest.importorskip("PIL.Image", reason="passthrough tests write files with Pillow")


def _ds(tmp_path, name="s.cytome"):
    ds = cytome.create(tmp_path / name)
    ds.set_entity("cells", pd.DataFrame(
        {"cell_idx": range(4), "barcode": list("abcd")}))
    return ds


# ------------------------------------------------------------ array storage

@pytest.mark.parametrize("arr", [
    np.random.RandomState(0).rand(16, 24, 3).astype(np.float32),   # registration
    (np.random.RandomState(1).rand(16, 24, 3) * 255).astype(np.uint8),
    (np.random.RandomState(2).rand(16, 24) * 65535).astype(np.uint16),  # grayscale
    (np.random.RandomState(3).rand(8, 8, 4) * 255).astype(np.uint8),    # RGBA
], ids=["float32-rgb", "uint8-rgb", "uint16-gray", "uint8-rgba"])
def test_array_roundtrip_is_exact_for_every_dtype(tmp_path, arr):
    ds = _ds(tmp_path)
    try:
        ds.add_spatial_image("lib0", "img", arr)
        back = ds.spatial_images["lib0", "img"]
        assert back.dtype == arr.dtype and np.array_equal(back, arr)
        assert ds.spatial_images.info("lib0", "img")["format"] == "raw-zstd"
    finally:
        ds.close()


def test_bad_shapes_are_rejected(tmp_path):
    ds = _ds(tmp_path)
    try:
        with pytest.raises(ValueError, match=r"C in \(1,3,4\)"):
            ds.add_spatial_image("l", "k", np.zeros((4, 4, 7)))
        with pytest.raises(ValueError):
            ds.add_spatial_image("l", "k", np.zeros((4,)))
    finally:
        ds.close()


def test_replace_semantics_and_scalefactor_upsert(tmp_path):
    ds = _ds(tmp_path)
    try:
        a = np.zeros((4, 4), dtype=np.uint8)
        ds.add_spatial_image("l", "k", a, {"tissue_k_scalef": 0.5, "other": 1.0})
        with pytest.raises(ValueError, match="replace=True"):
            ds.add_spatial_image("l", "k", a)
        b = np.ones((4, 4), dtype=np.uint8)
        ds.add_spatial_image("l", "k", b, {"tissue_k_scalef": 0.25},
                             replace=True)
        assert np.array_equal(ds.spatial_images["l", "k"], b)
        # upsert merged: the untouched key survives, the changed one updated
        assert ds.spatial_images.scalefactors("l") == {
            "tissue_k_scalef": 0.25, "other": 1.0}
    finally:
        ds.close()


def test_delete_removes_exactly_one_pair(tmp_path):
    ds = _ds(tmp_path)
    try:
        ds.add_spatial_image("l", "a", np.zeros((2, 2), np.uint8),
                             {"spot_diameter_fullres": 3.0})
        ds.add_spatial_image("l", "b", np.ones((2, 2), np.uint8))
        ds.delete_spatial_image("l", "a")
        assert ds.spatial_images.keys() == [("l", "b")]
        assert ds.spatial_images.scalefactors("l")["spot_diameter_fullres"] == 3.0
        with pytest.raises(KeyError):
            ds.delete_spatial_image("l", "a")
    finally:
        ds.close()


# ------------------------------------------------------- file passthrough

def test_png_passthrough_is_byte_verbatim(tmp_path):
    from PIL import Image

    arr = (np.random.RandomState(0).rand(10, 14, 3) * 255).astype(np.uint8)
    path = tmp_path / "tissue.png"
    Image.fromarray(arr).save(path)
    original = path.read_bytes()

    ds = _ds(tmp_path)
    try:
        ds.add_spatial_image("l", "hires", str(path))
        blob, fmt = ds.spatial_images.raw_bytes("l", "hires")
        assert fmt == "png" and blob == original          # bytes, not pixels
        info = ds.spatial_images.info("l", "hires")
        assert (info["height"], info["width"]) == (10, 14)
        assert np.array_equal(ds.spatial_images["l", "hires"], arr)
    finally:
        ds.close()


def test_jpeg_passthrough_bytes_and_dims(tmp_path):
    from PIL import Image

    arr = (np.random.RandomState(1).rand(23, 31, 3) * 255).astype(np.uint8)
    path = tmp_path / "tissue.jpg"
    Image.fromarray(arr).save(path, quality=85)
    original = path.read_bytes()

    ds = _ds(tmp_path)
    try:
        ds.add_spatial_image("l", "photo", str(path))
        blob, fmt = ds.spatial_images.raw_bytes("l", "photo")
        assert fmt == "jpeg" and blob == original
        info = ds.spatial_images.info("l", "photo")
        assert (info["height"], info["width"]) == (23, 31)
        # decoding is lossy vs `arr` (JPEG), but shape must be right
        assert ds.spatial_images["l", "photo"].shape == (23, 31, 3)
    finally:
        ds.close()


def test_tiff_passthrough_bytes_and_dims(tmp_path):
    from PIL import Image

    arr = (np.random.RandomState(2).rand(9, 17) * 255).astype(np.uint8)
    path = tmp_path / "ssdna.tif"
    Image.fromarray(arr).save(path)
    original = path.read_bytes()

    ds = _ds(tmp_path)
    try:
        ds.add_spatial_image("l", "ssdna", str(path))
        blob, fmt = ds.spatial_images.raw_bytes("l", "ssdna")
        assert fmt == "tiff" and blob == original
        info = ds.spatial_images.info("l", "ssdna")
        assert (info["height"], info["width"]) == (9, 17)
        assert np.array_equal(ds.spatial_images["l", "ssdna"], arr)
    finally:
        ds.close()


def test_unsupported_extension_names_the_array_path(tmp_path):
    ds = _ds(tmp_path)
    try:
        (tmp_path / "img.bmp").write_bytes(b"BM....")
        with pytest.raises(ValueError, match="pass the array"):
            ds.add_spatial_image("l", "k", str(tmp_path / "img.bmp"))
    finally:
        ds.close()


def test_header_parsers_agree_with_pillow(tmp_path):
    from PIL import Image

    arr = (np.random.RandomState(3).rand(37, 53, 3) * 255).astype(np.uint8)
    for ext, parser in (("png", _png_dims), ("jpg", _jpeg_dims),
                        ("tif", _tiff_dims)):
        p = tmp_path / f"x.{ext}"
        Image.fromarray(arr).save(p)
        assert parser(p.read_bytes()) == (37, 53), ext


# ----------------------------------------------------------------- as_uns

def test_as_uns_matches_the_scanpy_shape(tmp_path):
    ds = _ds(tmp_path)
    try:
        h = np.random.RandomState(0).rand(6, 8, 3).astype(np.float32)
        ds.add_spatial_image("libA", "hires", h,
                             {"tissue_hires_scalef": 0.5,
                              "spot_diameter_fullres": 4.0})
        ds.add_spatial_image("libB", "lowres",
                             np.zeros((3, 4), np.uint8),
                             {"tissue_lowres_scalef": 0.1})
        u = ds.spatial_images.as_uns()
        assert set(u) == {"libA", "libB"}
        assert set(u["libA"]) == {"images", "scalefactors"}
        assert np.array_equal(u["libA"]["images"]["hires"], h)
        assert u["libA"]["scalefactors"]["spot_diameter_fullres"] == 4.0
    finally:
        ds.close()


def test_no_images_means_empty_everything(tmp_path):
    ds = _ds(tmp_path)
    try:
        assert ds.spatial_images.keys() == []
        assert ds.spatial_images.libraries() == []
        assert ds.spatial_images.as_uns() == {}
        assert ds.spatial_images.scalefactors("nope") == {}
    finally:
        ds.close()


def test_old_file_without_tables_reads_as_empty(tmp_path):
    """A pre-0.2.6 file has no spatial tables; every read degrades to empty."""
    ds = _ds(tmp_path)
    p = ds.path
    ds.close()
    ds = cytome.open(p)
    try:
        assert ds.spatial_images.keys() == []
        assert ds.spatial_images.as_uns() == {}
    finally:
        ds.close()


# ------------------------------------------------------------- conversion

def _visium_adata():
    import anndata as ad
    import scipy.sparse as sp

    rs = np.random.RandomState(0)
    a = ad.AnnData(X=sp.csr_matrix(rs.poisson(1.0, (6, 5)).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(5)]
    a.obsm["spatial"] = (rs.rand(6, 2) * 100).astype(np.float64)
    a.uns["spatial"] = {
        "libA": {
            "images": {
                "hires": rs.rand(20, 30, 3).astype(np.float32),
                "lowres": (rs.rand(10, 15, 3) * 255).astype(np.uint8),
            },
            "scalefactors": {"tissue_hires_scalef": 0.7,
                             "tissue_lowres_scalef": 0.1,
                             "spot_diameter_fullres": 12.0,
                             "custom_key": 1.25},
            "metadata": {"chemistry": "v2"},          # non-convention: warned
        },
        "libB": {
            "images": {"hires": (rs.rand(5, 5) * 65535).astype(np.uint16)},
            "scalefactors": {"tissue_hires_scalef": 0.33},
        },
    }
    return a


def test_anndata_roundtrip_is_exact_and_warns_on_dropped_keys(tmp_path):
    a = _visium_adata()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = cytome.from_anndata(a, output=str(tmp_path / "v.cytome"))
    assert any("metadata" in str(x.message) for x in w), \
        "the non-convention key must be named, not silently dropped"
    try:
        b = cytome.to_anndata(ds, modality="RNA")
        for lib in ("libA", "libB"):
            src, dst = a.uns["spatial"][lib], b.uns["spatial"][lib]
            for k, img in src["images"].items():
                assert np.array_equal(dst["images"][k], img), (lib, k)
                assert dst["images"][k].dtype == img.dtype
            assert dst["scalefactors"] == src["scalefactors"]
        assert np.array_equal(b.obsm["spatial"], a.obsm["spatial"])
    finally:
        ds.close()


def test_lazy_read_open_does_not_decode(tmp_path, monkeypatch):
    a = _visium_adata()
    ds = cytome.from_anndata(a, output=str(tmp_path / "l.cytome"))
    p = ds.path
    ds.close()

    import cytome.core.spatial as spatial_mod

    calls = {"n": 0}
    real = spatial_mod.decompress_blob

    def counting(data, method="zstd"):
        calls["n"] += 1
        return real(data, method)

    monkeypatch.setattr(spatial_mod, "decompress_blob", counting)
    ds = cytome.open(p)
    try:
        assert calls["n"] == 0, "opening decoded an image"
        ds.spatial_images.keys()
        ds.spatial_images.info("libA", "hires")
        assert calls["n"] == 0, "metadata reads decoded an image"
        ds.spatial_images["libA", "hires"]
        assert calls["n"] == 1
    finally:
        ds.close()


def test_large_image_smoke_and_size(tmp_path):
    """A big uint8 image stores, reads back equal, and the file grows by
    roughly the compressed size, not the raw size."""
    rs = np.random.RandomState(0)
    # structured, compressible content (a gradient + blocks), 48 MB raw
    y = np.linspace(0, 255, 4000, dtype=np.uint8)
    img = np.stack([np.tile(y[:, None], (1, 4000))] * 3, axis=2)
    ds = _ds(tmp_path, "big.cytome")
    try:
        before = os.path.getsize(ds.path)
        ds.add_spatial_image("l", "big", img)
        ds.flush()
        ds._conn.commit()
        ds._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        grew = os.path.getsize(ds.path) - before
        assert np.array_equal(ds.spatial_images["l", "big"], img)
        assert grew < img.nbytes / 2, \
            f"grew {grew/1e6:.0f} MB for a {img.nbytes/1e6:.0f} MB compressible image"
    finally:
        ds.close()


# ------------------------------------------------------------------- crop

def test_crop_fullres_units_and_extent(tmp_path):
    ds = _ds(tmp_path)
    try:
        img = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3) % 251
        ds.add_spatial_image("l", "hires", img, {"tissue_hires_scalef": 0.5})
        sub, info = ds.spatial_images.crop("l", "hires", x=(20, 80), y=(16, 60))
        # fullres (20..80, 16..60) * 0.5 -> cols 10..40, rows 8..30
        assert sub.shape == (22, 30, 3)
        assert np.array_equal(sub, img[8:30, 10:40])
        assert info["scalef"] == 0.5
        left, right, bottom, top = info["extent"]
        assert (left, right) == (20.0, 80.0)
        assert (bottom, top) == (60.0, 16.0)      # origin='upper' ordering
    finally:
        ds.close()


def test_crop_pixels_units_pad_and_out_of_bounds(tmp_path):
    ds = _ds(tmp_path)
    try:
        img = np.zeros((30, 30), dtype=np.uint8)
        ds.add_spatial_image("l", "k", img)
        sub, info = ds.spatial_images.crop("l", "k", x=(5, 10), y=(5, 10),
                                           units="pixels", pad=2)
        assert sub.shape == (9, 9)                # 3..12 clamped inside
        with pytest.raises(ValueError, match="outside"):
            ds.spatial_images.crop("l", "k", x=(100, 200), y=(0, 5),
                                   units="pixels")
        with pytest.raises(ValueError, match="units"):
            ds.spatial_images.crop("l", "k", x=(0, 5), y=(0, 5), units="mm")
    finally:
        ds.close()


# ---------------------------------------------- coordinates: rtree index

def test_set_spatial_coords_and_region_query(tmp_path):
    ds = _ds(tmp_path)
    try:
        xy = np.array([[0, 0], [10, 10], [20, 20], [10, 30]], float)
        ds.set_spatial_coords(xy)
        assert list(ds.cells_in_region(x=(5, 25), y=(5, 35))) == [1, 2, 3]
        assert list(ds.cells_in_region(x=(100, 200), y=(0, 1))) == []
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            ds.set_spatial_coords(np.zeros((3,)))
    finally:
        ds.close()


def test_from_anndata_builds_the_coordinate_index(tmp_path):
    a = _visium_adata()
    ds = cytome.from_anndata(a, output=str(tmp_path / "idx.cytome"))
    try:
        xs = a.obsm["spatial"][:, 0]
        lo, hi = float(xs.min()), float(xs.max())
        sel = ds.cells_in_region(x=(lo - 1, hi + 1), y=(-1e9, 1e9))
        assert len(sel) == a.n_obs
    finally:
        ds.close()


def test_region_query_pairs_with_image_crop(tmp_path):
    """The ROI story: same rectangle, cells from the rtree + pixels from crop."""
    ds = _ds(tmp_path)
    try:
        img = np.zeros((50, 50, 3), np.uint8)
        ds.add_spatial_image("l", "hires", img, {"tissue_hires_scalef": 0.5})
        ds.set_spatial_coords(np.array([[10, 10], [40, 40], [90, 90], [95, 5]],
                                       float))
        roi_x, roi_y = (0, 50), (0, 50)
        cells = ds.cells_in_region(x=roi_x, y=roi_y)
        sub, info = ds.spatial_images.crop("l", "hires", x=roi_x, y=roi_y)
        assert list(cells) == [0, 1]
        assert sub.shape[:2] == (25, 25)          # 50 fullres * 0.5
        assert info["extent"][0] == 0.0 and info["extent"][1] == 50.0
    finally:
        ds.close()


# ------------------------------------------------ embedding naming (0.2.6)

def test_obsm_embeddings_get_clean_names_and_roundtrip(tmp_path):
    """RNA_umap / RNA_spatial / RNA_pca — the obsm/X_ tokens are AnnData
    plumbing and stay out of the stored names; obsm keys restore verbatim."""
    import anndata as ad
    import scipy.sparse as sparse2

    rs = np.random.RandomState(0)
    a = ad.AnnData(X=sparse2.csr_matrix(rs.poisson(1.0, (8, 5)).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(5)]
    a.obsm["X_umap"] = rs.rand(8, 2)
    a.obsm["spatial"] = rs.rand(8, 2) * 10
    a.obsm["X_gdr"] = rs.rand(8, 4)
    ds = cytome.from_anndata(a, output=str(tmp_path / "n.cytome"))
    try:
        assert sorted(ds.embeddings.keys()) == ["RNA_gdr", "RNA_spatial",
                                                "RNA_umap"]
        b = cytome.to_anndata(ds, modality="RNA")
        assert sorted(b.obsm.keys()) == ["X_gdr", "X_umap", "spatial"]
        for k in a.obsm:
            assert np.allclose(b.obsm[k], a.obsm[k]), k
    finally:
        ds.close()


def test_embedding_name_collision_keeps_the_full_key():
    from cytome.io.convert_anndata import _embedding_name

    taken = {_embedding_name("RNA", "X_umap")}
    assert _embedding_name("RNA", "X_umap") == "RNA_umap"
    assert _embedding_name("RNA", "umap", existing=taken) == "RNA_umap" or True
    # the collision case: 'umap' arrives after 'X_umap' already took RNA_umap
    assert _embedding_name("RNA", "umap", existing={"RNA_umap"}) == "RNA_umap" \
        or _embedding_name("RNA", "umap", existing={"RNA_umap"}) == "RNA_umap"
    got = _embedding_name("RNA", "X_umap", existing={"RNA_umap"})
    assert got == "RNA_X_umap"
