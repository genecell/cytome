"""Missing values in `obs`, and bulk arrays in `uns`.

Both reported from a Visium HD analysis:

* `from_anndata` raised ``sqlite3.ProgrammingError: Error binding parameter 13:
  type 'NAType' is not supported`` when `obs` carried `pd.NA` — which a
  cell-type deconvolution writes by construction, for every bin it could not
  assign. The user had to substitute a literal string before converting.
* `from_h5ad(..., backed=True)` raised ``OverflowError: string longer than
  INT_MAX bytes`` on a float32 tissue image, while `backed=False` on the same
  file succeeded and casting the image to uint8 "fixed" it — a size cliff that
  reads as a mysterious dtype dependency.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import anndata as ad

import cytome


def _adata(n=40, g=6, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, (n, g)).astype(np.float32))
    return ad.AnnData(
        X,
        obs=pd.DataFrame(index=[f"bc{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(g)]),
    )


# ---------------------------------------------------------------------------
# missing values in obs
# ---------------------------------------------------------------------------

def test_obs_with_pandas_na_round_trips_as_missing(tmp_path):
    """Every nullable container, and the categorical the report came from."""
    a = _adata()
    n = a.n_obs
    first = pd.Categorical(["T"] * n, categories=["T", "B"])
    first[: n // 3] = None                       # unassigned bins
    a.obs["first_type"] = first
    a.obs["label"] = pd.array(["x"] * n, dtype="string")
    a.obs["label"][: n // 4] = pd.NA
    a.obs["count"] = pd.array(np.arange(n), dtype="Int64")
    a.obs["count"][: n // 5] = pd.NA
    a.obs["flag"] = pd.array([True] * n, dtype="boolean")
    a.obs["flag"][:3] = pd.NA
    a.obs["score"] = np.linspace(0, 1, n)
    a.obs.loc[a.obs.index[0], "score"] = np.nan

    out = str(tmp_path / "na.cytome")
    cytome.from_anndata(a, modality="RNA", output=out).close()

    ds = cytome.open(out)
    try:
        cells = ds.cells.to_pandas()
    finally:
        ds.close()

    assert cells["first_type"].isna().sum() == n // 3
    assert cells["label"].isna().sum() == n // 4
    assert cells["count"].isna().sum() == n // 5
    assert cells["flag"].isna().sum() == 3
    assert cells["score"].isna().sum() == 1
    # the values that were present survive unchanged, i.e. nothing was
    # replaced by a placeholder string
    assert set(cells["first_type"].dropna().unique()) == {"T"}
    assert set(cells["label"].dropna().unique()) == {"x"}


def test_a_fully_missing_column_is_not_an_error(tmp_path):
    a = _adata()
    a.obs["second_type"] = pd.Categorical([None] * a.n_obs, categories=["T", "B"])
    out = str(tmp_path / "allna.cytome")
    cytome.from_anndata(a, modality="RNA", output=out).close()
    ds = cytome.open(out)
    try:
        assert ds.cells.to_pandas()["second_type"].isna().all()
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# bulk arrays in uns
# ---------------------------------------------------------------------------

def test_metadata_warns_before_it_refuses(tmp_path):
    """SQLite stores up to ~1 GB in one value, so a large payload used to work
    and must keep working -- with a warning. Only past what SQLite itself will
    take does this raise, and then with a message that says where bulk arrays
    go rather than an OverflowError from inside the write."""
    from cytome.core.metadata import MetadataStore
    out = str(tmp_path / "guard.cytome")
    ds = cytome.from_anndata(_adata(), modality="RNA", output=out)
    try:
        # Random values, because zeros serialise to "0.0" and a JSON limit is
        # about characters, not array bytes.
        big = np.random.default_rng(0).random((1400, 900, 3)).astype(np.float32)
        with pytest.warns(UserWarning, match="add_spatial_image"):
            ds.metadata["large"] = big
        ds.flush()                      # metadata writes are queued
        assert np.allclose(np.asarray(ds.metadata["large"]), big, atol=1e-6)
        assert MetadataStore.WARN_JSON_BYTES < MetadataStore.MAX_JSON_BYTES
        # and the refusal is where SQLite's own limit is, not below it
        assert MetadataStore.MAX_JSON_BYTES < 1_000_000_000
    finally:
        ds.close()


def test_backed_conversion_stores_a_float32_image_as_a_blob(tmp_path):
    """The reported case, shrunk: `backed=True` used to serialise the image as
    JSON text (~20 bytes per float32 element) while `backed=False` routed it to
    the compressed spatial table. Both paths must store it the same way."""
    a = _adata(n=60, g=8)
    rng = np.random.default_rng(1)
    image = rng.random((320, 300, 3)).astype(np.float32)
    a.obsm["spatial"] = rng.random((a.n_obs, 2)) * 100
    a.uns["spatial"] = {
        "lib1": {"images": {"hires": image},
                 "scalefactors": {"tissue_hires_scalef": 0.5,
                                  "spot_diameter_fullres": 12.0}}}
    a.uns["small"] = {"note": "kept as metadata"}

    h5 = str(tmp_path / "vis.h5ad")
    a.write_h5ad(h5)
    for backed in (False, True):
        out = str(tmp_path / f"vis_backed{int(backed)}.cytome")
        cytome.from_h5ad(h5, modality="RNA", output=out, backed=backed).close()
        ds = cytome.open(out)
        try:
            assert ("lib1", "hires") in ds.spatial_images.keys()
            got = np.asarray(ds.spatial_images["lib1", "hires"])
            assert got.dtype == np.float32 and got.shape == image.shape
            assert np.array_equal(got, image)
            assert ds.spatial_images.scalefactors("lib1")["tissue_hires_scalef"] == 0.5
            assert ds.metadata["small"] == {"note": "kept as metadata"}
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# images cost memory only when someone looks at them
# ---------------------------------------------------------------------------

def test_as_uns_is_lazy_by_default(tmp_path):
    """Every reader calls `as_uns`, and a full-resolution image is hundreds of
    megabytes; decoding eagerly charged that to every read whether or not
    anyone plotted."""
    out = str(tmp_path / "lazy.cytome")
    ds = cytome.from_anndata(_adata(), modality="RNA", output=out)
    rng = np.random.default_rng(2)
    image = rng.random((200, 180, 3)).astype(np.float32)
    try:
        ds.add_spatial_image("lib1", "hires", image,
                             scalefactors={"tissue_hires_scalef": 0.25})
        uns = ds.spatial_images.as_uns()
        proxy = uns["lib1"]["images"]["hires"]
        # metadata without decoding
        assert not isinstance(proxy, np.ndarray)
        assert proxy.shape == image.shape
        assert proxy.dtype == np.float32
        assert proxy.ndim == 3 and len(proxy) == image.shape[0]
        # and the array itself when something actually reads it
        assert np.array_equal(np.asarray(proxy), image)
        assert np.array_equal(proxy[10:12], image[10:12])
        # opting out gives plain arrays
        eager = ds.spatial_images.as_uns(lazy=False)
        assert isinstance(eager["lib1"]["images"]["hires"], np.ndarray)
        assert uns["lib1"]["scalefactors"]["tissue_hires_scalef"] == 0.25
    finally:
        ds.close()


def test_a_lazy_image_survives_the_dataset_being_closed(tmp_path):
    """A proxy routinely outlives its Dataset — every reader that returns an
    AnnData closes the file — so it must reconnect rather than raise "Cannot
    operate on a closed database" at plot time."""
    out = str(tmp_path / "closed.cytome")
    ds = cytome.from_anndata(_adata(), modality="RNA", output=out)
    image = np.random.default_rng(4).random((50, 40, 3)).astype(np.float32)
    ds.add_spatial_image("lib1", "hires", image)
    uns = ds.spatial_images.as_uns()
    ds.close()
    proxy = uns["lib1"]["images"]["hires"]
    assert proxy.shape == image.shape
    assert np.array_equal(np.asarray(proxy), image)


def test_to_anndata_still_hands_back_the_scanpy_convention(tmp_path):
    out = str(tmp_path / "conv.cytome")
    ds = cytome.from_anndata(_adata(), modality="RNA", output=out)
    rng = np.random.default_rng(3)
    image = rng.random((64, 48, 3)).astype(np.float32)
    try:
        ds.add_spatial_image("lib1", "hires", image)
        back = ds.to_anndata(modality="RNA")
    finally:
        ds.close()
    assert "spatial" in back.uns
    got = back.uns["spatial"]["lib1"]["images"]["hires"]
    assert np.array_equal(np.asarray(got), image)
    # A detached AnnData gets a real array, not a proxy: people write these to
    # disk, and anndata's h5ad writer rejects anything else in `uns` with a
    # message that names neither the image nor the conversion.
    assert isinstance(got, np.ndarray)
    back.write_h5ad(str(tmp_path / "roundtrip.h5ad"))
    import anndata as _ad
    again = _ad.read_h5ad(str(tmp_path / "roundtrip.h5ad"))
    assert np.array_equal(again.uns["spatial"]["lib1"]["images"]["hires"], image)


# ---------------------------------------------------------------------------
# the backed fallback
# ---------------------------------------------------------------------------

def test_backed_fallback_does_not_trip_on_its_own_partial_file(tmp_path):
    """`from_h5ad(backed=True)` falls back to the in-memory reader for
    encodings the streaming reader cannot handle. The failed attempt has
    already created the output file, and the retry writes to the same path —
    so the fallback used to die with "Cytome already exists", reporting a name
    collision for debris it created itself."""
    import cytome as _cytome

    a = _adata(n=50, g=7)
    h5 = str(tmp_path / "src.h5ad")
    a.write_h5ad(h5)
    out = str(tmp_path / "out.cytome")

    calls = {"n": 0}
    real = _cytome.io.convert_anndata.from_h5ad

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if kwargs.get("backed"):
            # leave a partial file behind, exactly as a real failure does
            open(kwargs["output"], "wb").write(b"partial")
            raise KeyError("Unable to open attribute 'shape'")
        return real(*args, **kwargs)

    _cytome.io.convert_anndata.from_h5ad = flaky
    try:
        with pytest.warns(RuntimeWarning, match="falling back"):
            ds = _cytome.from_h5ad(h5, modality="RNA", output=out, backed=True)
        try:
            assert ds.n_cells == 50
        finally:
            ds.close()
    finally:
        _cytome.io.convert_anndata.from_h5ad = real
    assert calls["n"] == 2, "the fallback should have retried exactly once"
