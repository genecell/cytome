"""Round 10 (2026-05-24) — ``from_h5ad(backed=True)`` peak-RAM + parity tests.

Five tests pin the new backed-mode contract:

1. ``test_from_h5ad_backed_lossless_roundtrip``
       Same h5ad in via ``from_anndata`` vs ``from_h5ad(backed=True)``
       produces structurally identical cytomes. The single strongest
       invariant — catches every silent data-loss regression.

2. ``test_from_h5ad_backed_peak_rss_bounded``
       Synthetic mid-size h5ad converted in a subprocess; peak RSS
       (via ``getrusage(RUSAGE_CHILDREN)``) must stay under a
       conservative threshold. Pins the "we're not eagerly loading
       anything" regression.

3. ``test_from_h5ad_backed_skip_kwargs``
       Each new ``write_*`` / ``skip_*`` kwarg is exercised; the
       resulting cytome is inspected to confirm the slot was/wasn't
       written.

4. ``test_from_h5ad_backed_uint16_preserved``
       uint16 X round-trips as uint16 through the streaming writer.
       Catches silent upcasts.

5. ``test_from_h5ad_backed_no_pyarrow_import``
       Asserts pyarrow is not imported by the conversion (user
       constraint — backed mode must work without pyarrow installed).
"""
from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome


anndata = pytest.importorskip("anndata")


# =====================================================================
# Fixture builders
# =====================================================================

def _build_rich_h5ad(tmp_path: Path, n_obs: int = 200, n_vars: int = 80, seed: int = 7):
    """Build a synthetic h5ad covering every slot from_h5ad needs to
    handle: X CSR, 2 layers CSR, raw with raw.var, multi obsm
    (dense + sparse), obsp, varm, varp, uns.
    """
    rng = np.random.default_rng(seed)

    X = sp.random(n_obs, n_vars, density=0.12, format="csr",
                  dtype=np.float32, random_state=seed).astype(np.float32)
    layer_raw = sp.random(n_obs, n_vars, density=0.12, format="csr",
                          dtype=np.float32, random_state=seed + 1).astype(np.float32)
    # uint16 layer to exercise non-float dtypes through the streaming writer
    layer_infog = sp.csr_matrix(
        (rng.integers(1, 10, size=X.nnz).astype(np.uint16),
         X.indices.copy(),
         X.indptr.copy()),
        shape=X.shape,
    )

    raw_X = sp.random(n_obs, n_vars, density=0.18, format="csr",
                      dtype=np.float32, random_state=seed + 2).astype(np.float32)
    raw_var = pd.DataFrame(
        {"gene_id": [f"ENS{i:05d}" for i in range(n_vars)],
         "raw_var_meta": rng.standard_normal(n_vars).astype(np.float32)},
        index=[f"g{i}" for i in range(n_vars)],
    )

    obs = pd.DataFrame({
        "barcode": [f"CELL{i:04d}" for i in range(n_obs)],
        "cell_type": pd.Categorical(rng.choice(["T", "B", "NK"], size=n_obs)),
        "n_genes": rng.integers(200, 5000, size=n_obs).astype(np.int32),
    })
    obs.index = obs["barcode"].astype(str)

    var = pd.DataFrame({
        "gene_id": [f"ENS{i:05d}" for i in range(n_vars)],
        "symbol": [f"GENE{i}" for i in range(n_vars)],
        "highly_variable": rng.random(n_vars) > 0.5,
    })
    var.index = var["gene_id"].astype(str)

    # obsm: small dense + larger dense + sparse (>500 cols => stored as matrix)
    obsm = {
        "X_pca": rng.standard_normal((n_obs, 5)).astype(np.float32),
        "X_umap": rng.standard_normal((n_obs, 2)).astype(np.float32),
        # NOTE: cytome's from_anndata uses .shape[1] > 500 as the
        # "store as matrix" trigger. Make this just over for coverage.
        "X_gene_act": sp.random(n_obs, 600, density=0.05, format="csr",
                                 dtype=np.float32, random_state=seed + 3),
    }

    obsp = {
        "connectivities": sp.random(n_obs, n_obs, density=0.05, format="csr",
                                     dtype=np.float32, random_state=seed + 4),
        "distances": sp.random(n_obs, n_obs, density=0.04, format="csr",
                                dtype=np.float32, random_state=seed + 5),
    }

    varm = {"PCs": rng.standard_normal((n_vars, 5)).astype(np.float32)}

    varp = {"corrgraph": sp.random(n_vars, n_vars, density=0.05, format="csr",
                                    dtype=np.float32, random_state=seed + 6)}

    uns = {"experiment": "synthetic_round10", "seed": int(seed)}

    a = anndata.AnnData(
        X=X, obs=obs, var=var,
        layers={"raw": layer_raw, "infog": layer_infog},
        obsm=obsm, obsp=obsp, varm=varm, varp=varp, uns=uns,
    )
    a.raw = anndata.AnnData(X=raw_X, obs=obs.copy(), var=raw_var)

    p = tmp_path / "rich.h5ad"
    a.write_h5ad(p)
    return p, a


# =====================================================================
# 1. Lossless round-trip parity vs from_anndata
# =====================================================================

def test_from_h5ad_backed_lossless_roundtrip(tmp_path):
    """The backed path MUST produce a cytome structurally identical
    to what ``from_anndata`` produces from the same source.

    This is the single strongest invariant. It catches every silent
    data-loss regression (raw / obsp / obsm-as-matrix / varm / varp /
    uns / layer_map / etc.).
    """
    h5ad_path, a = _build_rich_h5ad(tmp_path)

    out_mem = tmp_path / "mem.cytome"
    out_backed = tmp_path / "backed.cytome"

    cytome.from_anndata(a, modality="RNA", output=str(out_mem))
    cytome.from_h5ad(h5ad_path, output=str(out_backed),
                     modality="RNA", backed=True, verbose=False)

    ds_mem = cytome.open(out_mem)
    ds_backed = cytome.open(out_backed)

    # --- matrix_meta rows: same set of matrix names + shapes ---
    def _matrix_names(ds):
        rows = ds._conn.execute(
            "SELECT matrix_name, n_rows, n_cols, n_nonzero, dtype "
            "FROM matrix_meta ORDER BY matrix_name"
        ).fetchall()
        return {r[0]: r[1:] for r in rows}
    m_mem = _matrix_names(ds_mem)
    m_backed = _matrix_names(ds_backed)
    assert m_mem.keys() == m_backed.keys(), (
        f"matrix names differ:\n"
        f"  only in mem: {set(m_mem)-set(m_backed)}\n"
        f"  only in backed: {set(m_backed)-set(m_mem)}"
    )
    for name in m_mem:
        # n_rows, n_cols, n_nonzero, dtype must match. The dtype check
        # is the key catch for "silent upcast" regressions.
        assert m_mem[name] == m_backed[name], (
            f"matrix {name!r} differs: mem={m_mem[name]} backed={m_backed[name]}"
        )

    # --- _anndata_* metadata round-trip keys ---
    for k in ("_anndata_X_layer", "_anndata_layer_map", "_anndata_obsm_map",
              "_anndata_obsm_as_matrix", "_anndata_obsp_map",
              "_anndata_varm_map", "_anndata_varp_map"):
        v_mem = ds_mem.metadata.get(k)
        v_backed = ds_backed.metadata.get(k)
        assert v_mem == v_backed, (
            f"metadata[{k!r}] differs:\n  mem    = {v_mem}\n  backed = {v_backed}"
        )

    # --- raw_var table (cytome convention) ---
    raw_meta_mem = ds_mem.metadata.get("_anndata_raw")
    raw_meta_backed = ds_backed.metadata.get("_anndata_raw")
    assert raw_meta_mem == raw_meta_backed, (
        f"_anndata_raw metadata differs: mem={raw_meta_mem} backed={raw_meta_backed}"
    )
    assert raw_meta_backed is not None, "raw should be written by default"
    # _raw_var SQL table should exist and have same row count
    n_raw_var_mem = ds_mem._conn.execute("SELECT COUNT(*) FROM _raw_var").fetchone()[0]
    n_raw_var_backed = ds_backed._conn.execute("SELECT COUNT(*) FROM _raw_var").fetchone()[0]
    assert n_raw_var_mem == n_raw_var_backed > 0

    # --- embedding_meta + graphs parity ---
    n_emb_mem = ds_mem._conn.execute("SELECT COUNT(*) FROM embedding_meta").fetchone()[0]
    n_emb_backed = ds_backed._conn.execute("SELECT COUNT(*) FROM embedding_meta").fetchone()[0]
    assert n_emb_mem == n_emb_backed, (
        f"embedding count differs: mem={n_emb_mem} backed={n_emb_backed}"
    )

    # --- numeric equality of X (a couple of chunks) ---
    # Read the layer name the writer chose rather than assuming "counts":
    # since 0.3.0 a non-integer X is stored as RNA_data, precisely so that
    # "counts" means counts. The two paths must agree on the name as well as
    # on the values, which the _anndata_X_layer comparison above already pins.
    x_layer = ds_mem.metadata["_anndata_X_layer"].split("_", 1)[1]
    chunks_mem = list(ds_mem.iter_chunks(modality="RNA", layer=x_layer))
    chunks_backed = list(ds_backed.iter_chunks(modality="RNA", layer=x_layer))
    X_mem = sp.vstack([c for c, _ in chunks_mem]).toarray()
    X_backed = sp.vstack([c for c, _ in chunks_backed]).toarray()
    np.testing.assert_allclose(X_mem, X_backed, rtol=0, atol=0,
                                err_msg="X numeric values differ between paths")

    ds_mem.close()
    ds_backed.close()


# =====================================================================
# 2. Peak RSS bounded
# =====================================================================

def test_from_h5ad_backed_peak_rss_bounded(tmp_path):
    """Run conversion in a subprocess and assert peak RSS stays below
    a threshold well above the working set but well below the
    "loaded everything eagerly" floor.

    For the test fixture (~5 MB on disk), the regression behaviour
    would peak around the on-disk size; our streaming target is
    well under 100 MB working set. We set the threshold generously
    at 400 MB to absorb Python startup + scipy + anndata + h5py
    module imports without false-positives.
    """
    h5ad_path, _ = _build_rich_h5ad(tmp_path, n_obs=300, n_vars=120)
    out = tmp_path / "peak.cytome"
    metrics = tmp_path / "metrics.json"

    runner = tmp_path / "run.py"
    runner.write_text(textwrap.dedent(f"""
        import json, resource, sys
        sys.path.insert(0, {repr(str(Path(cytome.__file__).resolve().parents[1]))})
        import cytome
        # Baseline = RSS after all imports, BEFORE conversion. ru_maxrss is a
        # high-water mark, so (peak - baseline) isolates the conversion's
        # marginal RAM from the interpreter+library import footprint (which
        # varies wildly by env — ~400 MB with anndata 0.10 + scipy + h5py on
        # py3.12, vs the ~150 MB this test originally assumed).
        baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cytome.from_h5ad(
            {repr(str(h5ad_path))},
            output={repr(str(out))},
            modality='RNA', backed=True, verbose=False,
        )
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open({repr(str(metrics))}, 'w') as f:
            json.dump({{'peak_rss_mb': peak_kb / 1024,
                       'baseline_rss_mb': baseline_kb / 1024}}, f)
    """))

    proc = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"subprocess failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    with open(metrics) as f:
        m = json.load(f)
    peak_mb = m["peak_rss_mb"]
    baseline_mb = m["baseline_rss_mb"]
    delta_mb = peak_mb - baseline_mb

    # Assert on the MARGINAL RAM of the conversion (peak − post-import
    # baseline), not absolute RSS. The streaming backed path should add only
    # a small working set on this tiny fixture (~tens of MB); the regression
    # (loading the whole h5ad eagerly) would show up as a large delta. A
    # 150 MB delta budget absorbs chunk buffers + transient allocs while
    # still catching a full eager load. This is env-independent (unlike the
    # old absolute 400 MB threshold, which the import footprint alone hit).
    print(f"[bench] baseline={baseline_mb:.1f} MB, peak={peak_mb:.1f} MB, "
          f"delta={delta_mb:.1f} MB")
    assert delta_mb < 150, (
        f"backed conversion added {delta_mb:.1f} MB over the {baseline_mb:.1f} MB "
        f"import baseline (peak {peak_mb:.1f} MB); the backed path may be loading "
        f"something fully into RAM. Investigate with `psutil` instrumentation."
    )


# =====================================================================
# 3. write_*/skip_* kwargs exercised
# =====================================================================

@pytest.mark.parametrize("kwarg_name,kwarg_value,assertion_kind,target", [
    ("write_raw", False, "metadata_absent", "_anndata_raw"),
    ("write_layers", False, "layer_map_empty", None),
    ("write_obsm", False, "obsm_map_empty", None),
    ("write_obsp", False, "obsp_map_empty", None),
    ("write_varm", False, "varm_map_empty", None),
    ("write_varp", False, "varp_map_empty", None),
    ("skip_layers", ["infog"], "layer_skipped", "RNA_infog"),
    # RNA_X_umap, not RNA_obsm_X_umap: 0.2.6 dropped the "obsm_" infix. Under
    # the old name this row asserted a key that could never be present and so
    # would have passed with skip_obsm entirely broken.
    ("skip_obsm", ["X_umap"], "obsm_skipped", "RNA_X_umap"),
    ("skip_obsp", ["distances"], "obsp_skipped", "RNA_obsp_distances"),
])
def test_from_h5ad_backed_skip_kwargs(
    tmp_path, kwarg_name, kwarg_value, assertion_kind, target
):
    """Each opt-out kwarg drops the corresponding slot in the output cytome."""
    h5ad_path, _ = _build_rich_h5ad(tmp_path)
    out = tmp_path / f"skipped_{kwarg_name}.cytome"

    kwargs = {
        "modality": "RNA", "backed": True, "verbose": False,
        kwarg_name: kwarg_value,
    }
    cytome.from_h5ad(h5ad_path, output=str(out), **kwargs)
    ds = cytome.open(out)

    if assertion_kind == "metadata_absent":
        assert ds.metadata.get(target) in (None, {}), (
            f"{target} should be absent when {kwarg_name}={kwarg_value}"
        )
    elif assertion_kind == "layer_map_empty":
        layer_map = ds.metadata.get("_anndata_layer_map", {})
        assert layer_map == {}, f"layer_map should be empty: {layer_map}"
    elif assertion_kind == "obsm_map_empty":
        m1 = ds.metadata.get("_anndata_obsm_map", {})
        m2 = ds.metadata.get("_anndata_obsm_as_matrix", {})
        assert m1 == {} and m2 == {}, (
            f"both obsm maps should be empty: {m1}, {m2}"
        )
    elif assertion_kind == "obsp_map_empty":
        assert ds.metadata.get("_anndata_obsp_map", {}) == {}
    elif assertion_kind == "varm_map_empty":
        assert ds.metadata.get("_anndata_varm_map", {}) == {}
    elif assertion_kind == "varp_map_empty":
        assert ds.metadata.get("_anndata_varp_map", {}) == {}
    elif assertion_kind == "layer_skipped":
        layer_map = ds.metadata.get("_anndata_layer_map", {})
        assert target not in layer_map, (
            f"{target} should NOT appear in layer_map after skip_layers; "
            f"got {layer_map}"
        )
    elif assertion_kind == "obsm_skipped":
        m1 = ds.metadata.get("_anndata_obsm_map", {})
        m2 = ds.metadata.get("_anndata_obsm_as_matrix", {})
        assert target not in m1 and target not in m2
    elif assertion_kind == "obsp_skipped":
        assert target not in ds.metadata.get("_anndata_obsp_map", {})

    ds.close()


# =====================================================================
# 4. uint16 dtype preserved
# =====================================================================

def test_from_h5ad_backed_uint16_preserved(tmp_path):
    """uint16 X round-trips as uint16 through the streaming writer.

    The user's 200 GB h5ad has uint16 X (compact storage). A silent
    upcast to float32 would double RAM during downstream iter_chunks
    work — must catch any regression.
    """
    n_obs, n_vars = 60, 30
    X = sp.csr_matrix(
        np.random.default_rng(11).integers(0, 100, size=(n_obs, n_vars))
        .astype(np.uint16)
    )
    obs = pd.DataFrame(
        {"barcode": [f"c{i}" for i in range(n_obs)]},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"g{i}" for i in range(n_vars)]},
        index=[f"g{i}" for i in range(n_vars)],
    )
    a = anndata.AnnData(X=X, obs=obs, var=var)
    p = tmp_path / "uint16.h5ad"
    a.write_h5ad(p)

    out = tmp_path / "uint16.cytome"
    cytome.from_h5ad(p, output=str(out), modality="RNA",
                     backed=True, verbose=False)
    ds = cytome.open(out)
    # matrix_meta stores dtype as text
    dtype_stored = ds._conn.execute(
        "SELECT dtype FROM matrix_meta WHERE matrix_name = 'RNA_counts'"
    ).fetchone()[0]
    assert dtype_stored == "uint16", (
        f"uint16 silently upcast on backed path: dtype={dtype_stored}"
    )
    # And iter_chunks yields uint16 too
    for chunk, _ in ds.iter_chunks(modality="RNA", layer="counts"):
        assert chunk.dtype == np.uint16, (
            f"iter_chunks yielded {chunk.dtype} (expected uint16)"
        )
        break
    ds.close()


# =====================================================================
# 5. No pyarrow import
# =====================================================================

def test_from_h5ad_backed_no_pyarrow_import(tmp_path):
    """User constraint: backed-mode conversion must NOT pull pyarrow.

    Runs the conversion in a subprocess, then asserts pyarrow is not
    in sys.modules. Done in a subprocess so other tests' state can't
    pollute the import surface.
    """
    h5ad_path, _ = _build_rich_h5ad(tmp_path, n_obs=80, n_vars=30)
    out = tmp_path / "nopyarrow.cytome"
    result_file = tmp_path / "result.json"

    runner = tmp_path / "run.py"
    runner.write_text(textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {repr(str(Path(cytome.__file__).resolve().parents[1]))})
        import cytome
        cytome.from_h5ad(
            {repr(str(h5ad_path))},
            output={repr(str(out))},
            modality='RNA', backed=True, verbose=False,
        )
        # Filter pandas's own pyarrow-compat shim — it's a pure-Python
        # module that registers as 'pandas.compat.pyarrow' but does NOT
        # import the pyarrow package itself. The user constraint is no
        # ACTUAL pyarrow dependency. Anything starting with 'pyarrow.'
        # or equal to 'pyarrow' indicates a real import.
        actual_pyarrow = sorted(
            m for m in sys.modules
            if m == 'pyarrow' or m.startswith('pyarrow.')
        )
        with open({repr(str(result_file))}, 'w') as f:
            json.dump({{'pyarrow_modules': actual_pyarrow}}, f)
    """))
    proc = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"subprocess failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    with open(result_file) as f:
        r = json.load(f)
    assert r["pyarrow_modules"] == [], (
        f"pyarrow was imported by from_h5ad(backed=True): "
        f"{r['pyarrow_modules']}. The conversion path must not depend "
        f"on pyarrow (user constraint)."
    )


# =====================================================================
# 6. CSC h5ad raises NotImplementedError (clean failure mode)
# =====================================================================

def test_from_h5ad_backed_csc_falls_back_to_backed_false(tmp_path):
    """Round 12 (2026-05-27) behavior change: CSC-encoded h5ads NO
    LONGER raise NotImplementedError from the top-level ``from_h5ad``.
    Instead, the backed path catches the NotImplementedError and falls
    back to ``backed=False`` (in-memory load via anndata.read_h5ad) so
    the conversion succeeds with a RuntimeWarning.

    Pre-Round-12 this raised; Round 12 makes it work (at the cost of
    full-RAM load, with a warning). The underlying CSC streaming reader
    still raises NotImplementedError — but the top-level wrapper catches
    it.
    """
    import warnings as _warnings

    n_obs, n_vars = 30, 20
    X = sp.random(n_obs, n_vars, density=0.1, format="csc", dtype=np.float32,
                  random_state=0)
    obs = pd.DataFrame(
        {"barcode": [f"c{i}" for i in range(n_obs)]},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"g{i}" for i in range(n_vars)]},
        index=[f"g{i}" for i in range(n_vars)],
    )
    a = anndata.AnnData(X=X, obs=obs, var=var)
    p = tmp_path / "csc.h5ad"
    a.write_h5ad(p)

    out = tmp_path / "csc.cytome"
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        ds = cytome.from_h5ad(p, output=str(out), modality="RNA",
                              backed=True, verbose=False)

    fallback_warnings = [
        wi for wi in w if issubclass(wi.category, RuntimeWarning)
        and "backed" in str(wi.message).lower()
    ]
    assert fallback_warnings, (
        f"Expected RuntimeWarning about backed→backed=False fallback "
        f"for CSC h5ad. Got: {[str(wi.message) for wi in w]}"
    )
    # Conversion succeeded — n_cells matches what we wrote.
    assert ds.n_cells == n_obs
    ds.close()


# =====================================================================
# 7. Sparse obsm dispatch — pins BOTH branches (<=500 cols densified,
#    >500 cols stored as matrix). Regression for the pre-Round-10
#    np.asarray(sparse) bug that hit user files like NeuIPS BMMC
#    multiome (s1d1, s2d1_s3d10) where obsm['ATAC_gene_activity'] is
#    a 19K-col CSC sparse matrix.
# =====================================================================

def test_from_h5ad_backed_sparse_obsm_large_goes_to_add_matrix(tmp_path):
    """sparse obsm with shape[1] > 500 → stored via add_matrix +
    _anndata_obsm_as_matrix metadata map (NOT add_embedding).

    Pre-Round-10 this would have hit
    `IndexError: tuple index out of range` because the unconditional
    `ds.add_embedding(np.asarray(sparse_matrix))` produced a 0-d
    object ndarray that crashed at flush time in
    `_write_embedding_payload`. The Round 10 dispatch routes sparse
    obsm by shape[1] > 500 to `add_matrix` instead.
    """
    n_obs = 80
    n_vars = 20
    # Sparse obsm with shape[1] > 500 → triggers the add_matrix branch.
    sparse_obsm = sp.random(n_obs, 600, density=0.05, format="csc",
                            dtype=np.float32, random_state=11)

    X = sp.csr_matrix(np.random.default_rng(0).standard_normal(
        (n_obs, n_vars)).astype(np.float32))
    obs = pd.DataFrame(
        {"barcode": [f"c{i}" for i in range(n_obs)]},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"g{i}" for i in range(n_vars)]},
        index=[f"g{i}" for i in range(n_vars)],
    )
    a = anndata.AnnData(
        X=X, obs=obs, var=var,
        obsm={"ATAC_gene_activity": sparse_obsm},  # mirrors the user's file
    )
    h5ad_path = tmp_path / "sparse_obsm_large.h5ad"
    a.write_h5ad(h5ad_path)

    out = tmp_path / "sparse_obsm_large.cytome"
    # If pre-Round-10 logic regresses, this call raises IndexError.
    cytome.from_h5ad(h5ad_path, output=str(out), modality="RNA",
                     backed=True, verbose=False)

    ds = cytome.open(out)
    # >500-col sparse obsm should be in _anndata_obsm_as_matrix, NOT
    # _anndata_obsm_map.
    obsm_as_matrix = ds.metadata.get("_anndata_obsm_as_matrix", {})
    obsm_map = ds.metadata.get("_anndata_obsm_map", {})
    assert "RNA_ATAC_gene_activity" in obsm_as_matrix, (
        f"sparse obsm with shape[1]=600 should be stored via add_matrix "
        f"+ _anndata_obsm_as_matrix; got obsm_as_matrix={obsm_as_matrix}, "
        f"obsm_map={obsm_map}"
    )
    assert "RNA_ATAC_gene_activity" not in obsm_map
    # And the matrix actually exists in matrix_meta
    row = ds._conn.execute(
        "SELECT n_rows, n_cols FROM matrix_meta "
        "WHERE matrix_name = 'RNA_ATAC_gene_activity'"
    ).fetchone()
    assert row is not None, "matrix not written"
    assert row == (n_obs, 600)
    ds.close()


def test_from_h5ad_backed_sparse_obsm_small_densified_to_add_embedding(tmp_path):
    """sparse obsm with shape[1] <= 500 → densified via .toarray() and
    stored via add_embedding + _anndata_obsm_map (the embedding
    branch).

    This branch was also broken pre-Round-10 (same
    np.asarray(sparse) → 0-d object array bug). Round 10 now
    explicitly calls .toarray() first so add_embedding sees a real
    2D ndarray.
    """
    n_obs = 80
    n_vars = 20
    # shape[1] = 100, below the 500 threshold → densify branch.
    rng = np.random.default_rng(13)
    sparse_obsm_small = sp.csr_matrix(
        (rng.standard_normal(200).astype(np.float32),
         rng.integers(0, 100, size=200).astype(np.int32),
         np.arange(0, 200 + 1, 200 // n_obs)[:n_obs + 1].astype(np.int32)),
        shape=(n_obs, 100),
    )

    X = sp.csr_matrix(np.random.default_rng(0).standard_normal(
        (n_obs, n_vars)).astype(np.float32))
    obs = pd.DataFrame(
        {"barcode": [f"c{i}" for i in range(n_obs)]},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"g{i}" for i in range(n_vars)]},
        index=[f"g{i}" for i in range(n_vars)],
    )
    a = anndata.AnnData(
        X=X, obs=obs, var=var,
        obsm={"small_sparse": sparse_obsm_small},
    )
    h5ad_path = tmp_path / "sparse_obsm_small.h5ad"
    a.write_h5ad(h5ad_path)

    out = tmp_path / "sparse_obsm_small.cytome"
    # Would have hit IndexError pre-Round-10.
    cytome.from_h5ad(h5ad_path, output=str(out), modality="RNA",
                     backed=True, verbose=False)

    ds = cytome.open(out)
    # Small sparse → densified to embedding, NOT add_matrix.
    obsm_as_matrix = ds.metadata.get("_anndata_obsm_as_matrix", {})
    obsm_map = ds.metadata.get("_anndata_obsm_map", {})
    assert "RNA_small_sparse" in obsm_map, (
        f"small sparse obsm (shape[1]=100) should be densified and "
        f"stored via add_embedding + _anndata_obsm_map; got "
        f"obsm_map={obsm_map}, obsm_as_matrix={obsm_as_matrix}"
    )
    assert "RNA_small_sparse" not in obsm_as_matrix
    # And the embedding actually exists in embedding_meta
    row = ds._conn.execute(
        "SELECT n_rows, n_cols FROM embedding_meta "
        "WHERE array_name = 'RNA_small_sparse'"
    ).fetchone()
    assert row is not None, "embedding not written"
    assert row == (n_obs, 100)
    ds.close()


# =====================================================================
# 8. Modality registry routing — Round 11 (2026-05-26). Pre-Round-11
#    only routed ATAC vs everything-else; GA / tiles silently went to
#    the `genes` table. Round 11 routes via cytome.utils.modality.
# =====================================================================

@pytest.mark.parametrize("modality,expected_entity,expected_id_col", [
    ("RNA",   "genes",    "gene_id"),
    ("GA",    "GA_genes", "gene_id"),
    ("ATAC",  "peaks",    "peak_id"),
    ("tiles", "tiles",    "tile_id"),
])
def test_from_h5ad_backed_modality_routes_via_registry(
    tmp_path, modality, expected_entity, expected_id_col
):
    """Round 11 (2026-05-26): from_h5ad must route the var entity table
    through the modality registry, not a hardcoded ATAC-vs-other
    dichotomy. Pre-Round-11, modality='GA' silently wrote to `genes`
    (wrong) instead of `GA_genes`.
    """
    n_obs = 30
    n_vars = 12
    # ATAC needs peak-coord var_names for the auto-parser. tiles
    # requires chr/start/end_ columns in var (per cytome's tiles
    # schema with NOT NULL constraints on those columns).
    if modality == "ATAC":
        var_index = [f"chr1:{i*1000}-{i*1000+500}" for i in range(n_vars)]
        var_df = pd.DataFrame(index=var_index)
    elif modality == "tiles":
        var_index = [f"tile_{i}" for i in range(n_vars)]
        var_df = pd.DataFrame({
            "chr": ["chr1"] * n_vars,
            "start": [i * 1000 for i in range(n_vars)],
            "end_": [i * 1000 + 500 for i in range(n_vars)],
        }, index=var_index)
    else:
        var_index = [f"feat{i}" for i in range(n_vars)]
        var_df = pd.DataFrame(index=var_index)

    X = sp.csr_matrix(
        np.random.default_rng(0).standard_normal((n_obs, n_vars)).astype(np.float32)
    )
    a = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"barcode": [f"c{i}" for i in range(n_obs)]},
            index=[f"c{i}" for i in range(n_obs)],
        ),
        var=var_df,
    )
    p = tmp_path / f"{modality}.h5ad"
    a.write_h5ad(p)

    out = tmp_path / f"{modality}.cytome"
    cytome.from_h5ad(p, output=str(out), modality=modality,
                     backed=True, verbose=False)

    ds = cytome.open(out)
    # The var entity table must be the one the registry says, NOT 'genes'
    # for non-RNA modalities.
    row_count = ds._conn.execute(
        f"SELECT COUNT(*) FROM {expected_entity}"
    ).fetchone()[0]
    assert row_count == n_vars, (
        f"modality={modality!r} should populate {expected_entity!r} with "
        f"{n_vars} rows; got {row_count}"
    )
    # And the id column should be present in that table
    cols = [
        r[1] for r in ds._conn.execute(
            f"PRAGMA table_info({expected_entity})"
        ).fetchall()
    ]
    assert expected_id_col in cols, (
        f"modality={modality!r}: {expected_id_col!r} column missing "
        f"from {expected_entity!r}; got cols={cols}"
    )
    ds.close()
