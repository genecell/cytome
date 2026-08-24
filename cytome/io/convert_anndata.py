"""AnnData conversion for Cytome."""

from __future__ import annotations

import json
import re
import tempfile
import warnings
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sqlite3

from cytome.core.dataset import CytomeDataset
from cytome.core.measurement import MeasurementLayer
from cytome.io.chunked_io import write_sparse_chunked, _now_iso
from cytome.io.compression import compress_blob


_PEAK_COORD_RE = re.compile(r"^([^:\s]+):(\d+)-(\d+)$")


def make_unique_ids(ids) -> list:
    """scanpy-style uniquification: duplicate values get ``-1``, ``-2`` … suffixes.
    Used for the cytome feature id column, which carries a ``UNIQUE`` constraint."""
    seen: dict = {}
    out = []
    for n in (str(x) for x in ids):
        if n in seen:
            seen[n] += 1
            out.append(f"{n}-{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def warn_duplicate_symbols(names) -> None:
    """One-time UserWarning if the (display) symbol column has duplicates —
    name-based feature lookup then returns the first match."""
    s = pd.Index([str(x) for x in names])
    dup = s[s.duplicated()].unique()
    if len(dup):
        warnings.warn(
            f"{len(dup)} duplicate gene symbol(s) (e.g. {list(dup[:5])}). "
            "Name-based feature lookup (plotting / COSG) returns the first match; "
            "the unique key is kept in the id column.",
            UserWarning, stacklevel=2,
        )


# Modality → (var_entity_table, id_column) mapping moved to
# cytome.utils.modality.MODALITY_VAR_ENTITY. Re-exported here for
# backward compatibility with any consumer that imported the private
# names directly.
from cytome.utils.modality import (
    MODALITY_VAR_ENTITY as _MODALITY_VAR_ENTITY,
    modality_var_entity as _modality_var_entity,
)


def _parse_peak_coords_from_varnames(
    var: pd.DataFrame, var_names, verbose: bool = True,
) -> pd.DataFrame:
    """Auto-derive ``chr``/``start``/``end_`` columns from peak-string var_names.

    Triggered on ATAC import only when the cytome ``peaks`` schema's NOT NULL
    columns (``chr``, ``start``, ``end_``) are missing from ``adata.var``. ALL
    var_names must match the canonical ``"chr:start-end"`` regex — partial
    match falls through to the existing behaviour (the SQLite NOT NULL
    constraint will trip with an actionable diagnostic at flush time).

    Returns a (possibly modified) copy of ``var`` with the parsed columns
    inserted at the front. Emits a single ``UserWarning`` summarising what
    was derived; suppress by pre-populating ``chr``/``start``/``end_`` in
    ``adata.var`` before calling :func:`from_anndata`.
    """
    needed = {"chr", "start", "end_"}
    if needed.issubset(set(var.columns)):
        return var

    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for name in var_names:
        m = _PEAK_COORD_RE.match(str(name))
        if m is None:
            return var
        chroms.append(m.group(1))
        starts.append(int(m.group(2)))
        ends.append(int(m.group(3)))

    var = var.copy()
    if "chr" not in var.columns:
        var.insert(0, "chr", chroms)
    if "start" not in var.columns:
        var.insert(1 if "chr" in var.columns else 0, "start", starts)
    if "end_" not in var.columns:
        var.insert(2 if {"chr", "start"} <= set(var.columns) else 0, "end_", ends)

    if verbose:
        warnings.warn(
            f"cytome: auto-derived chr/start/end_ for {len(chroms):,} ATAC peaks "
            f"from var_names matching 'chr:start-end'. To suppress this warning, "
            f"pre-populate adata.var with these columns before from_anndata().",
            UserWarning, stacklevel=3,
        )
    return var


def _check_atac_var_has_peak_coords(var: pd.DataFrame) -> None:
    """Validate that an ATAC ``adata.var`` carries the peak-coordinate columns
    required by the cytome ``peaks`` schema (``chr``, ``start``, ``end_``).
    Called after :func:`_parse_peak_coords_from_varnames` has had a chance to
    auto-derive them from canonical var_names. Raises a ``ValueError`` with
    an actionable hint when any are still missing — much friendlier than the
    generic ``IntegrityError: NOT NULL constraint failed: peaks.chr`` that
    SQLite would emit later at flush time."""
    needed = ["chr", "start", "end_"]
    missing = [c for c in needed if c not in var.columns]
    if not missing:
        return
    raise ValueError(
        f"cytome.from_anndata(modality='ATAC'): adata.var is missing the "
        f"required column(s) {missing}. The cytome 'peaks' table requires "
        f"chr/start/end_ for every peak.\n\n"
        f"Two ways to fix:\n"
        f"  1. (Easiest) If your var_names are formatted as 'chr:start-end' "
        f"(e.g. 'chr1:3094641-3095334'), cytome derives these columns "
        f"automatically — but only when ALL var_names match the pattern. "
        f"It looks like some don't. Inspect with: \n"
        f"     adata.var_names[~adata.var_names.str.match(r'^[^:]+:\\d+-\\d+$')]\n\n"
        f"  2. Populate the columns explicitly before calling from_anndata:\n"
        f"     adata.var['chr']   = ...   # e.g. 'chr1'\n"
        f"     adata.var['start'] = ...   # int, 0-based or 1-based per your peak set\n"
        f"     adata.var['end_']  = ...   # int (note the trailing underscore)"
    )



def _write_uns_spatial(ds, spatial: dict) -> None:
    """Write a scanpy ``uns['spatial']`` dict into the spatial-image tables."""
    import warnings

    for lib, entry in spatial.items():
        if not isinstance(entry, dict):
            warnings.warn(f"uns['spatial'][{lib!r}] is not a dict; dropped")
            continue
        images = entry.get("images", {}) or {}
        sfs = entry.get("scalefactors", {}) or {}
        numeric_sfs = {k: float(v) for k, v in sfs.items()
                       if isinstance(v, (int, float, np.integer, np.floating))}
        dropped = [k for k in entry if k not in ("images", "scalefactors")]
        dropped += [k for k in sfs if k not in numeric_sfs]
        if dropped:
            warnings.warn(
                f"uns['spatial'][{lib!r}]: dropped non-convention keys "
                f"{sorted(dropped)} (only images + numeric scalefactors are "
                f"stored)")
        first = True
        for key, img in images.items():
            ds.add_spatial_image(str(lib), str(key), np.asarray(img),
                                 scalefactors=numeric_sfs if first else None,
                                 replace=True)
            first = False
        if first and numeric_sfs:
            # scalefactors but no images: keep them anyway
            from ..core.spatial import _ensure_tables
            _ensure_tables(ds._conn)
            for k, v in numeric_sfs.items():
                ds._conn.execute(
                    "INSERT OR REPLACE INTO spatial_scalefactors "
                    "(library_id, key, value) VALUES (?,?,?)",
                    (str(lib), k, v))


def _embedding_name(modality: str, obsm_key: str, existing=None) -> str:
    """The stored name for an ``obsm`` array: ``{modality}_{key}`` with the
    scanpy ``X_`` prefix dropped — ``RNA_umap``, ``RNA_spatial``, ``RNA_pca``,
    ``ATAC_umap`` — because the ``obsm``/``X_`` tokens are AnnData plumbing,
    not information. The exact original key is preserved in
    ``_anndata_obsm_map``, so ``to_anndata`` restores ``obsm`` verbatim
    regardless of the stored name; readers of files written by earlier
    versions still see the old ``{modality}_obsm_{key}`` names, which the
    basis resolvers already match.

    ``existing``: names already taken this conversion — on the (rare)
    collision like ``X_umap`` + ``umap`` both present, the later key keeps
    its full form instead of overwriting.
    """
    short = obsm_key[2:] if obsm_key.startswith("X_") else obsm_key
    name = f"{modality}_{short}"
    if existing is not None and name in existing:
        name = f"{modality}_{obsm_key}"
    return name


_MAIN_LAYER_FALLBACK = "data"


def _values_are_integer(matrix, n_probe: int = 20000) -> bool:
    """Whether a sparse/dense matrix holds integer values.

    Samples at most ``n_probe`` stored values: a normalized matrix has
    non-integers within the first few thousand.
    """
    data = getattr(matrix, "data", None)
    if data is None:
        data = np.asarray(matrix).ravel()
    data = np.asarray(data[:n_probe])
    if data.size == 0:
        return True                      # an all-zero matrix is integer
    if np.issubdtype(data.dtype, np.integer):
        return True
    return bool(np.allclose(data, np.round(data)))


def _resolve_main_matrix_name(modality, x_is_integer, counts_layer,
                              main_layer_name, stacklevel=3) -> str:
    """Name the matrix that ``adata.X`` becomes.

    The invariant this exists to hold: **``{modality}_counts`` holds raw
    integer counts, or it does not exist.** Before 0.3.0 ``adata.X`` went to
    ``{modality}_counts`` whatever it contained, so a normalized matrix was
    stored under a name every downstream default reads as counts -- which is
    how ``run_cosg_cytome(layer='auto')`` came to log-normalize an already
    log-normalized matrix.
    """
    if main_layer_name is not None:
        return f"{modality}_{main_layer_name}"
    if counts_layer is not None:
        # the raw counts are going to {modality}_counts, so X is something else
        return f"{modality}_{_MAIN_LAYER_FALLBACK}"
    if x_is_integer:
        return f"{modality}_counts"
    warnings.warn(
        f"adata.X does not hold integer counts, so it is stored as "
        f"'{modality}_{_MAIN_LAYER_FALLBACK}' rather than "
        f"'{modality}_counts' -- which downstream defaults read as raw "
        f"counts. Pass main_layer_name= to name it yourself, or "
        f"counts_layer= to say which layer holds the raw counts.",
        UserWarning, stacklevel=stacklevel,
    )
    return f"{modality}_{_MAIN_LAYER_FALLBACK}"


def from_anndata(
    adata,
    modality: str = "RNA",
    output: str | Path | None = None,
    counts_layer: str | None = None,
    main_layer_name: str | None = None,
    chunk_size: int | None = None,
    compression: str = "zstd",
    force: bool = False,
    write_raw: bool = True,
    write_layers: bool = True,
    write_obsm: bool = True,
    write_obsp: bool = True,
    write_varm: bool = True,
    write_varp: bool = True,
    write_uns: bool = True,
    skip_layers: Sequence[str] | None = None,
    skip_obsm: Sequence[str] | None = None,
    skip_obsp: Sequence[str] | None = None,
    skip_varm: Sequence[str] | None = None,
    skip_varp: Sequence[str] | None = None,
) -> CytomeDataset:
    """Convert AnnData to Cytome dataset.

    The ``write_*`` / ``skip_*`` arguments mirror :func:`from_h5ad`, which
    delegates here when ``backed=False``. They exist on both paths so that
    asking for a counts-only conversion means the same thing either way:
    before this, ``from_h5ad(..., backed=False, skip_layers=[...])`` accepted
    the argument and wrote every layer anyway.
    """
    del compression
    if output is None:
        handle = tempfile.NamedTemporaryFile(suffix=".cytome", delete=False)
        output = handle.name
        handle.close()
        force = True  # the tempfile we just created is ours to overwrite

    ds = CytomeDataset(output, mode="w", force=force)
    # Route the var entity table + id column via the modality registry
    # (single source of truth in cytome.utils.modality). Supports RNA →
    # genes, GA → GA_genes, ATAC → peaks, tiles → tiles. Pre-Round-11
    # this used a hardcoded ATAC-vs-other dichotomy, which silently
    # mis-routed GA / tiles inputs into the `genes` table.
    var_entity, id_col = _modality_var_entity(modality)

    obs = adata.obs.copy()
    if "barcode" not in obs.columns:
        obs.insert(0, "barcode", obs.index.astype(str))
    obs = obs.reset_index(drop=True)
    ds.set_entity("cells", obs)
    _store_column_meta(ds._conn, "cells", obs)

    var = adata.var.copy()
    if modality.upper() == "ATAC":
        var = _parse_peak_coords_from_varnames(var, adata.var_names)
        _check_atac_var_has_peak_coords(var)
        if id_col not in var.columns:
            var.insert(0, id_col, var.index.astype(str))
    else:
        # The cytome id column is UNIQUE. When it isn't already present, prefer an
        # existing unique id (10x 'gene_ids' = Ensembl) and keep the (possibly
        # duplicated) symbol — var_names — in a 'symbol' column for name-based
        # lookup. Otherwise fall back to var_names as the id. (Conditional, so the
        # common var_names-as-id case is unchanged / lossless.)
        if id_col not in var.columns:
            if "gene_ids" in var.columns:
                if "symbol" not in var.columns:
                    var.insert(0, "symbol", np.asarray(adata.var_names, dtype=str))
                    warn_duplicate_symbols(var["symbol"].values)
                var.insert(0, id_col, var["gene_ids"].astype(str).values)
            else:
                var.insert(0, id_col, np.asarray(adata.var_names, dtype=str))
        # Populate a 'symbol' display column from a recognised alias when the source
        # carries gene symbols under a non-standard name (CELLxGENE 'feature_name',
        # etc.) so downstream name resolution (COSG/dotplot/inferGRN) shows symbols,
        # not Ensembl ids. Only when 'symbol' isn't already a real column.
        if "symbol" not in var.columns:
            for _alias in ("gene_symbols", "gene_symbol", "gene_name",
                           "feature_name", "Symbol", "symbols"):
                if _alias in var.columns:
                    var.insert(0, "symbol", var[_alias].astype(str).values)
                    warn_duplicate_symbols(var["symbol"].values)
                    break
    # Enforce uniqueness of the id column only when it is actually violated.
    if id_col in var.columns:
        ids = var[id_col].astype(str).values
        uids = make_unique_ids(ids)
        if list(uids) != list(ids):
            warnings.warn(
                f"Duplicate {id_col} values were de-duplicated with -1/-2 suffixes "
                f"(the cytome {id_col} column is UNIQUE).", UserWarning, stacklevel=2)
            var[id_col] = uids
    var = var.reset_index(drop=True)
    ds.set_entity(var_entity, var)
    _store_column_meta(ds._conn, var_entity, var)

    matrix = adata.X if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))

    # `counts_layer` asserts where the raw counts are. Verify the assertion --
    # storing a normalized matrix under `{modality}_counts` is the whole bug
    # this release exists to close, and a caller who names the wrong layer
    # would reintroduce it by hand.
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"counts_layer={counts_layer!r} is not in adata.layers; "
                f"available: {sorted(adata.layers)}")
        _cl = adata.layers[counts_layer]
        _cl = _cl if sp.issparse(_cl) else sp.csr_matrix(np.asarray(_cl))
        if not _values_are_integer(_cl):
            raise ValueError(
                f"counts_layer={counts_layer!r} does not hold integer values, "
                f"so it is not raw counts. cytome stores raw counts as "
                f"'{modality}_counts' and every downstream default reads that "
                f"name as counts. Pass the layer that holds UMIs, or omit "
                f"counts_layer and name the matrix with main_layer_name=.")
        ds.add_matrix(f"{modality}_counts", _cl)

    x_matrix_name = _resolve_main_matrix_name(
        modality, _values_are_integer(matrix), counts_layer, main_layer_name)
    ds.add_matrix(x_matrix_name, matrix)

    # A layer promoted to `{modality}_counts` must not also be written under
    # its own name: it is the same matrix, and storing it twice doubles the
    # file for nothing. The promoted name is the canonical one.
    skip_layers_set = set(skip_layers or [])
    if counts_layer is not None:
        skip_layers_set.add(counts_layer)
    if counts_layer is not None:
        # already written as `{modality}_counts`; writing it again under its
        # AnnData name would store one matrix twice and recreate exactly the
        # near-identical pair this release exists to remove (RNA_count beside
        # RNA_counts, one letter apart).
        skip_layers_set.add(counts_layer)
    skip_obsm_set = set(skip_obsm or [])
    skip_obsp_set = set(skip_obsp or [])
    skip_varm_set = set(skip_varm or [])
    skip_varp_set = set(skip_varp or [])

    layer_map: dict[str, str] = {}
    if write_layers:
        for layer_name, layer_mat in adata.layers.items():
            if layer_name in skip_layers_set:
                continue
            cyt_name = f"{modality}_{layer_name}"
            layer = layer_mat if sp.issparse(layer_mat) else sp.csr_matrix(np.asarray(layer_mat))
            ds.add_matrix(cyt_name, layer)
            layer_map[cyt_name] = str(layer_name)

    obsm_map: dict[str, str] = {}
    obsm_as_matrix: dict[str, str] = {}
    for key, emb in (adata.obsm.items() if write_obsm else ()):
        if key in skip_obsm_set:
            continue
        cyt_key = _embedding_name(modality, key,
                                   existing=set(obsm_map) | set(obsm_as_matrix))
        if sp.issparse(emb):
            if emb.shape[1] > 500:
                # Large sparse obsm (e.g. gene activity) — store as matrix
                mat = emb.tocsr() if not sp.isspmatrix_csr(emb) else emb
                ds.add_matrix(cyt_key, mat)
                obsm_as_matrix[cyt_key] = str(key)
                continue
            emb = emb.toarray()
        else:
            emb = np.asarray(emb)
        ds.add_embedding(cyt_key, emb)
        obsm_map[cyt_key] = str(key)

    varm_map: dict[str, str] = {}
    for key, emb in (adata.varm.items() if write_varm else ()):
        if key in skip_varm_set:
            continue
        cyt_key = f"{modality}_varm_{key}"
        if sp.issparse(emb):
            emb = emb.toarray()
        else:
            emb = np.asarray(emb)
        ds.add_var_embedding(cyt_key, emb, entity=var_entity)
        varm_map[cyt_key] = str(key)

    obsp_map: dict[str, str] = {}
    for key, graph in (adata.obsp.items() if write_obsp else ()):
        if key in skip_obsp_set:
            continue
        cyt_key = f"{modality}_obsp_{key}"
        ds.add_graph(cyt_key, graph, axis="obs", entity_table="cells")
        obsp_map[cyt_key] = str(key)

    varp_map: dict[str, str] = {}
    for key, graph in (adata.varp.items() if write_varp else ()):
        if key in skip_varp_set:
            continue
        cyt_key = f"{modality}_varp_{key}"
        ds.add_var_graph(cyt_key, graph, entity_table=var_entity)
        varp_map[cyt_key] = str(key)

    if write_raw and adata.raw is not None:
        raw_name = f"{modality}_raw_X"
        raw_matrix = adata.raw.X if sp.issparse(adata.raw.X) else sp.csr_matrix(np.asarray(adata.raw.X))
        ds.add_matrix(raw_name, raw_matrix)
        _write_raw_var_table(ds._conn, adata.raw.var.copy())
        ds.metadata["_anndata_raw"] = {
            "matrix_name": raw_name,
            "var_table": "_raw_var",
        }

    ds.metadata["_anndata_X_layer"] = x_matrix_name
    ds.metadata["_anndata_layer_map"] = layer_map
    ds.metadata["_anndata_obsm_map"] = obsm_map
    ds.metadata["_anndata_obsm_as_matrix"] = obsm_as_matrix
    ds.metadata["_anndata_varm_map"] = varm_map
    ds.metadata["_anndata_obsp_map"] = obsp_map
    ds.metadata["_anndata_varp_map"] = varp_map

    if "spatial" in getattr(adata, "obsm", {}):
        # keep the queryable index in sync with the embedding it mirrors
        try:
            ds.set_spatial_coords(np.asarray(adata.obsm["spatial"]))
        except Exception as _e:      # pragma: no cover - index is best-effort
            import warnings as _w
            _w.warn(f"spatial coordinate index not built: {_e}")

    for key, value in (adata.uns.items() if write_uns else ()):
        if key == "spatial" and isinstance(value, dict):
            # Visium-convention images + scalefactors go to the spatial tables
            # (image arrays are not JSON and previously vanished in the
            # TypeError fallback below). Non-convention entries under a
            # library are named, never silently dropped.
            _write_uns_spatial(ds, value)
            continue
        try:
            ds.metadata[key] = value
        except TypeError:
            continue

    ds.flush()
    ds.provenance.log(
        operation="conversion",
        function_name="cytome.from_anndata",
        parameters={"modality": modality, "chunk_size": chunk_size},
        package_name="cytome",
        package_version="0.1.0",
        input_objects=["anndata"],
        output_objects=[x_matrix_name],
    )
    ds.flush()
    return ds


def _h5_values_are_integer(grp, n_probe: int = 20000) -> bool:
    """Whether an h5ad matrix group holds integer values.

    Reads at most ``n_probe`` stored values with an h5py slice, so the streaming
    path can hold the same naming invariant as the in-memory one without giving
    up streaming to do it.
    """
    try:
        data = grp["data"] if hasattr(grp, "keys") and "data" in grp else grp
        a = np.asarray(data[:n_probe], dtype="float64")
        if a.size == 0:
            return True
        a = a[np.isfinite(a)]
        return bool(a.size == 0 or np.allclose(a, np.round(a)))
    except Exception:
        return True          # cannot tell -> keep the historical name


def from_h5ad(
    h5ad_path: str | Path,
    output: str | Path,
    modality: str = "RNA",
    counts_layer: str | None = None,
    main_layer_name: str | None = None,
    backed: bool = False,
    chunk_size: int = 2048,
    storage_chunk_size: int = 128,
    compression: str = "zstd",
    verbose: bool = True,
    # ---- Round 10 (2026-05-24) per-category opt-outs ----
    write_raw: bool = True,
    write_layers: bool = True,
    write_obsm: bool = True,
    write_obsp: bool = True,
    write_varm: bool = True,
    write_varp: bool = True,
    write_uns: bool = True,
    skip_layers: list[str] | None = None,
    skip_obsm: list[str] | None = None,
    skip_obsp: list[str] | None = None,
    skip_varm: list[str] | None = None,
    skip_varp: list[str] | None = None,
    force: bool = False,
) -> CytomeDataset:
    """Convert an h5ad file to a Cytome dataset.

    Parameters
    ----------
    h5ad_path
        Path to the ``.h5ad`` file.
    output
        Path for the output ``.cytome`` file.
    modality
        Modality name (e.g. ``"RNA"``).
    backed
        If ``True``, stream the h5ad via h5py + cytome's
        ``create_layer_writer`` API. Peak RAM is bounded by
        ``chunk_size`` rows (sparse) or the largest dense obsm entry
        (whichever is bigger). If ``False``, defer to
        ``from_anndata`` (full in-memory load).
    chunk_size
        Number of rows per CSR read-chunk in backed mode.
    storage_chunk_size
        Rows per on-disk storage blob inside each chunk.
    compression
        Cytome storage compression ('zstd', 'lz4', 'zlib').
    verbose
        Print progress messages.
    write_raw
        Round 10: if False, drop ``adata.raw`` from the cytome
        (matches ``from_anndata``'s slot convention when True:
        ``{modality}_raw_X`` matrix + ``_raw_var`` table).
    write_layers
        Round 10: if False, drop all ``adata.layers`` entries.
    write_obsm / write_obsp / write_varm / write_varp / write_uns
        Round 10: per-category opt-outs. Defaults preserve everything.
    skip_layers / skip_obsm / skip_obsp / skip_varm / skip_varp
        Round 10: per-key skip lists within each category.

    Backed-mode peak-RAM fix (Round 10, 2026-05-24)
    -----------------------------------------------
    Pre-Round-10, the backed branch routed through
    ``anndata.read_h5ad(backed='r')``. That call lazy-loads only ``.X``
    and ``.raw.X``; everything else (layers / obsm / obsp / varm /
    varp / uns / obs / var) is eagerly read into memory before
    the function returns. On a 200 GB h5ad with two ~30 GB sparse
    layers and an 8 GB dense ``obsm['X_gdr']``, this OOM-killed at
    peak 109 GB.

    Round 10 bypasses AnnData's backed open entirely:

      * Sparse matrices (``X``, layers, ``raw.X``, sparse obsm /
        varm / obsp / varp): stream via h5py CSR slicing into
        cytome's public ``create_layer_writer`` API.
      * Small attrs (``obs``, ``var``, ``uns``, individual obsm
        keys): read via ``anndata.io.read_elem`` one-at-a-time and
        free between writes.

    No pyarrow dependency (verified — ``read_elem`` uses pure
    pandas+numpy+h5py).
    """
    if not backed:
        # anndata is an OPTIONAL extra -- cytome's own format needs none of it --
        # and it is imported lazily here, at call time. A bare ModuleNotFoundError
        # is a poor first impression for someone who just ran `pip install cytome`
        # and reached for the most obvious entry point, so name the extra.
        try:
            import anndata
        except ModuleNotFoundError as _e:               # pragma: no cover
            if _e.name != "anndata":
                raise
            raise ModuleNotFoundError(
                "Reading .h5ad requires the optional 'anndata' dependency:\n\n"
                "    pip install 'cytome[anndata]'\n\n"
                "Cytome's own .cytome format does not require it."
            ) from _e
        adata = anndata.read_h5ad(str(h5ad_path))
        return from_anndata(
            adata, modality=modality, output=output, force=force,
            write_raw=write_raw, write_layers=write_layers,
            write_obsm=write_obsm, write_obsp=write_obsp,
            write_varm=write_varm, write_varp=write_varp,
            write_uns=write_uns,
            skip_layers=skip_layers, skip_obsm=skip_obsm,
            skip_obsp=skip_obsp, skip_varm=skip_varm, skip_varp=skip_varp,
        )

    # === Backed (streaming) path — Round 10 rewrite ===
    import gc
    import time
    import h5py
    # `anndata.io` is the public home for read_elem only in anndata >= 0.11.
    # On anndata 0.10.x it lives in `anndata.experimental`. Fall back so the
    # backed path works across both (otherwise backed=True raised
    # ModuleNotFoundError: No module named 'anndata.io' on anndata < 0.11).
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem

    t_open = time.time()

    skip_layers_set = set(skip_layers or [])
    skip_obsm_set = set(skip_obsm or [])
    skip_obsp_set = set(skip_obsp or [])
    skip_varm_set = set(skip_varm or [])
    skip_varp_set = set(skip_varp or [])

    ds = CytomeDataset(output, mode="w", force=force)
    # Round 11 (2026-05-26): route via the modality registry. Same
    # change as the in-memory `from_anndata` path above — supports
    # GA → GA_genes and tiles → tiles, not just ATAC vs other.
    var_entity, id_col = _modality_var_entity(modality)

    with h5py.File(str(h5ad_path), "r") as f:

        if verbose:
            _emit_backed_inventory(
                f, modality, write_raw=write_raw, write_layers=write_layers,
                write_obsm=write_obsm, write_obsp=write_obsp,
                write_varm=write_varm, write_varp=write_varp,
                write_uns=write_uns,
                skip_layers=skip_layers_set, skip_obsm=skip_obsm_set,
                skip_obsp=skip_obsp_set, skip_varm=skip_varm_set,
                skip_varp=skip_varp_set,
                h5ad_path=h5ad_path, output=output,
            )

        # --- 1. obs / var via read_elem (no pyarrow; pure pandas) ---
        obs = read_elem(f["obs"])
        if "barcode" not in obs.columns:
            obs.insert(0, "barcode", obs.index.astype(str))
        obs = obs.reset_index(drop=True)
        ds.set_entity("cells", obs)
        _store_column_meta(ds._conn, "cells", obs)
        n_obs = len(obs)
        del obs

        var = read_elem(f["var"])
        if modality.upper() == "ATAC":
            var = _parse_peak_coords_from_varnames(var, var.index)
            _check_atac_var_has_peak_coords(var)
        if id_col not in var.columns:
            var.insert(0, id_col, var.index.astype(str))
        var = var.reset_index(drop=True)
        ds.set_entity(var_entity, var)
        _store_column_meta(ds._conn, var_entity, var)
        n_vars = len(var)
        del var
        gc.collect()
        ds.flush()

        if verbose:
            print(f"[from_h5ad backed] obs/var written ({n_obs:,} cells × "
                  f"{n_vars:,} {var_entity}). Streaming matrices...")

        # --- 2. X — stream via cytome's create_layer_writer ---
        # Same invariant as from_anndata: {modality}_counts holds raw integer
        # counts or does not exist. Probed with an h5py slice so streaming is
        # preserved.
        x_matrix_name = _resolve_main_matrix_name(
            modality, _h5_values_are_integer(f["X"]),
            counts_layer, main_layer_name, stacklevel=4)
        _stream_h5_csr_to_writer(
            ds, f["X"], matrix_name=x_matrix_name,
            chunk_size=chunk_size, storage_chunk_size=storage_chunk_size,
            compression=compression, row_entity="cells",
            col_entity=var_entity, verbose=verbose,
        )
        ds.metadata["_anndata_X_layer"] = x_matrix_name

        # --- 3. layers ---
        layer_map: dict[str, str] = {}
        if write_layers and "layers" in f:
            for layer_name in f["layers"].keys():
                if layer_name in skip_layers_set:
                    if verbose:
                        print(f"  [skip] layers/{layer_name}")
                    continue
                cyt_name = f"{modality}_{layer_name}"
                _stream_h5_csr_to_writer(
                    ds, f[f"layers/{layer_name}"], matrix_name=cyt_name,
                    chunk_size=chunk_size, storage_chunk_size=storage_chunk_size,
                    compression=compression, row_entity="cells",
                    col_entity=var_entity, verbose=verbose,
                )
                layer_map[cyt_name] = str(layer_name)
        ds.metadata["_anndata_layer_map"] = layer_map

        # --- 4. raw.X (cytome convention: {modality}_raw_X + _raw_var) ---
        if write_raw and "raw" in f and "X" in f["raw"]:
            raw_name = f"{modality}_raw_X"
            _stream_h5_csr_to_writer(
                ds, f["raw/X"], matrix_name=raw_name,
                chunk_size=chunk_size, storage_chunk_size=storage_chunk_size,
                compression=compression, row_entity="cells",
                col_entity=var_entity, verbose=verbose,
            )
            if "var" in f["raw"]:
                raw_var = read_elem(f["raw/var"])
                _write_raw_var_table(ds._conn, raw_var)
                del raw_var
                gc.collect()
            ds.metadata["_anndata_raw"] = {
                "matrix_name": raw_name,
                "var_table": "_raw_var",
            }

        # --- 5. obsm — read each key, write, free ---
        obsm_map: dict[str, str] = {}
        obsm_as_matrix: dict[str, str] = {}
        if write_obsm and "obsm" in f:
            for k in f["obsm"].keys():
                if k in skip_obsm_set:
                    if verbose:
                        print(f"  [skip] obsm/{k}")
                    continue
                arr = read_elem(f["obsm"][k])
                cyt_key = _embedding_name(
                    modality, k, existing=set(obsm_map) | set(obsm_as_matrix))
                if sp.issparse(arr):
                    if arr.shape[1] > 500:
                        mat = arr.tocsr() if not sp.isspmatrix_csr(arr) else arr
                        ds.add_matrix(cyt_key, mat)
                        obsm_as_matrix[cyt_key] = str(k)
                        del arr, mat
                        gc.collect()
                        continue
                    arr = arr.toarray()
                ds.add_embedding(cyt_key, np.asarray(arr))
                obsm_map[cyt_key] = str(k)
                del arr
                gc.collect()
        ds.metadata["_anndata_obsm_map"] = obsm_map
        ds.metadata["_anndata_obsm_as_matrix"] = obsm_as_matrix

        # --- 6. varm — same pattern but via add_var_embedding ---
        varm_map: dict[str, str] = {}
        if write_varm and "varm" in f:
            for k in f["varm"].keys():
                if k in skip_varm_set:
                    if verbose:
                        print(f"  [skip] varm/{k}")
                    continue
                arr = read_elem(f["varm"][k])
                cyt_key = f"{modality}_varm_{k}"
                if sp.issparse(arr):
                    arr = arr.toarray()
                ds.add_var_embedding(cyt_key, np.asarray(arr), entity=var_entity)
                varm_map[cyt_key] = str(k)
                del arr
                gc.collect()
        ds.metadata["_anndata_varm_map"] = varm_map

        # --- 7. obsp — read each, add_graph ---
        obsp_map: dict[str, str] = {}
        if write_obsp and "obsp" in f:
            for k in f["obsp"].keys():
                if k in skip_obsp_set:
                    if verbose:
                        print(f"  [skip] obsp/{k}")
                    continue
                graph = read_elem(f["obsp"][k])
                cyt_key = f"{modality}_obsp_{k}"
                ds.add_graph(cyt_key, graph, axis="obs", entity_table="cells")
                obsp_map[cyt_key] = str(k)
                del graph
                gc.collect()
        ds.metadata["_anndata_obsp_map"] = obsp_map

        # --- 8. varp — add_var_graph ---
        varp_map: dict[str, str] = {}
        if write_varp and "varp" in f:
            for k in f["varp"].keys():
                if k in skip_varp_set:
                    if verbose:
                        print(f"  [skip] varp/{k}")
                    continue
                graph = read_elem(f["varp"][k])
                cyt_key = f"{modality}_varp_{k}"
                ds.add_var_graph(cyt_key, graph, entity_table=var_entity)
                varp_map[cyt_key] = str(k)
                del graph
                gc.collect()
        ds.metadata["_anndata_varp_map"] = varp_map

        # --- 9. uns — per-key, skip unpickleable ---
        if write_uns and "uns" in f:
            uns = read_elem(f["uns"])
            for key, value in uns.items():
                try:
                    ds.metadata[key] = value
                except TypeError:
                    if verbose:
                        print(f"  [skip] uns/{key} (unpickleable)")
            del uns
            gc.collect()

    ds.flush()
    try:
        from cytome import __version__ as _cytome_version
    except Exception:
        _cytome_version = "0.0.0"
    ds.provenance.log(
        operation="conversion",
        function_name="cytome.from_h5ad",
        parameters={
            "modality": modality,
            "backed": True,
            "chunk_size": chunk_size,
            "storage_chunk_size": storage_chunk_size,
            "compression": compression,
            "write_raw": write_raw,
            "write_layers": write_layers,
            "write_obsm": write_obsm,
            "write_obsp": write_obsp,
            "write_varm": write_varm,
            "write_varp": write_varp,
            "write_uns": write_uns,
            "skip_layers": sorted(skip_layers_set),
            "skip_obsm": sorted(skip_obsm_set),
            "skip_obsp": sorted(skip_obsp_set),
            "skip_varm": sorted(skip_varm_set),
            "skip_varp": sorted(skip_varp_set),
        },
        package_name="cytome",
        package_version=_cytome_version,
        input_objects=[str(h5ad_path)],
        output_objects=[x_matrix_name],
    )
    ds.flush()
    if verbose:
        print(f"[from_h5ad backed] Done in {time.time()-t_open:.1f}s. "
              f"Output: {output}")
    return ds


def _stream_h5_csr_to_writer(
    ds: CytomeDataset, h5_csr_group, *, matrix_name: str,
    chunk_size: int, storage_chunk_size: int, compression: str,
    row_entity: str, col_entity: str, verbose: bool,
) -> None:
    """Stream a CSR sparse matrix from an h5ad group into cytome.

    Uses h5py slicing (no anndata involvement) + cytome's public
    ``create_layer_writer`` streaming API. Peak RAM is bounded by
    ``chunk_size * average_nnz_per_row * dtype_bytes``.

    The h5 group must have data/indices/indptr datasets per AnnData's
    CSR encoding (``encoding-type='csr_matrix'``).

    Round 10 (2026-05-24) — replaces the previous ``_write_sparse_backed``
    helper that wrote directly to SQL. Uses the public writer API so we
    pick up provenance + future schema migrations for free.
    """
    import time

    encoding = h5_csr_group.attrs.get("encoding-type", b"")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if "csc" in encoding.lower():
        raise NotImplementedError(
            f"{matrix_name}: CSC-encoded h5ad streaming not yet supported "
            f"in cytome's backed path. Workaround: re-write the h5ad with "
            f"row-major (CSR) encoding via "
            f"`anndata.AnnData(..., X=adata.X.tocsr()).write_h5ad(...)`, "
            f"or call `from_h5ad(..., backed=False)` at the cost of "
            f"full-RAM load."
        )

    shape = tuple(int(s) for s in h5_csr_group.attrs["shape"])
    n_rows, n_cols = shape
    indptr = h5_csr_group["indptr"][:]  # ~8 bytes × n_rows; bounded.
    dtype = h5_csr_group["data"].dtype

    writer = ds.create_layer_writer(
        matrix_name, n_rows, n_cols,
        dtype=dtype, compression=compression,
        row_entity=row_entity, col_entity=col_entity,
        overwrite=True, storage_chunk_size=storage_chunk_size,
    )

    t0 = time.time()
    total_nnz = 0
    for row_start in range(0, n_rows, chunk_size):
        row_end = min(row_start + chunk_size, n_rows)
        nnz_start = int(indptr[row_start])
        nnz_end = int(indptr[row_end])
        chunk_data = h5_csr_group["data"][nnz_start:nnz_end]
        chunk_indices = h5_csr_group["indices"][nnz_start:nnz_end]
        chunk_indptr = (indptr[row_start:row_end + 1] - indptr[row_start])
        chunk = sp.csr_matrix(
            (chunk_data, chunk_indices, chunk_indptr),
            shape=(row_end - row_start, n_cols),
        )
        writer.write_chunk(chunk, row_offset=row_start)
        total_nnz += int(chunk.nnz)
        del chunk, chunk_data, chunk_indices, chunk_indptr
    writer.finalize()

    if verbose:
        elapsed = time.time() - t0
        print(f"  [{matrix_name}] {n_rows:,} rows × {n_cols:,} cols, "
              f"{total_nnz:,} nnz, dtype={dtype}, {elapsed:.1f}s")


def _emit_backed_inventory(
    f, modality, *, write_raw, write_layers, write_obsm, write_obsp,
    write_varm, write_varp, write_uns,
    skip_layers, skip_obsm, skip_obsp, skip_varm, skip_varp,
    counts_layer=None, main_layer_name=None,
    h5ad_path, output,
):
    """Print a slot inventory before conversion starts (verbose=True only).

    Surfaces exactly what will/won't be written so users can ctrl-C
    and tweak skip-lists without wasting compute. Round 10 (2026-05-24).
    """
    print(f"[from_h5ad backed] {h5ad_path} -> {output}")

    def _shape_of(grp):
        if hasattr(grp, "attrs") and "shape" in grp.attrs:
            return tuple(int(s) for s in grp.attrs["shape"])
        if hasattr(grp, "shape"):
            return tuple(grp.shape)
        return None

    def _csr_dtype(grp):
        try:
            return str(grp["data"].dtype)
        except Exception:
            return "?"

    print(f"  Will write:")
    x_shape = _shape_of(f["X"]) if "X" in f else None
    if x_shape:
        # announce the name X will actually get, not the one it used to get
        _x_name = _resolve_main_matrix_name(
            modality, _h5_values_are_integer(f["X"]), counts_layer,
            main_layer_name, stacklevel=6) if "X" in f else f"{modality}_counts"
        print(f"    - X ({_x_name}): {x_shape[0]:,} x {x_shape[1]:,}, "
              f"dtype={_csr_dtype(f['X'])}")

    if write_layers and "layers" in f:
        layer_names = [k for k in f["layers"].keys() if k not in skip_layers]
        print(f"    - {len(layer_names)} layer(s): {layer_names}")
        if skip_layers:
            print(f"      [skip from skip_layers]: {sorted(skip_layers)}")
    elif "layers" in f:
        print(f"    - layers: SKIPPED (write_layers=False)")

    if write_raw and "raw" in f and "X" in f["raw"]:
        print(f"    - raw.X ({modality}_raw_X): {_shape_of(f['raw/X'])}, "
              f"dtype={_csr_dtype(f['raw/X'])}")
    elif "raw" in f and "X" in f["raw"]:
        print(f"    - raw.X: SKIPPED (write_raw=False)")

    for slot, write_flag, skip_set in (
        ("obsm", write_obsm, skip_obsm),
        ("obsp", write_obsp, skip_obsp),
        ("varm", write_varm, skip_varm),
        ("varp", write_varp, skip_varp),
    ):
        if slot in f:
            if write_flag:
                keys = [k for k in f[slot].keys() if k not in skip_set]
                print(f"    - {len(keys)} {slot} key(s): {keys}")
                if skip_set:
                    print(f"      [skip from skip_{slot}]: {sorted(skip_set)}")
            else:
                print(f"    - {slot}: SKIPPED (write_{slot}=False)")

    if "uns" in f and write_uns:
        try:
            n_uns = len(list(f["uns"].keys()))
            print(f"    - uns ({n_uns} entries; unpickleable skipped at write)")
        except Exception:
            pass


def to_anndata(
    ds: CytomeDataset,
    modality: str = "RNA",
    layer: str | None = None,
    include_embeddings: bool = True,
    include_cross_modality_embeddings: bool = True,
    cell_mask=None,
):
    """Convert Cytome dataset modality to AnnData.

    Parameters
    ----------
    ds
        CytomeDataset to convert.
    modality
        Modality name (e.g. ``"RNA"``).
    layer
        Specific layer to use as X. If ``None``, uses the default X layer.
    include_embeddings
        Whether to include obsm embeddings.
    cell_mask
        Optional boolean mask (length ``n_cells``) or sorted integer indices.
        When provided, only reads chunks containing selected cells,
        yielding significant RAM and time savings for partial exports.
    """
    try:
        import anndata
    except ImportError as exc:  # pragma: no cover
        raise ImportError("anndata is required for to_anndata") from exc

    from cytome.io.chunked_io import read_dense_rows

    # Validate modality up front (before any I/O) so an unknown modality
    # produces a clean ValueError rather than a deeper KeyError on the
    # matrix or entity lookup.
    var_entity, id_col = _modality_var_entity(modality)

    # Resolve cell_mask to sorted integer indices (or None for full export)
    keep_idx = None
    if cell_mask is not None:
        arr = np.asarray(cell_mask)
        if arr.dtype == bool:
            keep_idx = np.where(arr)[0]
        else:
            keep_idx = np.sort(np.asarray(arr, dtype=np.int64))

    if layer:
        x_matrix_name = f"{modality}_{layer}"
    else:
        # `_anndata_X_layer` is a single GLOBAL key written by the FIRST
        # from_anndata (typically RNA). Honor it ONLY when it belongs to the
        # requested modality; otherwise default to `{modality}_counts`. Without
        # this gate, to_anndata(modality="GA") grabs the RNA X matrix (wrong
        # column count) — see convert_anndata modality-scoping fix.
        default_x = f"{modality}_counts"
        recorded = _metadata_get(ds, "_anndata_X_layer", default=None)
        if (recorded and str(recorded).startswith(f"{modality}_")
                and _matrix_exists(ds._conn, recorded)):
            x_matrix_name = recorded
        elif _matrix_exists(ds._conn, default_x):
            x_matrix_name = default_x
        else:
            raise KeyError(
                f"to_anndata(modality={modality!r}): no X matrix found. Tried "
                f"{default_x!r}"
                + (f" and recorded {recorded!r}" if recorded else "")
                + f". Available for this modality: "
                + str([r[0] for r in ds._conn.execute(
                    "SELECT matrix_name FROM matrix_meta WHERE matrix_name LIKE ?",
                    (f"{modality}_%",)).fetchall()])
            )

    ml = MeasurementLayer(ds._conn, x_matrix_name)
    X = ml.rows(keep_idx) if keep_idx is not None else ml.to_memory()

    obs = _restore_column_dtypes(ds._conn, "cells", ds.cells.to_pandas())
    if keep_idx is not None:
        obs = obs.iloc[keep_idx].copy().reset_index(drop=True)
    if "barcode" in obs.columns:
        obs = obs.set_index("barcode", drop=False)

    # Build an EntityTable directly off the connection so we don't depend on
    # per-table @property accessors existing for every modality (e.g. tiles
    # has no ds.tiles accessor — `ds.tiles` would route through __getattr__
    # and return a Modality instead of an EntityTable). var_entity / id_col
    # were resolved up front from the modality registry.
    from cytome.core.entity import EntityTable
    var_tbl = EntityTable(ds._conn, var_entity)
    var = _restore_column_dtypes(ds._conn, var_entity, var_tbl.to_pandas())
    if id_col in var.columns:
        var = var.set_index(id_col, drop=False)

    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Collect names of obsm entries stored as matrices so they are not
    # mistaken for regular layers during fallback discovery.
    _obsm_matrix_names = set()
    obsm_as_matrix_meta = _metadata_get(ds, "_anndata_obsm_as_matrix", default={})
    if isinstance(obsm_as_matrix_meta, dict):
        _obsm_matrix_names = set(obsm_as_matrix_meta.keys())

    # Layers are feature-dimensioned, so they MUST match the requested modality.
    # Restrict discovery to `{modality}_%` matrices (the global `_anndata_layer_map`
    # is written for whichever modality was imported first and would otherwise
    # attach e.g. RNA_infog to a GA AnnData → column-count mismatch). Honor the
    # recorded adata-layer name when the map has one for this modality, and always
    # guard on the feature count.
    prefix = f"{modality}_"
    layer_map = _metadata_get(ds, "_anndata_layer_map", default={})
    layer_map = layer_map if isinstance(layer_map, dict) else {}
    rows = ds._conn.execute(
        "SELECT matrix_name FROM matrix_meta WHERE matrix_name LIKE ? ORDER BY matrix_name",
        (f"{prefix}%",),
    ).fetchall()
    for (name,) in rows:
        if name == x_matrix_name or name in _obsm_matrix_names:
            continue
        lyr = MeasurementLayer(ds._conn, name)
        # Skip layers with mismatched column count (compact HVG layers, etc.)
        if lyr.shape[1] != adata.shape[1]:
            continue
        adata_name = layer_map.get(name, name[len(prefix):])
        adata.layers[adata_name] = lyr.rows(keep_idx) if keep_idx is not None else lyr.to_memory()

    if include_embeddings:
        prefix = f"{modality}_"
        obsm_map = _metadata_get(ds, "_anndata_obsm_map", default={})
        obsm_map = obsm_map if isinstance(obsm_map, dict) else {}
        added_keys: set[str] = set()

        def _read_emb(k):
            return (read_dense_rows(ds._conn, k, keep_idx)
                    if keep_idx is not None else ds.embeddings[k])

        # (1) OWN-modality cell embeddings: honor the recorded adata name for
        #     this-modality map entries, then prefix-discover the rest. Cell
        #     embeddings are n_obs-dimensioned, so no feature-count issue — but
        #     the global map must still be gated so a GA export doesn't inherit
        #     RNA's obsm names.
        for cytome_key, adata_key in obsm_map.items():
            if str(cytome_key).startswith(prefix) and cytome_key in ds.embeddings.keys():
                adata.obsm[adata_key] = _read_emb(cytome_key)
                added_keys.add(cytome_key)
        for name in ds.embeddings.keys():
            if name in added_keys or "_TMP_" in name:
                continue
            if name.startswith(prefix):
                adata.obsm[f"X_{name[len(prefix):]}"] = _read_emb(name)
                added_keys.add(name)

        # (2) OWN-modality sparse obsm stored as matrices (feature-ish; gated).
        obsm_as_matrix = _metadata_get(ds, "_anndata_obsm_as_matrix", default={})
        if isinstance(obsm_as_matrix, dict):
            for cytome_key, adata_key in obsm_as_matrix.items():
                if str(cytome_key).startswith(prefix) and _matrix_exists(ds._conn, cytome_key):
                    ml = MeasurementLayer(ds._conn, cytome_key)
                    adata.obsm[adata_key] = ml.rows(keep_idx) if keep_idx is not None else ml.to_memory()
                    added_keys.add(cytome_key)

        # (3) CROSS-modality cell embeddings (other modalities). These are
        #     shared cell coordinates (n_obs rows), so they attach cleanly to any
        #     modality's AnnData. Named by their cytome key (`_obsm_`→`_`) to keep
        #     provenance; skipped if that collides with an own-modality obsm key.
        if include_cross_modality_embeddings:
            for name in ds.embeddings.keys():
                if name in added_keys or "_TMP_" in name:
                    continue
                disp = str(name).replace("_obsm_", "_")
                if disp in adata.obsm:
                    continue
                adata.obsm[disp] = _read_emb(name)
                added_keys.add(name)

        # (4) varm is VAR-dimensioned → strictly modality-scoped + shape-guarded.
        varm_map = _metadata_get(ds, "_anndata_varm_map", default={})
        if isinstance(varm_map, dict):
            for cytome_key, adata_key in varm_map.items():
                if not str(cytome_key).startswith(prefix):
                    continue
                if _embedding_exists(ds._conn, cytome_key):
                    ve = ds.var_embeddings[cytome_key]
                    if getattr(ve, "shape", (0,))[0] == adata.shape[1]:
                        adata.varm[adata_key] = ve

    obsp_map = _metadata_get(ds, "_anndata_obsp_map", default={})
    if isinstance(obsp_map, dict) and obsp_map:
        for cytome_key, adata_key in obsp_map.items():
            try:
                adata.obsp[adata_key] = ds.graphs[cytome_key].to_sparse()
            except KeyError:
                continue
    else:
        try:
            adata.obsp["connectivities"] = ds.graphs["knn"].to_sparse()
        except Exception:
            pass

    varp_map = _metadata_get(ds, "_anndata_varp_map", default={})
    if isinstance(varp_map, dict):
        for cytome_key, adata_key in varp_map.items():
            try:
                adata.varp[adata_key] = ds.var_graphs[cytome_key].to_sparse()
            except KeyError:
                continue

    for key, value in ds.metadata.items():
        if key.startswith("_anndata_"):
            continue
        adata.uns[key] = value

    _spatial_uns = ds.spatial_images.as_uns() if hasattr(ds, "spatial_images") else {}
    if _spatial_uns:
        adata.uns["spatial"] = _spatial_uns

    raw_info = _metadata_get(ds, "_anndata_raw", default=None)
    if isinstance(raw_info, dict):
        matrix_name = raw_info.get("matrix_name")
        if matrix_name and _matrix_exists(ds._conn, matrix_name):
            raw_X = MeasurementLayer(ds._conn, matrix_name).to_memory()
            raw_var = _read_raw_var_table(ds._conn)
            adata.raw = anndata.AnnData(X=raw_X, var=raw_var)

    return adata


def update_from_anndata(ds: CytomeDataset, adata, modality: str = "RNA") -> None:
    """Sync selected AnnData changes back into a Cytome dataset."""
    obs = adata.obs.copy()
    if "barcode" not in obs.columns:
        obs.insert(0, "barcode", obs.index.astype(str))
    ds.set_entity("cells", obs.reset_index(drop=True))
    _store_column_meta(ds._conn, "cells", obs.reset_index(drop=True))

    _taken: set = set()
    for key, emb in adata.obsm.items():
        cyt_key = _embedding_name(modality, key, existing=_taken)
        _taken.add(cyt_key)
        ds.add_embedding(cyt_key, np.asarray(emb))
    for key, layer in adata.layers.items():
        mat = layer if sp.issparse(layer) else sp.csr_matrix(np.asarray(layer))
        ds.add_matrix(f"{modality}_{key}", mat)

    ds.flush()


def _store_column_meta(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    """Record dtype metadata for DataFrame columns."""
    conn.execute("DELETE FROM _column_meta WHERE table_name = ?", (table_name,))
    for col in df.columns:
        series = df[col]
        dtype_tag = None
        categories = None
        if isinstance(series.dtype, pd.CategoricalDtype):
            cats = series.cat.categories.tolist()
            categories = json.dumps(cats)
            dtype_tag = "ordered_categorical" if bool(series.cat.ordered) else "categorical"
        elif pd.api.types.is_bool_dtype(series.dtype) or str(series.dtype) == "boolean":
            dtype_tag = "bool"
        elif pd.api.types.is_integer_dtype(series.dtype):
            dtype_tag = str(series.dtype)
        elif pd.api.types.is_float_dtype(series.dtype):
            dtype_tag = str(series.dtype)
        elif pd.api.types.is_string_dtype(series.dtype):
            dtype_tag = "string"
        if dtype_tag is None:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO _column_meta(table_name, column_name, dtype, categories)
            VALUES (?, ?, ?, ?)
            """,
            (table_name, str(col), dtype_tag, categories),
        )


def _restore_column_dtypes(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Restore pandas dtype metadata for a DataFrame."""
    try:
        rows = conn.execute(
            "SELECT column_name, dtype, categories FROM _column_meta WHERE table_name = ?",
            (table_name,),
        ).fetchall()
    except sqlite3.OperationalError:
        return df
    for col, dtype, categories_json in rows:
        if col not in df.columns:
            continue
        if dtype in {"categorical", "ordered_categorical"}:
            cats = json.loads(categories_json) if categories_json else []
            ordered = dtype == "ordered_categorical"
            df[col] = pd.Categorical(df[col], categories=cats, ordered=ordered)
        elif dtype == "bool":
            try:
                df[col] = df[col].astype(bool)
            except Exception:
                pass
        elif dtype.startswith("Int") or dtype.startswith("UInt"):
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                pass
        elif dtype in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}:
            try:
                if df[col].isna().any():
                    up = dtype.upper().replace("INT", "Int")
                    df[col] = df[col].astype(up)
                else:
                    df[col] = df[col].astype(dtype)
            except Exception:
                pass
        elif dtype in {"float32", "float64"}:
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                pass
    return df


def _write_raw_var_table(conn: sqlite3.Connection, var_df: pd.DataFrame) -> None:
    frame = var_df.copy()
    if "var_name" not in frame.columns:
        frame.insert(0, "var_name", frame.index.astype(str))
    frame = frame.reset_index(drop=True)
    if "var_idx" not in frame.columns:
        frame.insert(0, "var_idx", np.arange(frame.shape[0], dtype=np.int64))

    existing_cols = [c[1] for c in conn.execute("PRAGMA table_info(_raw_var)")]
    for col in frame.columns:
        if col in existing_cols:
            continue
        sql_type = _sqlite_type_for_series(frame[col])
        conn.execute(f"ALTER TABLE _raw_var ADD COLUMN {_quote_ident(col)} {sql_type}")

    conn.execute("DELETE FROM _raw_var")
    cols = frame.columns.tolist()
    placeholders = ",".join(["?"] * len(cols))
    quoted_cols = ", ".join(_quote_ident(c) for c in cols)
    rows = [tuple(_py_scalar(v) for v in row) for row in frame.to_numpy()]
    conn.executemany(f"INSERT INTO _raw_var ({quoted_cols}) VALUES ({placeholders})", rows)
    _store_column_meta(conn, "_raw_var", frame)


def _read_raw_var_table(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        df = pd.read_sql_query("SELECT * FROM _raw_var ORDER BY var_idx", conn)
    except Exception:
        df = pd.DataFrame()
    if "var_idx" in df.columns:
        df = df.drop(columns=["var_idx"])
    df = _restore_column_dtypes(conn, "_raw_var", df)
    if "var_name" in df.columns:
        df = df.set_index("var_name", drop=False)
    return df


def _sqlite_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series.dtype):
        return "REAL"
    return "TEXT"


def _metadata_get(ds: CytomeDataset, key: str, default: Any = None) -> Any:
    try:
        return ds.metadata[key]
    except KeyError:
        return default


def _matrix_exists(conn: sqlite3.Connection, matrix_name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    return bool(row and row[0] > 0)


def _embedding_exists(conn: sqlite3.Connection, array_name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    return bool(row and row[0] > 0)


def _py_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'
