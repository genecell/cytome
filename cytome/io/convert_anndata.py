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


#: the cells column that keeps obs_names when they are not the barcode column
_OBS_NAME_COL = "_obs_name"
#: the feature-table column that keeps var_names when no other column holds them
_VAR_NAME_COL = "_var_name"
#: col_entity of matrices with no feature table of the modality (see FORMAT_SPEC)
_RAW_ENTITY = "_raw_var"
_OBSM_ENTITY = "_obsm"


def _drop_incoming_key(df: pd.DataFrame, key: str, what: str) -> pd.DataFrame:
    """``df`` without cytome's own row key ``key`` (``cell_idx``, ``gene_idx``).

    The key is assigned by position. ``to_anndata`` of earlier versions exported
    it, and kept as the key it is wrong after any subset: genes 2, 3 and 4 of a
    three-column matrix, so a lookup of gene 2 read gene 4's column. SQLite
    names compare without case, so ``Cell_idx`` is the key too.
    """
    cols = [c for c in df.columns if str(c).lower() == key.lower()]
    if not cols:
        return df
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if not np.array_equal(vals, np.arange(len(df))):
            warnings.warn(
                f"{what}: the column {c!r} is cytome's row key, assigned by position; "
                f"its values ({vals[:3].tolist()} ...) are not positions and are dropped",
                UserWarning, stacklevel=4)
    return df.drop(columns=cols)


def _prepare_obs(obs: pd.DataFrame):
    """``(cells frame, column holding obs_names or None)`` for an AnnData's obs.

    ``barcode`` is the obs_names unless obs has its own ``barcode`` column; then
    obs_names that differ from it (unique names over merged libraries, say) are
    kept in ``_obs_name``, which :func:`to_anndata` makes the index again.
    """
    obs = _drop_incoming_key(obs.copy(), "cell_idx", "obs")
    names = np.asarray(pd.Index(obs.index).astype(str), dtype=object)
    index_col = None
    if "barcode" not in obs.columns:
        obs.insert(0, "barcode", names)
    elif not np.array_equal(obs["barcode"].astype(str).to_numpy(dtype=object), names):
        obs = obs.drop(columns=[_OBS_NAME_COL], errors="ignore")
        obs.insert(0, _OBS_NAME_COL, names)
        index_col = _OBS_NAME_COL
    return obs.reset_index(drop=True), index_col


def _prepare_var(var: pd.DataFrame, var_names, modality: str, id_col: str, key_col: str):
    """``(feature frame, column holding var_names or None)`` for an AnnData's var.

    One routine for the in-memory and the streaming conversion, which keyed
    genes differently and failed on duplicate names in the streaming one. The
    id column is unique (``gene_ids`` when the var has them, var_names
    otherwise, duplicates suffixed); var_names that end up in another column
    (``symbol`` for a 10x var) are recorded, so ``to_anndata`` restores them
    (it returned the Ensembl IDs as var_names).
    """
    var = _drop_incoming_key(var.copy(), key_col, "var")
    names = np.asarray(pd.Index(var_names).astype(str), dtype=object)
    if modality.upper() == "ATAC":
        var = _parse_peak_coords_from_varnames(var, var_names)
        _check_atac_var_has_peak_coords(var)
        if id_col not in var.columns:
            var.insert(0, id_col, names)
    else:
        # The cytome id column is UNIQUE. When it isn't already present, prefer an
        # existing unique id (10x 'gene_ids' = Ensembl) and keep the (possibly
        # duplicated) symbol — var_names — in a 'symbol' column for name-based
        # lookup. Otherwise fall back to var_names as the id.
        if id_col not in var.columns:
            if "gene_ids" in var.columns:
                if "symbol" not in var.columns:
                    var.insert(0, "symbol", names)
                    warn_duplicate_symbols(var["symbol"].values)
                var.insert(0, id_col, var["gene_ids"].astype(str).values)
            else:
                var.insert(0, id_col, names)
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
    ids = var[id_col].astype(str).values
    uids = make_unique_ids(ids)
    if list(uids) != list(ids):
        warnings.warn(
            f"Duplicate {id_col} values were de-duplicated with -1/-2 suffixes "
            f"(the cytome {id_col} column is UNIQUE).", UserWarning, stacklevel=3)
        var[id_col] = uids
    index_col = None
    if not np.array_equal(var[id_col].astype(str).to_numpy(dtype=object), names):
        for c in ("symbol",):
            if c in var.columns and np.array_equal(
                    var[c].astype(str).to_numpy(dtype=object), names):
                index_col = c
                break
        else:
            var = var.drop(columns=[_VAR_NAME_COL], errors="ignore")
            var.insert(0, _VAR_NAME_COL, names)
            index_col = _VAR_NAME_COL
    return var.reset_index(drop=True), index_col


def _same_matrix(a, b) -> bool:
    """Whether two sparse matrices hold the same values (canonical forms compared)."""
    if a.shape != b.shape:
        return False
    a, b = sp.csr_matrix(a), sp.csr_matrix(b)
    return (a != b).nnz == 0


class _MatrixNames:
    """The matrix names one conversion writes, refusing a name given twice.

    A non-integer X is stored as ``RNA_data``, and a layer named ``data`` was
    written under the same name and replaced it (a Seurat-derived h5ad: X =
    scale.data, layers data and counts); ``counts_layer=`` with
    ``main_layer_name='counts'`` replaced the verified counts with X. Two
    sources of one name are refused, unless they hold the same matrix, which
    is then stored once and the second name recorded as an alias.
    """

    def __init__(self, compare):
        self._compare = compare
        self.sources: dict[str, tuple[str, Any]] = {}

    def claim(self, name: str, source: str, matrix) -> bool:
        """True: write ``matrix`` as ``name``; False: it is already there."""
        if name in self.sources:
            prev_source, prev = self.sources[name]
            if self._compare(prev, matrix):
                return False
            raise ValueError(
                f"{source} and {prev_source} would both be stored as {name!r}, and they "
                f"differ. Pass main_layer_name= to name X (main_layer_name='scale_data', say), "
                f"or rename the layer.")
        self.sources[name] = (source, matrix)
        return True


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

    ``compression`` names the codec of the matrices (``zstd``, ``lz4`` or
    ``zlib``). Two sources that would be stored under one name (a layer called
    ``data`` beside a non-integer X) are refused unless they hold the same
    matrix; ``main_layer_name=`` names X.
    """
    from cytome.io.compression import _resolve_method
    _resolve_method(compression)
    if output is None:
        handle = tempfile.NamedTemporaryFile(suffix=".cytome", delete=False)
        output = handle.name
        handle.close()
        force = True  # the tempfile we just created is ours to overwrite

    # Route the var entity table + id column via the modality registry
    # (single source of truth in cytome.utils.modality). Supports RNA →
    # genes, GA → GA_genes, ATAC → peaks, tiles → tiles.
    var_entity, id_col = _modality_var_entity(modality)
    from cytome.core.dataset import _primary_key_for_table
    obs, obs_index = _prepare_obs(adata.obs)
    var, var_index = _prepare_var(adata.var, adata.var_names, modality, id_col,
                                  _primary_key_for_table(var_entity))

    def _csr(m):
        return m if sp.issparse(m) else sp.csr_matrix(np.asarray(m))

    matrix = _csr(adata.X)
    names = _MatrixNames(_same_matrix)
    plan: list[tuple[str, Any, str | None]] = []       # (cytome name, matrix, col_entity)
    layer_aliases: dict[str, str] = {}

    # `counts_layer` asserts where the raw counts are. Verify the assertion --
    # storing a normalized matrix under `{modality}_counts` is the whole bug
    # this release exists to close, and a caller who names the wrong layer
    # would reintroduce it by hand.
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"counts_layer={counts_layer!r} is not in adata.layers; "
                # sorted() over the raw keys raised TypeError on anndata
                # >= 0.13, which lists X under a None key -- so the error
                # explaining a bad counts_layer= died before it could print.
                f"available: {sorted(k for k in adata.layers if k is not None)}")
        _cl = _csr(adata.layers[counts_layer])
        if not _values_are_integer(_cl):
            raise ValueError(
                f"counts_layer={counts_layer!r} does not hold integer values, "
                f"so it is not raw counts. cytome stores raw counts as "
                f"'{modality}_counts' and every downstream default reads that "
                f"name as counts. Pass the layer that holds UMIs, or omit "
                f"counts_layer and name the matrix with main_layer_name=.")
        names.claim(f"{modality}_counts", f"counts_layer={counts_layer!r}", _cl)
        plan.append((f"{modality}_counts", _cl, None))

    x_matrix_name = _resolve_main_matrix_name(
        modality, _values_are_integer(matrix), counts_layer, main_layer_name)
    if names.claim(x_matrix_name, "adata.X", matrix):
        plan.append((x_matrix_name, matrix, None))

    # A layer promoted to `{modality}_counts` is not written again under its own
    # name: it is the same matrix (RNA_count beside RNA_counts, one letter apart).
    skip_layers_set = set(skip_layers or [])
    if counts_layer is not None:
        skip_layers_set.add(counts_layer)
    skip_obsm_set = set(skip_obsm or [])
    skip_obsp_set = set(skip_obsp or [])
    skip_varm_set = set(skip_varm or [])
    skip_varp_set = set(skip_varp or [])

    layer_map: dict[str, str] = {}
    if write_layers:
        for layer_name, layer_mat in adata.layers.items():
            # anndata >= 0.13 lists X under a `None` key in `.layers`; 0.10 did
            # not. Formatted into a name it becomes `{modality}_None`.
            if layer_name is None or layer_name in skip_layers_set:
                continue
            key = _metadata_name(str(layer_name))
            cyt_name = f"{modality}_{key}"
            layer = _csr(layer_mat)
            if names.claim(cyt_name, f"layers[{key!r}]", layer):
                plan.append((cyt_name, layer, None))
                layer_map[cyt_name] = key
            else:
                layer_aliases[key] = cyt_name

    obsm_map: dict[str, str] = {}
    obsm_as_matrix: dict[str, str] = {}
    obsm_dense: list[tuple[str, np.ndarray]] = []
    for key, emb in (adata.obsm.items() if write_obsm else ()):
        if key in skip_obsm_set:
            continue
        key = _metadata_name(str(key))
        cyt_key = _embedding_name(modality, key,
                                   existing=set(obsm_map) | set(obsm_as_matrix))
        if sp.issparse(emb):
            if emb.shape[1] > 500:
                # Large sparse obsm (e.g. gene activity) — store as matrix
                mat = emb.tocsr()
                if names.claim(cyt_key, f"obsm[{key!r}]", mat):
                    plan.append((cyt_key, mat, _OBSM_ENTITY))
                obsm_as_matrix[cyt_key] = key
                continue
            emb = emb.toarray()
        elif isinstance(emb, pd.DataFrame):
            if not all(pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)
                       for t in emb.dtypes):
                warnings.warn(
                    f"obsm[{key!r}] is a DataFrame with non-numeric columns "
                    f"({sorted({str(t) for t in emb.dtypes})}); an embedding holds numbers, "
                    f"so it is not stored (its columns belong in obs).", UserWarning, stacklevel=2)
                continue
            emb = emb.to_numpy(dtype=np.float64)
        else:
            emb = np.asarray(emb)
        obsm_dense.append((cyt_key, emb))
        obsm_map[cyt_key] = key

    raw_name = None
    if write_raw and adata.raw is not None:
        raw_name = f"{modality}_raw_X"
        raw_matrix = _csr(adata.raw.X)
        if names.claim(raw_name, "adata.raw.X", raw_matrix):
            plan.append((raw_name, raw_matrix, _RAW_ENTITY))

    # Everything is decided: only now is the store created, so a refused
    # conversion leaves no file behind. It is built as <output>.partial and
    # renamed when whole: a conversion killed half way left a store that opened
    # and passed validate().
    from cytome.io._atomic import building
    with building(output, force) as partial:
        ds = CytomeDataset(partial, mode="w", force=True)
        try:
            ds.set_entity("cells", obs)
            _store_column_meta(ds._conn, "cells", obs)
            ds.set_entity(var_entity, var)
            _store_column_meta(ds._conn, var_entity, var)
            for cyt_name, mat, col_entity in plan:
                ds.add_matrix(cyt_name, mat, compression=compression, col_entity=col_entity)
            for cyt_key, emb in obsm_dense:
                ds.add_embedding(cyt_key, emb, flush=False)

            varm_map: dict[str, str] = {}
            for key, emb in (adata.varm.items() if write_varm else ()):
                if key in skip_varm_set:
                    continue
                key = _metadata_name(str(key))
                cyt_key = f"{modality}_varm_{key}"
                if sp.issparse(emb):
                    emb = emb.toarray()
                elif isinstance(emb, pd.DataFrame):
                    emb = emb.to_numpy(dtype=np.float64)
                else:
                    emb = np.asarray(emb)
                ds.add_var_embedding(cyt_key, emb, entity=var_entity, flush=False)
                varm_map[cyt_key] = key

            obsp_map: dict[str, str] = {}
            for key, graph in (adata.obsp.items() if write_obsp else ()):
                if key in skip_obsp_set:
                    continue
                key = _metadata_name(str(key))
                cyt_key = f"{modality}_obsp_{key}"
                ds.add_graph(cyt_key, graph, axis="obs", entity_table="cells")
                obsp_map[cyt_key] = key

            varp_map: dict[str, str] = {}
            for key, graph in (adata.varp.items() if write_varp else ()):
                if key in skip_varp_set:
                    continue
                key = _metadata_name(str(key))
                cyt_key = f"{modality}_varp_{key}"
                ds.add_var_graph(cyt_key, graph, entity_table=var_entity)
                varp_map[cyt_key] = key

            if raw_name is not None:
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
            _record_names(ds, modality, obs_index, var_index, layer_aliases)

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
                    ds.metadata[_metadata_name(key)] = _metadata_value(value)
                except TypeError:
                    continue

            ds.flush()
            ds.provenance.log(
                operation="conversion",
                function_name="cytome.from_anndata",
                parameters={"modality": modality, "chunk_size": chunk_size,
                            "compression": compression},
                package_name="cytome",
                package_version=_cytome_version(),
                input_objects=["anndata"],
                output_objects=[x_matrix_name],
            )
            ds.flush()
        except BaseException:
            try:
                ds.close()
            except Exception:
                pass
            raise
        ds.close()
    return CytomeDataset(output, mode="r+")


def _cytome_version() -> str:
    try:
        from cytome import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _record_names(ds, modality, obs_index, var_index, layer_aliases) -> None:
    """Record where obs_names and var_names are kept, and the layers stored as
    another name (one per modality where it is a modality's), for to_anndata."""
    if obs_index is not None:
        ds.metadata["_anndata_obs_index"] = obs_index
    if var_index is not None:
        recorded = dict(_metadata_get(ds, "_anndata_var_index", default={}) or {})
        recorded[modality] = var_index
        ds.metadata["_anndata_var_index"] = recorded
    if layer_aliases:
        recorded = dict(_metadata_get(ds, "_anndata_layer_aliases", default={}) or {})
        recorded.update({f"{modality}:{k}": v for k, v in layer_aliases.items()})
        ds.metadata["_anndata_layer_aliases"] = recorded


def _h5_values_are_integer(grp, n_probe: int = 20000) -> bool:
    """Whether an h5ad matrix group holds integer values.

    Reads at most ``n_probe`` stored values with an h5py slice, so the streaming
    path can hold the same naming invariant as the in-memory one without giving
    up streaming to do it. For a dense 2-D dataset that is ``n_probe`` values,
    not rows: 20,000 full rows of a dense X were read before.
    """
    try:
        data = grp["data"] if hasattr(grp, "keys") and "data" in grp else grp
        if getattr(data, "ndim", 1) == 2:
            n_cols = max(1, int(data.shape[1]))
            a = np.asarray(data[:max(1, n_probe // n_cols)], dtype="float64").ravel()
        else:
            a = np.asarray(data[:n_probe], dtype="float64")
        if a.size == 0:
            return True
        a = a[np.isfinite(a)]
        return bool(a.size == 0 or np.allclose(a, np.round(a)))
    except Exception:
        return True          # cannot tell -> keep the historical name


def _h5_same_csr(a, b, chunk: int = 1 << 22) -> bool:
    """Whether two CSR groups of an h5ad file hold the same matrix, read a bounded
    slice at a time (only when two sources would be stored under one name)."""
    try:
        if tuple(a.attrs["shape"]) != tuple(b.attrs["shape"]):
            return False
        if a.name == b.name:
            return True
        if not np.array_equal(a["indptr"][:], b["indptr"][:]):
            return False
        n = int(a["data"].shape[0])
        if n != int(b["data"].shape[0]):
            return False
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            if not (np.array_equal(a["indices"][lo:hi], b["indices"][lo:hi])
                    and np.array_equal(a["data"][lo:hi], b["data"][lo:hi])):
                return False
        return True
    except Exception:
        return False


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
    counts_layer, main_layer_name
        As in :func:`from_anndata`, on both paths: ``counts_layer`` is checked to
        hold integers and stored as ``{modality}_counts`` (both paths stored it
        under its own name, unchecked), ``main_layer_name`` names X.
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

    Both paths write the same store: the cells and feature tables are prepared
    by one routine, and names are checked the same way (two sources of one
    matrix name are refused unless they hold the same matrix).

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
            counts_layer=counts_layer, main_layer_name=main_layer_name,
            compression=compression,
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
    from ._optional import require_h5py
    from cytome.core.dataset import _primary_key_for_table
    from cytome.io.compression import _resolve_method
    _resolve_method(compression)
    h5py = require_h5py("from_h5ad(backed=True)")
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
    if counts_layer is not None:
        skip_layers_set.add(counts_layer)
    skip_obsm_set = set(skip_obsm or [])
    skip_obsp_set = set(skip_obsp or [])
    skip_varm_set = set(skip_varm or [])
    skip_varp_set = set(skip_varp or [])

    # Round 11 (2026-05-26): route via the modality registry. Same
    # change as the in-memory `from_anndata` path above — supports
    # GA → GA_genes and tiles → tiles, not just ATAC vs other.
    var_entity, id_col = _modality_var_entity(modality)

    from cytome.io._atomic import building
    opened: list = []
    with building(output, force) as partial:
        try:
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
                        counts_layer=counts_layer, main_layer_name=main_layer_name,
                        h5ad_path=h5ad_path, output=output,
                    )

                # --- 0. the matrices and their names, decided before anything is written ---
                if counts_layer is not None:
                    if "layers" not in f or counts_layer not in f["layers"]:
                        avail = sorted(f["layers"].keys()) if "layers" in f else []
                        raise KeyError(f"counts_layer={counts_layer!r} is not in the file's layers; "
                                       f"available: {avail}")
                    if not _h5_values_are_integer(f["layers"][counts_layer]):
                        raise ValueError(
                            f"counts_layer={counts_layer!r} does not hold integer values, "
                            f"so it is not raw counts. cytome stores raw counts as "
                            f"'{modality}_counts' and every downstream default reads that "
                            f"name as counts. Pass the layer that holds UMIs, or omit "
                            f"counts_layer and name the matrix with main_layer_name=.")
                x_matrix_name = _resolve_main_matrix_name(
                    modality, _h5_values_are_integer(f["X"]),
                    counts_layer, main_layer_name, stacklevel=4)
                names = _MatrixNames(_h5_same_csr)
                streams: list[tuple[str, Any, str]] = []          # (name, h5 group, col_entity)
                layer_map: dict[str, str] = {}
                layer_aliases: dict[str, str] = {}
                if counts_layer is not None:
                    names.claim(f"{modality}_counts", f"counts_layer={counts_layer!r}",
                                f["layers"][counts_layer])
                    streams.append((f"{modality}_counts", f["layers"][counts_layer], var_entity))
                if names.claim(x_matrix_name, "X", f["X"]):
                    streams.append((x_matrix_name, f["X"], var_entity))
                if write_layers and "layers" in f:
                    for layer_name in f["layers"].keys():
                        if layer_name in skip_layers_set:
                            if verbose:
                                print(f"  [skip] layers/{layer_name}")
                            continue
                        key = _metadata_name(str(layer_name))
                        cyt_name = f"{modality}_{key}"
                        grp = f[f"layers/{layer_name}"]
                        if names.claim(cyt_name, f"layers[{key!r}]", grp):
                            streams.append((cyt_name, grp, var_entity))
                            layer_map[cyt_name] = key
                        else:
                            layer_aliases[key] = cyt_name
                raw_name = None
                if write_raw and "raw" in f and "X" in f["raw"]:
                    raw_name = f"{modality}_raw_X"
                    if names.claim(raw_name, "raw.X", f["raw/X"]):
                        streams.append((raw_name, f["raw/X"], _RAW_ENTITY))

                # --- 1. obs / var via read_elem (no pyarrow; pure pandas) ---
                obs, obs_index = _prepare_obs(read_elem(f["obs"]))
                var_raw = read_elem(f["var"])
                var, var_index = _prepare_var(var_raw, var_raw.index, modality, id_col,
                                              _primary_key_for_table(var_entity))
                del var_raw

                ds = CytomeDataset(partial, mode="w", force=True)
                opened.append(ds)
                ds.set_entity("cells", obs)
                _store_column_meta(ds._conn, "cells", obs)
                n_obs = len(obs)
                del obs
                ds.set_entity(var_entity, var)
                _store_column_meta(ds._conn, var_entity, var)
                n_vars = len(var)
                del var
                gc.collect()
                ds.flush()

                if verbose:
                    print(f"[from_h5ad backed] obs/var written ({n_obs:,} cells × "
                          f"{n_vars:,} {var_entity}). Streaming matrices...")

                # --- 2. X, the counts layer, layers, raw.X: streamed ---
                for cyt_name, grp, col_entity in streams:
                    _stream_h5_csr_to_writer(
                        ds, grp, matrix_name=cyt_name,
                        chunk_size=chunk_size, storage_chunk_size=storage_chunk_size,
                        compression=compression, row_entity="cells",
                        col_entity=col_entity, verbose=verbose,
                    )
                ds.metadata["_anndata_X_layer"] = x_matrix_name
                ds.metadata["_anndata_layer_map"] = layer_map

                if raw_name is not None:
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
                        key = _metadata_name(str(k))
                        cyt_key = _embedding_name(
                            modality, key, existing=set(obsm_map) | set(obsm_as_matrix))
                        if sp.issparse(arr):
                            if arr.shape[1] > 500:
                                mat = arr.tocsr()
                                if names.claim(cyt_key, f"obsm[{key!r}]", mat):
                                    ds.add_matrix(cyt_key, mat, compression=compression,
                                                  col_entity=_OBSM_ENTITY)
                                obsm_as_matrix[cyt_key] = key
                                del arr, mat
                                gc.collect()
                                continue
                            arr = arr.toarray()
                        elif isinstance(arr, pd.DataFrame):
                            if not all(pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)
                                       for t in arr.dtypes):
                                warnings.warn(
                                    f"obsm[{key!r}] is a DataFrame with non-numeric columns; an "
                                    f"embedding holds numbers, so it is not stored (its columns "
                                    f"belong in obs).", UserWarning, stacklevel=2)
                                continue
                            arr = arr.to_numpy(dtype=np.float64)
                        ds.add_embedding(cyt_key, np.asarray(arr))
                        obsm_map[cyt_key] = key
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
                        key = _metadata_name(str(k))
                        cyt_key = f"{modality}_varm_{key}"
                        if sp.issparse(arr):
                            arr = arr.toarray()
                        elif isinstance(arr, pd.DataFrame):
                            arr = arr.to_numpy(dtype=np.float64)
                        ds.add_var_embedding(cyt_key, np.asarray(arr), entity=var_entity)
                        varm_map[cyt_key] = key
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
                        key = _metadata_name(str(k))
                        cyt_key = f"{modality}_obsp_{key}"
                        ds.add_graph(cyt_key, graph, axis="obs", entity_table="cells")
                        obsp_map[cyt_key] = key
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
                        key = _metadata_name(str(k))
                        cyt_key = f"{modality}_varp_{key}"
                        ds.add_var_graph(cyt_key, graph, entity_table=var_entity)
                        varp_map[cyt_key] = key
                        del graph
                        gc.collect()
                ds.metadata["_anndata_varp_map"] = varp_map
                _record_names(ds, modality, obs_index, var_index, layer_aliases)

                # --- 9. uns — per-key, skip unpickleable ---
                if write_uns and "uns" in f:
                    # Read one key at a time. `read_elem(f["uns"])` materialises every
                    # key at once, which for a Visium object means a full-resolution
                    # tissue image in memory before anything decides where it goes.
                    uns_grp = f["uns"]
                    for key in list(uns_grp.keys()):
                        try:
                            value = read_elem(uns_grp[key])
                        except Exception as exc:      # pragma: no cover - exotic uns
                            if verbose:
                                print(f"  [skip] uns/{key} (unreadable: {exc})")
                            continue
                        if key == "spatial" and isinstance(value, dict):
                            # Same route as the in-memory path: images become
                            # compressed blobs in the spatial tables. Without this the
                            # backed path serialised the image as JSON text, which is
                            # ~20 bytes per float32 element and overflows sqlite3's
                            # 2 GB bind limit on any full-resolution image.
                            _write_uns_spatial(ds, value)
                            del value
                            gc.collect()
                            continue
                        try:
                            ds.metadata[_metadata_name(key)] = _metadata_value(value)
                        except (TypeError, ValueError, OverflowError) as exc:
                            if verbose:
                                print(f"  [skip] uns/{key} ({exc})")
                        del value
                    gc.collect()

            ds.flush()
            ds.provenance.log(
                operation="conversion",
                function_name="cytome.from_h5ad",
                parameters={
                    "modality": modality,
                    "backed": True,
                    "chunk_size": chunk_size,
                    "storage_chunk_size": storage_chunk_size,
                    "compression": compression,
                    "counts_layer": counts_layer,
                    "main_layer_name": main_layer_name,
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
                package_version=_cytome_version(),
                input_objects=[str(h5ad_path)],
                output_objects=[x_matrix_name],
            )
            ds.flush()
        except BaseException:
            for _ds in opened:
                try:
                    _ds.close()
                except Exception:
                    pass
            raise
        ds.close()
    if verbose:
        print(f"[from_h5ad backed] Done in {time.time()-t_open:.1f}s. "
              f"Output: {output}")
    return CytomeDataset(output, mode="r+")


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


def _h5ad_name(name):
    """``name`` as a key in an h5ad file: "/" written as "%2F"."""
    return name.replace("/", "%2F") if isinstance(name, str) and "/" in name else name


def _metadata_name(name):
    """A key read from an h5ad file as the metadata key it stands for: "%2F" back to "/"."""
    return name.replace("%2F", "/") if isinstance(name, str) and "%2F" in name else name


#: The colour :func:`to_anndata` gives a category kept from outside a column's
#: recorded categories, in ``uns['<column>_colors']``: a neutral grey, darker
#: than the light grey that scanpy and PIASO draw missing labels in.
KEPT_CATEGORY_COLOR = "#808080"


def _colour_kept_categories(conn, adata) -> None:
    """Extend each stored colour list of ``obs`` by the categories kept outside it.

    A value outside a column's recorded categories is kept, as a category after
    the recorded ones (:func:`cytome.core.column_meta.restore_categorical`). A
    colour list stored for the column -- scanpy's ``uns['<column>_colors']``,
    one colour per category, in order -- then has fewer colours than the column
    has categories, and scanpy and PIASO replace every colour with their default
    palette. Each kept category gets :data:`KEPT_CATEGORY_COLOR` instead, so the
    recorded categories keep theirs. A list of any other length is left as it
    is.
    """
    from cytome.core.column_meta import read_records

    for col, dtype, cats_json in read_records(conn, "cells"):
        if dtype not in ("categorical", "ordered_categorical") or col not in adata.obs.columns:
            continue
        key = _h5ad_name(f"{col}_colors")
        stored = adata.uns.get(key)
        values = adata.obs[col]
        if stored is None or not isinstance(values.dtype, pd.CategoricalDtype):
            continue
        try:
            n_recorded = len(json.loads(cats_json)) if cats_json else 0
        except ValueError:
            continue
        n_now = len(values.cat.categories)
        if isinstance(stored, (list, tuple, np.ndarray)) and len(stored) == n_recorded < n_now:
            extended = list(stored) + [KEPT_CATEGORY_COLOR] * (n_now - n_recorded)
            adata.uns[key] = (np.asarray(extended, dtype=object) if isinstance(stored, np.ndarray)
                              else extended)


def _uns_value(value):
    """A metadata value as it goes into ``uns``: no "/" in a mapping's keys or a table's columns.

    An h5ad (or zarr) file stores each key of a mapping and each column of a table as a name
    in the file, where "/" separates groups. anndata 0.13 refuses such a name by default
    (``anndata.settings.disallow_forward_slash_in_h5ad``), and earlier versions split it into
    nested groups, so a key "L2/3 IT" (a cell type, in the marker tables PIASO records)
    came back as "L2". Such names are written with "/" as "%2F", and :func:`from_anndata`
    turns "%2F" back into "/". A value with no such name is returned as it is.
    """
    if isinstance(value, dict):
        return _renamed(value, _h5ad_name, _uns_value)
    if isinstance(value, pd.DataFrame) and any(isinstance(c, str) and "/" in c for c in value.columns):
        return value.rename(columns=_h5ad_name)
    return value


def _metadata_value(value):
    """The reverse of :func:`_uns_value`, for a value read from ``uns``."""
    if isinstance(value, dict):
        return _renamed(value, _metadata_name, _metadata_value)
    if isinstance(value, pd.DataFrame) and any(isinstance(c, str) and "%2F" in c for c in value.columns):
        return value.rename(columns=_metadata_name)
    return value


def _renamed(mapping, rename, recurse):
    """``mapping`` with its keys renamed and its values converted, refusing two keys made one."""
    out = {}
    for k, v in mapping.items():
        nk = rename(k)
        if nk in out:
            raise ValueError(f"the keys {k!r} and {next(o for o in mapping if rename(o) == nk)!r} "
                             f"would both be stored as {nk!r} in an h5ad file; rename one of them.")
        out[nk] = recurse(v)
    return out


def _writable_frame(df):
    """Object columns with missing values become categoricals.

    A cytome column that is NULL for some or all cells comes back as an
    object column holding None, and anndata's h5ad writer cannot serialise
    that -- ``TypeError: Can't implicitly convert non-string objects to
    strings``, raised while writing a key the user never touched. Nullable
    ``string`` fares no better (no registered writer). ``Categorical`` is the
    representation anndata does write, and it is what scanpy would have used
    for these columns anyway, so `to_anndata(...)` output can be written to
    disk whether or not the source had missing labels.
    """
    if df is None or df.empty:
        return df
    out = None
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        if not series.isna().any():
            continue
        if out is None:
            out = df.copy()
        out[col] = pd.Categorical(series)
    return df if out is None else out


def to_anndata(
    ds: CytomeDataset,
    modality: str = "RNA",
    layer: str | None = None,
    include_embeddings: bool = True,
    include_cross_modality_embeddings: bool = True,
    cell_mask=None,
    layers=None,
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
        yielding significant RAM and time savings for partial exports. The
        graphs (obsp) and ``raw`` are cut to the same cells.
    layers
        Which other matrices of the modality to load as layers: ``None``
        (default) all of them, a list of names (the AnnData layer name or the
        store's, ``"infog"`` or ``"RNA_infog"``), or ``[]`` for none. A PIASO
        store holds four or five full RNA matrices; reading only X saves
        reading the others.

    obs_names and var_names are the ones the AnnData had when it was
    converted (recorded by :func:`from_anndata`); otherwise the ``barcode``
    and the feature id. The store's own row keys (``cell_idx``,
    ``gene_idx``) are not exported: they are positions, wrong after any
    subset of the AnnData.

    A categorical column comes back with its recorded categories, in their
    order. A value outside them (the column was written since) is kept, as a
    category after the recorded ones, with a warning. A colour list stored for
    the column (``uns['<column>_colors']``, one colour per recorded category)
    is extended with :data:`KEPT_CATEGORY_COLOR`, a neutral grey, for each
    such category: the recorded categories keep their colours, where scanpy
    and PIASO would otherwise replace them all with their default palette.
    """
    try:
        import anndata
    except ImportError as exc:  # pragma: no cover
        raise ImportError("anndata is required for to_anndata") from exc

    from cytome.io.chunked_io import read_dense_rows
    from cytome.core.dataset import _primary_key_for_table

    # Validate modality up front (before any I/O) so an unknown modality
    # produces a clean ValueError rather than a deeper KeyError on the
    # matrix or entity lookup.
    var_entity, id_col = _modality_var_entity(modality)

    # Resolve cell_mask to sorted integer indices (or None for full export)
    keep_idx = None
    if cell_mask is not None:
        arr = np.asarray(cell_mask)
        if arr.dtype == bool:
            if arr.shape[0] != ds.n_cells:
                raise ValueError(f"to_anndata: a boolean cell_mask of {arr.shape[0]} entries "
                                 f"for {ds.n_cells} cells")
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
    obs = obs.drop(columns=[c for c in obs.columns if str(c).lower() == "cell_idx"])
    if keep_idx is not None:
        obs = obs.iloc[keep_idx].copy().reset_index(drop=True)
    obs_index = _metadata_get(ds, "_anndata_obs_index", default=None)
    if obs_index and obs_index in obs.columns:
        obs = obs.set_index(obs_index, drop=True)
        obs.index.name = None
    elif "barcode" in obs.columns:
        obs = obs.set_index("barcode", drop=False)
    obs.index = obs.index.astype(str)

    # Build an EntityTable directly off the connection so we don't depend on
    # per-table @property accessors existing for every modality (e.g. tiles
    # has no ds.tiles accessor — `ds.tiles` would route through __getattr__
    # and return a Modality instead of an EntityTable). var_entity / id_col
    # were resolved up front from the modality registry.
    from cytome.core.entity import EntityTable
    var_tbl = EntityTable(ds._conn, var_entity)
    var = _restore_column_dtypes(ds._conn, var_entity, var_tbl.to_pandas())
    key = _primary_key_for_table(var_entity)
    var = var.drop(columns=[c for c in var.columns if str(c).lower() == key.lower()])
    var_index = (_metadata_get(ds, "_anndata_var_index", default={}) or {}).get(modality)
    if var_index and var_index in var.columns:
        var = var.set_index(var_index, drop=(var_index == _VAR_NAME_COL))
        var.index.name = None
    elif id_col in var.columns:
        var = var.set_index(id_col, drop=False)
    var.index = var.index.astype(str)

    obs = _writable_frame(obs)
    var = _writable_frame(var)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Collect names of obsm entries stored as matrices so they are not
    # mistaken for regular layers during fallback discovery.
    _obsm_matrix_names = set()
    obsm_as_matrix_meta = _metadata_get(ds, "_anndata_obsm_as_matrix", default={})
    if isinstance(obsm_as_matrix_meta, dict):
        _obsm_matrix_names = set(obsm_as_matrix_meta.keys())
    raw_info = _metadata_get(ds, "_anndata_raw", default=None)
    raw_matrix_name = raw_info.get("matrix_name") if isinstance(raw_info, dict) else None

    # Layers are feature-dimensioned, so they MUST match the requested modality.
    # Restrict discovery to `{modality}_%` matrices (the global `_anndata_layer_map`
    # is written for whichever modality was imported first and would otherwise
    # attach e.g. RNA_infog to a GA AnnData → column-count mismatch). Honor the
    # recorded adata-layer name when the map has one for this modality, and always
    # guard on the feature count. The raw matrix is adata.raw, not a layer.
    prefix = f"{modality}_"
    layer_map = _metadata_get(ds, "_anndata_layer_map", default={})
    layer_map = layer_map if isinstance(layer_map, dict) else {}
    wanted = None if layers is None else {str(x) for x in layers}

    def _wanted(cytome_name, adata_name):
        return wanted is None or cytome_name in wanted or adata_name in wanted

    rows = ds._conn.execute(
        "SELECT matrix_name FROM matrix_meta WHERE matrix_name LIKE ? ORDER BY matrix_name",
        (f"{prefix}%",),
    ).fetchall()
    for (name,) in rows:
        if name == x_matrix_name or name in _obsm_matrix_names or name == raw_matrix_name:
            continue
        adata_name = layer_map.get(name, name[len(prefix):])
        if not _wanted(name, adata_name):
            continue
        lyr = MeasurementLayer(ds._conn, name)
        # Skip layers with mismatched column count (compact HVG layers, etc.)
        if lyr.shape[1] != adata.shape[1]:
            continue
        adata.layers[_h5ad_name(adata_name)] = (lyr.rows(keep_idx) if keep_idx is not None
                                                else lyr.to_memory())
    # layers that were the same matrix as another, stored once
    aliases = _metadata_get(ds, "_anndata_layer_aliases", default={}) or {}
    for alias_key, cytome_name in aliases.items():
        mod, _, alias = str(alias_key).partition(":")
        if mod != modality or not _wanted(cytome_name, alias) or alias in adata.layers:
            continue
        if cytome_name == x_matrix_name:
            adata.layers[_h5ad_name(alias)] = adata.X.copy()
        elif _h5ad_name(layer_map.get(cytome_name, cytome_name[len(prefix):])) in adata.layers:
            adata.layers[_h5ad_name(alias)] = adata.layers[
                _h5ad_name(layer_map.get(cytome_name, cytome_name[len(prefix):]))].copy()

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
                adata.obsm[_h5ad_name(adata_key)] = _read_emb(cytome_key)
                added_keys.add(cytome_key)
        for name in ds.embeddings.keys():
            if name in added_keys or "_TMP_" in name:
                continue
            if name.startswith(prefix):
                adata.obsm[_h5ad_name(f"X_{name[len(prefix):]}")] = _read_emb(name)
                added_keys.add(name)

        # (2) OWN-modality sparse obsm stored as matrices (feature-ish; gated).
        obsm_as_matrix = _metadata_get(ds, "_anndata_obsm_as_matrix", default={})
        if isinstance(obsm_as_matrix, dict):
            for cytome_key, adata_key in obsm_as_matrix.items():
                if str(cytome_key).startswith(prefix) and _matrix_exists(ds._conn, cytome_key):
                    ml = MeasurementLayer(ds._conn, cytome_key)
                    adata.obsm[_h5ad_name(adata_key)] = (ml.rows(keep_idx) if keep_idx is not None
                                                         else ml.to_memory())
                    added_keys.add(cytome_key)

        # (3) CROSS-modality cell embeddings (other modalities). These are
        #     shared cell coordinates (n_obs rows), so they attach cleanly to any
        #     modality's AnnData. Named by their cytome key (`_obsm_`→`_`) to keep
        #     provenance; skipped if that collides with an own-modality obsm key.
        if include_cross_modality_embeddings:
            for name in ds.embeddings.keys():
                if name in added_keys or "_TMP_" in name:
                    continue
                disp = _h5ad_name(str(name).replace("_obsm_", "_"))
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
                        adata.varm[_h5ad_name(adata_key)] = ve

    def _cut(graph):
        """A cells × cells graph cut to the kept cells (rows and columns)."""
        if keep_idx is None:
            return graph
        return graph.tocsr()[keep_idx][:, keep_idx]

    obsp_map = _metadata_get(ds, "_anndata_obsp_map", default={})
    if isinstance(obsp_map, dict) and obsp_map:
        for cytome_key, adata_key in obsp_map.items():
            try:
                adata.obsp[_h5ad_name(adata_key)] = _cut(ds.graphs[cytome_key].to_sparse())
            except KeyError:
                continue
    else:
        try:
            adata.obsp["connectivities"] = _cut(ds.graphs["knn"].to_sparse())
        except Exception:
            pass

    varp_map = _metadata_get(ds, "_anndata_varp_map", default={})
    if isinstance(varp_map, dict):
        for cytome_key, adata_key in varp_map.items():
            try:
                adata.varp[_h5ad_name(adata_key)] = ds.var_graphs[cytome_key].to_sparse()
            except KeyError:
                continue

    for key, value in ds.metadata.items():
        if key.startswith("_anndata_"):
            continue
        adata.uns[_h5ad_name(key)] = _uns_value(value)   # no "/" in h5ad names (_uns_value)
    _colour_kept_categories(ds._conn, adata)

    # Materialised, not lazy: `to_anndata` hands back a *detached* object that
    # people write to disk, and anndata's h5ad writer type-checks the values in
    # `uns` -- a proxy there fails with "Can't implicitly convert non-string
    # objects to strings", a long way from the conversion that put it there.
    # Laziness belongs on `ds.spatial_images.as_uns()`, where the dataset is
    # open and the caller is exploring rather than serialising.
    _spatial_uns = (ds.spatial_images.as_uns(lazy=False)
                    if hasattr(ds, "spatial_images") else {})
    if _spatial_uns:
        adata.uns["spatial"] = _spatial_uns

    if raw_matrix_name and _matrix_exists(ds._conn, raw_matrix_name):
        raw_ml = MeasurementLayer(ds._conn, raw_matrix_name)
        raw_X = raw_ml.rows(keep_idx) if keep_idx is not None else raw_ml.to_memory()
        raw_var = _read_raw_var_table(ds._conn)
        adata.raw = anndata.AnnData(X=raw_X, obs=pd.DataFrame(index=adata.obs_names),
                                    var=raw_var)

    return adata


def update_from_anndata(ds: CytomeDataset, adata, modality: str = "RNA") -> None:
    """Sync selected AnnData changes back into a Cytome dataset.

    The AnnData must have the store's cells, in its order (``to_anndata``
    of the store, with columns, embeddings or layers added): the cells table
    is replaced by its obs, so a different number of cells, or other cells,
    is refused.
    """
    if adata.n_obs != ds.n_cells:
        raise ValueError(f"update_from_anndata: the AnnData has {adata.n_obs} cells and the "
                         f"store {ds.n_cells}; it must hold the store's cells, in its order")
    obs, obs_index = _prepare_obs(adata.obs)
    stored = np.asarray(ds.cells["barcode"]).astype(str)
    if not np.array_equal(obs["barcode"].astype(str).to_numpy(), stored):
        raise ValueError("update_from_anndata: the AnnData's barcodes are not the store's, "
                         "in the store's order")
    ds.set_entity("cells", obs)
    _store_column_meta(ds._conn, "cells", obs)
    if obs_index is not None:
        ds.metadata["_anndata_obs_index"] = obs_index

    _taken: set = set()
    for key, emb in adata.obsm.items():
        key = _metadata_name(str(key))
        cyt_key = _embedding_name(modality, key, existing=_taken)
        _taken.add(cyt_key)
        if sp.issparse(emb):
            emb = emb.toarray()
        ds.add_embedding(cyt_key, emb, flush=False)
    for key, layer in adata.layers.items():
        if key is None:                     # anndata >= 0.13 lists X here
            continue
        mat = layer if sp.issparse(layer) else sp.csr_matrix(np.asarray(layer))
        ds.add_matrix(f"{modality}_{_metadata_name(str(key))}", mat)

    ds.flush()


def _store_column_meta(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    """Record dtype metadata for DataFrame columns (see :mod:`cytome.core.column_meta`)."""
    from cytome.core.column_meta import store_column_meta
    store_column_meta(conn, table_name, df)


def _restore_column_dtypes(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Restore pandas dtype metadata for a DataFrame (see :mod:`cytome.core.column_meta`)."""
    from cytome.core.column_meta import restore_column_dtypes
    return restore_column_dtypes(conn, table_name, df)


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
    from cytome.core.column_meta import sql_type_of
    return sql_type_of(series)


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
    # the dataset's: a missing value (NaN, pd.NA, NaT) binds as NULL
    from cytome.core.dataset import _py_scalar as _scalar
    return _scalar(value)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'
