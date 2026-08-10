"""Modality registry + per-modality helpers for cytome.

Centralises the modality → (var-entity, id-column, candidate-name-columns)
mapping that previously lived inline in three different places (cytome's
``to_anndata``, piaso's ``_resolve_cytome_feature_values``, and COSG's
``_get_feature_table_info``). Extending the registry once here gives the
new modality to every downstream consumer.

This module has **no piaso / cosg dependency**. Piaso-specific pieces
(INFOG / TF-IDF / log1p chunk normalizers) stay in piaso and are
lazy-imported by COSG only when on-the-fly normalization is requested.
"""
from __future__ import annotations

import sqlite3
from typing import Tuple, List, Optional

import numpy as np


# Single source of truth for modality → entity-table routing.
# Order matters for callers that auto-detect which modality holds a feature
# (e.g. piaso plotEmbedding's resolver iterates this list to pick a unique
# modality). Open for proteins/spatial later.
MODALITY_REGISTRY: list = [
    # (modality_name,  entity_table, idx_col,    candidate id/name columns)
    ("RNA",   "genes",     "gene_idx", ("gene_id", "symbol", "feature_name", "gene_name")),
    ("GA",    "GA_genes",  "gene_idx", ("gene_id", "symbol", "feature_name", "gene_name")),
    ("ATAC",  "peaks",     "peak_idx", ("peak_id", "feature_name")),
    ("tiles", "tiles",     "tile_idx", ("tile_id", "feature_name")),
]

# Backward-compat shorthand mapping just modality → (entity_table, id_col).
# Equivalent to the dict that lived in cytome/io/convert_anndata.py.
MODALITY_VAR_ENTITY = {
    name: (entity, id_col.replace("_idx", "_id"))
    for name, entity, id_col, _ in MODALITY_REGISTRY
}


def modality_var_entity(modality: str) -> Tuple[str, str]:
    """Return (var_entity_table, id_column) for ``modality``.

    Case-sensitive on the canonical 4 modalities (RNA, GA, ATAC, tiles).
    Raises ``ValueError`` for unknown modality names.
    """
    if modality.upper() in {"RNA", "GA", "ATAC"}:
        modality = modality.upper()
    if modality not in MODALITY_VAR_ENTITY:
        raise ValueError(
            f"Unknown modality '{modality}'. Known: {list(MODALITY_VAR_ENTITY)}."
        )
    return MODALITY_VAR_ENTITY[modality]


def _registry_entry(modality: str):
    """Internal — return the full ``MODALITY_REGISTRY`` row for ``modality``."""
    if modality.upper() in {"RNA", "GA", "ATAC"}:
        modality = modality.upper()
    for entry in MODALITY_REGISTRY:
        if entry[0] == modality:
            return entry
    raise ValueError(
        f"Unknown modality '{modality}'. Known: "
        f"{[e[0] for e in MODALITY_REGISTRY]}."
    )


# Human-readable display-name columns — preferred over opaque ids (``gene_id``
# = Ensembl) when they are actually populated. ``peak_id`` / ``tile_id`` ARE the
# readable name for ATAC/tiles, so they're handled by falling through.
_READABLE_NAME_COLS = frozenset(
    {"symbol", "gene_name", "feature_name", "gene_symbols", "Symbol", "gene_symbol"}
)


def _col_is_populated(ds, table: str, col: str) -> bool:
    """True if ``table.col`` has at least one non-NULL, non-empty value."""
    try:
        row = ds._conn.execute(
            f'SELECT 1 FROM {table} WHERE "{col}" IS NOT NULL '
            f'AND TRIM(CAST("{col}" AS TEXT)) != \'\' LIMIT 1'
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False


def modality_feature_table_info(
    ds, modality: str, feature_name_col: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Return ``(feature_table, idx_col, name_col)`` for a modality.

    The display ``name_col`` is resolved with this precedence:

    1. **Explicit** ``feature_name_col`` (per-call override) — used if it is an
       actual column of the feature table (else ``ValueError``).
    2. **Per-cytome manifest** ``ds.metadata['{MODALITY}_name_col']`` — a sticky
       override that travels with the file, if set and present as a column.
    3. **Smart default** — prefer a *populated* human-readable column
       (``symbol`` / ``gene_name`` / ``feature_name`` …) over opaque ids
       (``gene_id`` = Ensembl); skip readable columns that exist but are entirely
       NULL/empty (e.g. converted objects, MTG). Falls back to the first existing
       registry candidate, else the second table column.

    This single source of truth makes COSG output, dotplot/embedding labels and
    inferGRN gene matching show symbols by default on cellranger-loaded cytomes,
    while staying correct for cytomes that store names in ``gene_id``.
    """
    _, feature_table, idx_col, candidate_cols = _registry_entry(modality)
    feat_cols = [
        c[1] for c in ds._conn.execute(
            f"PRAGMA table_info({feature_table})"
        ).fetchall()
    ]

    # (1) explicit per-call override.
    if feature_name_col is not None:
        if feature_name_col not in feat_cols:
            raise ValueError(
                f"feature_name_col='{feature_name_col}' is not a column of "
                f"'{feature_table}'. Available: {feat_cols}."
            )
        return feature_table, idx_col, feature_name_col

    # (2) sticky per-cytome manifest override (cheap read; no setter machinery).
    meta = getattr(ds, "metadata", None)
    if meta is not None:
        try:
            mkey = meta.get(f"{modality.upper()}_name_col")
        except Exception:
            mkey = None
        if mkey and mkey in feat_cols:
            return feature_table, idx_col, mkey

    # (3) smart default: populated-readable first, then any candidate, then id.
    ordered = ([c for c in candidate_cols if c in _READABLE_NAME_COLS]
               + [c for c in candidate_cols if c not in _READABLE_NAME_COLS])
    name_col = next(
        (c for c in ordered if c in feat_cols and _col_is_populated(ds, feature_table, c)),
        None,
    )
    if name_col is None:                       # nothing populated — keep old behaviour
        name_col = next((c for c in candidate_cols if c in feat_cols), None)
    if name_col is None:
        name_col = feat_cols[1] if len(feat_cols) > 1 else feat_cols[0]
    return feature_table, idx_col, name_col


def modality_has_feature(ds, modality: str, feature: str) -> Optional[Tuple[int, str]]:
    """If ``feature`` exists in this modality's entity table, return
    ``(feat_idx, id_col)``; else return ``None``. Cheap — checks each
    candidate id column once per modality.
    """
    _, feature_table, idx_col, candidate_cols = _registry_entry(modality)
    try:
        feat_cols = [
            c[1] for c in ds._conn.execute(
                f"PRAGMA table_info({feature_table})"
            ).fetchall()
        ]
    except sqlite3.OperationalError:
        return None
    for c in candidate_cols:
        if c not in feat_cols:
            continue
        rows = ds._conn.execute(
            f"SELECT {idx_col} FROM {feature_table} WHERE {c} = ? LIMIT 1",
            (str(feature),),
        ).fetchone()
        if rows is not None:
            return int(rows[0]), c
    return None


def read_feature_column(
    ds, modality: str, layer: str, feat_idx: int, batch_size: int = 2048,
) -> np.ndarray:
    """Stream a single column from ``{modality}_{layer}`` into a dense
    per-cell vector. Caller verifies the matrix exists.
    """
    n = ds.n_cells
    out = np.zeros(n, dtype=np.float32)
    for chunk, idxs in ds.iter_chunks(
        modality=modality, layer=layer, batch_size=batch_size,
    ):
        col = chunk[:, feat_idx]
        if hasattr(col, "toarray"):
            col = col.toarray().ravel()
        else:
            col = np.asarray(col).ravel()
        out[idxs] = col
    return out


def read_feature_columns(
    ds, modality: str, layer: str, feat_indices, batch_size: int = 2048,
) -> np.ndarray:
    """Stream MULTIPLE columns from ``{modality}_{layer}`` in a SINGLE pass.

    Returns a dense ``(n_cells, len(feat_indices))`` array, columns in the order
    of ``feat_indices``. One streaming pass for all requested features instead of
    one pass per feature — the batched analogue of :func:`read_feature_column`.
    Caller verifies the matrix exists.
    """
    n = ds.n_cells
    fi = np.asarray(feat_indices, dtype=np.int64)
    out = np.zeros((n, fi.shape[0]), dtype=np.float32)
    for chunk, idxs in ds.iter_chunks(
        modality=modality, layer=layer, batch_size=batch_size,
    ):
        sub = chunk[:, fi]
        if hasattr(sub, "toarray"):
            sub = sub.toarray()
        out[idxs, :] = np.asarray(sub, dtype=np.float32)
    return out


def modality_cell_depth(
    ds, modality: str, use_cached_stats: bool = True, batch_size: int = 2048,
) -> np.ndarray:
    """Return per-cell sum of ``{modality}_counts`` (i.e. UMIs for RNA,
    fragment counts for ATAC, etc.).

    Cached under ``ds.metadata['{modality}_cell_depth']``. ATAC short-
    circuits to ``cells.n_fragments`` when that column is non-empty
    (equivalent and already stored by the Rust importer).
    """
    cache_key = f"{modality}_cell_depth"
    if use_cached_stats:
        cached = ds.metadata.get(cache_key)
        if cached is not None:
            arr = np.asarray(cached, dtype=np.float64)
            # Self-heal a stale cache (e.g. a cytome filtered before the
            # subset-invalidation fix): a cell_depth whose length no longer
            # matches n_cells is from the pre-filter cell set — ignore it and
            # recompute below rather than return a wrong-length vector.
            if arr.shape[0] == int(ds.n_cells):
                return arr
            import warnings as _w
            _w.warn(
                f"Ignoring stale cached '{cache_key}' (length {arr.shape[0]} != "
                f"n_cells {ds.n_cells}); recomputing. This cytome was likely "
                "filtered before the cached-stats invalidation fix.",
                stacklevel=2,
            )
    if modality == "ATAC":
        try:
            n_frags = np.asarray(ds.cells["n_fragments"], dtype=np.float64)
            if n_frags.size == ds.n_cells and float(n_frags.sum()) > 0:
                ds.metadata[cache_key] = n_frags
                ds.flush()
                return n_frags
        except Exception:
            pass
    n_cells = int(ds.n_cells)
    depth = np.zeros(n_cells, dtype=np.float64)
    for chunk, idxs in ds.iter_chunks(
        modality=modality, layer="counts", batch_size=batch_size,
    ):
        depth[idxs] = np.asarray(chunk.sum(axis=1)).ravel()
    ds.metadata[cache_key] = depth
    ds.flush()
    return depth


__all__ = [
    "MODALITY_REGISTRY",
    "MODALITY_VAR_ENTITY",
    "modality_var_entity",
    "modality_feature_table_info",
    "modality_has_feature",
    "read_feature_column",
    "modality_cell_depth",
]
