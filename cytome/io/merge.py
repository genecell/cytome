"""Streaming-friendly dataset merge helpers."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cytome.core.dataset import CytomeDataset


def merge(
    inputs,
    output,
    batch_key: str = "batch",
    batch_labels: Sequence[str] | None = None,
    gene_strategy: str = "union",
    peak_strategy: str = "union",
    tile_strategy: str = "union",
    obs_columns: str | list[str] = "all",
    include_embeddings: bool = False,
    include_fragments: bool = True,
    include_graphs: bool = False,
    chunk_memory_mb: int = 256,
    force: bool = False,
    dtype=None,
):
    """Merge Cytome files into one output dataset.

    Parameters
    ----------
    batch_key
        The ``cells`` column that records each cell's input, filled with its
        label (``batch_labels``, by default each input's file stem). Default
        ``"batch"``. Pass ``"sample_id"`` to write the label there for every
        cell, replacing the cells' own samples.
    gene_strategy
        How to reconcile gene sets across inputs. ``'union'`` (default) keeps
        every gene seen in any input — genes absent from a given input become
        explicit zeros for its cells (atlas-friendly). ``'intersection'`` keeps
        only genes present in *all* inputs.
    peak_strategy
        Same as ``gene_strategy`` but for ATAC peaks. Default ``'union'``.
    tile_strategy
        Same as ``gene_strategy`` but for the genome-tiles feature axis. Default
        ``'union'``. (``GA`` gene-activity features reuse ``gene_strategy``.)
    obs_columns
        ``"all"`` (every column of any input, missing where an input lacks it),
        ``"shared"`` (the columns every input has) or a list. ``barcode``,
        ``sample_id`` and ``batch_key`` are always kept.
    include_embeddings
        Concatenate each cell embedding over the inputs: an input without it
        contributes rows of NaN, and an embedding whose width differs between
        inputs is left out, with a warning.
    dtype
        The value dtype of every merged matrix. ``None`` (default) promotes the
        inputs' dtypes (``np.result_type``: int32 and float32 give float64), so
        no value changes. A float written as an integer dtype must be integral.

    Notes
    -----
    **Cells and samples.** Barcodes are kept as given: a cell is named by its
    sample and its barcode, the pair the ``samples`` table and ``sample_id``
    record. A cell whose ``sample_id`` is empty gets its input's label there; a
    cell that has one keeps it (the default ``batch_key`` was ``"sample_id"``
    until 0.3.8, which replaced every cell's sample with the file stem). The
    ``samples`` table holds one row per sample of the merged cells, with the
    inputs' rows where they have them and each sample's cells counted. A
    barcode that repeats across inputs is reported.

    **Matrices** are written in one dtype per matrix, a row chunk of one input
    at a time, each projected onto the merged feature axis of its own feature
    table (the raw matrix onto the merged ``_raw_var``); fragments are merged
    per chromosome by their start, a chunk of each input at a time. The output
    is built as ``<output>.partial`` and renamed at the end, so a failure leaves
    no store under the output's name.
    """
    del include_graphs, chunk_memory_mb
    datasets = [_as_dataset(x) for x in inputs]
    close_after = [not isinstance(x, CytomeDataset) for x in inputs]
    try:
        return _merge(datasets, inputs, output, batch_key, batch_labels, gene_strategy,
                      peak_strategy, tile_strategy, obs_columns, include_embeddings,
                      include_fragments, force, dtype)
    finally:
        for ds, should_close in zip(datasets, close_after):
            if should_close:
                ds.close()


def _merge(datasets, inputs, output, batch_key, batch_labels, gene_strategy, peak_strategy,
           tile_strategy, obs_columns, include_embeddings, include_fragments, force, dtype):
    from cytome.io.subset import _remove_store_files, _sync_path
    if batch_labels is None:
        batch_labels = [Path(ds.path).stem for ds in datasets]
    batch_labels = [str(b) for b in batch_labels]
    if len(batch_labels) != len(datasets):
        raise ValueError(f"{len(batch_labels)} batch_labels for {len(datasets)} inputs")
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"Cytome already exists: {output}. Pass force=True to overwrite.")

    # Genome guard: merging across assemblies is never valid (coordinates, peaks
    # and tiles are incomparable). The genome is stored in the manifest
    # (set by the fragment importer's genome= arg). Refuse a cross-assembly merge;
    # carry the agreed genome onto the output so downstream (export_bigwig, …) work.
    _genomes = []
    for ds in datasets:
        try:
            _genomes.append(ds._manifest.get("genome"))
        except Exception:
            _genomes.append(None)
    _distinct_genomes = sorted({g for g in _genomes if g})
    if len(_distinct_genomes) > 1:
        raise ValueError(
            f"cytome.merge: cannot merge cytomes from different genomes: "
            f"{_distinct_genomes}. Coordinates / peaks / tiles are not comparable "
            f"across assemblies."
        )
    _merged_genome = _distinct_genomes[0] if _distinct_genomes else None
    if _merged_genome is None:
        warnings.warn(
            "cytome.merge: inputs have no recorded genome in their manifest; "
            "skipping the genome-consistency check.", stacklevel=3,
        )

    partial = Path(str(output) + ".partial")
    _remove_store_files(partial)
    out = CytomeDataset(partial, mode="w", force=True)
    try:
        if _merged_genome is not None:
            out._write_manifest_key("genome", _merged_genome)
        n_cells_per_ds = _merge_cells(datasets, out, batch_key, batch_labels, obs_columns)
        merged_axis = _merge_feature_tables(datasets, out, gene_strategy, peak_strategy,
                                            tile_strategy)
        out.flush()
        raw_axis = _merge_raw_var(datasets, out)

        for matrix_name in _collect_matrix_names(datasets):
            _stream_write_matrix(out, datasets, matrix_name, n_cells_per_ds, merged_axis,
                                 raw_axis, dtype)

        if include_embeddings:
            _merge_embeddings(datasets, out, n_cells_per_ds)
        _merge_var_embeddings(datasets, out, merged_axis)
        _merge_spatial(datasets, out, batch_labels)

        if include_fragments:
            _merge_fragments(datasets, out)

        _merge_metadata(datasets, out, batch_labels)
        out.provenance.log(
            operation="merge",
            function_name="cytome.merge",
            parameters={"batch_key": batch_key, "gene_strategy": gene_strategy,
                        "peak_strategy": peak_strategy, "tile_strategy": tile_strategy,
                        "obs_columns": obs_columns if isinstance(obs_columns, str)
                        else list(obs_columns),
                        "dtype": None if dtype is None else str(np.dtype(dtype))},
            package_name="cytome",
            package_version=_version(),
            input_objects=[str(getattr(x, "path", x)) for x in inputs],
            output_objects=[str(output)],
        )
        out.flush()
        out.close()
    except BaseException:
        try:
            out.close()
        except Exception:
            pass
        _remove_store_files(partial)
        raise
    _sync_path(partial)
    for sfx in ("-wal", "-shm", "-journal"):
        Path(str(output) + sfx).unlink(missing_ok=True)
    os.replace(partial, output)
    _sync_path(output.parent)
    return CytomeDataset(output, mode="r+")


def _version() -> str:
    try:
        from cytome import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _as_dataset(x) -> CytomeDataset:
    if isinstance(x, CytomeDataset):
        return x
    return CytomeDataset(x, mode="r")


# ---------------------------------------------------------------------------
# cells and samples
# ---------------------------------------------------------------------------

def _merge_cells(datasets, out, batch_key, batch_labels, obs_columns) -> list[int]:
    """Write the merged cells and samples tables; return each input's cell count."""
    from cytome.core.column_meta import restore_column_dtypes, store_column_meta

    frames = []
    for ds, label in zip(datasets, batch_labels):
        cells = restore_column_dtypes(ds._conn, "cells", ds.cells.to_pandas())
        cells = cells.drop(columns=[c for c in cells.columns if str(c).lower() == "cell_idx"])
        if batch_key == "sample_id":
            cells["sample_id"] = label                 # asked for: every cell's sample is its input
        else:
            cells[batch_key] = label
            sid = cells["sample_id"] if "sample_id" in cells.columns else pd.Series(
                [None] * len(cells), index=cells.index, dtype=object)
            empty = sid.isna() | (sid.astype(str) == "")
            cells["sample_id"] = sid.astype(object).where(~empty, label)
        frames.append(cells)

    keep_always = ["barcode", "sample_id", batch_key]
    if obs_columns == "shared":
        shared = set(frames[0].columns)
        for df in frames[1:]:
            shared &= set(df.columns)
        cols = [c for c in frames[0].columns if c in shared or c in keep_always]
    elif obs_columns == "all":
        cols = None
    else:
        cols = [c for c in obs_columns if any(c in f.columns for f in frames)]
        cols = [c for c in keep_always if c not in cols] + cols
    if cols is not None:
        frames = [f[[c for c in cols if c in f.columns]] for f in frames]

    merged = _concat_keeping_categories(frames)
    if "_obs_name" in merged.columns:
        # inputs without kept obs_names are named by their barcode
        merged["_obs_name"] = merged["_obs_name"].astype(object).where(
            merged["_obs_name"].notna(), merged["barcode"])

    # Barcodes are kept as given (a user's barcodes join to their own metadata);
    # a cell is its sample and its barcode.
    pairs = merged[["sample_id", "barcode"]].astype(str)
    dup = pairs.duplicated(keep=False)
    if dup.any():
        warnings.warn(
            f"cytome.merge: {int(dup.sum())} cells share a sample_id and a barcode with "
            f"another cell; give each input its own sample_id (its cells are named by the "
            f"pair).", UserWarning, stacklevel=4)
    origin = pd.Series(np.repeat(np.arange(len(frames)), [len(f) for f in frames]))
    n_across = int((origin.groupby(merged["barcode"].astype(str).to_numpy()).nunique() > 1).sum())
    if n_across:
        warnings.warn(
            f"cytome.merge: {n_across} barcodes occur in more than one input; they stay as "
            f"they are, and a cell is named by its sample_id and its barcode.",
            UserWarning, stacklevel=4)

    out.set_entity("cells", merged.reset_index(drop=True))
    out.flush()
    store_column_meta(out._conn, "cells", merged)

    # samples: one row per sample of the merged cells
    rows = {}
    for ds, label in zip(datasets, batch_labels):
        try:
            sdf = pd.read_sql_query("SELECT * FROM samples", ds._conn)
        except Exception:
            sdf = pd.DataFrame()
        for rec in sdf.to_dict("records"):
            sid = rec.get("sample_id")
            if sid is not None and sid not in rows:
                rec = {k: v for k, v in rec.items() if k not in ("sample_idx", "rowid")}
                if not rec.get("batch"):
                    rec["batch"] = label
                rows[sid] = rec
    counts = merged.groupby("sample_id", sort=False).size()
    batch_of = merged.groupby("sample_id", sort=False)[batch_key].first() \
        if batch_key in merged.columns and batch_key != "sample_id" else None
    for sid, n in counts.items():
        rec = rows.setdefault(sid, {"sample_id": sid})
        if not rec.get("batch"):
            rec["batch"] = batch_of.get(sid) if batch_of is not None else sid
        rec["n_cells"] = int(n)
    for sid, rec in rows.items():
        if sid not in counts.index:
            rec["n_cells"] = 0
    if rows:
        out.set_entity("samples", pd.DataFrame(list(rows.values())))
        out.flush()
    return [len(f) for f in frames]


def _concat_keeping_categories(frames):
    """``pd.concat`` that keeps a column categorical when every input that has it
    holds it as a categorical (the categories united, in order of appearance)."""
    merged = pd.concat(frames, ignore_index=True)
    for col in merged.columns:
        parts = [f[col] for f in frames if col in f.columns]
        if parts and all(isinstance(p.dtype, pd.CategoricalDtype) for p in parts):
            cats = []
            for p in parts:
                cats.extend(c for c in p.cat.categories if c not in cats)
            ordered = all(p.cat.ordered for p in parts) and all(
                list(p.cat.categories) == list(parts[0].cat.categories) for p in parts)
            merged[col] = pd.Categorical(merged[col].astype(object), categories=cats,
                                         ordered=ordered)
    return merged


# ---------------------------------------------------------------------------
# feature axes
# ---------------------------------------------------------------------------

def _merge_feature_tables(datasets, out, gene_strategy, peak_strategy, tile_strategy) -> dict:
    """Merge and write each modality's feature table;
    ``{modality: (entity, id_col, ids, axis unchanged)}``."""
    from cytome.core.column_meta import restore_column_dtypes, store_column_meta
    from cytome.utils.modality import MODALITY_VAR_ENTITY
    strategy = {"RNA": gene_strategy, "GA": gene_strategy,
                "ATAC": peak_strategy, "tiles": tile_strategy}
    merged_axis = {}
    for mod, (entity, id_col) in dict(MODALITY_VAR_ENTITY).items():
        frames = []
        for ds in datasets:
            fdf = _read_entity_frame(ds, entity)
            if fdf is not None and len(fdf) and id_col in fdf.columns:
                frames.append(restore_column_dtypes(ds._conn, entity, fdf))
        # Preserve-identical-axis fast path: when every input that has this
        # feature table shares the SAME ids in the SAME order (a fixed genome
        # tiling, or same-reference genes/peaks), keep that native order.
        shared_order = _identical_feature_axis(frames, id_col)
        if shared_order is not None:
            output_ids = shared_order
            feat_df = _clean_feature_frame(frames[0], id_col)
        else:
            output_ids, feat_df = _merge_feature_frames(frames, id_col, strategy[mod])
        if not output_ids:
            continue
        if entity in ("genes", "GA_genes") and "symbol" not in feat_df.columns:
            feat_df["symbol"] = feat_df[id_col]
        out.set_entity(entity, feat_df)
        out.flush()
        store_column_meta(out._conn, entity, feat_df)
        merged_axis[mod] = (entity, id_col, output_ids, shared_order is not None)
    return merged_axis


def _merge_raw_var(datasets, out):
    """The union of the inputs' raw var tables (``adata.raw.var``), by
    ``var_name``; None when no input has one."""
    frames = [f for f in (_read_entity_frame(ds, "_raw_var") for ds in datasets)
              if f is not None and len(f) and "var_name" in f.columns]
    if not frames:
        return None
    shared = _identical_feature_axis(frames, "var_name")
    if shared is not None:
        ids, df = shared, _clean_feature_frame(frames[0], "var_name")
    else:
        ids, df = _merge_feature_frames(frames, "var_name", "union")
    df = df.drop(columns=[c for c in df.columns if c == "var_idx"], errors="ignore")
    from cytome.io.convert_anndata import _write_raw_var_table
    _write_raw_var_table(out._conn, df.set_index("var_name", drop=False))
    out._conn.commit()
    return ids


def _read_entity_frame(ds: CytomeDataset, entity: str):
    """Return entity table ``entity`` as a DataFrame, or None if it doesn't
    exist. Reads by raw SQL — ``getattr(ds, 'tiles')`` / ``ds.GA`` resolve to a
    Modality via __getattr__, not the entity table, so the attribute path can't
    be used uniformly across genes/peaks/tiles/GA_genes."""
    exists = ds._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name=?", (entity,)
    ).fetchone()
    if exists is None:
        exists = ds._conn.execute(
            "SELECT 1 FROM sqlite_temp_master WHERE type IN ('table', 'view') AND name=?",
            (entity,)).fetchone()
    if exists is None:
        return None
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{entity}"', ds._conn)
    except Exception:
        return None
    if entity in getattr(ds._conn, "shim_views", ()) and "rowid" in df.columns:
        df = df.drop(columns=["rowid"])
    return df


def _collect_matrix_names(datasets: list[CytomeDataset]) -> list[str]:
    names = set()
    for ds in datasets:
        rows = ds._conn.execute("SELECT matrix_name FROM matrix_meta").fetchall()
        names.update(r[0] for r in rows)
    return sorted(names)


def _merge_features(feature_lists: list[list[str]], strategy: str) -> list[str]:
    if not feature_lists:
        return []
    if strategy == "intersection":
        s = set(feature_lists[0])
        for lst in feature_lists[1:]:
            s &= set(lst)
        return sorted(s)
    u = set()
    for lst in feature_lists:
        u.update(lst)
    return sorted(u)


def _identical_feature_axis(frames: list[pd.DataFrame], id_col: str):
    """Return the shared feature-id list iff every frame has the SAME ids in the
    SAME order; else None.

    When the inputs already agree on the feature axis (a fixed genome tiling, or
    same-reference genes/peaks), merge can keep that native order and skip the
    lexical sort + sparse reprojection entirely — cheaper, and it preserves
    genomic ordering (e.g. chr1, chr2, …, chr10 rather than chr1, chr10, …, chr2).
    """
    if not frames or id_col not in frames[0].columns:
        return None
    ref = list(frames[0][id_col].astype(str))
    for f in frames[1:]:
        if id_col not in f.columns or list(f[id_col].astype(str)) != ref:
            return None
    return ref


def _clean_feature_frame(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """A single feature frame readied for ``set_entity``: stringified id column,
    primary-key/index columns dropped (re-assigned by ``set_entity``)."""
    df = frame.copy()
    df[id_col] = df[id_col].astype(str)
    drop = [c for c in df.columns if c.endswith("_idx") or c == "_index"]
    if drop:
        df = df.drop(columns=drop)
    return df.reset_index(drop=True)


def _merge_feature_frames(frames: list[pd.DataFrame], id_col: str, strategy: str):
    """Merge feature tables, preserving per-feature columns (e.g. peak chr/start).

    Returns ``(sorted_feature_ids, dataframe)`` where the dataframe carries the
    first-seen row for each surviving feature, reindexed to the sorted id order.
    """
    if not frames:
        return [], pd.DataFrame()
    id_lists = [list(f[id_col].astype(str)) for f in frames]
    output_ids = _merge_features(id_lists, strategy)
    if not output_ids:
        return [], pd.DataFrame()
    # first-seen row per id across inputs (union of columns)
    rep: dict[str, dict] = {}
    for f in frames:
        recs = f.to_dict("records")
        for r in recs:
            fid = str(r[id_col])
            if fid not in rep:
                r = dict(r)
                r[id_col] = fid
                rep[fid] = r
    df = pd.DataFrame([rep[i] for i in output_ids])
    # drop source primary-key / index columns (re-assigned by set_entity)
    drop = [c for c in df.columns if c.endswith("_idx") or c == "_index"]
    if drop:
        df = df.drop(columns=drop)
    return output_ids, df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------

def _projection_matrix(in_features, out_idx, n_out, dtype):
    """0/1 sparse projection P (n_in × n_out): mat @ P reindexes columns."""
    in_pos = np.fromiter(
        (j for j, g in enumerate(in_features) if g in out_idx), dtype=np.int64
    )
    out_pos = np.fromiter((out_idx[in_features[j]] for j in in_pos), dtype=np.int64)
    return sp.csr_matrix(
        (np.ones(in_pos.size, dtype=dtype), (in_pos, out_pos)),
        shape=(len(in_features), n_out),
    )


def _column_entities(ds) -> dict:
    from cytome.utils.validation import matrix_column_entities
    return matrix_column_entities(ds._conn)


def _out_dtype(dtypes, requested):
    """The dtype of a merged matrix: the promotion of the inputs' (no value
    changes), or the one asked for."""
    if requested is not None:
        return np.dtype(requested)
    return np.result_type(*[np.dtype(d) for d in dtypes])


def _stream_write_matrix(out, datasets, matrix_name, n_cells_per_ds, merged_axis, raw_axis,
                         requested_dtype):
    """Merge a matrix onto the feature axis of its own feature table and stream it.

    One row chunk of one input at a time — (identity | sparse projection
    ``chunk @ P``) → the output's layer writer — so memory holds a chunk, not an
    input's whole matrix (an input stored column-major is read whole). An input
    without the matrix contributes empty rows. Every chunk is written in the
    one dtype of the merged matrix: chunks used to keep their input's dtype under
    the first input's, and an int32 count of 2 read back as 1,073,741,824 from a
    float32 merge.
    """
    from cytome.core.dataset import _infer_col_entity
    from cytome.io.chunked_io import read_sparse_rows_iter

    metas = {}
    for i, ds in enumerate(datasets):
        metas[i] = ds._conn.execute(
            "SELECT n_cols, n_nonzero, dtype, COALESCE(n_chunks, 0), COALESCE(has_csc, 0) "
            "FROM matrix_meta WHERE matrix_name = ?", (matrix_name,)).fetchone()
    present = [i for i, m in metas.items() if m is not None]
    if not present:
        return

    entities = {i: _column_entities(datasets[i]).get(matrix_name) for i in present}
    mod = matrix_name.split("_", 1)[0]
    axis = merged_axis.get(mod)
    ent_set = {entities[i] for i in present}
    if ent_set == {"_raw_var"} and raw_axis is not None:
        out_ids, entity, feat_col, table = raw_axis, "_raw_var", "var_name", "_raw_var"
    elif axis is not None and ent_set == {axis[0]}:
        out_ids, entity, feat_col, table = axis[2], axis[0], axis[1], axis[0]
    else:
        # no feature table to project by (a sparse obsm stored as a matrix, or
        # a matrix of no modality): kept only when every input has one width
        widths = {int(metas[i][0]) for i in present}
        if len(widths) != 1:
            warnings.warn(
                f"cytome.merge: {matrix_name} has no feature table to align its columns by "
                f"and its inputs differ in width ({sorted(widths)}); it is not merged.",
                UserWarning, stacklevel=4)
            return
        out_ids, table, feat_col = None, None, None
        entity = next(iter(ent_set)) if len(ent_set) == 1 else None
        if entity is None:
            entity = "_obsm" if any(e is None for e in ent_set) else _infer_col_entity(matrix_name)

    n_cols = len(out_ids) if out_ids is not None else int(metas[present[0]][0])
    out_idx = {g: k for k, g in enumerate(out_ids)} if out_ids is not None else None
    dtype = _out_dtype([metas[i][2] for i in present], requested_dtype)
    total_rows = int(sum(n_cells_per_ds))
    writer = out.create_layer_writer(matrix_name, n_rows=total_rows, n_cols=n_cols,
                                     dtype=dtype, col_entity=entity)
    row_offset = 0
    for i, (ds, n_cells) in enumerate(zip(datasets, n_cells_per_ds)):
        if metas[i] is None:
            writer.write_chunk(sp.csr_matrix((n_cells, n_cols), dtype=dtype), row_offset)
            row_offset += n_cells
            continue
        P = None
        if out_idx is not None:
            fdf = _read_entity_frame(ds, table)
            in_features = (list(fdf[feat_col].astype(str))
                           if fdf is not None and feat_col in fdf.columns else None)
            if in_features is None or len(in_features) != int(metas[i][0]):
                raise ValueError(
                    f"cytome.merge: {matrix_name} of input {i} has {int(metas[i][0])} columns "
                    f"and its feature table {table} "
                    f"{0 if in_features is None else len(in_features)} rows; its columns "
                    f"cannot be aligned")
            if in_features != out_ids:
                P = _projection_matrix(in_features, out_idx, n_cols, dtype)
        column_major = int(metas[i][3]) == 0 and bool(metas[i][4])
        if column_major:
            from cytome.core.measurement import MeasurementLayer
            blocks = [(0, MeasurementLayer(ds._conn, matrix_name).to_memory())]
        else:
            blocks = ((rs, chunk) for rs, _re, chunk in read_sparse_rows_iter(ds._conn, matrix_name))
        written = 0
        for rs, chunk in blocks:
            rs = min(int(rs), n_cells)
            if rs > written:                 # rows the input's chunks skip: zero rows
                writer.write_chunk(sp.csr_matrix((rs - written, n_cols), dtype=dtype),
                                   row_offset + written)
            rows = min(chunk.shape[0], n_cells - rs)
            if rows <= 0:
                continue
            block = chunk[:rows] if rows < chunk.shape[0] else chunk
            if P is not None:
                block = (block @ P).tocsr()
            writer.write_chunk(block, row_offset + rs)
            written = rs + rows
        if written < n_cells:
            writer.write_chunk(sp.csr_matrix((n_cells - written, n_cols), dtype=dtype),
                               row_offset + written)
        row_offset += n_cells
    writer.finalize()


# ---------------------------------------------------------------------------
# embeddings, spatial
# ---------------------------------------------------------------------------

def _merge_embeddings(datasets, out, n_cells_per_ds) -> None:
    """Each cell embedding, its rows concatenated over the inputs. (Only the
    last input's survived before: 2 rows for 5 cells.)"""
    names = []
    for ds in datasets:
        for name in ds.embeddings.keys():
            if name not in names:
                names.append(name)
    for name in names:
        widths = {int(ds._conn.execute("SELECT n_cols FROM embedding_meta WHERE array_name = ?",
                                       (name,)).fetchone()[0])
                  for ds in datasets if name in ds.embeddings}
        if len(widths) != 1:
            warnings.warn(f"cytome.merge: embedding {name!r} has widths {sorted(widths)} in the "
                          f"inputs; it is not merged.", UserWarning, stacklevel=4)
            continue
        width = widths.pop()
        parts, missing = [], []
        for k, (ds, n) in enumerate(zip(datasets, n_cells_per_ds)):
            if name in ds.embeddings:
                parts.append(ds.embeddings[name])
            else:
                parts.append(np.full((n, width), np.nan))
                missing.append(k)
        if missing:
            warnings.warn(f"cytome.merge: embedding {name!r} is missing from input(s) {missing}; "
                          f"their cells get NaN rows.", UserWarning, stacklevel=4)
        out.add_embedding(name, np.vstack(parts), flush=False)
    out.flush()


def _merge_var_embeddings(datasets, out, merged_axis) -> None:
    """Var embeddings follow features: carried from the first input that has
    each, when its modality's feature axis came out unchanged."""
    entity_of_axis = {v[0]: v for v in merged_axis.values()}
    done = set()
    for ds in datasets:
        for name, entity in ds._conn.execute(
                "SELECT array_name, entity FROM embedding_meta WHERE entity != 'cells'").fetchall():
            if name in done:
                continue
            axis = entity_of_axis.get(entity)
            if axis is None or not axis[3]:
                continue                       # the feature axis changed: rows would not align
            out.add_var_embedding(name, ds.var_embeddings[name], entity=entity, flush=False)
            done.add(name)
    out.flush()


def _merge_spatial(datasets, out, batch_labels) -> None:
    """Gene models (from the first input that has them: one assembly), spatial
    images and scale factors (all inputs'; a library id taken twice is renamed
    ``<label>:<library>``) and spatial coordinates (offset to the merged cells)."""
    from cytome.io.subset import _copy_feature_table, _table_has_rows
    for tbl in ("_gene_annotation", "_exon_annotation"):
        for ds in datasets:
            if _table_has_rows(ds._conn, tbl):
                _copy_feature_table(ds._conn, out._conn, tbl)
                break
    taken = set()
    images_seen = False
    for ds, label in zip(datasets, batch_labels):
        if not (_table_has_rows(ds._conn, "spatial_images")
                or _table_has_rows(ds._conn, "spatial_scalefactors")):
            continue
        from cytome.core.spatial import _ensure_tables
        if not images_seen:
            _ensure_tables(out._conn)
            images_seen = True
        libs = set()
        for tbl in ("spatial_images", "spatial_scalefactors"):
            if _table_has_rows(ds._conn, tbl):
                libs.update(r[0] for r in ds._conn.execute(f"SELECT library_id FROM {tbl}"))
        rename = {lib: (f"{label}:{lib}" if lib in taken else lib) for lib in libs}
        taken.update(rename.values())
        with out._conn:
            if _table_has_rows(ds._conn, "spatial_images"):
                for row in ds._conn.execute(
                        "SELECT library_id, img_key, image, format, height, width, channels, "
                        "dtype FROM spatial_images"):
                    out._conn.execute("INSERT INTO spatial_images VALUES (?,?,?,?,?,?,?,?)",
                                      (rename[row[0]], *row[1:]))
            if _table_has_rows(ds._conn, "spatial_scalefactors"):
                for lib, key, value in ds._conn.execute(
                        "SELECT library_id, key, value FROM spatial_scalefactors"):
                    out._conn.execute("INSERT OR REPLACE INTO spatial_scalefactors "
                                      "VALUES (?,?,?)", (rename[lib], key, value))
    # coordinates: each input's cells by position, offset
    rows, offset = [], 0
    for ds in datasets:
        pos = _cell_positions(ds)
        if _table_has_rows(ds._conn, "spatial_coords"):
            for ci, x, y, z in ds._conn.execute("SELECT cell_idx, x, y, z FROM spatial_coords"):
                p = pos.get(int(ci))
                if p is not None:
                    rows.append((offset + p, x, y, z))
        offset += len(pos)
    if rows:
        with out._conn:
            out._conn.executemany("INSERT INTO spatial_coords (cell_idx, x, y, z) VALUES (?,?,?,?)",
                                  rows)
            out._conn.executemany("INSERT INTO spatial_rtree (id, min_x, max_x, min_y, max_y) "
                                  "VALUES (?,?,?,?,?)", [(c, x, x, y, y) for c, x, y, _z in rows])


def _cell_positions(ds) -> dict:
    """``{cell_idx: row}`` of an input (its cell_idx need not be its rows)."""
    ci = np.asarray(ds.cells["cell_idx"], dtype=np.int64)
    return {int(c): i for i, c in enumerate(ci)}


# ---------------------------------------------------------------------------
# fragments
# ---------------------------------------------------------------------------

def _merge_fragments(datasets, out) -> None:
    """Merge ATAC fragments into the output's compressed ``fragment_chunks``.

    Each input's cells occupy a contiguous block of the merged cells; a
    fragment's ``cell_idx`` goes to its cell's row in that block (by position,
    so an input whose cell_idx is not its rows is still right). Per chromosome
    the inputs' start-sorted chunk streams are merged by start, a chunk of each
    input at a time (a k-way merge): memory holds about one chunk per input, not
    the chromosome of every input. An input whose chunks are not start-sorted
    has that chromosome loaded and sorted.
    """
    from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks
    from cytome.io.subset import _has_fragments

    CHUNK_SIZE = 500_000
    sources = []                                   # (FragmentStore, remap array)
    offset = 0
    chroms = []
    for ds in datasets:
        n = ds.n_cells
        if _has_fragments(ds):
            ci = np.asarray(ds.cells["cell_idx"], dtype=np.int64)
            remap = np.full(int(ci.max()) + 1 if ci.size else 0, -1, dtype=np.int64)
            remap[ci] = offset + np.arange(ci.size, dtype=np.int64)
            frags = ds.fragments
            sources.append((frags, remap))
            for chrom in frags.chromosomes:
                if chrom not in chroms:
                    chroms.append(chrom)
        offset += n
    if not sources:
        return

    with out._conn:
        for chrom in chroms:
            writer = _ChunkWriter(out._conn, chrom, CHUNK_SIZE, _write_block_chunks)
            streams = [_SortedStream(frags.iter_chromosome_chunks(chrom), remap)
                       for frags, remap in sources if chrom in frags.chromosomes]
            if not _kway(streams, writer):
                # an input not start-sorted: this chromosome loaded whole and sorted
                out._conn.execute("DELETE FROM fragment_chunks WHERE chrom = ?", (chrom,))
                writer = _ChunkWriter(out._conn, chrom, CHUNK_SIZE, _write_block_chunks)
                parts = []
                for frags, remap in sources:
                    if chrom not in frags.chromosomes:
                        continue
                    for s, e, c in frags.iter_chromosome_chunks(chrom):
                        parts.append(_remapped(s, e, c, remap))
                s = np.concatenate([p[0] for p in parts])
                e = np.concatenate([p[1] for p in parts])
                c = np.concatenate([p[2] for p in parts])
                order = np.argsort(s, kind="mergesort")
                writer.add(s[order], e[order], c[order])
            writer.close()
            if writer.n:
                _update_fragment_meta(out._conn, chrom, writer.n, max_end=writer.max_end)

    from cytome.index.builder import build_fragment_indices
    build_fragment_indices(out._conn)


def _remapped(s, e, c, remap):
    c = np.asarray(c, dtype=np.int64)
    ok = (c >= 0) & (c < remap.size)
    mapped = np.full(c.shape, -1, dtype=np.int64)
    mapped[ok] = remap[c[ok]]
    keep = mapped >= 0
    if not keep.all():
        s, e, mapped = s[keep], e[keep], mapped[keep]
    return (np.asarray(s, dtype=np.int32), np.asarray(e, dtype=np.int32),
            mapped.astype(np.int32))


class _SortedStream:
    """One input's chunks of a chromosome, remapped, checked to be start-sorted."""

    def __init__(self, it, remap):
        self._it, self._remap = it, remap
        self.buf = None                     # (starts, ends, cells), what is not yet emitted
        self.done = False
        self.unsorted = False
        self._last = None

    def fill(self) -> None:
        while not self.done and (self.buf is None or self.buf[0].size == 0):
            try:
                s, e, c = next(self._it)
            except StopIteration:
                self.done = True
                return
            if s.size == 0:
                continue
            if np.any(s[1:] < s[:-1]) or (self._last is not None and s[0] < self._last):
                self.unsorted = True
                self.done = True
                return
            self._last = int(s[-1])
            self.buf = _remapped(s, e, c, self._remap)


def _kway(streams, writer) -> bool:
    """Merge start-sorted streams into ``writer``; False if one turns out unsorted."""
    while True:
        for st in streams:
            st.fill()
        if any(st.unsorted for st in streams):
            return False
        live = [st for st in streams if st.buf is not None and st.buf[0].size]
        if not live:
            return True
        waiting = [st for st in live if not st.done]
        # every fragment still to come from a stream starts at or after the last
        # start it has buffered: up to the smallest of those, the order is final
        bound = min(int(st.buf[0][-1]) for st in waiting) if waiting else None
        parts = []
        for st in live:
            s, e, c = st.buf
            k = s.size if bound is None else int(np.searchsorted(s, bound, side="right"))
            if k:
                parts.append((s[:k], e[:k], c[:k]))
                st.buf = (s[k:], e[k:], c[k:])
        s = np.concatenate([p[0] for p in parts])
        e = np.concatenate([p[1] for p in parts])
        c = np.concatenate([p[2] for p in parts])
        order = np.argsort(s, kind="mergesort")
        writer.add(s[order], e[order], c[order])


class _ChunkWriter:
    """Fragments added in start order, written as full chunks as they fill."""

    def __init__(self, conn, chrom, chunk_size, write):
        self.conn, self.chrom, self.size, self._write = conn, chrom, chunk_size, write
        self.buf = ([], [], [])
        self.held = 0
        self.next_idx = 0
        self.n = 0
        self.max_end = None

    def add(self, s, e, c) -> None:
        if not s.size:
            return
        self.buf[0].append(s)
        self.buf[1].append(e)
        self.buf[2].append(c)
        self.held += s.size
        m = int(e.max())
        self.max_end = m if self.max_end is None else max(self.max_end, m)
        if self.held >= self.size:
            self._flush(full_only=True)

    def _flush(self, full_only: bool) -> None:
        if not self.held:
            return
        s = np.concatenate(self.buf[0])
        e = np.concatenate(self.buf[1])
        c = np.concatenate(self.buf[2])
        n = (s.size // self.size) * self.size if full_only else s.size
        if n:
            self._write(self.conn, self.chrom, s[:n], e[:n], c[:n], chunk_size=self.size,
                        compression="lz4", compression_level=1, chunk_idx_start=self.next_idx,
                        encoding=1)
            self.next_idx += -(-n // self.size)
            self.n += n
        self.buf = ([s[n:]], [e[n:]], [c[n:]])
        self.held = s.size - n

    def close(self) -> None:
        self._flush(full_only=False)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def _merge_metadata(datasets: list[CytomeDataset], out: CytomeDataset, labels: Sequence[str]) -> None:
    """Each input's metadata under ``<label>:<key>``; the AnnData conversion
    records (``_anndata_*``) are united unprefixed, so ``to_anndata`` of the
    merge still finds its layers, obsm and raw (they were prefixed and lost)."""
    united: dict = {}
    for ds, label in zip(datasets, labels):
        for key, value in ds.metadata.items():
            if key.startswith("_anndata_"):
                if isinstance(value, dict):
                    merged = dict(united.get(key) or {})
                    for k, v in value.items():
                        merged.setdefault(k, v)
                    united[key] = merged
                else:
                    united.setdefault(key, value)
                continue
            out.metadata[f"{label}:{key}"] = value
    raw = united.get("_anndata_raw")
    if isinstance(raw, dict) and raw.get("matrix_name") and \
            out.matrix_meta(raw["matrix_name"]) is None:
        united.pop("_anndata_raw")
    if "_anndata_obs_index" in united and united["_anndata_obs_index"] not in out.cells.columns:
        united.pop("_anndata_obs_index")
    for key, value in united.items():
        out.metadata[key] = value
