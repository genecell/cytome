"""Streaming-friendly dataset merge helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cytome.core.dataset import CytomeDataset


def merge(
    inputs,
    output,
    batch_key: str = "sample_id",
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
):
    """Merge Cytome files into one output dataset.

    Parameters
    ----------
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

    Notes
    -----
    Matrix merging takes a vectorized fast path when all inputs share an
    identical feature order (``vstack`` with no remap); otherwise each input's
    matrix is projected onto the merged feature axis with a single sparse
    matmul (no per-column Python loop). Fragments are remapped by a vectorized
    cell-index offset (inputs are concatenated in order, so each input's cells
    occupy a contiguous block) and streamed into the compressed
    ``fragment_chunks`` format.
    """
    del include_graphs, chunk_memory_mb
    datasets = [_as_dataset(x) for x in inputs]
    close_after = [not isinstance(x, CytomeDataset) for x in inputs]

    if batch_labels is None:
        batch_labels = [Path(ds.path).stem for ds in datasets]

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
        import warnings as _w
        _w.warn(
            "cytome.merge: inputs have no recorded genome in their manifest; "
            "skipping the genome-consistency check.", stacklevel=2,
        )

    # force=False (default) raises FileExistsError if `output` already exists,
    # so an expensive merged atlas is never silently truncated.
    out = CytomeDataset(output, mode="w", force=force)
    if _merged_genome is not None:
        out._write_manifest_key("genome", _merged_genome)

    # Per-modality feature axis: every modality with a non-empty feature table
    # (genes/GA_genes/peaks/tiles) is merged onto a unified axis and its var
    # table is carried over — NOT just genes+peaks. Read the entity tables by
    # raw SQL: ds.tiles / ds.GA hit __getattr__ → a Modality, not the table.
    from cytome.utils.modality import MODALITY_VAR_ENTITY
    _strategy = {
        "RNA": gene_strategy, "GA": gene_strategy,
        "ATAC": peak_strategy, "tiles": tile_strategy,
    }
    # modality -> (entity_table, id_col)
    _mod_entity = dict(MODALITY_VAR_ENTITY)

    cell_frames = []
    feature_frames = {m: [] for m in _mod_entity}   # modality -> [DataFrame, ...]
    n_cells_per_ds = []
    for ds, label in zip(datasets, batch_labels):
        cells = ds.cells.to_pandas()
        cells[batch_key] = label
        cell_frames.append(cells)
        n_cells_per_ds.append(len(cells))

        for mod, (entity, id_col) in _mod_entity.items():
            fdf = _read_entity_frame(ds, entity)
            if fdf is not None and len(fdf) and id_col in fdf.columns:
                feature_frames[mod].append(fdf)

    out_cells = pd.concat(cell_frames, ignore_index=True)
    if "cell_idx" in out_cells.columns:
        out_cells = out_cells.drop(columns=["cell_idx"])
    if obs_columns == "shared":
        shared = set(cell_frames[0].columns)
        for df in cell_frames[1:]:
            shared &= set(df.columns)
        out_cells = out_cells[list(shared)]
    elif isinstance(obs_columns, list):
        keep = [c for c in obs_columns if c in out_cells.columns]
        out_cells = out_cells[keep]
    out.set_entity("cells", out_cells)

    # dataset d's cells occupy rows [offset_d, offset_d + n_cells_per_ds[d]).
    cell_offsets = np.concatenate([[0], np.cumsum(n_cells_per_ds)])[:-1].astype(np.int64)

    # Merge + write each modality's feature table; remember the merged axis so
    # every matrix of that modality is projected onto it (not just *_counts).
    merged_axis = {}   # modality -> (entity, id_col, output_ids)
    for mod, (entity, id_col) in _mod_entity.items():
        frames = feature_frames[mod]
        # Preserve-identical-axis fast path: when every input that has this
        # feature table already shares the SAME ids in the SAME order (a fixed
        # genome tiling, or same-reference genes/peaks), keep that native order —
        # avoids a needless lexical re-sort + 5.4M-wide sparse reprojection, and
        # preserves genomic tile order. Otherwise fall back to the sorted
        # union/intersection merge.
        shared_order = _identical_feature_axis(frames, id_col)
        if shared_order is not None:
            output_ids = shared_order
            feat_df = _clean_feature_frame(frames[0], id_col)
        else:
            output_ids, feat_df = _merge_feature_frames(frames, id_col, _strategy[mod])
        if not output_ids:
            continue
        if entity in ("genes", "GA_genes") and "symbol" not in feat_df.columns:
            feat_df["symbol"] = feat_df[id_col]
        out.set_entity(entity, feat_df)
        merged_axis[mod] = (entity, id_col, output_ids)

    out.flush()  # ensure entity tables are on disk before streaming matrix chunks

    matrix_names = _collect_matrix_names(datasets)
    for matrix_name in matrix_names:
        mod = matrix_name.split("_", 1)[0]
        axis = merged_axis.get(mod)
        if axis is not None:
            entity, id_col, output_ids = axis
            _stream_write_matrix(out, datasets, matrix_name, output_ids,
                                 n_cells_per_ds, entity, id_col)
        else:
            # No merged feature axis for this modality (no feature table on any
            # input) — keep the native axis (identity vstack).
            _stream_write_matrix(out, datasets, matrix_name, None,
                                 n_cells_per_ds, None, None)

    if include_embeddings:
        for ds in datasets:
            for name in ds.embeddings.keys():
                out.add_embedding(name, ds.embeddings[name])

    if include_fragments:
        _merge_fragments(datasets, out, cell_offsets)

    _merge_metadata(datasets, out, batch_labels)
    out.provenance.log(
        operation="merge",
        function_name="cytome.merge",
        parameters={"batch_key": batch_key, "gene_strategy": gene_strategy, "peak_strategy": peak_strategy},
        package_name="cytome",
        package_version="0.1.0",
        input_objects=[str(getattr(x, "path", x)) for x in inputs],
        output_objects=[str(output)],
    )
    out.flush()

    for ds, should_close in zip(datasets, close_after):
        if should_close:
            ds.close()
    return out


def _as_dataset(x) -> CytomeDataset:
    if isinstance(x, CytomeDataset):
        return x
    return CytomeDataset(x, mode="r")


def _read_entity_frame(ds: CytomeDataset, entity: str):
    """Return entity table ``entity`` as a DataFrame, or None if it doesn't
    exist. Reads by raw SQL — ``getattr(ds, 'tiles')`` / ``ds.GA`` resolve to a
    Modality via __getattr__, not the entity table, so the attribute path can't
    be used uniformly across genes/peaks/tiles/GA_genes."""
    exists = ds._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (entity,)
    ).fetchone()
    if exists is None:
        return None
    try:
        return pd.read_sql_query(f"SELECT * FROM {entity}", ds._conn)
    except Exception:
        return None


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


def _layer_for(ds: CytomeDataset, matrix_name: str):
    """Resolve the MeasurementLayer for ``{modality}_{layer}`` if present."""
    row = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()
    if row is None:
        return None
    modality, layer = matrix_name.split("_", 1)
    return getattr(ds, modality).layer(layer)


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


def _append_csr_chunks(conn, matrix_name, csr, chunk_size, row_offset, chunk_idx,
                       total_nnz, compression):
    """Append one CSR block to ``matrix_chunks`` as compressed row-chunks.

    Rows are written at global positions ``row_offset + local``. Returns the
    advanced ``(chunk_idx, total_nnz)``. Holds only this block in memory.
    """
    from cytome.io.chunked_io import _INSERT_CHUNK_SQL
    from cytome.io.compression import compress_blob

    csr = csr.tocsr()
    dtype_str = str(csr.data.dtype)
    n_rows = csr.shape[0]
    batch = []
    for rs in range(0, n_rows, chunk_size):
        re = min(rs + chunk_size, n_rows)
        chunk = csr[rs:re]
        batch.append((
            matrix_name, chunk_idx, row_offset + rs, row_offset + re, int(chunk.nnz),
            compress_blob(chunk.data.tobytes(), method=compression),
            compress_blob(chunk.indices.tobytes(), method=compression),
            compress_blob(chunk.indptr.tobytes(), method=compression),
            dtype_str, compression,
        ))
        chunk_idx += 1
        total_nnz += int(chunk.nnz)
    if batch:
        conn.executemany(_INSERT_CHUNK_SQL, batch)
    return chunk_idx, total_nnz


def _stream_write_matrix(out, datasets, matrix_name, output_features, n_cells_per_ds,
                         entity, feat_col):
    """Merge a matrix onto a unified feature axis and stream it to disk.

    Processes ONE input at a time — load → (identity fast path | sparse
    projection ``mat @ P``) → append compressed row-chunks → free — so peak
    memory is a single input's matrix, never the full vstacked atlas (which
    OOMs at 10s-of-millions-cell scale). Datasets missing the matrix contribute
    an aligned all-zero block so rows stay aligned with the merged cells table.
    ``output_features=None`` ⇒ no remap (identical/native feature axis);
    otherwise ``entity``/``feat_col`` name the per-input feature table + id
    column used to project onto ``output_features``.
    """
    from cytome.core.dataset import _infer_col_entity
    from cytome.io.chunk_tuning import compute_chunk_size
    from cytome.io.chunked_io import _now_iso

    conn = out._conn
    metas = {}
    for i, ds in enumerate(datasets):
        metas[i] = ds._conn.execute(
            "SELECT n_cols, n_nonzero, dtype FROM matrix_meta WHERE matrix_name = ?",
            (matrix_name,),
        ).fetchone()
    if all(m is None for m in metas.values()):
        return

    if output_features is not None:
        n_cols = len(output_features)
        out_idx = {g: i for i, g in enumerate(output_features)}
    else:
        n_cols = next(int(m[0]) for m in metas.values() if m is not None)
        out_idx = None
        entity = _infer_col_entity(matrix_name)

    dtype = np.dtype(next((m[2] for m in metas.values() if m is not None), "float32"))
    total_rows = int(sum(n_cells_per_ds))
    total_nnz_est = sum(int(m[1]) for m in metas.values() if m is not None)
    chunk_size = compute_chunk_size(total_rows, n_cols, max(1, total_nnz_est))

    conn.execute("DELETE FROM matrix_chunks WHERE matrix_name = ?", (matrix_name,))
    conn.execute("DELETE FROM matrix_meta WHERE matrix_name = ?", (matrix_name,))

    chunk_idx = 0
    row_offset = 0
    total_nnz = 0
    for i, (ds, n_cells) in enumerate(zip(datasets, n_cells_per_ds)):
        if metas[i] is None:
            block = sp.csr_matrix((n_cells, n_cols), dtype=dtype)
        else:
            mat = _layer_for(ds, matrix_name).to_memory().tocsr()
            if output_features is None:
                block = mat
            else:
                fdf = _read_entity_frame(ds, entity)
                in_features = (
                    list(fdf[feat_col].astype(str))
                    if fdf is not None and feat_col in fdf.columns else None
                )
                if in_features is None:
                    block = sp.csr_matrix((mat.shape[0], n_cols), dtype=mat.dtype)
                elif in_features == output_features:
                    block = mat  # identity fast path — no remap
                else:
                    block = (mat @ _projection_matrix(in_features, out_idx, n_cols, mat.dtype)).tocsr()
            mat = None
        chunk_idx, total_nnz = _append_csr_chunks(
            conn, matrix_name, block, chunk_size, row_offset, chunk_idx, total_nnz, "zstd"
        )
        row_offset += block.shape[0]
        block = None

    conn.execute(
        """INSERT INTO matrix_meta(
            matrix_name, n_rows, n_cols, n_nonzero, dtype,
            row_entity, col_entity, chunk_size, n_chunks, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (matrix_name, int(row_offset), int(n_cols), int(total_nnz), str(dtype),
         "cells", entity, int(chunk_size), int(chunk_idx), _now_iso()),
    )
    conn.commit()


def _merge_fragments(datasets, out, cell_offsets) -> None:
    """Merge ATAC fragments into the output's compressed ``fragment_chunks``.

    Cells are concatenated in order, so dataset ``d``'s cells occupy the
    contiguous global block starting at ``cell_offsets[d]`` — the cell-index
    remap is a vectorized ``+offset`` (no barcode round-trip). For each
    chromosome we gather every input's (start, end, cell+offset), stable-sort by
    start, and write delta+LZ4 chunks via the shared converter, matching the
    importer/subset layout. Inputs without ATAC fragments are skipped.
    """
    from cytome.io.convert_fragments import _write_block_chunks, _update_fragment_meta

    CHUNK_SIZE = 500_000

    has_any = False
    chroms = []
    seen = set()
    frag_sources = []  # (frags_store, offset)
    for ds, offset in zip(datasets, cell_offsets):
        if "ATAC" not in ds.modalities:
            continue
        frags = ds.ATAC.fragments
        if not frags.has_compressed_chunks:
            # legacy per-row store still works via iter_chromosome_chunks
            pass
        frag_sources.append((frags, int(offset)))
        for chrom in frags.chromosomes:
            if chrom not in seen:
                seen.add(chrom)
                chroms.append(chrom)
        has_any = True
    if not has_any:
        return

    out._conn.execute(
        """CREATE TABLE IF NOT EXISTS fragment_chunks (
            id INTEGER PRIMARY KEY,
            chrom TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            row_start INTEGER NOT NULL,
            row_end INTEGER NOT NULL,
            n_fragments INTEGER NOT NULL,
            min_start INTEGER,
            starts_blob BLOB NOT NULL,
            ends_blob BLOB NOT NULL,
            cell_idx_blob BLOB NOT NULL,
            compression TEXT DEFAULT 'lz4',
            encoding INTEGER DEFAULT 1,
            UNIQUE(chrom, chunk_idx)
        )"""
    )
    out._conn.execute(
        """CREATE TABLE IF NOT EXISTS fragment_meta (
            chrom TEXT PRIMARY KEY,
            n_fragments INTEGER NOT NULL,
            table_name TEXT NOT NULL,
            rtree_name TEXT NOT NULL,
            min_start INTEGER,
            max_end INTEGER
        )"""
    )

    with out._conn:
        for chrom in chroms:
            parts_s, parts_e, parts_c = [], [], []
            for frags, offset in frag_sources:
                if chrom not in frags.chromosomes:
                    continue
                for src_s, src_e, src_c in frags.iter_chromosome_chunks(chrom):
                    if src_s.size == 0:
                        continue
                    parts_s.append(src_s.astype(np.int32, copy=False))
                    parts_e.append(src_e.astype(np.int32, copy=False))
                    parts_c.append((src_c.astype(np.int64) + offset).astype(np.int32))
            if not parts_s:
                continue
            starts = np.concatenate(parts_s)
            ends = np.concatenate(parts_e)
            cells = np.concatenate(parts_c)
            # global sort-by-start so chunks keep the importer/subset invariant
            order = np.argsort(starts, kind="mergesort")
            starts = starts[order]
            ends = ends[order]
            cells = cells[order]
            _write_block_chunks(
                out._conn, chrom, starts, ends, cells,
                chunk_size=CHUNK_SIZE, compression="lz4", compression_level=1,
                chunk_idx_start=0, encoding=1,
            )
            _update_fragment_meta(out._conn, chrom, len(starts))

    from cytome.index.builder import build_fragment_indices

    build_fragment_indices(out._conn)


def _merge_metadata(datasets: list[CytomeDataset], out: CytomeDataset, labels: Sequence[str]) -> None:
    for ds, label in zip(datasets, labels):
        for key, value in ds.metadata.items():
            out.metadata[f"{label}:{key}"] = value
