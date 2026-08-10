"""Streaming-friendly subset and downsample operations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cytome.core.dataset import CytomeDataset


def subset(
    ds: CytomeDataset,
    mask,
    output,
    include_fragments: bool = True,
    include_embeddings: bool = True,
    include_graphs: bool = False,
    copy_annotations: bool = True,
):
    """Subset cells to a new Cytome dataset.

    ``copy_annotations`` (default True) carries over the **cell-independent**
    auxiliary data that the cell filter does not touch — the imported GTF gene
    models (``_gene_annotation`` / ``_exon_annotation``), the ``GA_genes`` var
    table, and var (gene-axis) embeddings. Without this they would be silently
    dropped by the rebuild-from-scratch subset (a real concern: the QC pipeline
    calls ``filter_cells``). **Spatial coordinates** are cell-indexed data and
    are always subset + remapped (independent of ``copy_annotations``).
    **Cell×cell graphs** are intentionally dropped (with a warning) — removing
    cells makes such a graph incomplete, so it must be re-derived.
    """
    del include_graphs
    keep_idx = _resolve_keep_indices(ds.n_cells, mask)
    # force=True preserves the historical overwrite behaviour of subset/
    # filter_cells (the QC pipeline regenerates subset cytomes in place);
    # the create-guard targets the user-facing creators, not derived subsets.
    out = CytomeDataset(output, mode="w", force=True)

    cells = ds.cells.to_pandas().iloc[keep_idx].copy().reset_index(drop=True)
    cells["cell_idx"] = np.arange(cells.shape[0], dtype=np.int64)
    out.set_entity("cells", cells)

    if ds.n_genes:
        out.set_entity("genes", ds.genes.to_pandas())
    if ds.n_peaks:
        out.set_entity("peaks", ds.peaks.to_pandas())
    # Copy any other column-axis entity tables (tiles, samples, proteins, and —
    # when copying annotations — the GA_genes var table) that have rows. These
    # are gene/feature-axis (cell-independent), so they're copied in full.
    _entity_tables = ["tiles", "samples", "proteins"]
    if copy_annotations:
        _entity_tables.append("GA_genes")
    for tbl in _entity_tables:
        try:
            n = ds._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except Exception:
            continue
        if n:
            out.set_entity(tbl, _read_entity_dataframe(ds._conn, tbl))
    out.flush()

    _subset_matrices_streaming(ds, out, keep_idx)

    if include_embeddings:
        from cytome.io.chunked_io import read_dense_rows
        for emb_name in ds.embeddings.keys():
            out.add_embedding(emb_name, read_dense_rows(ds._conn, emb_name, keep_idx))

    if copy_annotations:
        # GTF gene models — genome-coordinate data, not cell-indexed → copy
        # verbatim. Without this, an imported GTF is lost on every filter_cells.
        for tbl in ("_gene_annotation", "_exon_annotation"):
            _copy_table_rows(ds._conn, out._conn, tbl)
        # Var (gene-axis) embeddings (varm) — gene-indexed → copy in full.
        for vname in ds.var_embeddings.keys():
            out.add_var_embedding(vname, ds.var_embeddings[vname])
        out.flush()

    # Spatial coordinates are CELL-indexed data (like embeddings/fragments) →
    # subset to the kept cells and remap cell_idx. Always carried over.
    _subset_spatial_coords(ds, out, keep_idx)

    # Warn about the remaining cell-axis auxiliary data that is NOT remapped:
    # cell graphs (dropped by design — cells are removed on both axes so a
    # cell×cell graph is no longer complete) and var graphs.
    _warn_dropped_categories(ds)

    if include_fragments and "ATAC" in ds.modalities:
        _subset_fragments(ds, out, keep_idx)

    # Copy metadata, but DROP cell-dependent cached normalization stats: they are
    # indexed/aggregated over the OLD cell set and become silently wrong (or raise
    # a broadcast error) once cells are removed. The `_ensure_*` / cell-depth
    # helpers recompute them lazily on next use, so dropping is self-healing.
    #   *_cell_depth      — per-cell vector (length = old n_cells)
    #   *_log1p_params    — carries cell_depth
    #   *_infog_params    — cell_depth + cell-aggregate counts_sum / inv_gene_depth
    #   *_tfidf_params    — cell_depth + cell-dependent idf
    #   infog_params      — legacy un-prefixed RNA alias
    def _is_cell_dependent_stat(key: str) -> bool:
        if key == "infog_params":
            return True
        return key.endswith(("_cell_depth", "_log1p_params",
                             "_infog_params", "_tfidf_params"))

    for key, value in ds.metadata.items():
        if _is_cell_dependent_stat(key):
            continue
        out.metadata[key] = value

    out.provenance.log(
        operation="subset",
        function_name="cytome.subset",
        parameters={"n_selected": int(len(keep_idx))},
        package_name="cytome",
        package_version="0.1.0",
        input_objects=[str(ds.path)],
        output_objects=[str(output)],
    )
    out.flush()

    # Post-subset consistency check
    from cytome.utils.validation import validate
    report = validate(out)
    if not report.passed:
        import warnings
        warnings.warn(
            f"Subset cytome failed validation: {report.checks_failed}. "
            f"Run ds.repair() on the output to fix.",
            stacklevel=2,
        )

    return out


def _subset_matrices_streaming(ds: CytomeDataset, out: CytomeDataset, keep_idx: np.ndarray) -> None:
    """Subset every matrix into ``out`` **without** materialising the whole
    matrix in RAM.

    The pre-existing path did ``MeasurementLayer.rows(keep_idx)`` → one full
    in-memory CSR per matrix, then ``add_matrix``. On a tile matrix
    (e.g. 200K cells × 5.4M tiles, ~7e9 nnz) that single allocation is tens of
    GB → OOM (the ADVIS/HEA ``filter_cells`` kill). This is the matrix analogue
    of the streaming fragment re-chunker: read the source in row-contiguous
    chunks, select the kept rows, and write them out incrementally via a
    ``ChunkedLayerWriter``. Peak RAM ≈ one bounded output buffer.

    ``keep_idx`` is always sorted ascending (``np.where`` / ``np.unique`` in
    ``_resolve_keep_indices``), and ``read_sparse_rows_iter`` yields chunks in
    row order, so the kept rows stream out in ``keep_idx`` order — matching the
    ``cells`` table (set as ``cells.iloc[keep_idx]``).
    """
    from cytome.io.chunked_io import read_sparse_rows_iter

    # Flush thresholds: bound the in-RAM output buffer. Cap on nnz keeps wide
    # sparse rows (tiles) tiny; cap on rows keeps narrow/dense matrices chunky.
    NNZ_CAP = 4_000_000
    ROW_CAP = 4096
    n_out = int(len(keep_idx))

    mat_rows = ds._conn.execute(
        "SELECT matrix_name, n_cols, dtype, col_entity FROM matrix_meta ORDER BY matrix_name"
    ).fetchall()

    out_modalities = set(out.modalities)
    for matrix_name, n_cols, dtype, col_entity in mat_rows:
        writer = out.create_layer_writer(
            matrix_name, n_rows=n_out, n_cols=int(n_cols),
            dtype=dtype, row_entity="cells", col_entity=col_entity,
        )
        out_off = 0
        buf: list = []
        buf_rows = 0
        buf_nnz = 0
        for row_start, row_end, chunk in read_sparse_rows_iter(ds._conn, matrix_name):
            lo = int(np.searchsorted(keep_idx, row_start, side="left"))
            hi = int(np.searchsorted(keep_idx, row_end, side="left"))
            if hi <= lo:
                continue
            local = (keep_idx[lo:hi] - row_start).astype(np.int64)
            sub = chunk[local]
            buf.append(sub)
            buf_rows += sub.shape[0]
            buf_nnz += int(sub.nnz)
            if buf_rows >= ROW_CAP or buf_nnz >= NNZ_CAP:
                block = sp.vstack(buf, format="csr") if len(buf) > 1 else buf[0].tocsr()
                writer.write_chunk(block, row_offset=out_off)
                out_off += block.shape[0]
                buf, buf_rows, buf_nnz = [], 0, 0
        if buf_rows:
            block = sp.vstack(buf, format="csr") if len(buf) > 1 else buf[0].tocsr()
            writer.write_chunk(block, row_offset=out_off)
            out_off += block.shape[0]
        writer.finalize()

        # ChunkedLayerWriter.finalize writes matrix_meta but not the modality
        # manifest key (add_matrix did). Register it so the output cytome
        # exposes the modality (e.g. "ATAC", "tiles", "RNA").
        if "_" in matrix_name:
            mod = matrix_name.split("_", 1)[0]
            if mod:
                out_modalities.add(mod)
    if out_modalities:
        out._write_manifest_key("modalities", sorted(out_modalities))
        out._manifest = out._read_manifest()


def downsample(
    ds: CytomeDataset,
    n_cells: int | None = None,
    fraction: float | None = None,
    method: str = "random",
    groupby: str | None = None,
    seed: int = 42,
    output=None,
):
    """Downsample cells randomly or stratified."""
    if n_cells is None and fraction is None:
        raise ValueError("Provide n_cells or fraction")
    if n_cells is not None and fraction is not None:
        raise ValueError("Provide only one of n_cells or fraction")

    rng = np.random.default_rng(seed)
    total = ds.n_cells
    target = int(n_cells) if n_cells is not None else max(1, int(total * float(fraction)))
    target = min(target, total)

    if method == "stratified":
        if not groupby:
            raise ValueError("groupby is required for stratified downsample")
        df = ds.cells.to_pandas()[["cell_idx", groupby]]
        parts = []
        for _, g in df.groupby(groupby, dropna=False):
            frac = len(g) / max(len(df), 1)
            n = max(1, int(round(target * frac)))
            idx = rng.choice(g["cell_idx"].to_numpy(), size=min(n, len(g)), replace=False)
            parts.append(idx)
        keep = np.unique(np.concatenate(parts))
        if keep.size > target:
            keep = rng.choice(keep, size=target, replace=False)
    else:
        keep = np.sort(rng.choice(np.arange(total), size=target, replace=False))

    if output is None:
        return keep
    return subset(ds, keep, output)


def _read_entity_dataframe(conn, table_name: str) -> pd.DataFrame:
    """Read a full entity table as a pandas DataFrame."""
    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def _copy_table_rows(src_conn, dst_conn, table: str) -> int:
    """Copy all rows of a base-schema table from src to dst verbatim.

    Used for genome-coordinate annotation tables (``_gene_annotation`` /
    ``_exon_annotation``) which are not cell-indexed, so they carry over
    unchanged through a cell subset. No-op if the table is missing or empty.
    """
    try:
        cur = src_conn.execute(f"SELECT * FROM {table}")
    except Exception:
        return 0
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        return 0
    placeholders = ",".join("?" * len(cols))
    collist = ",".join(f'"{c}"' for c in cols)
    dst_conn.execute(f"DELETE FROM {table}")
    dst_conn.executemany(
        f"INSERT INTO {table} ({collist}) VALUES ({placeholders})", rows)
    dst_conn.commit()
    return len(rows)


def _subset_spatial_coords(ds, out, keep_idx: np.ndarray) -> None:
    """Carry spatial coordinates over to the subset, remapping ``cell_idx`` to
    the new 0..n-1 cell indexing and rebuilding the spatial rtree index.

    No-op if the source has no spatial coordinates.
    """
    try:
        rows = ds._conn.execute(
            "SELECT cell_idx, x, y, z FROM spatial_coords"
        ).fetchall()
    except Exception:
        return
    if not rows:
        return
    old_to_new = {int(old): new for new, old in enumerate(keep_idx)}
    kept = [
        (old_to_new[int(ci)], x, y, z)
        for (ci, x, y, z) in rows
        if int(ci) in old_to_new
    ]
    if not kept:
        return
    out._conn.executemany(
        "INSERT INTO spatial_coords (cell_idx, x, y, z) VALUES (?,?,?,?)", kept)
    # Rebuild the rtree index (point bbox per cell) so spatial range queries work.
    try:
        out._conn.executemany(
            "INSERT INTO spatial_rtree (id, min_x, max_x, min_y, max_y) "
            "VALUES (?,?,?,?,?)",
            [(ci, x, x, y, y) for (ci, x, y, _z) in kept],
        )
    except Exception:
        pass
    out._conn.commit()


def _warn_dropped_categories(ds) -> None:
    """Warn when cell-axis graphs are present but dropped by subset. Cell×cell
    graphs are intentionally NOT carried over (cells are removed on both axes,
    so the graph is no longer complete) — surface it so the loss isn't silent.
    """
    import warnings
    for table, label in (("graph_edges", "cell/var graphs"),):
        try:
            n = ds._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            n = 0
        if n:
            warnings.warn(
                f"cytome.subset/filter_cells: {label} are present but NOT "
                "carried over to the subset (a cell×cell graph is no longer "
                "complete after cells are removed). Re-derive them after "
                "subsetting if needed.",
                stacklevel=3,
            )


def _resolve_keep_indices(n_cells: int, mask) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == bool:
        if arr.shape[0] != n_cells:
            raise ValueError(f"Boolean mask length mismatch: expected {n_cells}, got {arr.shape[0]}")
        return np.where(arr)[0]
    idx = np.asarray(arr, dtype=np.int64)
    if idx.size == 0:
        return idx
    if np.any(idx < 0) or np.any(idx >= n_cells):
        raise ValueError("Subset indices out of range")
    return np.unique(idx)


def _subset_fragments(ds: CytomeDataset, out: CytomeDataset, keep_idx: np.ndarray) -> None:
    """Subset fragments using compressed chunked storage (same format as import).

    Reads from source compressed chunks, filters by cell remap, re-sorts by
    start position, then writes compressed BLOBs (~500K fragments per chunk).
    This is ~10-20x faster than per-row INSERT for large datasets.
    """
    from cytome.io.compression import compress_blob
    from cytome.io.convert_fragments import _write_block_chunks, _update_fragment_meta

    CHUNK_SIZE = 500_000

    # Build vectorized cell_idx remap: src_cell_idx -> out_cell_idx (or -1)
    max_src_idx = int(ds.n_cells)
    remap = np.full(max_src_idx, -1, dtype=np.int32)

    src_cells = ds.cells.to_pandas()
    keep_cells = out.cells.to_pandas()
    barcode_to_out = {}
    for i, b in zip(keep_cells["cell_idx"], keep_cells["barcode"]):
        barcode_to_out[str(b)] = int(i)
    for i, b in zip(src_cells["cell_idx"], src_cells["barcode"]):
        new_idx = barcode_to_out.get(str(b), -1)
        if new_idx >= 0 and int(i) < max_src_idx:
            remap[int(i)] = np.int32(new_idx)

    # Ensure fragment_chunks table exists
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

    frags = ds.ATAC.fragments
    use_chunks = frags.has_compressed_chunks

    with out._conn:
        if use_chunks:
            _subset_from_chunks(frags, out._conn, remap, max_src_idx, CHUNK_SIZE)
        else:
            _subset_from_rows(frags, out._conn, remap, max_src_idx, CHUNK_SIZE)


def _subset_from_chunks(frags, conn, remap, max_src_idx, chunk_size):
    """Subset from compressed fragment_chunks (fast path).

    Streaming re-chunker: the source chunks are already globally sorted by
    (chrom, start), and dropping cells preserves that order, so we filter
    chunk-by-chunk and emit a new chunk every ``chunk_size`` kept fragments
    WITHOUT loading or sorting the whole chromosome. Peak RAM is one source
    chunk + one output buffer (~MBs) instead of the entire chromosome (GBs).

    A cheap guard verifies the source really is sorted; if a (legacy / hand-made)
    cytome violates that invariant for some chromosome, we fall back to the
    original load-and-sort path for that chromosome only.
    """
    from cytome.io.convert_fragments import _write_block_chunks, _update_fragment_meta

    for chrom in frags.chromosomes:
        buf_s, buf_e, buf_c = [], [], []
        buf_n = 0
        next_idx = 0
        n_keep_total = 0
        prev_last = None
        sorted_ok = True

        for src_s, src_e, src_c in frags.iter_chromosome_chunks(chrom):
            # --- sorted-source guard (O(n), no sort): within- and cross-chunk ---
            if src_s.size:
                if (src_s[1:] < src_s[:-1]).any() or (prev_last is not None and src_s[0] < prev_last):
                    sorted_ok = False
                    break
                prev_last = src_s[-1]

            valid = src_c < max_src_idx
            mapped = np.where(valid, remap[src_c], np.int32(-1))
            keep = mapped >= 0
            if not keep.any():
                continue
            buf_s.append(src_s[keep]); buf_e.append(src_e[keep]); buf_c.append(mapped[keep])
            buf_n += int(keep.sum())

            # flush whole chunks as the buffer fills (order preserved)
            if buf_n >= chunk_size:
                cs = np.concatenate(buf_s); ce = np.concatenate(buf_e); cc = np.concatenate(buf_c)
                n_full = (cs.size // chunk_size) * chunk_size
                _write_block_chunks(
                    conn, chrom, cs[:n_full].astype(np.int32),
                    ce[:n_full].astype(np.int32), cc[:n_full].astype(np.int32),
                    chunk_size=chunk_size, compression="lz4", compression_level=1,
                    chunk_idx_start=next_idx, encoding=1,
                )
                next_idx += n_full // chunk_size
                n_keep_total += n_full
                buf_s, buf_e, buf_c = [cs[n_full:]], [ce[n_full:]], [cc[n_full:]]
                buf_n = cs.size - n_full

        if not sorted_ok:
            _subset_chromosome_sorted(frags, conn, chrom, remap, max_src_idx, chunk_size)
            continue

        # flush the remainder (< chunk_size)
        if buf_n > 0:
            cs = np.concatenate(buf_s); ce = np.concatenate(buf_e); cc = np.concatenate(buf_c)
            _write_block_chunks(
                conn, chrom, cs.astype(np.int32), ce.astype(np.int32), cc.astype(np.int32),
                chunk_size=chunk_size, compression="lz4", compression_level=1,
                chunk_idx_start=next_idx, encoding=1,
            )
            n_keep_total += cs.size
        if n_keep_total > 0:
            _update_fragment_meta(conn, chrom, n_keep_total)


def _subset_chromosome_sorted(frags, conn, chrom, remap, max_src_idx, chunk_size):
    """Fallback for a single chromosome whose source chunks are NOT sorted:
    load the chromosome, sort by start, then re-chunk (the original behaviour)."""
    import warnings
    from cytome.io.convert_fragments import _write_block_chunks, _update_fragment_meta

    warnings.warn(
        f"cytome.subset: source fragment chunks for '{chrom}' are not start-sorted; "
        "falling back to in-memory sort for this chromosome (higher RAM). "
        "Re-import the cytome to restore the sorted-chunk invariant.",
        RuntimeWarning, stacklevel=2,
    )
    all_s, all_e, all_c = [], [], []
    for src_s, src_e, src_c in frags.iter_chromosome_chunks(chrom):
        valid = src_c < max_src_idx
        mapped = np.where(valid, remap[src_c], np.int32(-1))
        keep = mapped >= 0
        if keep.any():
            all_s.append(src_s[keep]); all_e.append(src_e[keep]); all_c.append(mapped[keep])
    if not all_s:
        return
    starts = np.concatenate(all_s); ends = np.concatenate(all_e); cells = np.concatenate(all_c)
    order = np.argsort(starts, kind="mergesort")
    starts, ends, cells = starts[order].astype(np.int32), ends[order].astype(np.int32), cells[order].astype(np.int32)
    _write_block_chunks(conn, chrom, starts, ends, cells, chunk_size=chunk_size,
                        compression="lz4", compression_level=1, chunk_idx_start=0, encoding=1)
    _update_fragment_meta(conn, chrom, len(starts))


def _subset_from_rows(frags, conn, remap, max_src_idx, chunk_size):
    """Subset from per-row fragment tables (legacy, DEPRECATED).

    Only reached for old cytomes that predate the compressed ``fragment_chunks``
    format. Re-import to the chunked format to use the bounded-RAM streaming path.
    """
    import warnings
    warnings.warn(
        "cytome.subset: this cytome stores fragments in legacy per-row tables; "
        "subsetting them is deprecated and may be removed in a future release. "
        "Re-import to the compressed fragment_chunks format.",
        DeprecationWarning, stacklevel=2,
    )
    from cytome.io.convert_fragments import _write_block_chunks, _update_fragment_meta

    for chrom, frag in frags.iter_chromosomes():
        cell_idx = frag["cell_idx"]
        valid = cell_idx < max_src_idx
        mapped = np.where(valid, remap[cell_idx.astype(np.int64)], np.int32(-1))
        keep = mapped >= 0
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue

        starts = frag["start"][keep].astype(np.int32)
        ends = frag["end_"][keep].astype(np.int32)
        cells = mapped[keep].astype(np.int32)

        # Re-sort by start position
        order = np.argsort(starts, kind="mergesort")
        starts = starts[order]
        ends = ends[order]
        cells = cells[order]

        _write_block_chunks(
            conn, chrom, starts, ends, cells,
            chunk_size=chunk_size,
            compression="lz4",
            compression_level=1,
            chunk_idx_start=0,
            encoding=1,
        )
        _update_fragment_meta(conn, chrom, n_keep)
