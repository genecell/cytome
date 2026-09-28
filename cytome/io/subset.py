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
    matrices: "list[str] | None" = None,
):
    """Subset cells to a new Cytome dataset.

    ``matrices`` names the matrices to carry (e.g. ``["ATAC_counts"]``);
    ``None`` carries every matrix in the store. A caller that clusters one
    modality of a store holding several -- peaks and tiles, say -- should
    name the one it needs: the others would otherwise be read and rewritten
    for nothing, once per subset. With ``matrices`` given, the output also
    carries only the feature tables those matrices are indexed by (``peaks``
    for ``ATAC_counts``), not the feature tables of other modalities.

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
    ds.flush()            # what the caller has written so far is part of the source
    keep_idx = _resolve_keep_indices(ds.n_cells, mask)
    out = _begin_output(ds, keep_idx, output, matrices, copy_annotations)
    _subset_matrices_streaming(ds, out, keep_idx, matrices=matrices)
    _finish_output(ds, out, keep_idx, output, include_embeddings, include_fragments,
                   copy_annotations)
    return out


def _begin_output(ds, keep_idx, output, matrices, copy_annotations):
    """A new store with the kept cells, the feature tables and the samples; no matrices yet."""
    # force=True preserves the historical overwrite behaviour of subset/
    # filter_cells (the QC pipeline regenerates subset cytomes in place);
    # the create-guard targets the user-facing creators, not derived subsets.
    out = CytomeDataset(output, mode="w", force=True)

    cells = ds.cells.to_pandas().iloc[keep_idx].copy().reset_index(drop=True)
    cells["cell_idx"] = np.arange(cells.shape[0], dtype=np.int64)
    out.set_entity("cells", cells)

    # Feature tables. A plain subset carries all of them. One that names its
    # matrices carries the feature tables of those matrices and no others: a
    # caller clustering the peak matrix has no use for five million tiles,
    # and copying them cost more than everything else in the subset together.
    carried = _feature_tables_of(ds, matrices)

    def _wanted(tbl: str) -> bool:
        return carried is None or tbl in carried

    # Feature tables do not depend on the cells kept, so they are copied row
    # for row, in batches, straight from one connection to the other: no
    # DataFrame of five million tiles, and every value arrives as stored.
    # ``samples`` describes the cells' samples and is always carried.
    out.flush()                                   # the cells table first
    _feature_tables = ["genes", "peaks", "tiles", "proteins"]
    if copy_annotations:
        _feature_tables.append("GA_genes")
    for tbl in _feature_tables:
        if _wanted(tbl):
            _copy_feature_table(ds._conn, out._conn, tbl)
    _copy_feature_table(ds._conn, out._conn, "samples")
    out.flush()

    return out


def _finish_output(ds, out, keep_idx, output, include_embeddings, include_fragments,
                   copy_annotations):
    """Everything a subset carries after its matrices: embeddings, annotations, spatial
    coordinates, fragments, metadata, provenance; then the consistency check."""

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



def _feature_tables_of(ds, matrices) -> "set[str] | None":
    """The feature tables the named matrices are indexed by, or ``None`` for
    "every table" when no matrices are named.

    A matrix records its column entity in ``matrix_meta.col_entity``; where
    an old store left it empty the modality's own table is used.
    """
    if matrices is None:
        return None
    from cytome.utils.modality import modality_feature_table_info
    tables: set[str] = set()
    for name in matrices:
        row = ds._conn.execute(
            "SELECT col_entity FROM matrix_meta WHERE matrix_name = ?", (name,)).fetchone()
        if row is None:
            continue                      # reported, with the list, by the matrix copy
        if row[0]:
            tables.add(str(row[0]))
        try:
            tables.add(modality_feature_table_info(ds, name.split("_", 1)[0])[0])
        except Exception:
            pass
    return tables


def _subset_matrices_streaming(ds: CytomeDataset, out: CytomeDataset, keep_idx: np.ndarray,
                               matrices: "list[str] | None" = None) -> None:
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
    if matrices is not None:
        wanted = set(matrices)
        present = {r[0] for r in mat_rows}
        missing = sorted(wanted - present)
        if missing:
            raise KeyError(f"subset(matrices=...): not in the store: {missing}; "
                           f"available: {sorted(present)}")
        mat_rows = [r for r in mat_rows if r[0] in wanted]

    out_modalities = set(out.modalities)
    for matrix_name, n_cols, dtype, col_entity in mat_rows:
        if "_" in matrix_name and matrix_name.split("_", 1)[0]:
            # Neither writer below registers the modality in the manifest.
            out_modalities.add(matrix_name.split("_", 1)[0])
        if _is_column_major(ds._conn, matrix_name):
            _subset_csc_matrix(ds, out, matrix_name, int(n_cols), dtype, col_entity, keep_idx)
            continue
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



def _is_column_major(conn, matrix_name: str) -> bool:
    row = conn.execute(
        "SELECT COALESCE(n_chunks, 0), COALESCE(has_csc, 0) FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    return bool(row) and row[0] == 0 and bool(row[1])


def _subset_csc_matrix(ds, out, matrix_name: str, n_cols: int, dtype, col_entity, keep_idx) -> None:
    """Row-select a column-major matrix, chunk by chunk, into ``out``.

    A CSC chunk holds all rows of a run of columns, so keeping rows is a
    fancy index on each chunk and the column ranges are unchanged. Nothing
    is transposed and one chunk is resident at a time. The result keeps the
    layout: CSC chunks only, ``matrix_meta.layout = 'csc'``.
    """
    from cytome.core.measurement import MeasurementLayer
    src = MeasurementLayer(ds._conn, matrix_name)
    keep = np.asarray(keep_idx, dtype=np.int64)
    out._conn.execute("DELETE FROM matrix_csc_chunks WHERE matrix_name = ?", (matrix_name,))
    total_nnz = 0
    n_chunks = 0
    for col_start, col_end, chunk in src.iter_columns():
        sub = sp.csc_matrix(chunk[keep])        # CSC stays CSC; rows in keep order
        sub.sort_indices()
        total_nnz += _write_csc_chunk(out._conn, matrix_name, n_chunks, col_start, col_end, sub, dtype)
        n_chunks += 1
    _write_csc_meta(ds, out._conn, matrix_name, int(len(keep)), n_cols, total_nnz, dtype,
                    col_entity, n_chunks)


def _write_csc_chunk(conn, matrix_name, chunk_idx, col_start, col_end, sub, dtype) -> int:
    from cytome.io.compression import compress_blob
    conn.execute(
        "INSERT INTO matrix_csc_chunks (matrix_name, chunk_idx, col_start, col_end, n_nonzero, "
        "data_blob, indices_blob, indptr_blob, dtype, compression) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (matrix_name, int(chunk_idx), int(col_start), int(col_end), int(sub.nnz),
         compress_blob(np.ascontiguousarray(sub.data, dtype=np.dtype(dtype)).tobytes(), "lz4"),
         compress_blob(sub.indices.astype(np.int32).tobytes(), "lz4"),
         compress_blob(sub.indptr.astype(np.int32).tobytes(), "lz4"),
         str(np.dtype(dtype)), "lz4"),
    )
    return int(sub.nnz)


def _write_csc_meta(ds, conn, matrix_name, n_rows, n_cols, total_nnz, dtype, col_entity, n_chunks):
    meta = ds._conn.execute(
        "SELECT csc_chunk_size, created_at FROM matrix_meta WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()
    try:
        conn.execute("ALTER TABLE matrix_meta ADD COLUMN layout TEXT DEFAULT 'csr'")
    except Exception:
        pass
    conn.execute("DELETE FROM matrix_meta WHERE matrix_name = ?", (matrix_name,))
    conn.execute(
        "INSERT INTO matrix_meta (matrix_name, n_rows, n_cols, n_nonzero, dtype, row_entity, col_entity, "
        "chunk_size, n_chunks, has_csc, csc_chunk_size, csc_n_chunks, created_at, layout) "
        "VALUES (?,?,?,?,?,'cells',?,128,0,1,?,?,?,'csc')",
        (matrix_name, int(n_rows), int(n_cols), int(total_nnz), str(np.dtype(dtype)), col_entity,
         meta[0] if meta else None, int(n_chunks), meta[1] if meta else None),
    )
    conn.commit()


def split(
    ds: CytomeDataset,
    labels,
    outputs: dict,
    matrices: "list[str]",
    include_fragments: bool = False,
    include_embeddings: bool = False,
    copy_annotations: bool = True,
):
    """Split the cells into one new store per group, reading each matrix once.

    ``labels`` holds one group label per cell; ``outputs`` maps each group to
    split off to the path of its store (cells of groups not in ``outputs``
    go nowhere). Each output is exactly what ``subset(ds, labels == group,
    output, matrices=matrices, ...)`` writes, cell order included.

    ``subset`` reads a whole column-major matrix for every group; this reads
    it once and hands every column chunk's rows to their groups' stores as
    it goes, so the cost of the matrices no longer grows with the number of
    groups. A row-major matrix is split group by group, as ``subset`` does.


    Returns the open output datasets, keyed by group. The caller closes
    them.
    """
    from cytome.core.measurement import MeasurementLayer
    ds.flush()
    labels = np.asarray(labels)
    if labels.shape[0] != ds.n_cells:
        raise ValueError(f"split: {labels.shape[0]} labels for {ds.n_cells} cells")
    groups = list(outputs)
    if not groups:
        return {}
    keys = labels.astype(str)
    gid = np.full(len(keys), -1, dtype=np.int64)
    keeps = []
    for g_i, g in enumerate(groups):
        m = keys == str(g)
        if not m.any():
            raise ValueError(f"split: no cell has the label {g!r}")
        gid[m] = g_i
        keeps.append(np.flatnonzero(m))
    rank = np.zeros(len(keys), dtype=np.int64)          # a cell's row in its group's store
    for keep in keeps:
        rank[keep] = np.arange(len(keep))

    stores = {g: _begin_output(ds, keeps[i], outputs[g], matrices, copy_annotations)
              for i, g in enumerate(groups)}
    mat_rows = {r[0]: r[1:] for r in ds._conn.execute(
        "SELECT matrix_name, n_cols, dtype, col_entity FROM matrix_meta").fetchall()}
    missing = sorted(set(matrices) - set(mat_rows))
    if missing:
        raise KeyError(f"split(matrices=...): not in the store: {missing}; available: {sorted(mat_rows)}")
    G = len(groups)
    for name in matrices:
        n_cols, dtype, col_entity = mat_rows[name]
        if not _is_column_major(ds._conn, name):
            for i, g in enumerate(groups):
                _subset_matrices_streaming(ds, stores[g], keeps[i], matrices=[name])
            continue
        n_cols = int(n_cols)
        for st in stores.values():
            st._conn.execute("DELETE FROM matrix_csc_chunks WHERE matrix_name = ?", (name,))
        totals = np.zeros(G, dtype=np.int64)
        n_chunks = 0
        for col_start, col_end, chunk in MeasurementLayer(ds._conn, name).iter_columns():
            c = int(col_end) - int(col_start)
            col = np.repeat(np.arange(c, dtype=np.int64), np.diff(chunk.indptr))
            g_of = gid[chunk.indices]
            # Entries grouped by their group, stably: within a group every column
            # keeps its entries in stored order, rows ascending as in subset.
            # A stable sort of small integers is a radix sort in numpy.
            order = np.argsort(g_of.astype(np.int16 if G < 32_000 else np.int64), kind="stable")
            bounds = np.searchsorted(g_of[order], np.arange(-1, G + 1))
            for i, g in enumerate(groups):
                o = order[bounds[i + 1]:bounds[i + 2]]
                indptr = np.zeros(c + 1, dtype=np.int64)
                np.cumsum(np.bincount(col[o], minlength=c), out=indptr[1:])
                sub = sp.csc_matrix((chunk.data[o], rank[chunk.indices[o]], indptr),
                                    shape=(len(keeps[i]), c))
                sub.sort_indices()
                totals[i] += _write_csc_chunk(stores[g]._conn, name, n_chunks, col_start, col_end,
                                              sub, dtype)
            n_chunks += 1
        for i, g in enumerate(groups):
            _write_csc_meta(ds, stores[g]._conn, name, len(keeps[i]), n_cols, int(totals[i]),
                            dtype, col_entity, n_chunks)
    for i, g in enumerate(groups):
        out = stores[g]
        modalities = set(out.modalities) | {m.split("_", 1)[0] for m in matrices if "_" in m}
        out._write_manifest_key("modalities", sorted(modalities))
        out._manifest = out._read_manifest()
        _finish_output(ds, out, keeps[i], outputs[g], include_embeddings, include_fragments,
                       copy_annotations)
    return stores


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


def _copy_feature_table(src_conn, dst_conn, table: str, batch: int = 50_000) -> int:
    """Copy a whole table from one store to another, as stored.

    Columns the destination lacks are added with the source's declared
    type. The rows are copied by SQLite itself, from the source file
    attached read-only to the destination connection: no Python object per
    row, and memory does not grow with the table. Where the source cannot be
    attached (an in-memory source), rows move in batches through Python. A
    table that is missing or empty in the source is left alone. Returns the
    number of rows copied.
    """
    try:
        src_cols = src_conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
        if not src_cols or not src_conn.execute(f"SELECT 1 FROM {_q(table)} LIMIT 1").fetchone():
            return 0
        dst_cols = dst_conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
    except Exception:
        return 0
    if not dst_cols:
        return 0                                  # not a table of this schema
    have = {c[1].lower() for c in dst_cols}
    for _cid, name, decl, *_rest in src_cols:
        if name.lower() not in have:
            dst_conn.execute(f"ALTER TABLE {_q(table)} ADD COLUMN {_q(name)} {decl or ''}".rstrip())
            have.add(name.lower())
    names = [c[1] for c in src_cols]
    collist = ", ".join(_q(n) for n in names)
    dst_conn.execute(f"DELETE FROM {_q(table)}")
    src_file = src_conn.execute("PRAGMA database_list").fetchone()[2]
    if src_file:
        dst_conn.commit()                         # ATTACH is refused inside a transaction
        # Attached by path: the connections are opened without URI support. It
        # is only read, by the same SQLite library as the source's connection.
        dst_conn.execute("ATTACH DATABASE ? AS cytome_src", (str(src_file),))
        try:
            cur = dst_conn.execute(
                f"INSERT INTO main.{_q(table)} ({collist}) "
                f"SELECT {collist} FROM cytome_src.{_q(table)} ORDER BY rowid")
            n = cur.rowcount
            dst_conn.commit()
        finally:
            dst_conn.execute("DETACH DATABASE cytome_src")
        return int(n)
    insert = f"INSERT INTO {_q(table)} ({collist}) VALUES ({','.join('?' * len(names))})"
    cur = src_conn.execute(f"SELECT {collist} FROM {_q(table)} ORDER BY rowid")
    n = 0
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        dst_conn.executemany(insert, rows)
        n += len(rows)
    dst_conn.commit()
    return n


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


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
    for table, label in (("graph_edges", "cell/var graphs"), ("graph_meta", "cell/var graphs")):
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
