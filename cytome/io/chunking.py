"""Chunked matrix assembly and cytome writing.

Builds a cell x feature matrix from a stream of ``(cell_idx, col_idx)`` hits
without ever holding the whole thing in memory: hits are bucketed into
per-chunk temp files, each chunk is assembled independently, and the chunks are
written straight into a cytome.

This code used to live in ``piaso.preprocessing._streaming_io``, where
``cytome.io.convert_fragments`` imported it -- so cytome depended on PIASO at
runtime without declaring it, and ``pip install cytome`` on its own produced a
package whose fragment-tiling path raised ImportError.

It belongs here. ``_write_chunks_to_cytome`` takes a cytome dataset and writes
cytome format, and in PIASO's public distribution nothing except cytome ever
used any of it. PIASO still uses these for fragment and peak quantification,
but the dependency now points one way: PIASO -> cytome.

Not exported in ``cytome.__all__`` -- this is shared internals, not user API.
The import path ``cytome.io.chunking`` and these symbol names are nonetheless
stable, because PIASO imports them; ``tests/test_public_api_contract.py`` pins
them for exactly that reason.
"""
from __future__ import annotations

import array
import math
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


MAX_CHUNK_FILES = 500
DEFAULT_CHUNK_SIZE = 2000


def _compute_chunk_params(n_cells: int) -> Tuple[int, int]:
    """Return ``(chunk_size, n_chunks)`` keeping *n_chunks* <= MAX_CHUNK_FILES."""
    chunk_size = max(DEFAULT_CHUNK_SIZE, math.ceil(n_cells / MAX_CHUNK_FILES))
    n_chunks = math.ceil(n_cells / chunk_size)
    return chunk_size, n_chunks


# ===================================================================
#  ChunkBucketWriter
# ===================================================================

class ChunkBucketWriter:
    """
    Buffered writer that routes (cell_idx, col_idx) hits to per-cell-chunk
    temp files.

    Each chunk covers *chunk_size* cells.  A hit for *cell_idx* goes to
    ``chunk_files[cell_idx // chunk_size]``.  Within each file hits are
    stored as ``(local_row, col)`` int32 pairs.

    File handles are opened lazily — chunks with no hits produce no file.

    Memory
    ------
    ``n_chunks x BUFFER_FLUSH x 8`` bytes worst case.
    250 chunks x 50 000 buffer ~ 100 MB.
    """

    BUFFER_FLUSH = 50_000  # pairs per chunk before flushing

    def __init__(self, n_chunks: int, chunk_size: int, tmpdir: str):
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.tmpdir = tmpdir
        self.buffers: List[array.array] = [
            array.array("i") for _ in range(n_chunks)
        ]
        self.handles: List[Optional[object]] = [None] * n_chunks

    def add(self, cell_idx: int, col_idx: int):
        """Route one hit to the appropriate chunk buffer."""
        chunk_id = cell_idx // self.chunk_size
        local_row = cell_idx % self.chunk_size
        buf = self.buffers[chunk_id]
        buf.append(local_row)
        buf.append(col_idx)
        if len(buf) >= self.BUFFER_FLUSH * 2:
            self._flush(chunk_id)

    def flush_arrays(self, cell_indices: np.ndarray, col_indices: np.ndarray):
        """Route numpy arrays of hits to chunk bucket files (vectorized).

        Partitions by chunk_id using argsort + searchsorted, then writes
        each partition directly to its chunk file.  10-50x faster than
        calling add() per hit.
        """
        if len(cell_indices) == 0:
            return

        chunk_ids = cell_indices // self.chunk_size
        local_rows = (cell_indices % self.chunk_size).astype(np.int32)
        col_arr = col_indices.astype(np.int32)

        order = np.argsort(chunk_ids, kind='stable')
        sorted_chunk_ids = chunk_ids[order]
        sorted_rows = local_rows[order]
        sorted_cols = col_arr[order]

        boundaries = np.searchsorted(sorted_chunk_ids,
                                     np.arange(self.n_chunks + 1))

        for cid in range(self.n_chunks):
            start, end = int(boundaries[cid]), int(boundaries[cid + 1])
            if start == end:
                continue
            pairs = np.empty(2 * (end - start), dtype=np.int32)
            pairs[0::2] = sorted_rows[start:end]
            pairs[1::2] = sorted_cols[start:end]
            if self.handles[cid] is None:
                path = os.path.join(self.tmpdir, f"chunk_{cid}.bin")
                self.handles[cid] = open(path, "ab")
            pairs.tofile(self.handles[cid])

    def close(self):
        """Flush all remaining buffers and close file handles."""
        for i in range(self.n_chunks):
            self._flush(i)
            if self.handles[i] is not None:
                self.handles[i].close()
                self.handles[i] = None

    def _flush(self, chunk_id: int):
        buf = self.buffers[chunk_id]
        if not buf:
            return
        if self.handles[chunk_id] is None:
            path = os.path.join(self.tmpdir, f"chunk_{chunk_id}.bin")
            self.handles[chunk_id] = open(path, "ab")
        np.array(buf, dtype=np.int32).tofile(self.handles[chunk_id])
        self.buffers[chunk_id] = array.array("i")


# ===================================================================
#  Chunk -> CSR
# ===================================================================

def _process_chunk(
    chunk_path: str,
    chunk_rows: int,
    n_cols: int,
    binary: bool,
) -> csr_matrix:
    """
    Read one chunk file and convert to CSR matrix.

    ``csr_matrix((data, (rows, cols)), shape=...)`` automatically sums
    duplicate ``(row, col)`` entries — multiple distinct fragments from the
    same cell overlapping the same feature are summed to give the count of
    distinct overlapping fragments.

    For *binary* mode the result is clipped to 0/1 afterwards.
    """
    if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
        return csr_matrix((chunk_rows, n_cols), dtype=np.float32)

    pairs = np.fromfile(chunk_path, dtype=np.int32).reshape(-1, 2)
    rows = pairs[:, 0]
    cols = pairs[:, 1]
    vals = np.ones(len(rows), dtype=np.float32)

    chunk_csr = csr_matrix(
        (vals, (rows, cols)), shape=(chunk_rows, n_cols), dtype=np.float32,
    )

    if binary:
        chunk_csr.data = np.minimum(chunk_csr.data, 1.0)

    return chunk_csr


# ===================================================================
#  CSR assembly — in RAM
# ===================================================================

def _assemble_csr(
    chunks: List[csr_matrix],
    n_rows: int,
    n_cols: int,
) -> csr_matrix:
    """
    Concatenate CSR chunks into a single matrix by splicing arrays.

    More efficient than ``scipy.sparse.vstack`` (no intermediate COO).
    """
    if not chunks:
        return csr_matrix((n_rows, n_cols), dtype=np.float32)

    total_nnz = sum(c.nnz for c in chunks)
    if total_nnz == 0:
        return csr_matrix((n_rows, n_cols), dtype=np.float32)

    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0

    nnz_off = 0
    row_off = 0

    for chunk in chunks:
        cr = chunk.shape[0]
        cn = chunk.nnz

        if cn > 0:
            data[nnz_off : nnz_off + cn] = chunk.data
            indices[nnz_off : nnz_off + cn] = chunk.indices

        indptr[row_off + 1 : row_off + cr + 1] = chunk.indptr[1:] + nnz_off

        nnz_off += cn
        row_off += cr

    return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


# ===================================================================
#  CSR assembly — direct to h5ad (constant memory)
# ===================================================================

def _parse_peak_metadata(var_names: List[str]) -> pd.DataFrame:
    """Parse peak names like 'chr1:100-200' into a DataFrame with chrom/start/end."""
    return _parse_feature_metadata(var_names, "peaks")


def _parse_feature_metadata(var_names: List[str], col_entity: str = "peaks") -> pd.DataFrame:
    """Parse feature names like 'chr1:100-200' into a DataFrame with chrom/start/end.

    Uses the correct ID column name based on col_entity (peak_id, tile_id, etc.).
    """
    id_col_map = {"peaks": "peak_id", "tiles": "tile_id"}
    id_col = id_col_map.get(col_entity, f"{col_entity.rstrip('s')}_id")

    chroms, starts, ends = [], [], []
    for name in var_names:
        try:
            chrom, coords = name.split(":", 1)
            s, e = coords.split("-")
            chroms.append(chrom)
            starts.append(int(s))
            ends.append(int(e))
        except (ValueError, IndexError):
            chroms.append("")
            starts.append(0)
            ends.append(0)
    return pd.DataFrame({
        id_col: var_names,
        "chr": chroms,
        "start": starts,
        "end_": ends,
    })


def _write_chunks_to_cytome(
    ds,
    chunks_dir: str,
    n_chunks: int,
    chunk_size: int,
    n_cells: int,
    n_cols: int,
    binary: bool,
    obs_names: List[str],
    var_names: List[str] = None,
    feature_df: "pd.DataFrame" = None,
    measurement: str = "counts",
    col_entity: str = "peaks",
    modality: str = "ATAC",
):
    """
    Write chunk files directly to Cytome.  Peak RAM = one chunk (~30 MB).

    Uses Cytome's ChunkedLayerWriter for zstd-compressed chunked storage.

    Parameters
    ----------
    ds : CytomeDataset
        Open dataset in read-write mode.
    chunks_dir : str
        Directory containing ``chunk_*.bin`` files.
    n_chunks : int
        Number of chunks.
    chunk_size : int
        Cells per chunk.
    n_cells : int
        Total cell count.
    n_cols : int
        Feature count (peaks or tiles).
    binary : bool
        Clip non-zero values to 1.
    obs_names : list of str
        Cell barcodes in row order.
    var_names : list of str, optional
        Feature names (peak coords or tile coords).  Parsed into a DataFrame
        via ``_parse_feature_metadata``.  Ignored when *feature_df* is given.
    feature_df : pd.DataFrame, optional
        Pre-built feature metadata DataFrame.  When provided, used directly
        instead of parsing *var_names*.  Avoids constructing + re-parsing
        5M+ tile name strings (~400 MB savings).
    measurement : str
        Name for the measurement layer (e.g., "counts", "tiles").
    col_entity : str
        Column entity type ("peaks" or "tiles").
    modality : str
        Modality prefix for the layer name (e.g., "ATAC", "tiles").
    """
    layer_name = f"{modality}_{measurement}"

    writer = ds.create_layer_writer(
        layer_name=layer_name,
        n_rows=n_cells,
        n_cols=n_cols,
        dtype=np.float32,
        compression="zstd",
        col_entity=col_entity,
        overwrite=True,
    )

    for chunk_id in range(n_chunks):
        chunk_rows = min(chunk_size, n_cells - chunk_id * chunk_size)
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}.bin")
        chunk_csr = _process_chunk(chunk_path, chunk_rows, n_cols, binary)
        writer.write_chunk(chunk_csr, row_offset=chunk_id * chunk_size)
        del chunk_csr

    # Write entity metadata BEFORE finalize() so that if the process is
    # interrupted between finalize (which commits matrix_meta) and flush,
    # the entity table is already consistent.  If crash after flush but
    # before finalize: matrix_meta absent → clean "not found" error.
    # If crash after finalize: both committed → consistent.

    # Write cell metadata if not already present
    if ds.n_cells == 0:
        ds.set_entity("cells", pd.DataFrame({"barcode": obs_names}))

    # Write peak/tile/feature metadata
    if feature_df is not None:
        ds.set_entity(col_entity, feature_df)
    else:
        if var_names is None:
            raise ValueError("Either var_names or feature_df must be provided")
        feat_df = _parse_feature_metadata(var_names, col_entity)
        ds.set_entity(col_entity, feat_df)

    ds.flush()

    # Finalize writes matrix_meta and commits — the last step, so a crash
    # before this leaves no matrix_meta (clean error) rather than a
    # matrix_meta/entity mismatch (cryptic error).
    writer.finalize()
