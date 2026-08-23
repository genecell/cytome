# Cytome on-disk format specification (v1)

A `.cytome` file is a **plain SQLite 3 database**. Any language with a SQLite driver can read the
structural data directly; the count matrices and embeddings are stored as compressed chunk blobs
that a reader decodes with the small codec described below. This spec is the contract that keeps the
Python writer (`cytome`) and the R reader (`cytome`) in sync; the cross-language conformance test
(`tests/testthat/test-conformance.R`) enforces it.

## Versioning
The schema version lives in the `_schema_migrations` table. This document describes the layout used
by cytome ≥ the version that introduced `matrix_chunks` / `dense_chunks` (chunked, compressed CSR).

## Entity tables (plain SQL)
- `cells` — one row per cell. Columns include `cell_idx` (0-based row id), `barcode`, `sample_id`,
  and arbitrary obs columns. This is the obs / `colData`.
- `genes`, `peaks`, `tiles`, `samples`, `proteins` — feature/sample tables (var / `rowData`).
- Read with a single `SELECT *`.

## Matrices — `matrix_meta` + `matrix_chunks`
- `matrix_meta(matrix_name, n_rows, n_cols, dtype, row_entity, col_entity, chunk_size, n_chunks, …)`.
  Matrices are stored **cells × features** (`row_entity = "cells"`, `col_entity = "genes"`/`"peaks"`).
- `matrix_chunks(matrix_name, chunk_idx, row_start, row_end, n_nonzero, data_blob, indices_blob,
  indptr_blob, dtype, compression)` — one row per **CSR row-block** `[row_start, row_end)`:
  - `data_blob`    = the CSR `data` array (`float32`/`float64`) → numpy `.tobytes()`, then compressed.
  - `indices_blob` = the CSR `indices` array (column ids, `int32`, 0-based) → bytes → compressed.
  - `indptr_blob`  = the CSR `indptr` array (`int32`, length `= (row_end-row_start)+1`) → bytes → compressed.
  - `dtype` = value dtype; `compression` = `"zstd"` | `"lz4"` | `"zlib"` (see Codecs).
- **Reconstruction**: decode each chunk → its CSR row-block; stack chunks in `chunk_idx` order →
  the full `cells × features` CSR; transpose for a features × cells assay.

## Embeddings — `embedding_meta` + `dense_chunks`
- `embedding_meta(array_name, n_rows, n_cols, dtype, entity, chunk_size, n_chunks, …)`.
- `dense_chunks(array_name, chunk_idx, row_start, row_end, n_cols, data_blob, dtype, compression)` —
  one row per dense **row-major** block; `data_blob` = the `(row_end-row_start) × n_cols` values
  (row-major) → bytes → compressed.

## Fragments — `fragments_<chrom>` (+ R*-tree)
- Per-chromosome tables `fragments_chr1`, … with columns `rowid, start, end_, cell_idx, dup_count`,
  plus an SQLite R*-tree virtual table (`fragments_<chrom>_rtree`) for fast interval queries.

## Graphs / misc
- `graph_knn`, `graph_edges`, `graph_peak_gene` — KNN / eGRN / peak–gene links (plain SQL).
- `_manifest`, `_provenance`, `_metadata`, `_column_meta` — provenance and modality registry.

## Codecs (must match `cytome/io/compression.py`)
All three operate on the per-blob bytes:
- **`zstd`** — a zstd frame; the uncompressed size is in the frame header
  (`ZSTD_getFrameContentSize` → `ZSTD_decompress`).
- **`lz4`** — Python `lz4.block.compress(store_size=True)`: a **4-byte little-endian** uncompressed
  size header followed by a raw LZ4 **block** (decode: read header, then `LZ4_decompress_safe`).
  This is the LZ4 *block* format, **not** the LZ4 *frame* format that most generic lz4 libraries
  default to — hence cytome ships its own tiny block decoder.
- **`zlib`** — `zlib.compress()` (zlib-wrapped deflate, no stored size): grow the output buffer
  and `uncompress()` until it fits.

Endianness: arrays are little-endian (cytome runs on x86_64).

## Index dtypes & the 32-bit limit
`indices`/`indptr` are `int32` in practice. R's `dgCMatrix` is limited to ~2.1 billion nonzeros, so
the in-memory reader (`read_cytome_matrix`) is for cytomes that fit; larger stores use the streaming
path (`cytome_stream`, one chunk at a time) or, in a future version, a lazy/`DelayedArray` backend.

## Spatial images — `spatial_images` + `spatial_scalefactors` (format ≥ 0.2.6)

Two additive tables (absent in older files; readers must treat a missing
table as "no images"):

- `spatial_images(library_id, img_key, image BLOB, format, height, width,
  channels, dtype)`, PK `(library_id, img_key)`.
  - `format='raw-zstd'`: `image` = the C-contiguous array bytes compressed
    with zstd. Decode: zstd-decompress → interpret as `dtype` → reshape to
    `(height, width)` or `(height, width, channels)`. Exact for every dtype.
  - `format='png' | 'jpeg' | 'tiff'`: `image` = the source file's bytes,
    **verbatim** (no re-encode). `height`/`width` are parsed from the file
    header at write time; `channels=0` and `dtype=''` mean "unknown until
    decoded". Decoding requires an image library (Python: Pillow/tifffile;
    R: the `png`/`jpeg` packages). For TIFF only the first IFD's plane is
    described and decoded; pyramidal/OME multiscale is out of scope.
- `spatial_scalefactors(library_id, key, value REAL)`, PK
  `(library_id, key)` — the scanpy per-library scalefactor dict as rows
  (`tissue_<img_key>_scalef`, `spot_diameter_fullres`, and any custom keys;
  rows preserve keys the writer did not anticipate).

Spot coordinates are **not** stored here: they are the `spatial` embedding
(full-resolution pixel units). `image pixels = coords ×
tissue_<img_key>_scalef`.

### Coordinates index (same release)

`spatial_coords(cell_idx, x, y, z)` + the `spatial_rtree` R*-tree have been
part of the base schema; 0.2.6 adds the writer (`ds.set_spatial_coords`,
populated automatically by `from_anndata` when `obsm['spatial']` exists) and
the indexed range query (`ds.cells_in_region(x=, y=)`). Units are the same
full-resolution pixels as the `spatial` embedding, which remains the array
analysis and plotting consume — the table is its queryable index.
