"""Import/export 10x fragments.tsv.gz files."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import sqlite3

from cytome.io.compression import compress_blob
from cytome.index.builder import build_fragment_indices
from cytome.index.rtree import query_fragment_rtree

logger = logging.getLogger(__name__)

BATCH_SIZE = 100000


def import_fragments(
    conn: sqlite3.Connection,
    fragments_path: str | Path,
    cell_name_to_idx: Dict[str, int],
    build_index: bool = True,
    keep_chroms: str = "standard",
) -> None:
    """Import 10x fragments file into per-chromosome fragment tables.

    .. deprecated::
        This function uses the legacy per-row table format which is 10-50x
        slower and uses 10x more storage than the streaming chunk format.
        Use the Rust importer (``cytome-import-fragments``) or
        ``import_fragments_streaming()`` instead.

    Parameters
    ----------
    conn
        Open SQLite connection.
    fragments_path
        Path to fragments file (gz/bgz/plain).
    cell_name_to_idx
        Mapping from barcode string to cell index.
    build_index
        Whether to build R-tree indices after insertion.
    """
    import warnings
    warnings.warn(
        "import_fragments() uses the legacy per-row table format which is "
        "10-50x slower and uses 10x more storage than the streaming format. "
        "Use the Rust importer (cytome-import-fragments) or "
        "import_fragments_streaming() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = Path(fragments_path)
    records: Dict[str, List[Tuple[int, int, int, int]]] = {}

    from cytome.utils.genome import CHROM_ORDER
    _standard = (keep_chroms == "standard")
    _dropped_chroms: set = set()
    for chrom, start, end, barcode, dup_count in _iter_fragment_lines(path):
        if _standard and chrom not in CHROM_ORDER:   # skip scaffolds (would crash the per-chrom rtree)
            _dropped_chroms.add(chrom)
            continue
        cell_idx = cell_name_to_idx.get(barcode)
        if cell_idx is None:
            continue
        records.setdefault(chrom, []).append((start, end, cell_idx, dup_count))
    if _dropped_chroms:
        warnings.warn(
            f"import_fragments: skipped fragments on {len(_dropped_chroms)} non-standard chromosome(s) "
            f"(keep_chroms='standard'), e.g. {sorted(_dropped_chroms)[:3]}.", stacklevel=2)

    with conn:
        for chrom, rows in records.items():
            _ensure_fragment_tables(conn, chrom)
            rows.sort(key=lambda x: x[0])
            table = f"fragments_{chrom}"
            conn.execute(f"DELETE FROM {table}")
            for batch_start in range(0, len(rows), BATCH_SIZE):
                batch = rows[batch_start : batch_start + BATCH_SIZE]
                conn.executemany(
                    f"INSERT INTO {table}(start, end_, cell_idx, dup_count) VALUES (?, ?, ?, ?)",
                    batch,
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO fragment_meta(
                    chrom, n_fragments, table_name, rtree_name, min_start, max_end
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chrom,
                    len(rows),
                    table,
                    f"{table}_rtree",
                    min(r[0] for r in rows) if rows else None,
                    max(r[1] for r in rows) if rows else None,
                ),
            )

        if build_index:
            build_fragment_indices(conn)


def export_fragments(
    conn: sqlite3.Connection,
    output_path: str | Path,
    cell_idx_to_name: Dict[int, str],
    format: str = "10x",
    barcode_filter: Optional[Iterable[str]] = None,
    region: Optional[Tuple[str, int, int]] = None,
) -> Path:
    """Export fragments to 10x-format file.

    Parameters
    ----------
    conn
        Open SQLite connection.
    output_path
        Output path.
    cell_idx_to_name
        Mapping from cell index to barcode.
    format
        Output format. Only ``10x`` is supported.
    barcode_filter
        Optional barcode whitelist.
    region
        Optional ``(chrom, start, end)`` region constraint.
    """
    if format != "10x":
        raise ValueError("Only format='10x' is supported")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    barcode_filter_set = set(barcode_filter) if barcode_filter is not None else None

    try:
        import pysam  # type: ignore

        with pysam.BGZFile(str(out), "wb") as handle:
            for line in _iter_export_lines(conn, cell_idx_to_name, barcode_filter_set, region):
                handle.write(line.encode())

        try:
            pysam.tabix_index(str(out), preset="bed", force=True)
        except Exception as exc:  # pragma: no cover - index optional in CI
            logger.warning("Failed to build tabix index for %s: %s", out, exc)
    except ImportError:
        with gzip.open(out, "wt") as handle:
            for line in _iter_export_lines(conn, cell_idx_to_name, barcode_filter_set, region):
                handle.write(line)

    return out


def _validate_column(conn: sqlite3.Connection, table: str, column: str) -> None:
    """Validate that a column exists in a table to prevent SQL injection."""
    valid_cols = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in valid_cols:
        raise ValueError(
            f"Column '{column}' not found in table '{table}'. "
            f"Valid columns: {sorted(valid_cols)}"
        )


def export_fragments_by_group(
    conn: sqlite3.Connection,
    output_dir: str | Path,
    cell_idx_to_name: Dict[int, str],
    groupby_column: str,
    groupby_values: Sequence[str],
    cells_table: str = "cells",
    format: str = "10x",
) -> List[Path]:
    """Export fragments for each value in a cell grouping column."""
    _validate_column(conn, cells_table, groupby_column)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    for value in groupby_values:
        rows = conn.execute(
            f"SELECT cell_idx, barcode FROM {cells_table} WHERE {groupby_column} = ?",
            (value,),
        ).fetchall()
        barcodes = [r[1] for r in rows if r[1] is not None]
        out = output_root / f"{groupby_column}_{_safe_name(str(value))}.fragments.tsv.gz"
        out_paths.append(
            export_fragments(
                conn,
                out,
                cell_idx_to_name=cell_idx_to_name,
                format=format,
                barcode_filter=barcodes,
                region=None,
            )
        )
    return out_paths


def _iter_export_lines(
    conn: sqlite3.Connection,
    cell_idx_to_name: Dict[int, str],
    barcode_filter_set: Optional[set[str]],
    region: Optional[Tuple[str, int, int]],
):
    chroms = _chromosomes_with_data(conn)
    if region is not None:
        chroms = [region[0]] if region[0] in chroms else []

    # Subset cytomes (from subset/filter_cells) store fragments ONLY in the
    # compressed ``fragment_chunks`` table — there are no per-row
    # ``fragments_{chrom}`` tables. Detect the layout once and decode chunks
    # when needed (the region branch already handles chunks via the rtree).
    use_chunks = _has_fragment_chunks(conn)

    for chrom in sorted(chroms, key=_chrom_sort_key):
        if region is not None:
            rows = query_fragment_rtree(conn, chrom, int(region[1]), int(region[2]))
        elif use_chunks:
            rows = _chunk_export_rows(conn, chrom)
        else:
            rows = conn.execute(
                f"SELECT start, end_, cell_idx, dup_count FROM fragments_{chrom} ORDER BY start"
            ).fetchall()

        for start, end, cell_idx, dup_count in rows:
            barcode = cell_idx_to_name.get(int(cell_idx))
            if barcode is None:
                continue
            if barcode_filter_set is not None and barcode not in barcode_filter_set:
                continue
            yield f"{chrom}\t{int(start)}\t{int(end)}\t{barcode}\t{int(dup_count)}\n"


def _iter_fragment_lines(path: Path):
    opener = gzip.open if path.suffix in {".gz", ".bgz"} else open
    mode = "rt"
    with opener(path, mode) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            barcode = fields[3]
            dup_count = int(fields[4]) if len(fields) >= 5 else 1
            yield chrom, start, end, barcode, dup_count


def _ensure_fragment_tables(conn: sqlite3.Connection, chrom: str) -> None:
    table = f"fragments_{chrom}"
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            rowid INTEGER PRIMARY KEY,
            start INTEGER NOT NULL,
            end_ INTEGER NOT NULL,
            cell_idx INTEGER NOT NULL,
            dup_count INTEGER DEFAULT 1
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_cell ON {table}(cell_idx)")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {table}_rtree USING rtree(
            id,
            min_start, max_start,
            min_end, max_end
        )
        """
    )


def _chromosomes_with_data(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT chrom FROM fragment_meta WHERE n_fragments > 0").fetchall()
    return [r[0] for r in rows]


def _has_fragment_chunks(conn: sqlite3.Connection) -> bool:
    """True if fragments are stored in the compressed ``fragment_chunks`` table
    (the layout a subset/filter_cells cytome uses — no per-row tables)."""
    try:
        row = conn.execute("SELECT COUNT(*) FROM fragment_chunks").fetchone()
        return row is not None and row[0] > 0
    except sqlite3.OperationalError:
        return False


def _chunk_export_rows(conn: sqlite3.Connection, chrom: str):
    """Yield ``(start, end, cell_idx, dup_count)`` for a chromosome by decoding
    the compressed ``fragment_chunks`` (used when there are no per-row tables).
    Fragments are start-sorted to match the per-row export ordering (10x-style).
    """
    from cytome.io.compression import decompress_blob, decode_starts, decode_ends

    rows = conn.execute(
        "SELECT starts_blob, ends_blob, cell_idx_blob, compression, "
        "COALESCE(encoding, 0) FROM fragment_chunks WHERE chrom = ? ORDER BY chunk_idx",
        (chrom,),
    ).fetchall()
    parts_s, parts_e, parts_c = [], [], []
    for starts_b, ends_b, cells_b, comp, enc in rows:
        starts = decode_starts(starts_b, comp, enc)
        ends = decode_ends(ends_b, comp, starts, enc)
        cells = np.frombuffer(decompress_blob(cells_b, comp), dtype=np.int32).copy()
        parts_s.append(starts)
        parts_e.append(ends)
        parts_c.append(cells)
    if not parts_s:
        return
    s = np.concatenate(parts_s)
    e = np.concatenate(parts_e)
    c = np.concatenate(parts_c)
    order = np.argsort(s, kind="stable")
    for i in order:
        yield int(s[i]), int(e[i]), int(c[i]), 1


def _chrom_sort_key(chrom: str) -> Tuple[int, str]:
    if chrom.startswith("chr"):
        tail = chrom[3:]
        if tail.isdigit():
            return (int(tail), chrom)
        if tail == "X":
            return (23, chrom)
        if tail == "Y":
            return (24, chrom)
        if tail == "M":
            return (25, chrom)
    return (1000, chrom)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


# ---------------------------------------------------------------------------
# Streaming compressed fragment import
# ---------------------------------------------------------------------------

def import_fragments_streaming(
    conn: sqlite3.Connection,
    fragments_path: str | Path,
    cell_name_to_idx: Dict[str, int],
    chunk_size: int = 500_000,
    compression: str = "lz4",
    compression_level: int = 1,
    deduplicate: str = "auto",
    assume_sorted: bool = True,
    max_block_frags: int = 50_000_000,
    chr_filter: Optional[set] = None,
    verbose: bool = False,
    genome: Optional[str] = None,
    tile_sizes: Optional[List[int]] = None,
    tile_output_cytome=None,
    tile_binary: bool = False,
) -> Dict[str, int]:
    """Import fragments as compressed chunks with block-based streaming.

    .. note::
        The Rust importer (``cytome-import-fragments``) is 2-3x faster,
        supports multi-file k-way merge, and uses bounded memory.
        This Python function is kept as a fallback for environments
        without the Rust binary.

    v2.4: Processes in fixed-size blocks (default 50M fragments) instead
    of whole chromosomes. Peak RAM = one block (~700 MB) instead of one
    chromosome (~1.8 MB for chr2 on MouseCortex). Optionally computes
    tile matrices inline, eliminating the separate tile quantification step.

    Parameters
    ----------
    conn
        Open SQLite connection.
    fragments_path
        Path to fragments file (.tsv.gz, bgzip or gzip).
    cell_name_to_idx
        Mapping from barcode to cell index.
    chunk_size
        Fragments per compressed chunk (default 100K).
    compression
        Compression method (``"lz4"``, ``"zlib"``, or ``"zstd"``).
    compression_level
        Compression level (default 1 = fast, nearly same ratio for int32).
    deduplicate
        ``"auto"`` (check first block), ``"always"``, or ``"never"``.
    assume_sorted
        If True, skip sorting (verified on first block). 10x files are
        pre-sorted by position within each chromosome.
    max_block_frags
        Maximum fragments per block before flushing. Controls peak RAM.
        Default 50M = ~700 MB peak working set.
    chr_filter
        Optional set of chromosomes to keep. If None, keeps chr1-22 + chrX/Y.
    verbose
        Print progress.
    genome
        Optional genome name stored in cytome manifest.
    tile_sizes
        List of tile sizes to compute inline (e.g. ``[500]``).
        When provided, tile matrices are computed during import,
        eliminating the separate tile quantification step.
    tile_output_cytome
        CytomeDataset to write tile matrices to. Required when
        ``tile_sizes`` is not None.
    tile_binary
        Clip tile counts to 0/1.

    Returns
    -------
    dict
        ``{chrom: n_fragments}`` per chromosome.
    """
    path = Path(fragments_path)
    if chr_filter is None:
        chr_filter = set(f"chr{i}" for i in range(1, 23)) | {"chrX", "chrY"}

    # Enable WAL mode for faster writes during import
    _prev_journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Ensure encoding column exists (backward compat with older schemas)
    try:
        conn.execute("SELECT encoding FROM fragment_chunks LIMIT 0")
    except Exception:
        try:
            conn.execute(
                "ALTER TABLE fragment_chunks ADD COLUMN encoding INTEGER DEFAULT 0"
            )
        except Exception:
            pass  # table may not exist yet; schema creation handles it

    stats: Dict[str, int] = {}
    n_dropped_chr = 0

    # Auto-detection state (resolved on first block)
    _checked_sort = False
    _checked_dedup = False
    _do_dedup = deduplicate == "always"
    _do_sort = not assume_sorted

    # Block accumulation
    current_chrom: Optional[str] = None
    s_list: List[np.ndarray] = []
    e_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []
    block_count = 0
    block_chunk_idx: Dict[str, int] = {}  # {chrom: next_chunk_idx}

    # Incremental WAL checkpoint (v2.6): keep WAL < 2 GB
    _flush_count = 0
    _CHECKPOINT_INTERVAL = 5  # PASSIVE checkpoint every 5 block flushes

    # Tile computation state
    tile_writers = {}
    tile_col_offsets = {}
    tile_chrom_bin_info = {}
    tile_chrom_order = []
    tile_total_cols = {}
    tile_total_hits = {}

    if tile_sizes and tile_output_cytome is not None:
        from piaso.preprocessing._streaming_io import (
            _compute_chunk_params, ChunkBucketWriter,
        )
        n_cells = len(cell_name_to_idx)
        _tile_chunk_size, _tile_n_chunks = _compute_chunk_params(n_cells)

        for ts in tile_sizes:
            tmp = tempfile.mkdtemp(prefix=f"piaso_tile_{ts}_")
            tile_writers[ts] = ChunkBucketWriter(
                _tile_n_chunks, _tile_chunk_size, tmp
            )
            tile_col_offsets[ts] = {}
            tile_chrom_bin_info[ts] = {}
            tile_total_cols[ts] = 0
            tile_total_hits[ts] = 0

    # Per-cell fragment counter (post-dedup, written after import)
    n_cells = len(cell_name_to_idx)
    cell_frag_counts = np.zeros(n_cells, dtype=np.int64) if n_cells > 0 else None

    def _flush_block():
        nonlocal block_count, s_list, e_list, c_list, _flush_count
        nonlocal _checked_sort, _checked_dedup, _do_dedup, _do_sort
        nonlocal cell_frag_counts   # augmented-assigned below; must not be treated as local
        if not s_list:
            return

        starts = np.concatenate(s_list)
        ends = np.concatenate(e_list)
        cells = np.concatenate(c_list)

        # Auto-detect sort order (first block only)
        if not _checked_sort and assume_sorted:
            _checked_sort = True
            if not _verify_sorted(starts):
                _do_sort = True
                if verbose:
                    print("  [fragments] WARNING: Input not sorted, enabling sort")

        # Auto-detect duplicates (first block only)
        if not _checked_dedup and deduplicate == "auto":
            _checked_dedup = True
            n_dups = _count_duplicates_sample(starts, ends, cells)
            _do_dedup = n_dups > 0
            if verbose:
                if _do_dedup:
                    print(f"  [fragments] Auto-detected {n_dups} duplicates, "
                          f"enabling dedup")
                else:
                    print("  [fragments] No duplicates detected, skipping dedup")

        # Conditional sort
        if _do_sort:
            order = np.argsort(starts)
            starts = starts[order]
            ends = ends[order]
            cells = cells[order]
            del order

        # Conditional dedup
        if _do_dedup:
            starts, ends, cells = _deduplicate_arrays(starts, ends, cells, verbose)

        # Accumulate per-cell fragment counts (post-dedup)
        if cell_frag_counts is not None and len(cells) > 0:
            counts = np.bincount(cells.astype(np.intp), minlength=n_cells)
            cell_frag_counts += counts[:n_cells]

        # Write compressed chunks (append mode)
        ci = block_chunk_idx.get(current_chrom, 0)
        ci = _write_block_chunks(
            conn, current_chrom, starts, ends, cells,
            chunk_size, compression, compression_level, ci,
        )
        block_chunk_idx[current_chrom] = ci

        # Accumulate fragment count
        stats[current_chrom] = stats.get(current_chrom, 0) + len(starts)

        # Inline tile computation
        if tile_sizes:
            for ts in tile_sizes:
                _compute_tile_hits_inline(
                    starts, ends, cells, current_chrom, ts,
                    tile_writers[ts], tile_col_offsets[ts],
                    tile_chrom_bin_info[ts], tile_total_cols,
                    tile_total_hits, tile_chrom_order,
                )

        if verbose:
            print(f"  [fragments] {current_chrom}: flushed block "
                  f"({len(starts):,} fragments)")

        del starts, ends, cells
        s_list, e_list, c_list = [], [], []
        block_count = 0

        # Incremental WAL checkpoint (v2.6)
        _flush_count += 1
        if _flush_count % _CHECKPOINT_INTERVAL == 0:
            conn.commit()  # close open transaction before checkpoint
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            if verbose:
                print(f"  [fragments] WAL checkpoint (PASSIVE) after {_flush_count} flushes")

    # ── Main parsing loop ───────────────────────────────────────
    handle = _open_for_pandas(path)
    try:
        for chunk_df in pd.read_csv(
            handle, sep='\t', header=None,
            names=['chrom', 'start', 'end', 'barcode', 'dup'],
            usecols=[0, 1, 2, 3],
            dtype={'chrom': str, 'start': np.int32,
                   'end': np.int32, 'barcode': str},
            chunksize=500_000,
            comment='#',
        ):
            # Map barcodes to cell_idx (vectorized)
            chunk_df['cell_idx'] = chunk_df['barcode'].map(cell_name_to_idx)
            chunk_df = chunk_df.dropna(subset=['cell_idx'])
            if len(chunk_df) == 0:
                continue
            chunk_df['cell_idx'] = chunk_df['cell_idx'].astype(np.int32)

            # Filter chromosomes
            n_before = len(chunk_df)
            chunk_df = chunk_df[chunk_df['chrom'].isin(chr_filter)]
            n_dropped_chr += n_before - len(chunk_df)
            if len(chunk_df) == 0:
                continue

            # Group by chromosome and accumulate
            for chrom, grp in chunk_df.groupby('chrom', sort=False):
                if chrom != current_chrom:
                    _flush_block()  # flush remaining block for prev chrom
                    current_chrom = chrom

                s_list.append(grp['start'].values)
                e_list.append(grp['end'].values)
                c_list.append(grp['cell_idx'].values)
                block_count += len(grp)

                # Block size limit: flush mid-chromosome
                if block_count >= max_block_frags:
                    _flush_block()

    finally:
        if hasattr(handle, 'close'):
            handle.close()

    # Flush last block
    _flush_block()

    # Update fragment_meta for all chromosomes
    for chrom, n_frags in stats.items():
        _update_fragment_meta(conn, chrom, n_frags)

    # Store genome version in manifest if provided
    if genome:
        import json as _json
        try:
            conn.execute(
                "UPDATE _manifest SET value = ? WHERE key = 'genome'",
                (_json.dumps(genome),)
            )
        except sqlite3.OperationalError:
            pass

    conn.commit()

    # Write per-cell fragment counts to cells table
    if cell_frag_counts is not None:
        n_nonzero = int(np.count_nonzero(cell_frag_counts))
        if verbose:
            print(f"  [fragments] Writing per-cell fragment counts "
                  f"({n_nonzero} cells with fragments)")
        cursor = conn.cursor()
        for cell_idx in range(n_cells):
            count = int(cell_frag_counts[cell_idx])
            if count > 0:
                cursor.execute(
                    "UPDATE cells SET n_fragments = ? WHERE cell_idx = ?",
                    (count, cell_idx),
                )
        conn.commit()

    if verbose:
        total = sum(stats.values())
        print(f"  [fragments] Imported {total:,} fragments across "
              f"{len(stats)} chromosomes")
        if n_dropped_chr > 0:
            total_parsed = total + n_dropped_chr
            print(f"  [fragments] Filtered {n_dropped_chr:,} non-chr "
                  f"fragments ({n_dropped_chr / total_parsed * 100:.1f}%)")

    # Finalize tile matrices
    if tile_sizes and tile_output_cytome is not None:
        _finalize_tiles(
            tile_sizes, tile_writers, tile_col_offsets,
            tile_chrom_bin_info, tile_total_cols, tile_total_hits,
            tile_chrom_order, tile_output_cytome, tile_binary,
            cell_name_to_idx, verbose,
        )

    # Restore journal mode after import
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if _prev_journal.lower() != "wal":
        conn.execute(f"PRAGMA journal_mode={_prev_journal}")
    conn.execute("PRAGMA synchronous=FULL")

    return stats


def _open_for_pandas(path: Path):
    """Open fragment file for pd.read_csv. Returns file-like object.

    NOTE: pysam.BGZFile segfaults when passed to pd.read_csv's C parser,
    so we always use gzip.open (which can read bgzip files since bgzip
    is gzip-compatible).
    """
    if path.suffix in {'.gz', '.bgz'}:
        return gzip.open(path, 'rb')
    return open(path, 'rb')


def _verify_sorted(starts, n_check=100_000):
    """Check that the first n_check starts are non-decreasing."""
    n = min(len(starts), n_check)
    if n < 2:
        return True
    return not np.any(starts[:n - 1] > starts[1:n])


def _count_duplicates_sample(starts, ends, cells, n_sample=100_000):
    """Estimate duplicate count from a sample."""
    n = min(len(starts), n_sample)
    arr = np.stack([starts[:n], ends[:n], cells[:n]], axis=1)
    return n - len(np.unique(arr, axis=0))


def _deduplicate_arrays(starts, ends, cells, verbose=False):
    """Remove duplicate (start, end, cell_idx) triples."""
    arr = np.stack([starts, ends, cells], axis=1)
    unique_arr = np.unique(arr, axis=0)
    n_before = len(starts)
    if len(unique_arr) < n_before:
        if verbose:
            print(f"    Removed {n_before - len(unique_arr):,} duplicates")
        return unique_arr[:, 0].copy(), unique_arr[:, 1].copy(), unique_arr[:, 2].copy()
    return starts, ends, cells


def _write_block_chunks(
    conn, chrom, starts, ends, cells,
    chunk_size, compression, compression_level, chunk_idx_start,
    encoding=1,
):
    """Write compressed chunks for one block (append mode).

    encoding 0: raw int32 (legacy).
    encoding 1: delta-encoded starts, length-encoded ends (v2.5).

    Returns the next chunk_idx for this chromosome.
    """
    n_frags = len(starts)
    chunk_idx = chunk_idx_start
    row_offset = chunk_idx_start * chunk_size

    for i in range(0, n_frags, chunk_size):
        end_i = min(i + chunk_size, n_frags)
        chunk_s = starts[i:end_i]
        chunk_e = ends[i:end_i]
        chunk_c = cells[i:end_i]

        if encoding == 1:
            # Delta-encode starts: store diffs (first value = absolute position)
            deltas = np.diff(chunk_s, prepend=np.int32(0))
            s_blob = compress_blob(deltas.tobytes(), compression,
                                   level=compression_level)
            # Length-encode ends: store (end - start)
            lengths = (chunk_e - chunk_s).astype(np.int32)
            e_blob = compress_blob(lengths.tobytes(), compression,
                                   level=compression_level)
        else:
            s_blob = compress_blob(chunk_s.tobytes(), compression,
                                   level=compression_level)
            e_blob = compress_blob(chunk_e.tobytes(), compression,
                                   level=compression_level)

        c_blob = compress_blob(chunk_c.tobytes(), compression,
                               level=compression_level)
        min_start = int(chunk_s[0]) if len(chunk_s) > 0 else 0
        conn.execute(
            """INSERT INTO fragment_chunks
               (chrom, chunk_idx, row_start, row_end, n_fragments,
                min_start, starts_blob, ends_blob, cell_idx_blob,
                compression, encoding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chrom, chunk_idx, row_offset + i, row_offset + end_i,
             end_i - i, min_start, s_blob, e_blob, c_blob,
             compression, encoding),
        )
        chunk_idx += 1

    return chunk_idx


def _update_fragment_meta(conn, chrom, n_frags):
    """Update fragment_meta after all blocks for a chromosome are written."""
    # Get min_start and max_end from the stored chunks
    row = conn.execute(
        "SELECT MIN(row_start), MAX(row_end) FROM fragment_chunks WHERE chrom = ?",
        (chrom,),
    ).fetchone()

    # Get actual coordinate bounds from first and last chunks
    first_chunk = conn.execute(
        "SELECT starts_blob, compression, COALESCE(encoding, 0) "
        "FROM fragment_chunks "
        "WHERE chrom = ? ORDER BY chunk_idx ASC LIMIT 1",
        (chrom,),
    ).fetchone()
    last_chunk = conn.execute(
        "SELECT starts_blob, ends_blob, compression, COALESCE(encoding, 0) "
        "FROM fragment_chunks "
        "WHERE chrom = ? ORDER BY chunk_idx DESC LIMIT 1",
        (chrom,),
    ).fetchone()

    min_start = None
    max_end = None
    if first_chunk:
        from cytome.io.compression import decode_starts
        starts_arr = decode_starts(first_chunk[0], first_chunk[1], first_chunk[2])
        if len(starts_arr) > 0:
            min_start = int(starts_arr[0])
    if last_chunk:
        from cytome.io.compression import decode_starts, decode_ends
        last_starts = decode_starts(last_chunk[0], last_chunk[2], last_chunk[3])
        last_ends = decode_ends(last_chunk[1], last_chunk[2], last_starts, last_chunk[3])
        if len(last_ends) > 0:
            max_end = int(last_ends[-1])

    conn.execute(
        """INSERT OR REPLACE INTO fragment_meta
           (chrom, n_fragments, table_name, rtree_name, min_start, max_end)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chrom, n_frags,
         f"fragment_chunks:{chrom}", f"fragments_{chrom}_rtree",
         min_start, max_end),
    )


def _compute_tile_hits_inline(
    starts, ends, cells, chrom, tile_size,
    writer, col_offsets, chrom_bin_info, total_cols_dict,
    total_hits_dict, chrom_order,
):
    """Compute tile hits for one block and route to ChunkBucketWriter."""
    bin_starts = starts // tile_size
    bin_ends = (ends - 1) // tile_size

    # Register new chromosome if first block for this chrom+tile_size
    if chrom not in col_offsets:
        min_bin = int(bin_starts.min())
        max_bin = int(bin_ends.max())
        n_bins = max_bin - min_bin + 1
        col_offsets[chrom] = (total_cols_dict.get(tile_size, 0), min_bin)
        chrom_bin_info[chrom] = [min_bin, max_bin, n_bins]
        if chrom not in [c for c in chrom_order if isinstance(c, str) and c == chrom]:
            chrom_order.append((chrom, tile_size))
        total_cols_dict[tile_size] = total_cols_dict.get(tile_size, 0) + n_bins
    else:
        # Update max_bin if this block extends beyond previous blocks
        info = chrom_bin_info[chrom]
        new_max = int(bin_ends.max())
        if new_max > info[1]:
            old_n_bins = info[2]
            info[1] = new_max
            info[2] = new_max - info[0] + 1
            added = info[2] - old_n_bins
            total_cols_dict[tile_size] = total_cols_dict.get(tile_size, 0) + added

    col_offset, min_bin = col_offsets[chrom]

    # Single-tile fragments (majority, fully vectorized)
    single = bin_starts == bin_ends
    if single.any():
        s_cells = cells[single]
        s_cols = (bin_starts[single] - min_bin + col_offset).astype(np.int32)
        writer.flush_arrays(s_cells, s_cols)
        total_hits_dict[tile_size] = total_hits_dict.get(tile_size, 0) + len(s_cells)

    # Multi-tile fragments (expand)
    multi_mask = ~single
    if multi_mask.any():
        m_cells = cells[multi_mask]
        m_bin_starts = bin_starts[multi_mask]
        m_bin_ends = bin_ends[multi_mask]
        m_rows = []
        m_cols = []
        for ci, bs, be in zip(m_cells, m_bin_starts, m_bin_ends):
            bins_range = np.arange(int(bs), int(be) + 1)
            m_rows.append(np.full(len(bins_range), ci, dtype=np.int32))
            m_cols.append((bins_range - min_bin + col_offset).astype(np.int32))
        if m_rows:
            m_r = np.concatenate(m_rows)
            m_c = np.concatenate(m_cols)
            writer.flush_arrays(m_r, m_c)
            total_hits_dict[tile_size] = total_hits_dict.get(tile_size, 0) + len(m_r)


def _finalize_tiles(
    tile_sizes, tile_writers, tile_col_offsets,
    tile_chrom_bin_info, tile_total_cols, tile_total_hits,
    tile_chrom_order, tile_output_cytome, tile_binary,
    cell_name_to_idx, verbose,
):
    """Finalize tile matrices: close writers, build metadata, write to cytome."""
    import pandas as pd
    from piaso.preprocessing._streaming_io import (
        _compute_chunk_params, _write_chunks_to_cytome,
    )

    n_cells = len(cell_name_to_idx)
    chunk_size, n_chunks = _compute_chunk_params(n_cells)
    ds = tile_output_cytome

    obs_names = [r[0] for r in ds._conn.execute(
        "SELECT barcode FROM cells ORDER BY cell_idx"
    ).fetchall()]

    for ts in tile_sizes:
        tile_writers[ts].close()
        n_tiles = tile_total_cols[ts]

        # Build feature DataFrame
        tile_ids = []
        chroms_list = []
        starts_list = []
        ends_list = []

        # Iterate chroms in order they were seen
        seen = set()
        for entry in tile_chrom_order:
            if isinstance(entry, tuple):
                chrom, t = entry
                if t != ts:
                    continue
            else:
                chrom = entry
            if chrom in seen:
                continue
            seen.add(chrom)
            if chrom not in tile_chrom_bin_info[ts]:
                continue
            info = tile_chrom_bin_info[ts][chrom]
            min_bin = info[0]
            n_bins = info[2]
            for b in range(min_bin, min_bin + n_bins):
                tile_ids.append(f"{chrom}:{b * ts + 1}-{(b + 1) * ts}")
                chroms_list.append(chrom)
                starts_list.append(b * ts + 1)
                ends_list.append((b + 1) * ts)

        feature_df = pd.DataFrame({
            "tile_id": tile_ids,
            "chr": chroms_list,
            "start": starts_list,
            "end_": ends_list,
        })

        _write_chunks_to_cytome(
            ds=ds,
            chunks_dir=tile_writers[ts].tmpdir,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            n_cells=n_cells,
            n_cols=n_tiles,
            binary=tile_binary,
            obs_names=obs_names,
            var_names=None,
            feature_df=feature_df,
            measurement="counts",
            col_entity="tiles",
            modality="tiles",
        )

        shutil.rmtree(tile_writers[ts].tmpdir, ignore_errors=True)

        if verbose:
            print(f"  [tiles] size={ts}: {n_tiles:,} tiles, "
                  f"{tile_total_hits[ts]:,} hits")


# Legacy wrapper for backward compat
def _write_chrom_chunks(
    conn, chrom, starts, ends, cells,
    chunk_size, compression, deduplicate, verbose,
):
    """Sort, deduplicate, and write compressed chunks for one chromosome.
    Legacy wrapper for backward compatibility.
    """
    order = np.argsort(starts)
    starts = starts[order]; ends = ends[order]; cells = cells[order]

    if deduplicate and len(starts) > 0:
        starts, ends, cells = _deduplicate_arrays(starts, ends, cells, verbose)

    conn.execute("DELETE FROM fragment_chunks WHERE chrom = ?", (chrom,))

    n_frags = len(starts)
    _write_block_chunks(conn, chrom, starts, ends, cells,
                        chunk_size, compression, 6, 0)

    conn.execute(
        """INSERT OR REPLACE INTO fragment_meta
           (chrom, n_fragments, table_name, rtree_name, min_start, max_end)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chrom, n_frags,
         f"fragment_chunks:{chrom}", f"fragments_{chrom}_rtree",
         int(starts[0]) if n_frags > 0 else None,
         int(ends[-1]) if n_frags > 0 else None),
    )
