"""Fragment storage interface for Cytome."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import sqlite3

from cytome.io.compression import decompress_blob, decode_starts, decode_ends
from cytome.index.rtree import query_fragment_rtree
from cytome.io.convert_fragments import export_fragments, export_fragments_by_group, _validate_column


class FragmentStore:
    """Access fragment tables and coordinate queries."""

    def __init__(self, conn: sqlite3.Connection, dataset) -> None:
        self._conn = conn
        self._dataset = dataset

    @property
    def n_fragments(self) -> int:
        """Total number of fragments across chromosomes."""
        row = self._conn.execute("SELECT COALESCE(SUM(n_fragments), 0) FROM fragment_meta").fetchone()
        return int(row[0]) if row is not None else 0

    @property
    def chromosomes(self) -> List[str]:
        """Chromosomes with stored fragments."""
        rows = self._conn.execute(
            "SELECT chrom FROM fragment_meta WHERE n_fragments > 0 ORDER BY chrom"
        ).fetchall()
        return [str(r[0]) for r in rows]

    def query_region(self, chrom: str, start: int, end: int) -> Dict[str, np.ndarray]:
        """Query fragments overlapping one genomic region."""
        rows = query_fragment_rtree(self._conn, chrom, int(start), int(end))
        return _rows_to_fragment_dict(rows)

    def query_regions(self, regions_list: Sequence[Tuple[str, int, int]]) -> List[Dict[str, np.ndarray]]:
        """Query fragments for multiple regions."""
        return [self.query_region(chrom, start, end) for chrom, start, end in regions_list]

    def query_cells(self, cell_indices: Sequence[int], chrom: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Query fragments for selected cell indices."""
        if not cell_indices:
            return _rows_to_fragment_dict([])
        if chrom is not None:
            return self.query_cells_on_chrom(chrom, cell_indices)

        all_rows = []
        for c in self.chromosomes:
            rows = self._query_rows_by_cells(c, cell_indices)
            all_rows.extend(rows)
        all_rows.sort(key=lambda x: (x[4], x[0]))
        return _rows_to_fragment_dict([(s, e, ci, dc) for s, e, ci, dc, _chrom in all_rows])

    def query_cells_on_chrom(self, chrom: str, cell_indices: Sequence[int]) -> Dict[str, np.ndarray]:
        """Query fragments for selected cells on one chromosome."""
        rows = self._query_rows_by_cells(chrom, cell_indices)
        return _rows_to_fragment_dict([(s, e, ci, dc) for s, e, ci, dc, _ in rows])

    def export(
        self,
        output: str | Path,
        format: str = "10x",
        barcode_filter: Optional[Sequence[str]] = None,
        region: Optional[Tuple[str, int, int]] = None,
    ) -> Path:
        """Export fragments to disk."""
        return export_fragments(
            self._conn,
            output_path=output,
            cell_idx_to_name=self._cell_idx_to_name(),
            format=format,
            barcode_filter=barcode_filter,
            region=region,
        )

    def export_by_group(self, groupby: str, output_dir: str | Path, format: str = "10x"):
        """Export fragments to one file per grouping value."""
        _validate_column(self._conn, "cells", groupby)
        vals = self._conn.execute(
            f"SELECT DISTINCT {groupby} FROM cells WHERE {groupby} IS NOT NULL"
        ).fetchall()
        values = [str(v[0]) for v in vals]
        return export_fragments_by_group(
            self._conn,
            output_dir=output_dir,
            cell_idx_to_name=self._cell_idx_to_name(),
            groupby_column=groupby,
            groupby_values=values,
            cells_table="cells",
            format=format,
        )

    @property
    def has_compressed_chunks(self) -> bool:
        """Check if compressed fragment chunks exist."""
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM fragment_chunks"
            ).fetchone()
            return row is not None and row[0] > 0
        except sqlite3.OperationalError:
            return False

    def iter_chromosome_chunks(self, chrom: str) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (starts, ends, cell_idx) numpy arrays per compressed chunk.

        Each chunk contains ~500K fragments as int32 arrays.
        Handles encoding=0 (raw) and encoding=1 (delta/length).
        """
        rows = self._conn.execute(
            "SELECT starts_blob, ends_blob, cell_idx_blob, compression, "
            "COALESCE(encoding, 0) as encoding "
            "FROM fragment_chunks WHERE chrom = ? ORDER BY chunk_idx",
            (chrom,),
        ).fetchall()
        for starts_b, ends_b, cells_b, comp, enc in rows:
            starts = decode_starts(starts_b, comp, enc)
            ends = decode_ends(ends_b, comp, starts, enc)
            cells = np.frombuffer(decompress_blob(cells_b, comp), dtype=np.int32).copy()
            yield starts, ends, cells

    def iter_chromosomes_chunked(self) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (chrom, starts, ends, cell_idx) per chromosome from compressed chunks.

        Concatenates all chunks for each chromosome into single arrays.
        Peak RAM = one chromosome's fragments.
        """
        for chrom in self.chromosomes:
            all_starts, all_ends, all_cells = [], [], []
            for s, e, c in self.iter_chromosome_chunks(chrom):
                all_starts.append(s)
                all_ends.append(e)
                all_cells.append(c)
            if all_starts:
                yield (chrom,
                       np.concatenate(all_starts),
                       np.concatenate(all_ends),
                       np.concatenate(all_cells))

    def iter_chromosomes(self) -> Iterator[Tuple[str, Dict[str, np.ndarray]]]:
        """Iterate over complete chromosome fragment blocks.

        Uses compressed chunks if available, otherwise falls back to
        per-row SQLite tables.
        """
        if self.has_compressed_chunks:
            for chrom, starts, ends, cells in self.iter_chromosomes_chunked():
                yield chrom, {
                    "start": starts.astype(np.int64),
                    "end_": ends.astype(np.int64),
                    "cell_idx": cells.astype(np.int64),
                    "dup_count": np.ones(len(starts), dtype=np.int64),
                }
        else:
            for chrom in self.chromosomes:
                rows = self._conn.execute(
                    f"SELECT start, end_, cell_idx, dup_count FROM fragments_{chrom} ORDER BY start"
                ).fetchall()
                yield chrom, _rows_to_fragment_dict(rows)

    def _cell_idx_to_name(self) -> Dict[int, str]:
        rows = self._conn.execute("SELECT cell_idx, barcode FROM cells").fetchall()
        return {int(i): str(b) for i, b in rows if b is not None}

    def _query_rows_by_cells(self, chrom: str, cell_indices: Sequence[int]):
        if len(cell_indices) == 0:
            return []
        placeholders = ",".join("?" for _ in cell_indices)
        try:
            rows = self._conn.execute(
                f"""
                SELECT start, end_, cell_idx, dup_count
                FROM fragments_{chrom}
                WHERE cell_idx IN ({placeholders})
                ORDER BY start
                """,
                tuple(int(i) for i in cell_indices),
            ).fetchall()
            if rows:
                return [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), chrom) for r in rows]
        except sqlite3.OperationalError:
            pass

        # Fallback: scan compressed chunks (subset outputs with empty per-row tables)
        cell_arr = np.array(list(set(int(i) for i in cell_indices)), dtype=np.int32)
        results = []
        for starts, ends, cells in self.iter_chromosome_chunks(chrom):
            mask = np.isin(cells, cell_arr)
            if mask.any():
                for s, e, c in zip(starts[mask], ends[mask], cells[mask]):
                    results.append((int(s), int(e), int(c), 1, chrom))
        results.sort(key=lambda x: x[0])
        return results


def _rows_to_fragment_dict(rows) -> Dict[str, np.ndarray]:
    if not rows:
        empty_i = np.array([], dtype=np.int64)
        return {
            "start": empty_i,
            "end_": empty_i,
            "cell_idx": empty_i,
            "dup_count": empty_i,
        }
    start = np.array([r[0] for r in rows], dtype=np.int64)
    end_ = np.array([r[1] for r in rows], dtype=np.int64)
    cell_idx = np.array([r[2] for r in rows], dtype=np.int64)
    dup_count = np.array([r[3] for r in rows], dtype=np.int64)
    return {
        "start": start,
        "end_": end_,
        "cell_idx": cell_idx,
        "dup_count": dup_count,
    }
