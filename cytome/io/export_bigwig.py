"""Pseudo-bulk coverage export to BigWig/bedGraph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from cytome.utils.genome import get_chrom_sizes

logger = logging.getLogger(__name__)


def export_coverage(
    dataset,
    groupby: str,
    output_dir: str | Path,
    format: str = "bigwig",
    normalize: str = "cpm",
    bin_size: int = 10,
    region: Optional[Tuple[str, int, int]] = None,
):
    """Export pseudo-bulk insertion coverage per group.

    Parameters
    ----------
    dataset
        Open ``CytomeDataset``.
    groupby
        Column name in cells table.
    output_dir
        Output directory.
    format
        ``bigwig`` or ``bedgraph``.
    normalize
        One of ``cpm``, ``rpkm``, ``raw``.
    bin_size
        Genomic bin size.
    region
        Optional region constraint.
    """
    if format not in {"bigwig", "bedgraph"}:
        raise ValueError("format must be 'bigwig' or 'bedgraph'")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    groups = dataset.cells.to_pandas()[groupby].dropna().unique().tolist()
    out_paths = []
    for group in groups:
        out_paths.append(
            _export_group_coverage(
                dataset=dataset,
                groupby=groupby,
                group_name=str(group),
                output_dir=output_root,
                format=format,
                normalize=normalize,
                bin_size=bin_size,
                region=region,
            )
        )
    return out_paths


def export_coverage_region(
    dataset,
    groupby: str,
    region: Tuple[str, int, int],
    output_dir: str | Path,
    format: str = "bigwig",
    normalize: str = "cpm",
    bin_size: int = 10,
):
    """Export pseudo-bulk coverage restricted to one region."""
    return export_coverage(
        dataset=dataset,
        groupby=groupby,
        output_dir=output_dir,
        format=format,
        normalize=normalize,
        bin_size=bin_size,
        region=region,
    )


def cache_coverage(dataset, groupby: str, bin_size: int, normalize: str) -> None:
    """Cache binned coverage arrays in ``coverage_cache`` table."""
    groups = dataset.cells.to_pandas()[groupby].dropna().unique().tolist()
    chrom_sizes = _dataset_chrom_sizes(dataset)
    with dataset._conn:
        for group in groups:
            values_by_chrom = _compute_group_coverage(
                dataset=dataset,
                groupby=groupby,
                group_name=str(group),
                bin_size=bin_size,
                normalize=normalize,
                region=None,
            )
            for chrom, values in values_by_chrom.items():
                if chrom not in chrom_sizes:
                    continue
                dataset._conn.execute(
                    """
                    INSERT OR REPLACE INTO coverage_cache(
                        group_name, groupby_key, chrom, bin_size, normalize, values_blob
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(group),
                        groupby,
                        chrom,
                        int(bin_size),
                        normalize,
                        values.astype(np.float32).tobytes(),
                    ),
                )


def get_cached_coverage(
    dataset,
    group_name: str,
    chrom: str,
    bin_size: int,
    normalize: str,
) -> np.ndarray:
    """Fetch cached coverage values for one group and chromosome."""
    row = dataset._conn.execute(
        """
        SELECT values_blob FROM coverage_cache
        WHERE group_name = ? AND chrom = ? AND bin_size = ? AND normalize = ?
        """,
        (group_name, chrom, int(bin_size), normalize),
    ).fetchone()
    if row is None:
        raise KeyError(
            f"No cached coverage for group={group_name}, chrom={chrom}, "
            f"bin_size={bin_size}, normalize={normalize}"
        )
    return np.frombuffer(row[0], dtype=np.float32)


def _export_group_coverage(
    dataset,
    groupby: str,
    group_name: str,
    output_dir: Path,
    format: str,
    normalize: str,
    bin_size: int,
    region: Optional[Tuple[str, int, int]],
) -> Path:
    values_by_chrom = _compute_group_coverage(
        dataset=dataset,
        groupby=groupby,
        group_name=group_name,
        bin_size=bin_size,
        normalize=normalize,
        region=region,
    )
    suffix = "bw" if format == "bigwig" else "bedgraph"
    out = output_dir / f"{groupby}_{_safe_name(group_name)}.{suffix}"
    if format == "bigwig":
        _write_bigwig(out, values_by_chrom, bin_size, dataset)
    else:
        _write_bedgraph(out, values_by_chrom, bin_size)
    return out


def _compute_group_coverage(
    dataset,
    groupby: str,
    group_name: str,
    bin_size: int,
    normalize: str,
    region: Optional[Tuple[str, int, int]],
) -> Dict[str, np.ndarray]:
    df = dataset.cells.query(f"{groupby} == '{group_name}'")
    if df.empty:
        return {}
    cell_indices = df["cell_idx"].to_numpy(dtype=np.int64)
    cell_set = set(cell_indices.tolist())

    chrom_sizes = _dataset_chrom_sizes(dataset)
    if region is not None:
        chrom_sizes = {region[0]: chrom_sizes.get(region[0], region[2])}

    # Check which fragment access method is available
    use_chunks = _has_fragment_chunks(dataset)

    out: Dict[str, np.ndarray] = {}
    total_fragments = 0
    for chrom, size in chrom_sizes.items():
        bins = int(np.ceil(size / bin_size))
        coverage = np.zeros(bins, dtype=np.float64)
        if region is not None and chrom != region[0]:
            out[chrom] = coverage
            continue

        if use_chunks:
            # v2.7+ cytomes: use fragment_chunks (one-pass, all fragments)
            from cytome.io.compression import decompress_blob, decode_starts, decode_ends
            chunk_rows = dataset._conn.execute(
                "SELECT starts_blob, ends_blob, cell_idx_blob, compression, "
                "COALESCE(encoding, 0) as encoding "
                "FROM fragment_chunks WHERE chrom = ? ORDER BY chunk_idx",
                (chrom,),
            ).fetchall()

            for starts_b, ends_b, cells_b, comp, enc in chunk_rows:
                starts = decode_starts(starts_b, comp, enc)
                ends = decode_ends(ends_b, comp, starts, enc)
                cells = np.frombuffer(decompress_blob(cells_b, comp), dtype=np.int32).copy()
                if len(starts) == 0:
                    continue

                # Filter to group cells
                mask = np.array([c in cell_set for c in cells], dtype=bool)
                if not mask.any():
                    continue

                s = starts[mask]
                e = ends[mask]
                total_fragments += len(s)

                start_bins = np.clip(s // bin_size, 0, bins - 1)
                end_bins = np.clip(e // bin_size, 0, bins - 1)
                np.add.at(coverage, start_bins, 1)
                np.add.at(coverage, end_bins, 1)
        else:
            # Legacy: per-row fragments_{chrom} tables
            frag = dataset.ATAC.fragments.query_cells_on_chrom(chrom, cell_indices)
            if frag["start"].size == 0:
                out[chrom] = coverage
                continue

            starts = frag["start"]
            ends = frag["end_"]
            total_fragments += starts.size
            start_bins = np.clip(starts // bin_size, 0, bins - 1)
            end_bins = np.clip(ends // bin_size, 0, bins - 1)
            np.add.at(coverage, start_bins, 1)
            np.add.at(coverage, end_bins, 1)

        if region is not None:
            r0, r1 = int(region[1] // bin_size), int(np.ceil(region[2] / bin_size))
            mask_r = np.zeros_like(coverage)
            mask_r[r0:r1] = 1
            coverage *= mask_r

        out[chrom] = coverage

    if normalize == "raw":
        return {k: v.astype(np.float32) for k, v in out.items()}

    if total_fragments <= 0:
        return {k: v.astype(np.float32) for k, v in out.items()}

    scale = total_fragments / 1_000_000.0
    normed = {k: (v / scale) for k, v in out.items()}
    if normalize == "rpkm":
        bin_kb = bin_size / 1000.0
        normed = {k: (v / bin_kb) for k, v in normed.items()}
    elif normalize != "cpm":
        raise ValueError("normalize must be one of 'raw', 'cpm', 'rpkm'")

    return {k: v.astype(np.float32) for k, v in normed.items()}


def _has_fragment_chunks(dataset) -> bool:
    """Check if cytome has fragment_chunks table (v2.7+ Rust-imported)."""
    try:
        row = dataset._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='fragment_chunks'"
        ).fetchone()
        return row[0] > 0
    except Exception:
        return False


def _dataset_chrom_sizes(dataset) -> Dict[str, int]:
    genome = dataset._manifest.get("genome") or "hg38"
    try:
        return get_chrom_sizes(genome)
    except Exception:
        return get_chrom_sizes("hg38")


def _write_bigwig(path: Path, values_by_chrom: Dict[str, np.ndarray], bin_size: int, dataset) -> None:
    try:
        import pyBigWig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyBigWig is required for BigWig export. Install with `pip install pyBigWig`."
        ) from exc

    chrom_sizes = _dataset_chrom_sizes(dataset)
    with pyBigWig.open(str(path), "w") as bw:
        header = [(chrom, int(chrom_sizes.get(chrom, arr.size * bin_size))) for chrom, arr in values_by_chrom.items()]
        bw.addHeader(header)
        for chrom, values in values_by_chrom.items():
            nonzero = np.flatnonzero(values)
            if nonzero.size == 0:
                continue
            starts = (nonzero * bin_size).astype(np.int64)
            ends = ((nonzero + 1) * bin_size).astype(np.int64)
            bw.addEntries(
                [chrom] * nonzero.size,
                starts.tolist(),
                ends=ends.tolist(),
                values=values[nonzero].astype(float).tolist(),
            )


def _write_bedgraph(path: Path, values_by_chrom: Dict[str, np.ndarray], bin_size: int) -> None:
    with open(path, "wt") as handle:
        for chrom, values in values_by_chrom.items():
            nonzero = np.flatnonzero(values)
            for idx in nonzero:
                start = int(idx * bin_size)
                end = int((idx + 1) * bin_size)
                handle.write(f"{chrom}\t{start}\t{end}\t{float(values[idx])}\n")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
