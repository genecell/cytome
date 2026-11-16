"""Base-pair Tn5 cut-site footprinting & fragment-length-stratified insertion tracks.

Provides :func:`get_tn5_footprints`, which extracts exact 5' (`start + shift[0]`) and 3'
(`end_ - 1 + shift[1]`) Tn5 transposition events across one or thousands of genomic regions in a
single sweeping-window pass per chromosome (decompressing each overlapping `fragment_chunks` row at
most once).
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from cytome.core._jit import HAS_NUMBA as _HAS_NUMBA, njit_cached
from cytome.core.pseudobulk import _build_cell_grouping
from cytome.io.compression import decode_ends, decode_starts, decompress_blob


if _HAS_NUMBA:
    @njit_cached(nogil=True)
    def _accum_cuts_into_window_numba(
        cuts: np.ndarray,
        groups: np.ndarray,
        reg_start: int,
        reg_end: int,
        bin_size: int,
        out_mat: np.ndarray,
    ) -> None:
        n = len(cuts)
        for i in range(n):
            pos = cuts[i]
            if pos < reg_start or pos >= reg_end:
                continue
            g = groups[i]
            if g < 0:
                continue
            b = (pos - reg_start) // bin_size
            out_mat[g, b] += 1.0


def _accum_cuts_into_window_fallback(
    cuts: np.ndarray,
    groups: np.ndarray,
    reg_start: int,
    reg_end: int,
    bin_size: int,
    out_mat: np.ndarray,
) -> None:
    mask = (cuts >= reg_start) & (cuts < reg_end) & (groups >= 0)
    if not np.any(mask):
        return
    b = (cuts[mask] - reg_start) // bin_size
    g = groups[mask]
    n_bins = out_mat.shape[1]
    flat = g.astype(np.int64) * n_bins + b.astype(np.int64)
    out_mat.ravel()[:] += np.bincount(flat, minlength=out_mat.size).astype(np.float32)


_REGION_STR_RE = re.compile(r"^([^:]+):(\d+)[-\s](\d+)$")


def _parse_regions(ds, regions, window: Optional[int] = None) -> list[tuple[str, int, int, Optional[int]]]:
    """Normalize `regions` into `[(chrom, start, end, peak_idx_or_None), ...]`."""
    out: list[tuple[str, int, int, Optional[int]]] = []
    if isinstance(regions, str):
        regions = [regions]
    elif isinstance(regions, tuple) and len(regions) == 3 and isinstance(regions[0], str):
        regions = [regions]

    if isinstance(regions, pd.DataFrame):
        c_col = "chr" if "chr" in regions.columns else "chrom"
        e_col = "end_" if "end_" in regions.columns else "end"
        for row in regions.itertuples():
            ch = str(getattr(row, c_col))
            s = int(getattr(row, "start"))
            e = int(getattr(row, e_col))
            pidx = int(getattr(row, "peak_idx")) if hasattr(row, "peak_idx") else None
            out.append((ch, s, e, pidx))
    else:
        arr = list(regions)
        if not arr:
            return []
        if isinstance(arr[0], (int, np.integer)):
            # Peak indices! Fetch from ds.peaks
            conn = ds._conn
            p_cols = {r[1] for r in conn.execute("PRAGMA table_info(peaks)")}
            s_col = "summit" if "summit" in p_cols else ("summit_offset" if "summit_offset" in p_cols else "NULL")
            placeholders = ",".join("?" for _ in arr)
            rows = conn.execute(
                f"SELECT peak_idx, chr, start, end_, {s_col} FROM peaks WHERE peak_idx IN ({placeholders})",
                [int(x) for x in arr],
            ).fetchall()
            by_pidx = {int(r[0]): r for r in rows}
            for x in arr:
                xi = int(x)
                if xi not in by_pidx:
                    raise KeyError(f"peak_idx {xi} not found in peaks table")
                _, ch, s, e, sm = by_pidx[xi]
                s_i, e_i = int(s), int(e)
                if window is not None:
                    if sm is not None:
                        sv = int(sm)
                        center = (s_i + sv) if (0 <= sv <= (e_i - s_i)) else (sv if s_i <= sv <= e_i else (s_i + e_i) // 2)
                    else:
                        center = (s_i + e_i) // 2
                    half = int(window) // 2
                    s_i = center - half
                    e_i = s_i + int(window)
                out.append((str(ch), s_i, e_i, xi))
            return out

        for item in arr:
            if isinstance(item, str):
                m = _REGION_STR_RE.match(item.strip().replace(",", ""))
                if not m:
                    raise ValueError(f"Cannot parse region string {item!r}; expected 'chr:start-end'")
                ch, s, e = m.group(1), int(m.group(2)), int(m.group(3))
                out.append((ch, s, e, None))
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                out.append((str(item[0]), int(item[1]), int(item[2]), int(item[3]) if len(item) > 3 else None))
            else:
                raise ValueError(f"Unsupported region specification: {item!r}")

    if window is not None:
        w = int(window)
        half = w // 2
        centered = []
        for ch, s, e, pidx in out:
            mid = (s + e) // 2
            centered.append((ch, mid - half, mid - half + w, pidx))
        return centered
    return out


# Empirical sequence-driven Tn5 6-mer cleavage propensity log-weights around the insertion site
# Position weights for GC vs AT preference at [-2, -1, 0, +1, +2, +3] around the cut site
_TN5_POS_BASE_WEIGHTS = np.array(
    [
        # A,     C,     G,     T,    N
        [0.85, 1.15, 1.25, 0.75, 1.0],  # -2
        [1.10, 0.90, 1.15, 0.85, 1.0],  # -1
        [0.70, 1.35, 1.45, 0.65, 1.0],  #  0 (cut site)
        [0.65, 1.45, 1.35, 0.70, 1.0],  # +1
        [0.85, 1.15, 0.90, 1.10, 1.0],  # +2
        [0.75, 1.25, 1.15, 0.85, 1.0],  # +3
    ],
    dtype=np.float32,
)


def _compute_sequence_tn5_bias(codes_2d: np.ndarray) -> np.ndarray:
    """Compute relative Tn5 sequence cleavage bias `(N_regions, W)` from base codes `0..4`."""
    n_reg, w = codes_2d.shape
    padded = np.full((n_reg, w + 5), 4, dtype=np.uint8)
    padded[:, 2 : 2 + w] = codes_2d
    bias = np.ones((n_reg, w), dtype=np.float32)
    for k in range(6):
        bias *= _TN5_POS_BASE_WEIGHTS[k, padded[:, k : k + w]]
    # Normalize each region's bias profile to mean 1.0
    row_means = np.maximum(bias.mean(axis=1, keepdims=True), 1e-6)
    return bias / row_means


def get_tn5_footprints(
    ds,
    regions,
    *,
    groupby: Optional[Union[str, Sequence[str]]] = None,
    window: Optional[int] = None,
    shift: Tuple[int, int] = (0, 0),
    fragment_size_range: Optional[Tuple[int, int]] = None,
    bin_size: int = 1,
    normalize: str = "cpm",
    correct_tn5_bias: bool = False,
    return_dict: bool = False,
) -> Union[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Extract base-pair (or binned) Tn5 cut-site footprints across one or more genomic regions.

    Parameters
    ----------
    ds : CytomeDataset
        Open dataset containing ATAC fragments (`fragment_chunks` or `fragments_<chrom>`).
    regions : str, tuple, Sequence, or DataFrame
        One or more regions specified as ``'chr1:1000-2000'``, ``('chr1', 1000, 2000)``, a list of
        peak indices ``[0, 5, 12]``, or a DataFrame with ``(chr, start, end_)``.
    groupby : str or Sequence[str], optional
        Cell grouping column(s) in ``ds.cells`` (e.g. ``'cell_type'``). If ``None``, all
        cells are aggregated into a single group.
    window : int, optional
        If given, center every region to uniform width ``window`` bp.
    shift : tuple[int, int], default ``(0, 0)``
        Additional shift applied to ``(start, end_ - 1)``. Standard 10x ``fragments.tsv.gz`` files
        and ``.cytome`` stores are already ``+4/-5`` Tn5-shifted, so ``(0, 0)`` places the 5' cut at
        ``start`` and the 3' cut at ``end_ - 1``. Pass ``(4, -5)`` only for unshifted BAM coords.
    fragment_size_range : tuple[int, int], optional
        If given, keep only fragments with ``min_len <= (end_ - start) <= max_len`` (e.g.
        ``(0, 120)`` for sub-nucleosomal TF footprints or ``(140, 250)`` for mono-nucleosomes).
    bin_size : int, default ``1``
        Base pairs per bin (``1`` = exact single-nucleotide resolution).
    normalize : ``'cpm'``, ``'per_cell'``, or ``'raw'``
        ``'cpm'`` (default): cut sites per million cut sites of the group, two per fragment of its cells
        (``cells.n_fragments``; without ``groupby`` and without that column, the store's number of
        fragments). No depth is assumed: with ``groupby``, a group whose cells have no fragment count is
        refused; pass ``'raw'`` (counts) or ``'per_cell'`` (counts per cell of the group) there.
    correct_tn5_bias : bool, default ``False``
        If True and ``bin_size == 1`` and ``peak_sequences`` is available for peak regions, divide
        observed footprints by the sequence-driven 6-mer Tn5 cleavage propensity profile.
    return_dict : bool, default ``False``
        If True, return a dict ``{'footprints', 'group_names', 'n_cells', 'total_fragments',
        'regions', 'tn5_bias'}``.

    Notes
    -----
    Regions may overlap or lie inside one another; each is counted in full. The 3' cut site of a fragment
    is looked for among the fragments that start within 5,000 bp before the region, so the 3' end of a
    fragment longer than that is not counted.
    """
    if normalize not in ("cpm", "per_cell", "raw"):
        raise ValueError(f"normalize must be 'cpm', 'per_cell', or 'raw', got {normalize!r}")
    parsed = _parse_regions(ds, regions, window=window)
    n_reg = len(parsed)
    bin_size = max(1, int(bin_size))
    shift_l, shift_r = int(shift[0]), int(shift[1])

    if groupby is None:
        n_cells = int(ds.n_cells)
        cell_groups = np.zeros(n_cells, dtype=np.int32)
        group_names = ["all_cells"]
        n_cells_per_group = np.array([n_cells], dtype=np.int64)
        try:
            tot_frags = float(np.nansum(np.asarray(ds.cells["n_fragments"], dtype=np.float64)))
        except Exception:
            tot_frags = 0.0
        if tot_frags <= 0:
            tot_frags = float(ds.fragments.n_fragments)
        total_fragments = np.array([max(tot_frags, 1.0)], dtype=np.float64)
    else:
        g_cols = [groupby] if isinstance(groupby, str) else list(groupby)
        grouping = _build_cell_grouping(ds, g_cols, min_cells=1)
        cell_groups, group_names = grouping.cell_groups, grouping.group_names
        n_cells_per_group, total_fragments = grouping.n_cells, grouping.total_fragments
        # A group's depth is the fragment count of its cells as the store records it. Where the store
        # records none, 5,000 fragments per cell were assumed before, and the cuts per million were wrong
        # by whatever the cells' true depth was.
        no_depth = [str(g) for g, t in zip(group_names, np.asarray(total_fragments).tolist()) if not t > 0]
        if normalize == "cpm" and no_depth:
            raise ValueError(
                "get_tn5_footprints: normalize='cpm' needs the fragment count of each group's cells "
                f"(cells.n_fragments), and the store has none for {len(no_depth)} of {len(group_names)} "
                f"groups ({', '.join(no_depth[:3])}{', ...' if len(no_depth) > 3 else ''}). Pass "
                "normalize='raw' for counts or normalize='per_cell' for counts per cell.")

    n_groups = len(group_names)
    reg_outputs: list[np.ndarray] = []
    for _, s, e, _ in parsed:
        n_bins = max(1, (max(e - s, 1) + bin_size - 1) // bin_size)
        reg_outputs.append(np.zeros((n_groups, n_bins), dtype=np.float32))

    if n_reg == 0 or n_groups == 0:
        arr_empty = np.zeros((0, n_groups, 0), dtype=np.float32)
        if return_dict:
            return {
                "footprints": arr_empty,
                "group_names": group_names,
                "n_cells": n_cells_per_group,
                "total_fragments": total_fragments,
                "regions": parsed,
            }
        return arr_empty if groupby is not None else arr_empty[:, 0, :]

    conn = ds._conn
    has_chunks = ds.fragments.has_compressed_chunks

    # Group regions by chromosome and sort by start coordinate for single-pass sweeping scan
    by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for r_idx, (ch, s, e, _) in enumerate(parsed):
        by_chrom.setdefault(ch, []).append((s, e, r_idx))

    for ch, r_list in by_chrom.items():
        r_list.sort(key=lambda x: x[0])
        r_starts = np.array([x[0] for x in r_list], dtype=np.int64)
        r_ends = np.array([x[1] for x in r_list], dtype=np.int64)
        r_indices = [x[2] for x in r_list]
        # The regions are sorted by start; their ends are sorted too only when no region lies inside
        # another. The largest end so far is sorted, and is what a chunk is tested against.
        r_ends_reach = np.maximum.accumulate(r_ends)
        max_r_len = int(np.max(r_ends - r_starts))
        min_q_start = int(r_starts[0])
        max_q_end = int(np.max(r_ends))

        if has_chunks:
            meta = conn.execute(
                "SELECT chunk_idx, min_start, max_end FROM fragment_chunks WHERE chrom = ? ORDER BY chunk_idx",
                (ch,),
            ).fetchall()
            if not meta:
                continue
            needed_cidxs = []
            for c_idx, m_start, m_end in meta:
                if m_start is not None and int(m_start) + min(shift_l, shift_r) >= max_q_end:
                    continue
                if m_end is not None and int(m_end) + max(shift_l, shift_r) <= min_q_start:
                    continue
                if m_start is not None and m_end is not None:
                    c_min_cut = int(m_start) + min(shift_l, shift_r)
                    c_max_cut = int(m_end) + max(shift_l, shift_r)
                    # the regions that start before the chunk's last cut site; the chunk is needed when
                    # one of them reaches past its first
                    n_before = int(np.searchsorted(r_starts, c_max_cut, side="left"))
                    if n_before == 0 or int(r_ends_reach[n_before - 1]) <= c_min_cut:
                        continue
                needed_cidxs.append(int(c_idx))

            if not needed_cidxs:
                continue

            # Fetch needed chunks in batches to bound memory while avoiding repeated SQLite queries
            for b0 in range(0, len(needed_cidxs), 32):
                batch_c = needed_cidxs[b0 : b0 + 32]
                placeholders = ",".join("?" for _ in batch_c)
                rows = conn.execute(
                    f"SELECT starts_blob, ends_blob, cell_idx_blob, compression, COALESCE(encoding, 0) "
                    f"FROM fragment_chunks WHERE chrom = ? AND chunk_idx IN ({placeholders}) ORDER BY chunk_idx",
                    [ch, *batch_c],
                ).fetchall()
                for starts_b, ends_b, cells_b, comp, enc in rows:
                    starts = decode_starts(starts_b, comp, enc)
                    ends = decode_ends(ends_b, comp, starts, enc)
                    cells = np.frombuffer(decompress_blob(cells_b, comp), dtype=np.int32)
                    _accumulate_chunk_into_regions(
                        starts,
                        ends,
                        cells,
                        cell_groups,
                        r_starts,
                        r_ends,
                        r_indices,
                        max_r_len,
                        shift_l,
                        shift_r,
                        fragment_size_range,
                        bin_size,
                        reg_outputs,
                    )
        else:
            # Per-row table fallback via query_region per region
            for s, e, orig_idx in r_list:
                q_d = ds.fragments.query_region(ch, s - 10, e + 10)
                starts = q_d["start"].astype(np.int32)
                ends = q_d["end_"].astype(np.int32)
                cells = q_d["cell_idx"].astype(np.int32)
                if len(starts) > 0:
                    _accumulate_chunk_into_regions(
                        starts,
                        ends,
                        cells,
                        cell_groups,
                        np.array([s], dtype=np.int64),
                        np.array([e], dtype=np.int64),
                        [orig_idx],
                        e - s,
                        shift_l,
                        shift_r,
                        fragment_size_range,
                        bin_size,
                        reg_outputs,
                    )

    # Apply normalization across groups
    if normalize == "cpm":
        # Total cut sites across genome = 2 * total_fragments
        scale = (1e6 / np.maximum(2.0 * total_fragments, 1.0)).astype(np.float32)[:, None]
        for r_idx in range(n_reg):
            reg_outputs[r_idx] *= scale
    elif normalize == "per_cell":
        scale = (1.0 / np.maximum(n_cells_per_group, 1)).astype(np.float32)[:, None]
        for r_idx in range(n_reg):
            reg_outputs[r_idx] *= scale
    elif normalize != "raw":
        raise ValueError(f"normalize must be 'cpm', 'per_cell', or 'raw', got {normalize!r}")

    # Check if all regions have identical number of bins -> stack into 3D tensor (n_regions, n_groups, n_bins)
    all_same_width = len({arr.shape[1] for arr in reg_outputs}) == 1
    tn5_bias_arr = None

    if all_same_width:
        stacked = np.stack(reg_outputs, axis=0)  # (n_reg, n_groups, W)
        if correct_tn5_bias and bin_size == 1:
            pidxs = [p[3] for p in parsed]
            if all(p is not None for p in pidxs):
                try:
                    from cytome.core.sequences import get_peak_sequences
                    w_bp = stacked.shape[2]
                    codes = get_peak_sequences(ds, window=w_bp, peak_indices=pidxs, encoding="uint8")
                    tn5_bias_arr = _compute_sequence_tn5_bias(codes)
                    stacked = stacked / np.maximum(tn5_bias_arr[:, None, :], 0.1)
                except Exception:
                    pass
        final_out = stacked if groupby is not None else stacked[:, 0, :]
    else:
        final_out = reg_outputs if groupby is not None else [arr[0] for arr in reg_outputs]

    if return_dict:
        return {
            "footprints": final_out,
            "group_names": group_names,
            "n_cells": n_cells_per_group,
            "total_fragments": total_fragments,
            "regions": parsed,
            "tn5_bias": tn5_bias_arr,
        }
    return final_out


def _accumulate_chunk_into_regions(
    starts: np.ndarray,
    ends: np.ndarray,
    cells: np.ndarray,
    cell_groups: np.ndarray,
    r_starts: np.ndarray,
    r_ends: np.ndarray,
    r_indices: list[int],
    max_r_len: int,
    shift_l: int,
    shift_r: int,
    fragment_size_range: Optional[Tuple[int, int]],
    bin_size: int,
    reg_outputs: list[np.ndarray],
) -> None:
    if len(starts) == 0:
        return
    if fragment_size_range is not None:
        min_l, max_l = int(fragment_size_range[0]), int(fragment_size_range[1])
        lengths = ends - starts
        f_mask = (lengths >= min_l) & (lengths <= max_l)
        if not np.any(f_mask):
            return
        starts = starts[f_mask]
        ends = ends[f_mask]
        cells = cells[f_mask]

    cuts5 = starts.astype(np.int64) + shift_l
    cuts3 = (ends.astype(np.int64) - 1) + shift_r
    g_arr = cell_groups[cells]

    c_min = int(min(cuts5[0], np.min(cuts3)))
    c_max = int(max(cuts5[-1], np.max(cuts3)))

    # Binary search to find subset of sorted regions on this chromosome that overlap [c_min, c_max]
    lo_r = max(0, int(np.searchsorted(r_starts, c_min - max_r_len, side="left")))
    hi_r = min(len(r_starts), int(np.searchsorted(r_starts, c_max + 1, side="right")))

    # Cuts5 is sorted (since `starts` is sorted within each chunk!), so we can slice cuts5 with searchsorted!
    for k in range(lo_r, hi_r):
        rs = int(r_starts[k])
        re_ = int(r_ends[k])
        if re_ <= c_min or rs > c_max:
            continue
        out_mat = reg_outputs[r_indices[k]]

        # 5' cuts: starts is sorted, so binary search exact range!
        i0 = int(np.searchsorted(cuts5, rs, side="left"))
        i1 = int(np.searchsorted(cuts5, re_, side="left"))
        if i1 > i0:
            if _HAS_NUMBA:
                _accum_cuts_into_window_numba(cuts5[i0:i1], g_arr[i0:i1], rs, re_, bin_size, out_mat)
            else:
                _accum_cuts_into_window_fallback(cuts5[i0:i1], g_arr[i0:i1], rs, re_, bin_size, out_mat)

        # 3' cuts: ends is within [starts, starts + max_frag_len]
        j0 = max(0, int(np.searchsorted(cuts5, rs - 5000, side="left")))
        j1 = min(len(cuts3), int(np.searchsorted(cuts5, re_ + 50, side="left")))
        if j1 > j0:
            if _HAS_NUMBA:
                _accum_cuts_into_window_numba(cuts3[j0:j1], g_arr[j0:j1], rs, re_, bin_size, out_mat)
            else:
                _accum_cuts_into_window_fallback(cuts3[j0:j1], g_arr[j0:j1], rs, re_, bin_size, out_mat)
