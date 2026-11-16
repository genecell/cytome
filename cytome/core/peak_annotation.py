"""Vectorized dual-track TSS distance & genomic region classifier for ``peaks``.

Provides :func:`annotate_peaks_tss`, which populates:
- ``nearest_pc_gene``, ``nearest_pc_gene_id``, ``distance_to_pc_tss`` (protein-coding TSS track)
- ``nearest_any_gene``, ``distance_to_any_tss`` (all-biotype TSS track)
- ``nearest_gene``, ``nearest_gene_id`` (= the protein-coding gene when on same chromosome, else the any-biotype one)
- ``distance_to_tss`` (= signed distance to ``nearest_gene`` TSS: negative = upstream, positive = downstream)
- ``annotation`` and ``peak_type`` (``'Promoter'``, ``'Exonic'``, ``'Intronic'``, ``'Distal'``)
- ``gc_content`` (when ``peak_sequences`` is present in the store)

Genes are named as the reference names them, and identified by version-stripped Ensembl ID
(``*_gene_id``): :func:`peak_gene_index` joins peaks to a store's genes by that ID, whatever symbol either
side uses. The reference is data: a GTF passed as ``gtf_path`` (imported into the store), a gene annotation
the store already holds, or one copied from another store. Nothing is searched for on disk and no genome is
guessed.
"""

from __future__ import annotations

import re

import numpy as np

from cytome.io.gtf_import import chrom_alternatives, ensure_gtf_imported

_PREDICTED_SYM_RE = re.compile(r"^(Gm\d+|.*Rik|LOC\d+|ENS[A-Z]*G\d+|RGD\d+)$")


def _dedup_tss_track(
    records: list[tuple[int, int, str, str, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate ``(tss, strand_sign, symbol, gene_id, is_pc)`` by ``(tss, strand_sign)``.

    When multiple transcripts or overlapping readthrough/predicted genes share an identical
    ``(tss, strand_sign)``, keeps the highest-priority representative:
    1. ``is_pc == True`` (protein-coding preferred over non-coding)
    2. Canonical gene symbol preferred over predicted/numbered symbols (``Gm\\d+``, ``*Rik``, ``LOC\\d+``)
    3. Lexicographically smallest ``(symbol, gene_id)`` for deterministic tie-breaking.
    """
    if not records:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
        )
    best_by_key: dict[tuple[int, int], tuple[tuple[int, int, str, str], str, str]] = {}
    for tss, st_sign, sym, gid, is_pc in records:
        key = (int(tss), int(st_sign))
        is_pred = 1 if _PREDICTED_SYM_RE.match(str(sym)) else 0
        rank = (0 if is_pc else 1, is_pred, str(sym), str(gid))
        prev = best_by_key.get(key)
        if prev is None or rank < prev[0]:
            best_by_key[key] = (rank, str(sym), str(gid))

    n = len(best_by_key)
    pos = np.empty(n, dtype=np.int64)
    sign = np.empty(n, dtype=np.int64)
    names = np.empty(n, dtype=object)
    ids = np.empty(n, dtype=object)
    for i, ((t_pos, t_sign), (_, sym, gid)) in enumerate(best_by_key.items()):
        pos[i] = t_pos
        sign[i] = t_sign
        names[i] = sym
        ids[i] = gid
    return pos, sign, names, ids


def _merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort and merge overlapping `[start, end)` intervals in `O(N log N)`."""
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(starts, kind="mergesort")
    s_sorted = starts[order].astype(np.int64)
    e_sorted = ends[order].astype(np.int64)
    merged_s = [int(s_sorted[0])]
    merged_e = [int(e_sorted[0])]
    for i in range(1, len(s_sorted)):
        si = int(s_sorted[i])
        ei = int(e_sorted[i])
        if si <= merged_e[-1]:
            if ei > merged_e[-1]:
                merged_e[-1] = ei
        else:
            merged_s.append(si)
            merged_e.append(ei)
    return np.asarray(merged_s, dtype=np.int64), np.asarray(merged_e, dtype=np.int64)


def _overlaps_merged(
    q_starts: np.ndarray,
    q_ends: np.ndarray,
    m_starts: np.ndarray,
    m_ends: np.ndarray,
) -> np.ndarray:
    """Vectorized overlap check of query `[q_starts, q_ends)` against disjoint merged `[m_starts, m_ends)`."""
    if len(m_starts) == 0 or len(q_starts) == 0:
        return np.zeros(len(q_starts), dtype=bool)
    # Find last merged interval with m_start < q_end
    idx = np.searchsorted(m_starts, q_ends, side="left") - 1
    valid = idx >= 0
    out = np.zeros(len(q_starts), dtype=bool)
    if np.any(valid):
        v_idx = idx[valid]
        out[valid] = m_ends[v_idx] > q_starts[valid]
    return out


def _nearest_tss_on_chrom(
    summits: np.ndarray,
    tss_pos: np.ndarray,
    tss_strand_sign: np.ndarray,
    tss_names: np.ndarray,
    tss_ids: np.ndarray,
    promoter_window: tuple[int, int] = (2000, 500),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest TSS for each peak summit in ``O(P log T)`` using the Exact Strand-Split Theorem.

    Splits TSSs into strictly increasing Plus-strand (``+1``) and Minus-strand (``-1``) coordinate
    arrays, runs ``np.searchsorted`` on each strand independently to obtain up to 4 candidates
    ``{left+, right+, left-, right-}``, and selects the candidate minimizing the lexicographic
    key ``(~in_prom, abs(signed_distance))`` where ``in_prom = (-up_bp <= signed_distance <= +down_bp)``.

    Returns ``(nearest_name, nearest_id, signed_distance)`` where ``signed_distance`` is
    ``(summit - tss) * (+1 if strand == '+' else -1)`` (upstream of TSS is negative, downstream
    into the gene body is positive).
    """
    n_p = len(summits)
    if len(tss_pos) == 0 or n_p == 0:
        return (
            np.full(n_p, None, dtype=object),
            np.full(n_p, None, dtype=object),
            np.full(n_p, None, dtype=object),
        )

    up_bp, down_bp = int(promoter_window[0]), int(promoter_window[1])
    cand_names = []
    cand_ids = []
    cand_sdist = []

    for sign_val in (1, -1):
        s_mask = tss_strand_sign == sign_val
        if not np.any(s_mask):
            continue
        pos_sub = tss_pos[s_mask]
        name_sub = tss_names[s_mask]
        id_sub = tss_ids[s_mask]
        order = np.argsort(pos_sub, kind="mergesort")
        pos_s = pos_sub[order]
        name_s = name_sub[order]
        id_s = id_sub[order]

        idx_r = np.searchsorted(pos_s, summits, side="left")
        idx_l = np.clip(idx_r - 1, 0, len(pos_s) - 1)
        idx_rc = np.clip(idx_r, 0, len(pos_s) - 1)

        for idx_arr in (idx_l, idx_rc):
            cand_names.append(name_s[idx_arr])
            cand_ids.append(id_s[idx_arr])
            cand_sdist.append((summits - pos_s[idx_arr]) * sign_val)

    # Stack candidates of shape (K, n_p), where K in {2, 4}
    sdist_mat = np.vstack(cand_sdist)  # int64 (K, n_p)
    abs_mat = np.abs(sdist_mat)
    in_prom_mat = (sdist_mat >= -up_bp) & (sdist_mat <= down_bp)

    # Lexicographic priority: in-promoter candidates beat non-promoter candidates; ties broken by |d|
    max_span = int(abs_mat.max()) + 10 if abs_mat.size else 1
    score_mat = np.where(in_prom_mat, abs_mat, abs_mat + max_span)
    best_k = np.argmin(score_mat, axis=0)
    col_idx = np.arange(n_p)

    names_mat = np.vstack(cand_names)
    ids_mat = np.vstack(cand_ids)
    return names_mat[best_k, col_idx], ids_mat[best_k, col_idx], sdist_mat[best_k, col_idx]


_ENS_VERSION_RE = re.compile(r"^(ENS[A-Z]*G\d+)\.\d+(?:_PAR_Y)?$")


def versionless_gene_id(gene_id) -> str | None:
    """An Ensembl gene ID without its version (``ENSG00000141510.18`` -> ``ENSG00000141510``); any other ID
    as it is; None for none."""
    if gene_id is None:
        return None
    g = str(gene_id)
    if not g:
        return None
    m = _ENS_VERSION_RE.match(g)
    return m.group(1) if m else g


def annotate_peaks_tss(
    ds,
    gtf_path=None,
    *,
    copy_from_dataset=None,
    promoter_window: tuple[int, int] = (2000, 500),
    compute_gc: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict:
    """Annotate ``ds.peaks`` with dual-track TSS distances, region classes, and GC content.

    Parameters
    ----------
    ds : CytomeDataset
        Open dataset (writable).
    gtf_path : str or Path, optional
        The GTF of the store's genome, imported into the store when it holds no gene annotation. A store that
        holds one uses it (one store, one annotation).
    copy_from_dataset : CytomeDataset or str, optional
        A store whose gene annotation is copied when ``ds`` holds none and no ``gtf_path`` is given.
    promoter_window : tuple[int, int]
        ``(upstream_bp, downstream_bp)`` around TSS defining a ``'Promoter'`` peak
        (default ``(2000, 500)``, i.e. ``[-2000, +500]`` bp relative to strand-aware TSS).
    compute_gc : bool
        If True and ``peak_sequences`` is present in ``ds``, also compute per-peak ``gc_content``
        (in a 500 bp summit window).
    overwrite : bool, default False
        Replace an annotation the peaks already carry. Without it, peaks with an ``annotation`` or
        ``nearest_gene`` are left as they are and ``ValueError`` is raised.

    Returns
    -------
    dict
        ``n_peaks``, ``n_with_pc_gene``, ``annotation_counts``, ``has_gc_content``, and ``reference``: where
        the gene annotation came from (``status`` ``'existing'``, ``'imported'`` or ``'copied'``, and the GTF
        the store records).

    Raises
    ------
    FileNotFoundError
        The store holds no gene annotation and none was given (``gtf_path`` or ``copy_from_dataset``), or
        ``gtf_path`` does not exist. The peaks are not changed: before, every peak was written as ``'Distal'``.
    ValueError
        The peaks are already annotated (see ``overwrite``), or the reference names none of the peaks'
        chromosomes.
    """
    ds.require_writable("cytome.annotate_peaks_tss")
    if hasattr(ds, "peaks") and hasattr(ds.peaks, "_settle"):
        ds.peaks._settle()
    if not overwrite:
        _pcols = {r[1] for r in ds._conn.execute("PRAGMA table_info(peaks)")}
        _done = [c for c in ("annotation", "nearest_gene") if c in _pcols]
        if _done:
            n_done = ds._conn.execute(
                "SELECT COUNT(*) FROM peaks WHERE " + " OR ".join(f"{c} IS NOT NULL" for c in _done)
            ).fetchone()[0]
            if n_done:
                raise ValueError(
                    f"{n_done} peaks already carry an annotation; pass overwrite=True to replace it"
                )
    held_before = ds._conn.execute("SELECT COUNT(*) FROM _gene_annotation").fetchone()[0] > 0
    record_before = ds.metadata.get("gene_annotation")
    gtf_info = ensure_gtf_imported(ds, gtf_path=gtf_path, copy_from_dataset=copy_from_dataset, verbose=verbose)
    conn = ds._conn
    recorded = (ds.metadata.get("gene_annotation") or {}).get("gtf_path")
    reference = {"status": gtf_info.get("status"), "gtf_path": recorded}

    def _refused(exc: Exception) -> Exception:
        """A reference that came into the store with this call and cannot annotate its peaks leaves again.

        It stayed before, and a store that holds an annotation uses it: the next call, with the right GTF,
        was answered by the wrong one. A reference the store held already is left in it.
        """
        if held_before or gtf_info.get("status") not in ("imported", "copied"):
            return exc
        conn.execute("DELETE FROM _gene_annotation")
        conn.execute("DELETE FROM _exon_annotation")
        conn.commit()
        if record_before is not None:
            ds.metadata["gene_annotation"] = record_before
        elif ds.metadata.get("gene_annotation") is not None:
            del ds.metadata["gene_annotation"]
        ds.flush()
        from cytome.core.provenance import record_edit
        record_edit(ds, "cytome.annotate_peaks_tss",
                    {"gtf_path": gtf_path, "copy_from_dataset": copy_from_dataset, "reference": reference},
                    outputs=["_gene_annotation", "_exon_annotation"],
                    notes=f"the reference given was taken out again, the peaks unchanged: {exc}")
        return exc

    if gtf_info.get("status") == "empty":
        hint = f" This store records the GTF {recorded}." if recorded else ""
        raise FileNotFoundError(
            "annotate_peaks_tss: the store holds no gene annotation; pass gtf_path= (a GTF of the store's "
            f"genome) or copy_from_dataset=.{hint} The peaks were not changed."
        )

    n_peaks = conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0]
    if n_peaks == 0:
        return {"n_peaks": 0, "reference": reference}

    peak_cols = {r[1] for r in conn.execute("PRAGMA table_info(peaks)")}
    summit_col = "summit" if "summit" in peak_cols else ("summit_offset" if "summit_offset" in peak_cols else None)
    if summit_col:
        peak_rows = conn.execute(
            f"SELECT peak_idx, chr, start, end_, {summit_col} FROM peaks ORDER BY rowid"
        ).fetchall()
    else:
        peak_rows = conn.execute(
            "SELECT peak_idx, chr, start, end_, NULL FROM peaks ORDER BY rowid"
        ).fetchall()

    gene_rows = conn.execute(
        "SELECT gene_id, chrom, start, end, strand, gene_name, gene_type FROM _gene_annotation"
    ).fetchall()
    exon_rows = conn.execute(
        "SELECT gene_id, chrom, start, end, strand, feature FROM _exon_annotation "
        "WHERE feature IN ('exon', 'transcript')"
    ).fetchall()

    # Build chromosome lookup harmonized to peak chromosome names
    peak_chroms = {str(r[1]) for r in peak_rows}

    def _norm_chrom(c: str) -> str:
        if c in peak_chroms:
            return c
        for alt in chrom_alternatives(c):
            if alt in peak_chroms:
                return alt
        return c

    # Collect gene metadata & TSS positions per chromosome; a gene is named as the reference names it and
    # identified by its version-stripped ID
    gene_info = {}
    pc_tss_raw: dict[str, list[tuple[int, int, str, str, bool]]] = {}
    any_tss_raw: dict[str, list[tuple[int, int, str, str, bool]]] = {}
    gene_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {}
    exon_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {}

    for gid, chrom, start, end, strand, gname, gtype in gene_rows:
        nc = _norm_chrom(str(chrom))
        s = int(start)
        e = int(end)
        st_sign = -1 if strand == "-" else 1
        tss = (e - 1) if strand == "-" else s
        gid_s = versionless_gene_id(gid) or ""
        sym = str(gname) if (gname is not None and str(gname) != "") else gid_s
        is_pc = str(gtype or "") == "protein_coding"
        gene_info[gid] = (nc, st_sign, sym, gid_s, is_pc)
        any_tss_raw.setdefault(nc, []).append((tss, st_sign, sym, gid_s, is_pc))
        if is_pc:
            pc_tss_raw.setdefault(nc, []).append((tss, st_sign, sym, gid_s, True))
        gene_intervals_by_chrom.setdefault(nc, []).append((s, e))

    # Also include transcript-level TSSs from _exon_annotation (captures alternative promoters!)
    for gid, chrom, start, end, strand, feat in exon_rows:
        nc = _norm_chrom(str(chrom))
        s = int(start)
        e = int(end)
        if feat == "exon":
            exon_intervals_by_chrom.setdefault(nc, []).append((s, e))
        elif feat == "transcript" and gid in gene_info:
            _, st_sign, sym, gid_s, is_pc = gene_info[gid]
            tss = (e - 1) if (strand == "-" or st_sign == -1) else s
            any_tss_raw.setdefault(nc, []).append((tss, st_sign, sym, gid_s, is_pc))
            if is_pc:
                pc_tss_raw.setdefault(nc, []).append((tss, st_sign, sym, gid_s, True))

    if not any_tss_raw:
        raise _refused(FileNotFoundError(
            "annotate_peaks_tss: the store's gene annotation has no genes. The peaks were not changed."
        ))

    # Deduplicate TSS tracks per (tss_pos, strand_sign) on each chromosome
    pc_tss_by_chrom = {nc: _dedup_tss_track(recs) for nc, recs in pc_tss_raw.items() if recs}
    any_tss_by_chrom = {nc: _dedup_tss_track(recs) for nc, recs in any_tss_raw.items() if recs}
    if not any(nc in any_tss_by_chrom for nc in peak_chroms):
        if held_before:
            kept = ("The store held this annotation already" + (f" (read from {recorded})" if recorded else "")
                    + "; cytome.import_gtf(ds, <a GTF of the peaks' genome>, force=True) replaces it.")
        else:
            kept = "The reference given was not kept in the store."
        raise _refused(ValueError(
            "annotate_peaks_tss: the TSS reference names none of the peaks' chromosomes "
            f"(peaks: {sorted(peak_chroms)[:3]}; reference: {sorted(any_tss_by_chrom)[:3]}). "
            f"The peaks were not changed. {kept}"
        ))

    # Pre-merge exon and gene intervals per chromosome
    merged_exons = {}
    for nc, ivals in exon_intervals_by_chrom.items():
        arr = np.asarray(ivals, dtype=np.int64)
        merged_exons[nc] = _merge_intervals(arr[:, 0], arr[:, 1])

    merged_genes = {}
    for nc, ivals in gene_intervals_by_chrom.items():
        arr = np.asarray(ivals, dtype=np.int64)
        merged_genes[nc] = _merge_intervals(arr[:, 0], arr[:, 1])

    # Vectorized extraction for all peaks
    N = len(peak_rows)
    p_pks = np.fromiter((int(r[0]) for r in peak_rows), dtype=np.int64, count=N)
    p_chroms = np.array([str(r[1]) for r in peak_rows], dtype=object)
    p_starts = np.fromiter((int(r[2]) for r in peak_rows), dtype=np.int64, count=N)
    p_ends = np.fromiter((int(r[3]) for r in peak_rows), dtype=np.int64, count=N)
    p_mid = (p_starts + p_ends) // 2
    p_sm = np.fromiter(
        (-1 if r[4] is None else int(r[4]) for r in peak_rows), dtype=np.int64, count=N
    )
    has_sm = p_sm >= 0
    is_rel = has_sm & (p_sm <= (p_ends - p_starts))
    is_abs = has_sm & (~is_rel) & (p_sm >= p_starts) & (p_sm <= p_ends)
    p_summits = np.where(is_rel, p_starts + p_sm, np.where(is_abs, p_sm, p_mid))

    nearest_pc_gene = np.full(N, None, dtype=object)
    nearest_pc_id = np.full(N, None, dtype=object)
    dist_pc_tss = np.full(N, None, dtype=object)
    nearest_any_gene = np.full(N, None, dtype=object)
    nearest_any_id = np.full(N, None, dtype=object)
    dist_any_tss = np.full(N, None, dtype=object)
    annotations = np.full(N, "Distal", dtype=object)

    up_bp, down_bp = int(promoter_window[0]), int(promoter_window[1])

    for nc in sorted(peak_chroms):
        mask = p_chroms == nc
        idxs = np.flatnonzero(mask)
        if len(idxs) == 0:
            continue
        sub_summits = p_summits[idxs]
        sub_starts = p_starts[idxs]
        sub_ends = p_ends[idxs]

        # 1. Protein-coding TSS track
        pc_track = pc_tss_by_chrom.get(nc)
        if pc_track is not None and len(pc_track[0]) > 0:
            pc_pos, pc_sign, pc_sym, pc_id = pc_track
            n_sym, n_id, s_dist = _nearest_tss_on_chrom(
                sub_summits, pc_pos, pc_sign, pc_sym, pc_id, promoter_window=(up_bp, down_bp)
            )
            nearest_pc_gene[idxs] = n_sym
            nearest_pc_id[idxs] = n_id
            dist_pc_tss[idxs] = s_dist

        # 2. All-gene TSS track
        any_track = any_tss_by_chrom.get(nc)
        if any_track is not None and len(any_track[0]) > 0:
            a_pos, a_sign, a_sym, a_id = any_track
            na_sym, na_id, sa_dist = _nearest_tss_on_chrom(
                sub_summits, a_pos, a_sign, a_sym, a_id, promoter_window=(min(up_bp, 1000), down_bp)
            )
            nearest_any_gene[idxs] = na_sym
            nearest_any_id[idxs] = na_id
            dist_any_tss[idxs] = sa_dist

        # 3. Region classification: Promoter > Exonic > Intronic > Distal
        # Consistent with primary (nearest_gene, distance_to_tss): use protein-coding promoter
        # window when protein-coding genes exist on this chromosome, else all-biotype TSS window.
        is_prom = np.zeros(len(idxs), dtype=bool)
        if pc_track is not None and len(pc_track[0]) > 0:
            d_pc = dist_pc_tss[idxs].astype(np.int64)
            is_prom |= (d_pc >= -up_bp) & (d_pc <= down_bp)
        elif any_track is not None and len(any_track[0]) > 0:
            d_any = dist_any_tss[idxs].astype(np.int64)
            is_prom |= (d_any >= -min(up_bp, 1000)) & (d_any <= down_bp)

        m_ex_s, m_ex_e = merged_exons.get(nc, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
        is_exonic = (~is_prom) & _overlaps_merged(sub_starts, sub_ends, m_ex_s, m_ex_e)

        m_gn_s, m_gn_e = merged_genes.get(nc, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
        is_intronic = (~is_prom) & (~is_exonic) & _overlaps_merged(sub_starts, sub_ends, m_gn_s, m_gn_e)

        sub_ann = np.full(len(idxs), "Distal", dtype=object)
        sub_ann[is_intronic] = "Intronic"
        sub_ann[is_exonic] = "Exonic"
        sub_ann[is_prom] = "Promoter"
        annotations[idxs] = sub_ann

    # Primary nearest_gene and distance_to_tss prefer protein-coding when available
    has_pc = nearest_pc_gene != None  # noqa: E711
    nearest_gene = np.where(has_pc, nearest_pc_gene, nearest_any_gene)
    nearest_gene_id = np.where(has_pc, nearest_pc_id, nearest_any_id)
    distance_to_tss = np.where(dist_pc_tss != None, dist_pc_tss, dist_any_tss)  # noqa: E711

    # Optional GC content from embedded peak_sequences
    gc_arr = None
    if compute_gc:
        try:
            n_seq_chunks = conn.execute("SELECT COUNT(*) FROM peak_sequences").fetchone()[0]
            if n_seq_chunks > 0:
                from cytome.core.sequences import get_peak_sequences
                codes = get_peak_sequences(ds, window=500, center="summit", encoding="uint8")
                is_gc = (codes == 1) | (codes == 2)
                is_valid = codes < 4
                denom = np.maximum(is_valid.sum(axis=1), 1)
                gc_arr = (is_gc.sum(axis=1) / denom).astype(np.float32)
        except Exception:
            gc_arr = None

    # Ensure extra columns exist on peaks table
    new_cols = [
        ("peak_type", "TEXT", "object"),
        ("nearest_gene_id", "TEXT", "object"),
        ("nearest_pc_gene", "TEXT", "object"),
        ("nearest_pc_gene_id", "TEXT", "object"),
        ("distance_to_pc_tss", "INTEGER", "int64"),
        ("nearest_any_gene", "TEXT", "object"),
        ("distance_to_any_tss", "INTEGER", "int64"),
    ]
    if gc_arr is not None:
        new_cols.append(("gc_content", "REAL", "float64"))

    for cname, sql_type, meta_dtype in new_cols:
        if cname not in peak_cols:
            conn.execute(f'ALTER TABLE peaks ADD COLUMN "{cname}" {sql_type}')
        conn.execute(
            "INSERT OR REPLACE INTO _column_meta(table_name, column_name, dtype, categories) "
            "VALUES ('peaks', ?, ?, NULL)",
            (cname, meta_dtype),
        )

    # Streamed batch update of peaks table in 65,536-row blocks inside a single transaction
    def _to_py_int(x):
        return int(x) if x is not None else None

    block_size = 65536
    with conn:
        if gc_arr is not None:
            sql_upd = """
                UPDATE peaks SET
                    annotation = ?,
                    peak_type = ?,
                    nearest_gene = ?,
                    nearest_gene_id = ?,
                    distance_to_tss = ?,
                    nearest_pc_gene = ?,
                    nearest_pc_gene_id = ?,
                    distance_to_pc_tss = ?,
                    nearest_any_gene = ?,
                    distance_to_any_tss = ?,
                    gc_content = ?
                WHERE peak_idx = ?
            """
            for b_start in range(0, N, block_size):
                b_end = min(b_start + block_size, N)
                conn.executemany(
                    sql_upd,
                    [
                        (
                            annotations[i],
                            annotations[i],
                            nearest_gene[i],
                            nearest_gene_id[i],
                            _to_py_int(distance_to_tss[i]),
                            nearest_pc_gene[i],
                            nearest_pc_id[i],
                            _to_py_int(dist_pc_tss[i]),
                            nearest_any_gene[i],
                            _to_py_int(dist_any_tss[i]),
                            float(gc_arr[i]),
                            int(p_pks[i]),
                        )
                        for i in range(b_start, b_end)
                    ],
                )
        else:
            sql_upd = """
                UPDATE peaks SET
                    annotation = ?,
                    peak_type = ?,
                    nearest_gene = ?,
                    nearest_gene_id = ?,
                    distance_to_tss = ?,
                    nearest_pc_gene = ?,
                    nearest_pc_gene_id = ?,
                    distance_to_pc_tss = ?,
                    nearest_any_gene = ?,
                    distance_to_any_tss = ?
                WHERE peak_idx = ?
            """
            for b_start in range(0, N, block_size):
                b_end = min(b_start + block_size, N)
                conn.executemany(
                    sql_upd,
                    [
                        (
                            annotations[i],
                            annotations[i],
                            nearest_gene[i],
                            nearest_gene_id[i],
                            _to_py_int(distance_to_tss[i]),
                            nearest_pc_gene[i],
                            nearest_pc_id[i],
                            _to_py_int(dist_pc_tss[i]),
                            nearest_any_gene[i],
                            _to_py_int(dist_any_tss[i]),
                            int(p_pks[i]),
                        )
                        for i in range(b_start, b_end)
                    ],
                )

    unique_ann, counts_ann = np.unique(annotations, return_counts=True)
    ann_summary = {str(k): int(v) for k, v in zip(unique_ann, counts_ann)}
    n_with_pc = int(np.sum(nearest_pc_gene != None))  # noqa: E711
    from cytome.core.provenance import record_edit
    record_edit(
        ds, "cytome.annotate_peaks_tss",
        {"gtf_path": gtf_path, "copy_from_dataset": copy_from_dataset, "promoter_window": promoter_window,
         "compute_gc": compute_gc, "overwrite": overwrite, "reference": reference},
        outputs=["peaks"], inputs=["_gene_annotation", "_exon_annotation"])
    return {
        "n_peaks": N,
        "n_with_pc_gene": n_with_pc,
        "annotation_counts": ann_summary,
        "has_gc_content": gc_arr is not None,
        "reference": reference,
    }


def peak_gene_index(ds, column: str = "nearest_gene_id", table: str = "genes") -> np.ndarray:
    """Each peak's gene as a position in a gene table: the join of peaks to the store's genes by gene ID.

    Parameters
    ----------
    ds : CytomeDataset
    column : str
        The peaks' gene-ID column: ``'nearest_gene_id'`` (default) or ``'nearest_pc_gene_id'``, as
        :func:`annotate_peaks_tss` writes them.
    table : str
        ``'genes'`` (the RNA features) or ``'GA_genes'``.

    Returns
    -------
    np.ndarray of int64, one per peak in ``peak_idx`` order
        The gene's ``gene_idx`` (its column in the table's matrices), -1 where the peak names no gene or one
        the table does not hold.

    The table's IDs are its ``gene_ids`` column when it has one (Cell Ranger's Ensembl IDs), else its
    ``gene_id``; both sides are compared without Ensembl versions. Symbols are never compared, so a gene
    renamed between annotation releases (GENCODE vM23 against vM25) still matches.
    """
    conn = ds._conn
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(peaks)")}
    if column not in pcols:
        raise KeyError(f"peaks has no column {column!r}: run annotate_peaks_tss first")
    gcols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
    if not gcols:
        raise KeyError(f"the store has no {table!r} table")
    id_col = "gene_ids" if "gene_ids" in gcols else "gene_id"
    pos: dict[str, int] = {}
    for idx, gid in conn.execute(f'SELECT gene_idx, "{id_col}" FROM "{table}" ORDER BY gene_idx'):
        key = versionless_gene_id(gid)
        if key is not None:
            pos.setdefault(key, int(idx))
    names = [r[0] for r in conn.execute(f'SELECT "{column}" FROM peaks ORDER BY peak_idx')]
    return np.fromiter((pos.get(versionless_gene_id(g), -1) for g in names), dtype=np.int64, count=len(names))
