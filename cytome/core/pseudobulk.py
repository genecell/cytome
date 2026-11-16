"""Materialized multi-axis pseudobulk cubes: sufficient statistics of a matrix by cell group.

Provides:
- :class:`PseudobulkCube`: per group and feature, the sum, the sum of squares and the number of non-zero cells;
  per group, the number of cells and the sum of the cells' totals and of their squares. They add over groups,
  so marginalizing across grouping axes is exact.
- :func:`aggregate_pseudobulk`: one streaming pass over CSR (``matrix_chunks``) or CSC
  (``matrix_csc_chunks``) chunks, cached in ``pseudobulk_cubes``.

Statistics computed from a cube live outside this module, as COSG's markers do in COSG (``cosg.score_cube``).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd

from cytome.core._jit import HAS_NUMBA as _HAS_NUMBA, njit_cached
from cytome.io.compression import compress_blob, decompress_blob
from cytome.utils.modality import modality_var_entity


if _HAS_NUMBA:
    @njit_cached(nogil=True)
    def _accum_csr_numba(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        row_groups: np.ndarray,
        out_sum: np.ndarray,
        out_sum_sq: np.ndarray,
        out_nnz: np.ndarray,
    ) -> None:
        n_rows = len(row_groups)
        for r in range(n_rows):
            g = row_groups[r]
            if g < 0:
                continue
            start = indptr[r]
            end = indptr[r + 1]
            for idx in range(start, end):
                c = indices[idx]
                v = float(data[idx])
                if v != 0.0:
                    out_sum[g, c] += v
                    out_sum_sq[g, c] += v * v
                    out_nnz[g, c] += 1

    @njit_cached(nogil=True)
    def _accum_csc_numba(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        cell_groups: np.ndarray,
        col_offset: int,
        out_sum: np.ndarray,
        out_sum_sq: np.ndarray,
        out_nnz: np.ndarray,
    ) -> None:
        n_cols = len(indptr) - 1
        for c_local in range(n_cols):
            c = col_offset + c_local
            start = indptr[c_local]
            end = indptr[c_local + 1]
            for idx in range(start, end):
                r = indices[idx]
                g = cell_groups[r]
                if g < 0:
                    continue
                v = float(data[idx])
                if v != 0.0:
                    out_sum[g, c] += v
                    out_sum_sq[g, c] += v * v
                    out_nnz[g, c] += 1

    @njit_cached(nogil=True)
    def _add_row_sums_numba(indices: np.ndarray, data: np.ndarray, out: np.ndarray) -> None:
        for k in range(indices.shape[0]):
            out[indices[k]] += float(data[k])


def _accum_csr_fallback(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_groups: np.ndarray,
    out_sum: np.ndarray,
    out_sum_sq: np.ndarray,
    out_nnz: np.ndarray,
) -> None:
    counts_per_row = np.diff(indptr)
    rows = np.repeat(np.arange(len(row_groups), dtype=np.int64), counts_per_row)
    g = row_groups[rows]
    valid = (g >= 0) & (data != 0)
    if not np.any(valid):
        return
    gv = g[valid]
    cv = indices[valid]
    dv = data[valid].astype(np.float64, copy=False)
    flat_idx = gv.astype(np.int64) * out_sum.shape[1] + cv.astype(np.int64)
    n_tot = out_sum.size
    out_sum.ravel()[:] += np.bincount(flat_idx, weights=dv, minlength=n_tot)
    out_sum_sq.ravel()[:] += np.bincount(flat_idx, weights=dv * dv, minlength=n_tot)
    out_nnz.ravel()[:] += np.bincount(flat_idx, minlength=n_tot).astype(np.int32)


def _accum_csc_fallback(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    cell_groups: np.ndarray,
    col_offset: int,
    out_sum: np.ndarray,
    out_sum_sq: np.ndarray,
    out_nnz: np.ndarray,
) -> None:
    n_cols = len(indptr) - 1
    counts_per_col = np.diff(indptr)
    cols = np.repeat(np.arange(col_offset, col_offset + n_cols, dtype=np.int64), counts_per_col)
    g = cell_groups[indices]
    valid = (g >= 0) & (data != 0)
    if not np.any(valid):
        return
    gv = g[valid]
    cv = cols[valid]
    dv = data[valid].astype(np.float64, copy=False)
    flat_idx = gv.astype(np.int64) * out_sum.shape[1] + cv
    n_tot = out_sum.size
    out_sum.ravel()[:] += np.bincount(flat_idx, weights=dv, minlength=n_tot)
    out_sum_sq.ravel()[:] += np.bincount(flat_idx, weights=dv * dv, minlength=n_tot)
    out_nnz.ravel()[:] += np.bincount(flat_idx, minlength=n_tot).astype(np.int32)


def _add_row_sums(indices: np.ndarray, data: np.ndarray, out: np.ndarray) -> None:
    """``out[indices[k]] += data[k]``: each cell's total over a CSC chunk's entries."""
    if _HAS_NUMBA:
        _add_row_sums_numba(indices, data, out)
    else:
        np.add.at(out, indices, data.astype(np.float64, copy=False))


def _add_chunk_depths(indptr: np.ndarray, data: np.ndarray, row_groups: np.ndarray,
                      tot: np.ndarray, tot_sq: np.ndarray) -> None:
    """Add a CSR chunk's cells' totals, and their squares, to their groups."""
    n_rows = len(indptr) - 1
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
    rs = np.bincount(rows, weights=data.astype(np.float64, copy=False), minlength=n_rows)
    ok = row_groups >= 0
    if ok.any():
        np.add.at(tot, row_groups[ok], rs[ok])
        np.add.at(tot_sq, row_groups[ok], rs[ok] * rs[ok])


#: format of a cached cube; a cube cached with another is recomputed
_CUBE_VERSION = 2


@dataclass
class PseudobulkCube:
    """Materialized multi-axis pseudobulk sufficient-statistics tensor.

    Stores exact cell-level additive sufficient statistics per group ``g`` and feature ``f``:
    - ``n_cells[g]``: number of cells in group ``g``
    - ``total_counts[g]``: the sum over the group's cells of each cell's total in this matrix, ``d_c``
    - ``total_counts_sq[g]``: the sum over the group's cells of ``d_c`` squared (the spread of the cells'
      totals: ``n_cells * total_counts_sq / total_counts**2`` is 1 plus their squared coefficient of variation)
    - ``total_fragments[g]``: the group's fragments, from ``cells.n_fragments`` when the store records them
      (0 otherwise); a property of the cells, not of this matrix
    - ``sum[g, f]``: sum of matrix entries $S_{g,f} = \\sum_{c \\in g} X_{c,f}$
    - ``sum_sq[g, f]``: sum of squared entries $Q_{g,f} = \\sum_{c \\in g} X_{c,f}^2$
    - ``nnz[g, f]``: number of non-zero cells $N_{g,f} = \\sum_{c \\in g} \\mathbb{I}(X_{c,f} > 0)$
    """

    cube_key: str
    matrix_name: str
    col_entity: str
    group_cols: list[str]
    group_names: list[Any]
    n_cells: np.ndarray
    total_counts: np.ndarray
    total_fragments: np.ndarray
    sum: np.ndarray
    sum_sq: np.ndarray
    nnz: np.ndarray
    digest: dict = field(default_factory=dict)
    created_at: str = ""
    total_counts_sq: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.sum.shape

    @property
    def n_groups(self) -> int:
        return int(self.sum.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.sum.shape[1])

    def group_labels(self, sep: str = "__") -> list[str]:
        """Human-readable string labels for each group."""
        out = []
        for g in self.group_names:
            if isinstance(g, (tuple, list)):
                out.append(sep.join(str(x) for x in g))
            else:
                out.append(str(g))
        return out

    def marginalize(self, keep_axes: Union[str, Sequence[str]]) -> "PseudobulkCube":
        """Collapse out any grouping axes not in ``keep_axes`` by exact additive marginalization."""
        if isinstance(keep_axes, str):
            keep_list = [keep_axes]
        else:
            keep_list = list(keep_axes)
        if not keep_list:
            raise ValueError("keep_axes must contain at least one grouping column")
        for col in keep_list:
            if col not in self.group_cols:
                raise KeyError(f"Axis {col!r} not in cube group_cols {self.group_cols}")
        if keep_list == self.group_cols:
            return self

        axis_indices = [self.group_cols.index(c) for c in keep_list]
        single_axis_out = len(keep_list) == 1

        # Map each existing group tuple to its projected key
        proj_keys = []
        for g in self.group_names:
            tup = g if isinstance(g, (tuple, list)) else (g,)
            if single_axis_out:
                proj_keys.append(tup[axis_indices[0]])
            else:
                proj_keys.append(tuple(tup[i] for i in axis_indices))

        unique_keys = []
        key_to_idx: dict[Any, int] = {}
        remap = np.empty(self.n_groups, dtype=np.int64)
        for i, k in enumerate(proj_keys):
            idx = key_to_idx.get(k)
            if idx is None:
                idx = len(unique_keys)
                key_to_idx[k] = idx
                unique_keys.append(k)
            remap[i] = idx

        m_groups = len(unique_keys)
        n_feat = self.n_features
        new_n_cells = np.zeros(m_groups, dtype=np.int64)
        new_tot_counts = np.zeros(m_groups, dtype=np.float64)
        new_tot_frags = np.zeros(m_groups, dtype=np.float64)
        new_tot_counts_sq = (
            np.zeros(m_groups, dtype=np.float64)
            if self.total_counts_sq is not None
            else None
        )
        new_sum = np.zeros((m_groups, n_feat), dtype=np.float32)
        new_sum_sq = np.zeros((m_groups, n_feat), dtype=np.float32)
        new_nnz = np.zeros((m_groups, n_feat), dtype=np.int32)

        for i in range(self.n_groups):
            gi = remap[i]
            new_n_cells[gi] += self.n_cells[i]
            new_tot_counts[gi] += self.total_counts[i]
            new_tot_frags[gi] += self.total_fragments[i]
            if new_tot_counts_sq is not None and self.total_counts_sq is not None:
                new_tot_counts_sq[gi] += self.total_counts_sq[i]
            new_sum[gi] += self.sum[i]
            new_sum_sq[gi] += self.sum_sq[i]
            new_nnz[gi] += self.nnz[i]

        new_key = f"{self.matrix_name}::{'+'.join(keep_list)}"
        return PseudobulkCube(
            cube_key=new_key,
            matrix_name=self.matrix_name,
            col_entity=self.col_entity,
            group_cols=keep_list,
            group_names=unique_keys,
            n_cells=new_n_cells,
            total_counts=new_tot_counts,
            total_fragments=new_tot_frags,
            sum=new_sum,
            sum_sq=new_sum_sq,
            nnz=new_nnz,
            digest=dict(self.digest),
            created_at=self.created_at,
            total_counts_sq=new_tot_counts_sq,
        )

    def _resolve_depths(self, depth_source: str = "matrix_counts") -> np.ndarray:
        """Return group sequencing depths ``(G,)`` in ``float64``."""
        if depth_source == "fragments" and np.all(self.total_fragments > 0):
            return np.maximum(self.total_fragments.astype(np.float64), 1.0)
        row_sums = self.total_counts.astype(np.float64)
        if np.any(row_sums <= 0):
            row_sums = np.maximum(self.sum.sum(axis=1, dtype=np.float64), 1.0)
        return np.maximum(row_sums, 1.0)

    def cpm(
        self,
        scale_factor: float = 1e6,
        depth_source: str = "matrix_counts",
        prior_counts: float = 0.0,
    ) -> np.ndarray:
        """Exact or Empirical-Bayes shrunk Counts Per Million ``(G, F)`` in ``float32``."""
        denom = self._resolve_depths(depth_source=depth_source)
        if prior_counts and float(prior_counts) > 0.0:
            s = self.sum.astype(np.float64)
            global_sum = s.sum(axis=0, keepdims=True)
            global_tot = max(float(denom.sum()), 1.0)
            global_rate = global_sum / global_tot
            d = denom[:, None]
            shrunk_rate = (s + float(prior_counts) * global_rate) / (d + float(prior_counts))
            return (float(scale_factor) * shrunk_rate).astype(np.float32)
        scale = (float(scale_factor) / denom).astype(np.float32)[:, None]
        return self.sum * scale

    def log1p_cpm(
        self,
        scale_factor: float = 1e6,
        base: float = np.e,
        depth_source: str = "matrix_counts",
        prior_counts: float = 0.0,
    ) -> np.ndarray:
        """Exact ``log(1 + CPM) / log(base)`` of shape ``(G, F)`` in ``float32``."""
        out = np.log1p(
            self.cpm(scale_factor=scale_factor, depth_source=depth_source, prior_counts=prior_counts)
        )
        if base != np.e and abs(base - np.e) > 1e-6:
            out /= np.float32(np.log(base))
        return out

    def mean(self) -> np.ndarray:
        """Per-cell mean within each group ``(G, F)`` in ``float32``."""
        denom = np.maximum(self.n_cells, 1).astype(np.float32)[:, None]
        return self.sum / denom

    def var(self) -> np.ndarray:
        """Unbiased within-group sample variance across cells ``(G, F)`` in ``float32``."""
        n = np.maximum(self.n_cells, 1).astype(np.float64)[:, None]
        s = self.sum.astype(np.float64)
        sq = self.sum_sq.astype(np.float64)
        num = np.maximum(sq - (s * s) / n, 0.0)
        denom = np.maximum(n - 1.0, 1.0)
        return (num / denom).astype(np.float32)

    def fraction_expressing(self) -> np.ndarray:
        """Fraction of cells with non-zero counts per group ``(G, F)`` in ``float32``."""
        denom = np.maximum(self.n_cells, 1).astype(np.float32)[:, None]
        return self.nnz.astype(np.float32) / denom

    def to_dataframe(
        self,
        metric: str = "log1p_cpm",
        feature_names: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Convert the cube into a ``(n_groups, n_features)`` DataFrame."""
        if metric == "log1p_cpm":
            mat = self.log1p_cpm(**kwargs)
        elif metric == "cpm":
            mat = self.cpm(**kwargs)
        elif metric == "sum":
            mat = self.sum
        elif metric == "nnz":
            mat = self.nnz
        else:
            raise ValueError(f"Unknown metric {metric!r}")
        return pd.DataFrame(mat, index=self.group_labels(), columns=feature_names)


def _normalize_group_cols(by: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(by, str):
        return [by]
    cols = [str(c) for c in by]
    if not cols:
        raise ValueError("`by` must specify at least one cell metadata column")
    return cols


class CellGrouping(NamedTuple):
    """What :func:`_build_cell_grouping` returns.

    A tuple, so it still unpacks, but callers read it by name: adding a field must not break them
    (``get_tn5_footprints`` unpacked five values when a sixth was added).
    """

    cell_groups: np.ndarray
    group_names: list
    n_cells: np.ndarray
    total_fragments: np.ndarray
    group_hash: str


def _labels_hash(cols_data: Sequence[np.ndarray], group_cols: Sequence[str]) -> str:
    """Hash of every cell's labels, so relabelling any cell changes it.

    It hashed every 2,048th of the cells before: relabelling a few hundred cells of a large store was
    missed about one time in three, and a stale cube was served.
    """
    hasher = hashlib.blake2b(digest_size=12)
    for col, arr in zip(group_cols, cols_data):
        hasher.update(col.encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(pd.util.hash_pandas_object(pd.Series(arr), index=False).to_numpy().tobytes())
    return hasher.hexdigest()


def _group_key(g) -> tuple:
    """A group name as a tuple of strings, equal for a name read back from a cube's JSON."""
    return tuple(str(x) for x in (g if isinstance(g, (tuple, list)) else (g,)))


def _with_fresh_depths(cube: "PseudobulkCube", grouping: CellGrouping) -> Optional["PseudobulkCube"]:
    """Give a cached cube the fragment totals read from ``cells`` now, matched to its groups by name.

    The cells' totals in the matrix (``total_counts``, ``total_counts_sq``) are part of the cube; the
    fragments come from the cells table, which can change after the cube was cached. A marginalized superset
    lists its groups in another order than a fresh grouping, which sorts them, so they are matched by name.
    ``None`` when a cached group no longer exists, so the cube is recomputed.
    """
    pos = {_group_key(g): i for i, g in enumerate(grouping.group_names)}
    idx = [pos.get(_group_key(g)) for g in cube.group_names]
    if any(i is None for i in idx):
        return None
    cube.total_fragments = np.asarray(grouping.total_fragments, dtype=np.float64)[np.asarray(idx, dtype=np.int64)]
    return cube


def _same_groups_and_cells(cube: "PseudobulkCube", grouping: CellGrouping) -> bool:
    """Whether a cube holds the grouping's groups with the grouping's numbers of cells.

    A cube over more columns leaves out every cell with no value in one of them. Marginalized to fewer
    columns it then counts fewer cells than the grouping asked for, or lacks a group. Its labels are the
    current ones (their hash was compared), so the same groups with the same numbers of cells are the same
    cells.
    """
    have = {_group_key(g): int(n) for g, n in zip(cube.group_names, np.asarray(cube.n_cells).tolist())}
    want = {_group_key(g): int(n) for g, n in zip(grouping.group_names, np.asarray(grouping.n_cells).tolist())}
    return have == want


def _same_matrix(cached_digest: dict, digest: dict) -> bool:
    """Whether a cached cube was summed from the matrix as it is stored now.

    The shape and the number of non-zeros do not tell a matrix from the same matrix written again with other
    values (a normalized layer computed again). The time the matrix was written does, and every writer
    records it (``matrix_meta.created_at``). A cube cached before that time was kept has no record of it and
    is matched as before.
    """
    if "matrix_created_at" not in cached_digest:
        return True
    return cached_digest.get("matrix_created_at") == digest.get("matrix_created_at")


def _build_cell_grouping(ds, group_cols: list[str], min_cells: int = 1) -> CellGrouping:
    """Map each cell ``0..n_cells-1`` to a group index ``0..G-1`` (or ``-1`` if excluded/NA)."""
    n_cells = int(ds.n_cells)
    cols_data = [np.asarray(ds.cells[col]) for col in group_cols]
    group_hash = _labels_hash(cols_data, group_cols)

    if len(group_cols) == 1:
        raw = cols_data[0]
        s = pd.Series(raw)
        valid_mask = s.notna().to_numpy() & (s.astype(str).to_numpy() != "") & (s.astype(str).to_numpy() != "nan")
        codes, uniques = pd.factorize(s[valid_mask], sort=True)
        counts = np.bincount(codes, minlength=len(uniques))
        keep_u = counts >= int(min_cells)
        remap = np.full(len(uniques), -1, dtype=np.int32)
        kept_names = [uniques[i].item() if hasattr(uniques[i], "item") else uniques[i] for i in np.flatnonzero(keep_u)]
        remap[keep_u] = np.arange(len(kept_names), dtype=np.int32)
        cell_groups = np.full(n_cells, -1, dtype=np.int32)
        cell_groups[valid_mask] = remap[codes]
        group_names = kept_names
    else:
        df = pd.DataFrame({c: cols_data[i] for i, c in enumerate(group_cols)})
        valid_mask = np.ones(n_cells, dtype=bool)
        for c in group_cols:
            sc = df[c]
            valid_mask &= sc.notna().to_numpy() & (sc.astype(str).to_numpy() != "") & (sc.astype(str).to_numpy() != "nan")
        sub = df.loc[valid_mask, group_cols]
        grouped = sub.groupby(group_cols, sort=True, observed=True).ngroup()
        # Extract unique sorted tuples
        cats = sub.drop_duplicates().sort_values(group_cols)
        uniques = [
            tuple(x.item() if hasattr(x, "item") else x for x in row)
            for row in cats.itertuples(index=False, name=None)
        ]
        codes = grouped.to_numpy(dtype=np.int32)
        counts = np.bincount(codes, minlength=len(uniques))
        keep_u = counts >= int(min_cells)
        remap = np.full(len(uniques), -1, dtype=np.int32)
        group_names = [uniques[i] for i in np.flatnonzero(keep_u)]
        remap[keep_u] = np.arange(len(group_names), dtype=np.int32)
        cell_groups = np.full(n_cells, -1, dtype=np.int32)
        cell_groups[valid_mask] = remap[codes]

    n_groups = len(group_names)
    n_cells_per_group = np.bincount(cell_groups[cell_groups >= 0], minlength=n_groups).astype(np.int64)

    # Fragments per group, from cells.n_fragments when the store records them. Each cell's total in the
    # aggregated matrix, the depth that matrix's statistics need, is accumulated by aggregate_pseudobulk.
    total_fragments = np.zeros(n_groups, dtype=np.float64)
    if "n_fragments" in set(ds.cells.columns):
        nf = np.asarray(ds.cells["n_fragments"], dtype=np.float64)
        ok = (cell_groups >= 0) & np.isfinite(nf)
        total_fragments = np.bincount(cell_groups[ok], weights=nf[ok], minlength=n_groups).astype(np.float64)

    return CellGrouping(cell_groups, group_names, n_cells_per_group, total_fragments, group_hash)


def _save_cube_to_sqlite(conn, cube: PseudobulkCube, compression: str = "zstd") -> None:
    now = datetime.now(timezone.utc).isoformat()
    cube.created_at = cube.created_at or now
    digest_to_save = dict(cube.digest)
    if cube.total_counts_sq is not None:
        digest_to_save["total_counts_sq"] = np.asarray(
            cube.total_counts_sq, dtype=np.float64
        ).tolist()
    conn.execute(
        """
        INSERT OR REPLACE INTO pseudobulk_cubes (
            cube_key, matrix_name, col_entity, group_cols_json, group_names_json,
            n_groups, n_features, n_cells_blob, total_counts_blob, total_fragments_blob,
            sum_blob, sum_sq_blob, nnz_blob, compression, digest_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cube.cube_key,
            cube.matrix_name,
            cube.col_entity,
            json.dumps(cube.group_cols),
            json.dumps([list(g) if isinstance(g, tuple) else g for g in cube.group_names]),
            cube.n_groups,
            cube.n_features,
            compress_blob(np.ascontiguousarray(cube.n_cells, dtype=np.int64).tobytes(), compression),
            compress_blob(np.ascontiguousarray(cube.total_counts, dtype=np.float64).tobytes(), compression),
            compress_blob(np.ascontiguousarray(cube.total_fragments, dtype=np.float64).tobytes(), compression),
            compress_blob(np.ascontiguousarray(cube.sum, dtype=np.float32).tobytes(), compression),
            compress_blob(np.ascontiguousarray(cube.sum_sq, dtype=np.float32).tobytes(), compression),
            compress_blob(np.ascontiguousarray(cube.nnz, dtype=np.int32).tobytes(), compression),
            compression,
            json.dumps(digest_to_save),
            cube.created_at,
        ),
    )
    conn.commit()


def _load_cube_from_row(row) -> PseudobulkCube:
    (
        cube_key,
        matrix_name,
        col_entity,
        group_cols_json,
        group_names_json,
        n_groups,
        n_features,
        n_cells_blob,
        total_counts_blob,
        total_fragments_blob,
        sum_blob,
        sum_sq_blob,
        nnz_blob,
        compression,
        digest_json,
        created_at,
    ) = row
    group_cols = json.loads(group_cols_json)
    raw_names = json.loads(group_names_json)
    if len(group_cols) > 1:
        group_names = [tuple(x) if isinstance(x, list) else (x,) for x in raw_names]
    else:
        group_names = [x[0] if isinstance(x, list) and len(x) == 1 else x for x in raw_names]
    G, F = int(n_groups), int(n_features)
    digest = json.loads(digest_json) if digest_json else {}
    tot_frags_sq = None
    if "total_counts_sq" in digest and digest["total_counts_sq"] is not None:
        arr = np.asarray(digest["total_counts_sq"], dtype=np.float64)
        if len(arr) == G:
            tot_frags_sq = arr
    return PseudobulkCube(
        cube_key=cube_key,
        matrix_name=matrix_name,
        col_entity=col_entity,
        group_cols=group_cols,
        group_names=group_names,
        n_cells=np.frombuffer(decompress_blob(n_cells_blob, compression), dtype=np.int64).copy(),
        total_counts=np.frombuffer(decompress_blob(total_counts_blob, compression), dtype=np.float64).copy(),
        total_fragments=np.frombuffer(decompress_blob(total_fragments_blob, compression), dtype=np.float64).copy(),
        sum=np.frombuffer(decompress_blob(sum_blob, compression), dtype=np.float32).reshape(G, F).copy(),
        sum_sq=np.frombuffer(decompress_blob(sum_sq_blob, compression), dtype=np.float32).reshape(G, F).copy(),
        nnz=np.frombuffer(decompress_blob(nnz_blob, compression), dtype=np.int32).reshape(G, F).copy(),
        digest=digest,
        created_at=created_at or "",
        total_counts_sq=tot_frags_sq,
    )


def _resolve_matrix_name(ds, modality: str, layer: str) -> tuple[str, dict]:
    candidates = [f"{modality}_{layer}", layer] if modality else [layer]
    for name in candidates:
        meta = ds.matrix_meta(name)
        if meta is not None:
            return name, meta
    raise KeyError(f"Matrix for modality={modality!r}, layer={layer!r} not found in dataset")


def aggregate_pseudobulk(
    ds,
    modality: str = "ATAC",
    layer: str = "counts",
    by: Union[str, Sequence[str]] = "cell_type",
    *,
    min_cells: int = 1,
    cache: bool = True,
    force: bool = False,
    threads: int = 0,
    transform=None,
    transform_name: Optional[str] = None,
    transform_version: Optional[str] = None,
) -> PseudobulkCube:
    """Compute or load a multi-axis :class:`PseudobulkCube` from raw compressed matrix chunks.

    Parameters
    ----------
    ds : CytomeDataset
        Open dataset.
    modality : str
        Modality name (``'ATAC'``, ``'RNA'``, ``'GA'``, ``'tiles'``).
    layer : str
        Layer name (default ``'counts'``).
    by : str or Sequence[str]
        One or more cell metadata columns in ``ds.cells`` (e.g. ``'cell_type'`` or
        ``['cell_type', 'age_bin']``).
    min_cells : int
        Minimum cells required per group (default 1).
    cache : bool
        If True (default), read from / write to ``pseudobulk_cubes`` in SQLite. A cached cube is reused
        while the cells' labels, ``min_cells``, the transform and the time the matrix was written are the
        ones it was made with. A cached cube over more columns than ``by`` is marginalized instead of
        recomputed when it holds the same cells: a cell with no value in one of the other columns is not in
        it, and the cube asked for is then computed.
    force : bool
        If True, recompute even if a valid cached cube exists.
    threads : int
        Worker threads for parallel chunk decompression and accumulation (``0`` = auto up to 8).
    transform : callable, optional
        Applied to each chunk before its entries are summed, so the cube holds the sums of a normalized layer
        computed on the fly (COSG passes its own normalizers: ``cosg.score_cube``). A row chunk is passed as
        ``transform(chunk, row_indices)``: a CSR matrix of those cells and their positions. A column chunk, on
        a matrix stored column-major, as ``transform(chunk, row_indices, col_start=c0)``: a CSR matrix of every
        cell over the chunk's features, every cell's position, and the chunk's first column. It returns a
        matrix of the same shape. The cells' totals (``total_counts``, ``total_counts_sq``) stay those of the
        stored matrix. Runs on one thread, as the transform need not be thread-safe.
    transform_name : str, optional
        Names the transform in the cache key and digest. Without it, a transformed cube is neither read from
        nor written to the cache.
    transform_version : str, optional
        A fingerprint of what the transform applies (its statistics, the version of its code). A cached cube
        made with another fingerprint is recomputed, and replaces it under the same key, so the cache keeps one
        cube per grouping and transform.
    """
    import os

    matrix_name, meta = _resolve_matrix_name(ds, modality, layer)
    group_cols = _normalize_group_cols(by)
    cube_key = f"{matrix_name}::{'+'.join(group_cols)}"
    if transform is not None:
        if transform_name:
            cube_key += f"::{transform_name}"
        else:
            cache = False
    conn = ds._conn

    grouping = _build_cell_grouping(ds, group_cols, min_cells=min_cells)
    cell_groups, group_names, n_cells_per_group, total_fragments, group_hash = grouping
    n_groups = len(group_names)
    n_features = int(meta["n_cols"])
    col_entity = str(meta.get("col_entity") or modality_var_entity(modality) or "features")
    digest = {
        "n_cells": int(ds.n_cells),
        "n_features": n_features,
        "n_nonzero": int(meta["n_nonzero"]),
        "matrix_created_at": meta.get("created_at"),
        "min_cells": int(min_cells),
        "group_hash": group_hash,
        "cube_version": _CUBE_VERSION,
        "transform": transform_name if transform is not None else None,
        "transform_version": transform_version if transform is not None else None,
    }

    if cache and not force:
        try:
            row = conn.execute(
                "SELECT * FROM pseudobulk_cubes WHERE cube_key = ?", (cube_key,)
            ).fetchone()
            if row is not None:
                cube = _load_cube_from_row(row)
                if (
                    cube.digest.get("n_cells") == digest["n_cells"]
                    and cube.digest.get("n_features") == digest["n_features"]
                    and cube.digest.get("n_nonzero") == digest["n_nonzero"]
                    and _same_matrix(cube.digest, digest)
                    and cube.digest.get("group_hash") == digest["group_hash"]
                    and cube.digest.get("min_cells", 1) == digest["min_cells"]
                    and cube.digest.get("cube_version") == digest["cube_version"]
                    and cube.digest.get("transform") == digest["transform"]
                    and cube.digest.get("transform_version") == digest["transform_version"]
                ):
                    fresh = _with_fresh_depths(cube, grouping)
                    if fresh is not None:
                        return fresh
            # A cached superset cube on the same matrix whose labels are still current is marginalized
            if min_cells <= 1:
                sup_rows = conn.execute(
                    "SELECT * FROM pseudobulk_cubes WHERE matrix_name = ?", (matrix_name,)
                ).fetchall()
                for srow in sup_rows:
                    scube = _load_cube_from_row(srow)
                    if not (
                        scube.digest.get("n_cells") == digest["n_cells"]
                        and scube.digest.get("n_features") == digest["n_features"]
                        and scube.digest.get("n_nonzero") == digest["n_nonzero"]
                        and _same_matrix(scube.digest, digest)
                        and scube.digest.get("min_cells", 1) <= 1
                        and scube.digest.get("cube_version") == digest["cube_version"]
                        and scube.digest.get("transform") == digest["transform"]
                        and scube.digest.get("transform_version") == digest["transform_version"]
                        and set(group_cols).issubset(set(scube.group_cols))
                    ):
                        continue
                    sup_cols = list(scube.group_cols)
                    sup_hash = _labels_hash([np.asarray(ds.cells[c]) for c in sup_cols], sup_cols)
                    if scube.digest.get("group_hash") != sup_hash:
                        continue
                    marginal = scube.marginalize(group_cols)
                    if not _same_groups_and_cells(marginal, grouping):
                        continue                    # it leaves out cells that this grouping holds
                    fresh = _with_fresh_depths(marginal, grouping)
                    if fresh is not None:
                        return fresh
        except Exception:
            pass

    if n_groups == 0:
        empty_cube = PseudobulkCube(
            cube_key=cube_key,
            matrix_name=matrix_name,
            col_entity=col_entity,
            group_cols=group_cols,
            group_names=[],
            n_cells=np.zeros(0, dtype=np.int64),
            total_counts=np.zeros(0, dtype=np.float64),
            total_fragments=np.zeros(0, dtype=np.float64),
            sum=np.zeros((0, n_features), dtype=np.float32),
            sum_sq=np.zeros((0, n_features), dtype=np.float32),
            nnz=np.zeros((0, n_features), dtype=np.int32),
            digest=digest,
            total_counts_sq=np.zeros(0, dtype=np.float64),
        )
        return empty_cube

    n_csr = conn.execute(
        "SELECT COUNT(*) FROM matrix_chunks WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()[0]
    n_csc = conn.execute(
        "SELECT COUNT(*) FROM matrix_csc_chunks WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()[0]

    n_workers = threads if threads > 0 else min(8, os.cpu_count() or 4)
    # Limit thread count if (n_workers * G * F * 20 bytes) would exceed 2 GB
    bytes_per_worker = max(1, n_groups * n_features * 20)
    max_workers_mem = max(1, int(2_000_000_000 // bytes_per_worker))
    n_workers = max(1, min(n_workers, max_workers_mem))

    tot_c = np.zeros(n_groups, dtype=np.float64)
    tot_c_sq = np.zeros(n_groups, dtype=np.float64)
    if transform is not None:
        import scipy.sparse as sp

        sum_f64 = np.zeros((n_groups, n_features), dtype=np.float64)
        sum_sq_f64 = np.zeros((n_groups, n_features), dtype=np.float64)
        nnz_i32 = np.zeros((n_groups, n_features), dtype=np.int32)
        if n_csr > 0:
            rows = conn.execute(
                "SELECT row_start, row_end, n_nonzero, data_blob, indices_blob, indptr_blob, dtype, compression "
                "FROM matrix_chunks WHERE matrix_name = ? ORDER BY chunk_idx",
                (matrix_name,),
            )
            for r0, r1, nnz_c, d_blob, i_blob, p_blob, dtype_s, comp in rows:
                rg = cell_groups[r0:r1]
                if not np.any(rg >= 0):
                    continue
                data = np.frombuffer(decompress_blob(d_blob, comp), dtype=np.dtype(dtype_s))
                indices = np.frombuffer(decompress_blob(i_blob, comp), dtype=np.int32)
                indptr = np.frombuffer(decompress_blob(p_blob, comp), dtype=np.int32)
                _add_chunk_depths(indptr, data, rg, tot_c, tot_c_sq)
                chunk = sp.csr_matrix((data, indices, indptr), shape=(r1 - r0, n_features))
                out = transform(chunk, np.arange(r0, r1, dtype=np.int64))
                out = out.tocsr() if sp.issparse(out) else sp.csr_matrix(np.asarray(out))
                out.sort_indices()
                t_data = np.asarray(out.data, dtype=np.float64)
                t_ind = np.asarray(out.indices, dtype=np.int32)
                t_ptr = np.asarray(out.indptr, dtype=np.int32)
                if _HAS_NUMBA:
                    _accum_csr_numba(t_ptr, t_ind, t_data, rg, sum_f64, sum_sq_f64, nnz_i32)
                else:
                    _accum_csr_fallback(t_ptr, t_ind, t_data, rg, sum_f64, sum_sq_f64, nnz_i32)
        elif n_csc > 0:
            # A column chunk holds every cell of its features. The transform gets it as CSR, with every cell's
            # position and the chunk's first column (COSG's normalizers take a column chunk so), and each
            # cell's total adds up over the chunks.
            n_rows = len(cell_groups)
            all_rows = np.arange(n_rows, dtype=np.int64)
            row_sums = np.zeros(n_rows, dtype=np.float64)
            cols = conn.execute(
                "SELECT col_start, col_end, data_blob, indices_blob, indptr_blob, dtype, compression "
                "FROM matrix_csc_chunks WHERE matrix_name = ? ORDER BY chunk_idx",
                (matrix_name,),
            )
            for c0, c1, d_blob, i_blob, p_blob, dtype_s, comp in cols:
                data = np.frombuffer(decompress_blob(d_blob, comp), dtype=np.dtype(dtype_s))
                indices = np.frombuffer(decompress_blob(i_blob, comp), dtype=np.int32)
                indptr = np.frombuffer(decompress_blob(p_blob, comp), dtype=np.int32)
                _add_row_sums(indices, data, row_sums)
                chunk = sp.csc_matrix((data, indices, indptr), shape=(n_rows, int(c1) - int(c0))).tocsr()
                try:
                    out = transform(chunk, all_rows, col_start=int(c0))
                except TypeError as exc:
                    if "col_start" not in str(exc):
                        raise
                    raise TypeError(
                        f"{matrix_name!r} is stored column-major, so the transform is given column chunks and "
                        f"must accept col_start= (the chunk's first column)"
                    ) from exc
                out = out.tocsc() if sp.issparse(out) else sp.csc_matrix(np.asarray(out))
                out.sort_indices()
                t_data = np.asarray(out.data, dtype=np.float64)
                t_ind = np.asarray(out.indices, dtype=np.int32)
                t_ptr = np.asarray(out.indptr, dtype=np.int32)
                if _HAS_NUMBA:
                    _accum_csc_numba(t_ptr, t_ind, t_data, cell_groups, int(c0), sum_f64, sum_sq_f64, nnz_i32)
                else:
                    _accum_csc_fallback(t_ptr, t_ind, t_data, cell_groups, int(c0), sum_f64, sum_sq_f64, nnz_i32)
            ok = cell_groups >= 0
            tot_c = np.bincount(cell_groups[ok], weights=row_sums[ok], minlength=n_groups).astype(np.float64)
            tot_c_sq = np.bincount(cell_groups[ok], weights=row_sums[ok] ** 2, minlength=n_groups).astype(np.float64)
        else:
            raise RuntimeError(f"Matrix {matrix_name!r} has neither CSR nor CSC chunks")
    elif n_csr > 0:
        sum_f64 = np.zeros((n_groups, n_features), dtype=np.float64)
        sum_sq_f64 = np.zeros((n_groups, n_features), dtype=np.float64)
        nnz_i32 = np.zeros((n_groups, n_features), dtype=np.int32)

        chunk_meta = conn.execute(
            "SELECT chunk_idx, row_start, row_end, n_nonzero FROM matrix_chunks "
            "WHERE matrix_name = ? ORDER BY chunk_idx",
            (matrix_name,),
        ).fetchall()

        # Filter to chunks that have nonzeros and at least one valid cell
        active_chunks = [
            c_idx
            for (c_idx, r0, r1, nnz_c) in chunk_meta
            if nnz_c > 0 and np.any(cell_groups[r0:r1] >= 0)
        ]

        def _process_csr_batch(c_indices):
            loc_sum = np.zeros((n_groups, n_features), dtype=np.float64)
            loc_sq = np.zeros((n_groups, n_features), dtype=np.float64)
            loc_nnz = np.zeros((n_groups, n_features), dtype=np.int32)
            loc_tot = np.zeros(n_groups, dtype=np.float64)
            loc_tot_sq = np.zeros(n_groups, dtype=np.float64)
            for idx_i in c_indices:
                c_idx, r0, r1, nnz_c, d_blob, i_blob, p_blob, dtype_s, comp = rows_batch[int(idx_i)]
                rg = cell_groups[r0:r1]
                data = np.frombuffer(decompress_blob(d_blob, comp), dtype=np.dtype(dtype_s))
                indices = np.frombuffer(decompress_blob(i_blob, comp), dtype=np.int32)
                indptr = np.frombuffer(decompress_blob(p_blob, comp), dtype=np.int32)
                if _HAS_NUMBA:
                    _accum_csr_numba(indptr, indices, data, rg, loc_sum, loc_sq, loc_nnz)
                else:
                    _accum_csr_fallback(indptr, indices, data, rg, loc_sum, loc_sq, loc_nnz)
                _add_chunk_depths(indptr, data, rg, loc_tot, loc_tot_sq)
            return loc_sum, loc_sq, loc_nnz, loc_tot, loc_tot_sq

        batch_step = max(8, n_workers * 4)
        for b0 in range(0, len(active_chunks), batch_step):
            batch_ids = active_chunks[b0 : b0 + batch_step]
            placeholders = ",".join("?" for _ in batch_ids)
            rows_batch = conn.execute(
                f"SELECT chunk_idx, row_start, row_end, n_nonzero, data_blob, indices_blob, "
                f"indptr_blob, dtype, compression FROM matrix_chunks "
                f"WHERE matrix_name = ? AND chunk_idx IN ({placeholders}) ORDER BY chunk_idx",
                [matrix_name, *batch_ids],
            ).fetchall()
            if n_workers <= 1 or len(rows_batch) <= 1:
                for row_c in rows_batch:
                    c_idx, r0, r1, nnz_c, d_blob, i_blob, p_blob, dtype_s, comp = row_c
                    rg = cell_groups[r0:r1]
                    data = np.frombuffer(decompress_blob(d_blob, comp), dtype=np.dtype(dtype_s))
                    indices = np.frombuffer(decompress_blob(i_blob, comp), dtype=np.int32)
                    indptr = np.frombuffer(decompress_blob(p_blob, comp), dtype=np.int32)
                    if _HAS_NUMBA:
                        _accum_csr_numba(indptr, indices, data, rg, sum_f64, sum_sq_f64, nnz_i32)
                    else:
                        _accum_csr_fallback(indptr, indices, data, rg, sum_f64, sum_sq_f64, nnz_i32)
                    _add_chunk_depths(indptr, data, rg, tot_c, tot_c_sq)
            else:
                splits = np.array_split(np.arange(len(rows_batch)), min(n_workers, len(rows_batch)))
                with ThreadPoolExecutor(max_workers=len(splits)) as ex:
                    for ls, lsq, lnnz, ltot, ltot_sq in ex.map(_process_csr_batch, [s for s in splits if len(s) > 0]):
                        sum_f64 += ls
                        sum_sq_f64 += lsq
                        nnz_i32 += lnnz
                        tot_c += ltot
                        tot_c_sq += ltot_sq
    elif n_csc > 0:
        # CSC chunks partition the feature columns into disjoint [col_start, col_end) ranges!
        # Threads can write directly into disjoint column slices of shared (G, F) arrays without locks!
        sum_f64 = np.zeros((n_groups, n_features), dtype=np.float64)
        sum_sq_f64 = np.zeros((n_groups, n_features), dtype=np.float64)
        nnz_i32 = np.zeros((n_groups, n_features), dtype=np.int32)

        chunk_meta = conn.execute(
            "SELECT chunk_idx, col_start, col_end, n_nonzero FROM matrix_csc_chunks "
            "WHERE matrix_name = ? ORDER BY chunk_idx",
            (matrix_name,),
        ).fetchall()
        active_chunks = [c_idx for (c_idx, c0, c1, nnz_c) in chunk_meta if nnz_c > 0]

        n_csc_workers = threads if threads > 0 else min(8, os.cpu_count() or 4)
        n_cells_all = len(cell_groups)
        # Each cell's total sums over every column chunk; each worker adds into its own array.
        row_sum_parts = [np.zeros(n_cells_all, dtype=np.float64) for _ in range(max(1, n_csc_workers))]

        def _process_csc_row(row_c, part):
            c_idx, c0, c1, nnz_c, d_blob, i_blob, p_blob, dtype_s, comp = row_c
            data = np.frombuffer(decompress_blob(d_blob, comp), dtype=np.dtype(dtype_s))
            indices = np.frombuffer(decompress_blob(i_blob, comp), dtype=np.int32)
            indptr = np.frombuffer(decompress_blob(p_blob, comp), dtype=np.int32)
            if _HAS_NUMBA:
                _accum_csc_numba(indptr, indices, data, cell_groups, int(c0), sum_f64, sum_sq_f64, nnz_i32)
            else:
                _accum_csc_fallback(indptr, indices, data, cell_groups, int(c0), sum_f64, sum_sq_f64, nnz_i32)
            _add_row_sums(indices, data, part)

        def _process_csc_group(args):
            rows, part = args
            for row_c in rows:
                _process_csc_row(row_c, part)

        batch_step = max(16, n_csc_workers * 4)
        for b0 in range(0, len(active_chunks), batch_step):
            batch_ids = active_chunks[b0 : b0 + batch_step]
            placeholders = ",".join("?" for _ in batch_ids)
            rows_batch = conn.execute(
                f"SELECT chunk_idx, col_start, col_end, n_nonzero, data_blob, indices_blob, "
                f"indptr_blob, dtype, compression FROM matrix_csc_chunks "
                f"WHERE matrix_name = ? AND chunk_idx IN ({placeholders}) ORDER BY chunk_idx",
                [matrix_name, *batch_ids],
            ).fetchall()
            if n_csc_workers <= 1 or len(rows_batch) <= 1:
                for row_c in rows_batch:
                    _process_csc_row(row_c, row_sum_parts[0])
            else:
                k = min(n_csc_workers, len(rows_batch))
                with ThreadPoolExecutor(max_workers=k) as ex:
                    list(ex.map(_process_csc_group, [(rows_batch[j::k], row_sum_parts[j]) for j in range(k)]))
        row_sums = row_sum_parts[0]
        for part in row_sum_parts[1:]:
            row_sums += part
        ok = cell_groups >= 0
        tot_c = np.bincount(cell_groups[ok], weights=row_sums[ok], minlength=n_groups).astype(np.float64)
        tot_c_sq = np.bincount(cell_groups[ok], weights=row_sums[ok] ** 2, minlength=n_groups).astype(np.float64)
    else:
        raise RuntimeError(f"Matrix {matrix_name!r} has neither CSR nor CSC chunks")

    cube = PseudobulkCube(
        cube_key=cube_key,
        matrix_name=matrix_name,
        col_entity=col_entity,
        group_cols=group_cols,
        group_names=group_names,
        n_cells=n_cells_per_group,
        total_counts=tot_c,
        total_fragments=total_fragments,
        sum=sum_f64.astype(np.float32),
        sum_sq=sum_sq_f64.astype(np.float32),
        nnz=nnz_i32,
        digest=digest,
        total_counts_sq=tot_c_sq,
    )
    if cache and not ds.read_only:
        try:
            _save_cube_to_sqlite(conn, cube)
        except Exception:
            pass
    return cube
