"""Renumber cells to their positions, carrying the tables that name a cell by its ``cell_idx``.

Matrices, embeddings, graphs and fragments refer to a cell by its position: its row in ``cells`` read in
``rowid`` order (``tests/test_fragments_follow_cell_position.py``). ``spatial_coords`` and ``cell_sets``
refer to it by its ``cell_idx`` value. Since entity keys are kept at ``0..n-1`` the two are the same. A
store written before that can have gaps in ``cell_idx``, and renumbering its cells then has to carry these
two tables along.

Fragments are not rewritten, unless the store's own per-cell fragment counts (``cells.n_fragments``) show
that they were written by ``cell_idx`` value; :func:`fragment_cell_reading` decides that once for the
whole store.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _has_table(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (name,)
    ).fetchone() is not None


@dataclass
class CellValueRemap:
    """New rows of the tables keyed by ``cell_idx`` value, worked out before the cells are renumbered."""

    spatial: Optional[list]
    cell_sets: Optional[list]
    n_spatial_dropped: int = 0
    n_set_members_dropped: int = 0


def plan_cell_value_remap(conn, old_ids) -> CellValueRemap:
    """Map ``spatial_coords`` and ``cell_sets`` from the ``cell_idx`` values ``old_ids`` to positions.

    ``old_ids[i]`` is the ``cell_idx`` of the cell at position ``i``. A row or a set member naming a
    ``cell_idx`` that is not among ``old_ids`` is dropped and counted. ``spatial`` or ``cell_sets`` is
    ``None`` when that table is absent or empty, so there is nothing to rewrite.
    """
    pos_of = {int(c): i for i, c in enumerate(old_ids)}
    spatial, n_sp_drop = None, 0
    if _has_table(conn, "spatial_coords"):
        rows = conn.execute("SELECT cell_idx, x, y, z FROM spatial_coords").fetchall()
        if rows:
            spatial = []
            for ci, x, y, z in rows:
                p = pos_of.get(int(ci))
                if p is None:
                    n_sp_drop += 1
                else:
                    spatial.append((p, x, y, z))
    sets, n_mem_drop = None, 0
    if _has_table(conn, "cell_sets"):
        rows = conn.execute("SELECT set_name, cell_indices FROM cell_sets").fetchall()
        if rows:
            sets = []
            for name, members in rows:
                idx = json.loads(members) if members else []
                kept = [pos_of[int(i)] for i in idx if int(i) in pos_of]
                n_mem_drop += len(idx) - len(kept)
                sets.append((name, json.dumps(kept)))
    return CellValueRemap(spatial, sets, n_sp_drop, n_mem_drop)


def apply_cell_value_remap(conn, plan: CellValueRemap) -> None:
    """Write the rows of :func:`plan_cell_value_remap`, once ``cells`` holds the new numbers.

    Foreign keys must be deferred (``PRAGMA defer_foreign_keys``) by the caller's transaction.
    """
    if plan.spatial is not None:
        conn.execute("DELETE FROM spatial_coords")
        conn.executemany(
            "INSERT INTO spatial_coords (cell_idx, x, y, z) VALUES (?, ?, ?, ?)", plan.spatial
        )
        if _has_table(conn, "spatial_rtree"):
            conn.execute("DELETE FROM spatial_rtree")
            conn.executemany(
                "INSERT INTO spatial_rtree (id, min_x, max_x, min_y, max_y) VALUES (?, ?, ?, ?, ?)",
                [(p, x, x, y, y) for p, x, y, _z in plan.spatial],
            )
    if plan.cell_sets is not None:
        conn.executemany(
            "UPDATE cell_sets SET cell_indices = ? WHERE set_name = ?",
            [(js, name) for name, js in plan.cell_sets],
        )
    if plan.n_spatial_dropped or plan.n_set_members_dropped:
        warnings.warn(
            f"renumbering cells dropped {plan.n_spatial_dropped} spatial row(s) and "
            f"{plan.n_set_members_dropped} cell-set member(s) that named no cell of the table",
            stacklevel=3,
        )


def fragment_cell_reading(conn, old_ids) -> str:
    """How the store's fragments name their cells: ``'none'``, ``'position'`` or ``'cell_idx'``.

    Decided once for the whole store. Each cell's fragments are counted under both readings, and the reading
    whose counts equal ``cells.n_fragments`` wins. Without that column, fragment ids all below ``n`` read as
    positions (what every writer produces) and larger ids as ``cell_idx`` values. Raises ``ValueError`` when
    neither reading fits, or when both fit but assign fragments to different cells.
    """
    from cytome.io.compression import decompress_blob

    if not _has_table(conn, "fragment_chunks"):
        return "none"
    old = np.asarray(old_ids, dtype=np.int64)
    n = old.size
    lut = np.full(int(old.max()) + 1 if n else 1, -1, dtype=np.int64)
    lut[old] = np.arange(n, dtype=np.int64)
    cnt_pos = np.zeros(n, dtype=np.int64)
    cnt_cid = np.zeros(n, dtype=np.int64)
    seen = False
    pos_possible = cid_possible = identity = True
    for blob, comp in conn.execute("SELECT cell_idx_blob, compression FROM fragment_chunks"):
        ids = np.frombuffer(decompress_blob(blob, comp), dtype=np.int32).astype(np.int64)
        if ids.size == 0:
            continue
        seen = True
        if int(ids.min()) < 0:
            raise ValueError("fragment_chunks holds negative cell ids; the store cannot be renumbered")
        if int(ids.max()) >= n:
            pos_possible = False
        elif pos_possible:
            cnt_pos += np.bincount(ids, minlength=n)
        mapped = np.where(ids < lut.size, lut[np.minimum(ids, lut.size - 1)], -1)
        if (mapped < 0).any():
            cid_possible = False
        elif cid_possible:
            cnt_cid += np.bincount(mapped, minlength=n)
            identity = identity and bool(np.array_equal(mapped, ids))
    if not seen:
        return "none"
    nf = None
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cells)")}
    if "n_fragments" in cols:
        vals = [r[0] for r in conn.execute("SELECT n_fragments FROM cells ORDER BY rowid")]
        if vals and all(v is not None for v in vals) and sum(vals) > 0:
            nf = np.asarray(vals, dtype=np.int64)
    if nf is not None:
        pos_ok = pos_possible and bool(np.array_equal(cnt_pos, nf))
        cid_ok = cid_possible and bool(np.array_equal(cnt_cid, nf))
    else:
        pos_ok = pos_possible
        cid_ok = cid_possible and not pos_possible
    if pos_ok and (not cid_ok or identity):
        return "position"
    if cid_ok and not pos_ok:
        return "cell_idx"
    raise ValueError(
        "the store's fragments fit neither reading of their cell ids (by position or by cell_idx), "
        "checked against cells.n_fragments; the cells were left as they are"
    )


def rewrite_fragment_cells_to_positions(conn, old_ids) -> int:
    """Rewrite every fragment chunk's cell ids from ``cell_idx`` values to positions; returns chunks written."""
    from cytome.io.compression import compress_blob, decompress_blob

    old = np.asarray(old_ids, dtype=np.int64)
    lut = np.full(int(old.max()) + 1, -1, dtype=np.int64)
    lut[old] = np.arange(old.size, dtype=np.int64)
    rows = conn.execute("SELECT id, cell_idx_blob, compression FROM fragment_chunks").fetchall()
    for cid, blob, comp in rows:
        ids = np.frombuffer(decompress_blob(blob, comp), dtype=np.int32).astype(np.int64)
        new = lut[ids].astype(np.int32)
        conn.execute(
            "UPDATE fragment_chunks SET cell_idx_blob = ? WHERE id = ?",
            (compress_blob(new.tobytes(), comp), int(cid)),
        )
    return len(rows)
