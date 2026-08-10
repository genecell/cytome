"""Sparse graph storage wrappers."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import sqlite3


class GraphStore:
    """Graph-backed sparse adjacency matrix interface."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        graph_name: str,
        axis: str = "obs",
        entity_table: str = "cells",
    ) -> None:
        self._conn = conn
        self._graph_name = graph_name
        self._axis = axis
        self._entity_table = entity_table

    def to_sparse(self) -> sp.csr_matrix:
        """Return graph as CSR sparse matrix."""
        if self._graph_name == "knn":
            rows = self._conn.execute(
                "SELECT cell_i, cell_j, weight FROM graph_knn"
            ).fetchall()
            if not rows:
                return sp.csr_matrix((0, 0), dtype=np.float32)
            i = np.array([r[0] for r in rows], dtype=np.int32)
            j = np.array([r[1] for r in rows], dtype=np.int32)
            w = np.array([r[2] for r in rows], dtype=np.float32)
            n = int(max(i.max(), j.max()) + 1)
            return sp.csr_matrix((w, (i, j)), shape=(n, n))

        if self._graph_name == "peak_gene":
            rows = self._conn.execute(
                "SELECT peak_idx, gene_idx, correlation FROM graph_peak_gene"
            ).fetchall()
            if not rows:
                return sp.csr_matrix((0, 0), dtype=np.float32)
            i = np.array([r[0] for r in rows], dtype=np.int32)
            j = np.array([r[1] for r in rows], dtype=np.int32)
            w = np.array([r[2] if r[2] is not None else 0.0 for r in rows], dtype=np.float32)
            shape = (int(i.max() + 1), int(j.max() + 1))
            return sp.csr_matrix((w, (i, j)), shape=shape)

        try:
            rows = self._conn.execute(
                """
                SELECT row_idx, col_idx, value
                FROM graph_edges
                WHERE graph_name = ? AND axis = ?
                """,
                (self._graph_name, self._axis),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise KeyError(f"Unknown graph: {self._graph_name}") from exc
        if not rows:
            raise KeyError(f"Unknown graph: {self._graph_name}")
        i = np.array([r[0] for r in rows], dtype=np.int32)
        j = np.array([r[1] for r in rows], dtype=np.int32)
        w = np.array([r[2] for r in rows], dtype=np.float32)
        n = int(max(i.max(), j.max()) + 1) if i.size else 0
        return sp.csr_matrix((w, (i, j)), shape=(n, n))

    def write_sparse(self, matrix: sp.spmatrix) -> None:
        """Write sparse graph matrix to backing graph table."""
        coo = matrix.tocoo()
        if self._graph_name == "knn":
            self._conn.execute("DELETE FROM graph_knn")
            self._conn.executemany(
                "INSERT INTO graph_knn(cell_i, cell_j, weight) VALUES (?, ?, ?)",
                list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist())),
            )
            return

        try:
            self._conn.execute(
                "DELETE FROM graph_edges WHERE graph_name = ? AND axis = ?",
                (self._graph_name, self._axis),
            )
        except sqlite3.OperationalError as exc:
            raise KeyError(f"Unknown graph: {self._graph_name}") from exc
        self._conn.executemany(
            """
            INSERT INTO graph_edges(graph_name, axis, entity_table, row_idx, col_idx, value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    self._graph_name,
                    self._axis,
                    self._entity_table,
                    int(r),
                    int(c),
                    float(v),
                )
                for r, c, v in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist())
            ],
        )
