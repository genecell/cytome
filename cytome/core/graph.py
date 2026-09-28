"""Sparse graph storage wrappers."""

from __future__ import annotations

from cytome.core import _handles

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
        _handles.bind(self, conn)
        self._graph_name = graph_name
        self._axis = axis
        self._entity_table = entity_table

    # A chunk holds about this many entries: a few megabytes compressed.
    _CHUNK_NNZ = 1 << 20

    def _chunked_meta(self):
        try:
            return self._conn.execute(
                "SELECT n_rows, n_cols, n_nonzero, dtype, n_chunks FROM graph_meta "
                "WHERE graph_name = ? AND axis = ?", (self._graph_name, self._axis)).fetchone()
        except sqlite3.OperationalError:
            return None

    @property
    def shape(self) -> tuple[int, int]:
        meta = self._chunked_meta()
        if meta is not None:
            return int(meta[0]), int(meta[1])
        return self.to_sparse().shape

    def iter_chunks(self):
        """``(row_start, row_end, csr block)`` in row order, one stored chunk at a
        time; for a consumer that does not need the whole graph at once."""
        from cytome.io.compression import decompress_blob
        meta = self._chunked_meta()
        if meta is None:
            A = self.to_sparse()
            yield 0, A.shape[0], A
            return
        n_cols, dtype = int(meta[1]), np.dtype(meta[3])
        for r0, r1, d, i, p, comp in self._conn.execute(
                "SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression "
                "FROM graph_chunks WHERE graph_name = ? AND axis = ? ORDER BY chunk_idx",
                (self._graph_name, self._axis)):
            yield int(r0), int(r1), sp.csr_matrix(
                (np.frombuffer(decompress_blob(d, comp), dtype=dtype),
                 np.frombuffer(decompress_blob(i, comp), dtype=np.int32),
                 np.frombuffer(decompress_blob(p, comp), dtype=np.int64)),
                shape=(int(r1) - int(r0), n_cols))

    def _read_chunked(self, meta) -> sp.csr_matrix:
        """The whole graph from its chunks: the arrays are joined, never one
        Python object per entry."""
        from cytome.io.compression import decompress_blob
        n_rows, n_cols, nnz, dtype = int(meta[0]), int(meta[1]), int(meta[2]), np.dtype(meta[3])
        data = np.empty(nnz, dtype=dtype)
        indices = np.empty(nnz, dtype=np.int32)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        at = 0
        for r0, r1, d, i, p, comp in self._conn.execute(
                "SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression "
                "FROM graph_chunks WHERE graph_name = ? AND axis = ? ORDER BY chunk_idx",
                (self._graph_name, self._axis)):
            ptr = np.frombuffer(decompress_blob(p, comp), dtype=np.int64)
            k = int(ptr[-1])
            data[at:at + k] = np.frombuffer(decompress_blob(d, comp), dtype=dtype)
            indices[at:at + k] = np.frombuffer(decompress_blob(i, comp), dtype=np.int32)
            indptr[int(r0) + 1:int(r1) + 1] = ptr[1:] + at
            at += k
        if at != nnz:
            raise ValueError(f"graph {self._graph_name!r}: {at} entries in its chunks, {nnz} recorded")
        return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))

    def to_sparse(self) -> sp.csr_matrix:
        """Return graph as CSR sparse matrix."""
        meta = self._chunked_meta()
        if meta is not None:
            return self._read_chunked(meta)
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
        """Write the graph as compressed chunks of rows, replacing any earlier
        version of it (chunked or in the per-edge tables of older files).

        ``peak_gene`` keeps its own table, which has more columns than a
        weight."""
        if self._graph_name == "peak_gene":
            return self._write_rows(matrix)
        from cytome.io.compression import compress_blob
        A = sp.csr_matrix(matrix)
        A.sum_duplicates()
        A.sort_indices()
        dtype = A.dtype if A.dtype in (np.float32, np.float64) else np.dtype(np.float32)
        data = np.ascontiguousarray(A.data, dtype=dtype)
        indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        indptr = np.asarray(A.indptr, dtype=np.int64)
        name, axis = self._graph_name, self._axis
        self._drop()
        n_rows, n_chunks, r0 = A.shape[0], 0, 0
        while r0 < n_rows or (n_rows == 0 and n_chunks == 0):
            # rows up to about _CHUNK_NNZ entries, at least one row
            r1 = int(np.searchsorted(indptr, indptr[r0] + self._CHUNK_NNZ, side="right")) - 1
            r1 = min(max(r1, r0 + 1), n_rows)
            lo, hi = int(indptr[r0]), int(indptr[r1])
            ptr = indptr[r0:r1 + 1] - lo
            self._conn.execute(
                "INSERT INTO graph_chunks (graph_name, axis, chunk_idx, row_start, row_end, n_nonzero, "
                "data_blob, indices_blob, indptr_blob, compression) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (name, axis, n_chunks, r0, r1, hi - lo,
                 compress_blob(data[lo:hi].tobytes(), "lz4"),
                 compress_blob(indices[lo:hi].tobytes(), "lz4"),
                 compress_blob(np.ascontiguousarray(ptr).tobytes(), "lz4"), "lz4"))
            n_chunks += 1
            r0 = r1
            if n_rows == 0:
                break
        self._conn.execute(
            "INSERT INTO graph_meta (graph_name, axis, entity_table, n_rows, n_cols, n_nonzero, dtype, n_chunks) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (name, axis, self._entity_table, int(A.shape[0]), int(A.shape[1]), int(A.nnz),
             str(np.dtype(dtype)), n_chunks))

    def _drop(self) -> None:
        name, axis = self._graph_name, self._axis
        self._conn.execute("DELETE FROM graph_chunks WHERE graph_name = ? AND axis = ?", (name, axis))
        self._conn.execute("DELETE FROM graph_meta WHERE graph_name = ? AND axis = ?", (name, axis))
        self._conn.execute("DELETE FROM graph_edges WHERE graph_name = ? AND axis = ?", (name, axis))
        if name == "knn" and axis == "obs":
            self._conn.execute("DELETE FROM graph_knn")

    def _write_rows(self, matrix: sp.spmatrix) -> None:
        """The per-edge tables, as written before 0.3.6."""
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
