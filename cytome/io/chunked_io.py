"""Chunked sparse and dense array IO for Cytome.

The stored form of a sparse matrix (``matrix_chunks`` / ``matrix_csc_chunks``)
is the contract of ``docs/FORMAT_SPEC.md``: per chunk, a canonical CSR (or CSC)
block -- one entry per (row, column), indices sorted -- with ``int32`` indices
and chunk-local ``int32`` index pointers, values little-endian in the dtype the
chunk's ``dtype`` column names, which is the matrix's ``matrix_meta.dtype``.
The writers here produce exactly that and refuse what they cannot store
(a value past int32, an object dtype, a float written into an integer layer
that is not integral).

The readers are lenient where a store written by an earlier version departs
from it, and say so: an index blob is read as int32 or int64 by its length, a
chunk is decoded by its own ``dtype`` (``merge`` of mixed inputs wrote int32
chunks under a float32 matrix), rows are placed by ``row_start`` (not by
``chunk_idx``), and a full read uses an int64 index pointer once the matrix
holds more than 2**31 - 1 entries.
"""

from __future__ import annotations

import functools
import warnings
from datetime import datetime, timezone
from typing import Iterator, Tuple

import numpy as np
import scipy.sparse as sp
import sqlite3

from .compression import compress_blob, decompress_blob  # noqa: F401

_INSERT_CHUNK_SQL = """
INSERT INTO matrix_chunks(
    matrix_name, chunk_idx, row_start, row_end, n_nonzero,
    data_blob, indices_blob, indptr_blob, dtype, compression
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INT32_MAX = int(np.iinfo(np.int32).max)
_I4 = np.dtype("<i4")
_I8 = np.dtype("<i8")
#: chunk ids per SELECT ... IN (...): SQLite's stock limit on bound parameters
#: is 32,766 (999 before 3.32), and a masked read can need every chunk
_IN_BATCH = 500


def _values_are_integer(dtype_str, sample) -> int | None:
    """1 / 0 / None for "these values are integers".

    Decided by the writer, once, while the data is in hand. Consumers that
    re-derive it from a stored sample get it wrong in ways that are hard to
    notice -- a probe that silently answers "cannot tell" reads exactly like
    "not integer" at the call site.
    """
    import numpy as _np
    try:
        if _np.issubdtype(_np.dtype(dtype_str), _np.integer):
            return 1
        a = _np.asarray(sample, dtype="float64")
        if a.size == 0:
            return 1                      # an all-zero matrix is integral
        return int(bool(_np.allclose(a, _np.round(a))))
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# The stored form: helpers shared by every writer and reader
# --------------------------------------------------------------------------

def _index_dtype(n: int):
    """The index dtype that holds ``n``: int32 up to 2**31 - 1, else int64."""
    return np.int64 if int(n) > _INT32_MAX else np.int32


@functools.lru_cache(maxsize=64)
def _np_dtype(name: str) -> np.dtype:
    return np.dtype(name)


def storage_dtype(dtype) -> np.dtype:
    """The dtype values are stored in: ``dtype`` in little-endian byte order.

    Refuses what a chunk cannot hold (object, strings, structured dtypes):
    their ``tobytes()`` is pointers or padding, which no reader can decode.
    """
    dt = np.dtype(dtype)
    if dt.kind not in "biufc" or dt.fields is not None:
        raise TypeError(f"a cytome matrix holds numbers; dtype {dt} cannot be stored")
    return dt.newbyteorder("<") if dt.itemsize > 1 else dt


def _check_cast(values: np.ndarray, target: np.dtype, what: str) -> np.ndarray:
    """``values`` as ``target``, refusing a cast that changes a value.

    Floats into an integer dtype must be integral and in range, and an integer
    must fit a narrower integer; float to float (float64 to float32) narrows
    precision by request and is allowed.
    """
    if values.dtype == target:
        return values
    out = values.astype(target)
    src_kind, dst_kind = values.dtype.kind, target.kind
    if dst_kind in "iu" and src_kind in "fciu" and values.size:
        if src_kind in "fc" or values.dtype.itemsize > target.itemsize or src_kind != dst_kind:
            if not np.array_equal(out.astype(values.dtype), values):
                raise ValueError(
                    f"{what}: values of dtype {values.dtype} do not fit {target} "
                    f"(not integral or out of range); store them as a float dtype")
    return out


def _canonical_csr(block, dtype: np.dtype, what: str):
    """``(data, indices, indptr)`` of ``block`` in the stored form.

    One entry per (row, column) with the indices of each row sorted (scipy's
    canonical form; a copy when ``block`` is not), ``data`` as ``dtype``,
    ``indices`` int32 (a column past 2**31 - 1 is refused) and ``indptr`` int64
    over the whole block: the chunk-local int32 index pointers are cut from it
    by :func:`_chunk_payload`. Works for ``csr_matrix`` and ``csr_array``, whose
    index arrays stay int64 through slicing.
    """
    csr = block.tocsr() if sp.issparse(block) else sp.csr_matrix(np.asarray(block))
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    data = np.ascontiguousarray(_check_cast(np.asarray(csr.data), dtype, what))
    indices = np.asarray(csr.indices)
    if indices.dtype != np.int32:
        if indices.size and int(indices.max()) > _INT32_MAX:
            raise ValueError(f"{what}: a column index past 2**31 - 1 cannot be stored")
        indices = indices.astype(np.int32)
    return data, np.ascontiguousarray(indices), np.asarray(csr.indptr, dtype=np.int64)


def _chunk_payload(data, indices, indptr, r0: int, r1: int, compression: str, what: str):
    """``(n_nonzero, data_blob, indices_blob, indptr_blob)`` of rows ``[r0, r1)``
    of a canonical block (see :func:`_canonical_csr`)."""
    lo, hi = int(indptr[r0]), int(indptr[r1])
    if hi - lo > _INT32_MAX:
        raise ValueError(f"{what}: {hi - lo:,} entries in one chunk is past an int32 "
                         f"index pointer; write fewer rows per chunk")
    local = (indptr[r0:r1 + 1] - lo).astype(_I4)
    return (hi - lo,
            compress_blob(data[lo:hi].tobytes(), method=compression),
            compress_blob(indices[lo:hi].astype(_I4, copy=False).tobytes(), method=compression),
            compress_blob(local.tobytes(), method=compression))


def _frombuffer_index(raw: bytes, n: int, what: str) -> np.ndarray:
    """An index blob of ``n`` values: int32 as specified, or int64 as some
    earlier writers stored it (decided by its length)."""
    nb = len(raw)
    if nb == 4 * n:
        return np.frombuffer(raw, dtype=_I4)
    if nb == 8 * n:
        arr = np.frombuffer(raw, dtype=_I8)
        if arr.size == 0 or int(arr.max()) <= _INT32_MAX:
            return arr.astype(np.int32)
        return arr
    raise ValueError(f"{what}: {nb} bytes cannot be {n} int32 or int64 values; "
                     f"the chunk is damaged")


#: matrices already reported as holding chunks of another dtype (one warning each)
_MIXED_REPORTED: set = set()


def _frombuffer_data(raw: bytes, chunk_dtype, meta_dtype: np.dtype, nnz: int,
                     what: str, matrix_name: str = "") -> np.ndarray:
    """A data blob of ``nnz`` values, decoded by the chunk's own dtype and
    returned as the matrix's.

    The label beside a chunk is what its bytes are. ``merge`` of inputs of
    different dtypes (cytome 0.3.8 and earlier) wrote each input's chunks in
    their own dtype under the first input's ``matrix_meta.dtype``, and decoding
    them all by that dtype turned a count of 2 into 1,073,741,824. Such a chunk
    is decoded by its label and converted, with a warning (and a second one if
    the conversion changes a value). When the label disagrees with the blob's
    length (no writer should allow that) the matrix's dtype is tried.
    """
    # the label names the byte order too ('>f4' for big-endian bytes)
    cd = _np_dtype(str(chunk_dtype)) if chunk_dtype else meta_dtype
    if len(raw) != nnz * cd.itemsize:
        if len(raw) == nnz * meta_dtype.itemsize:
            cd = meta_dtype
        else:
            raise ValueError(f"{what}: {len(raw)} bytes cannot be {nnz} values of {cd}; "
                             f"the chunk is damaged")
    arr = np.frombuffer(raw, dtype=cd)
    if arr.dtype == meta_dtype:
        return arr
    out = arr.astype(meta_dtype)
    if (cd.kind, cd.itemsize) != (meta_dtype.kind, meta_dtype.itemsize):   # not just byte order
        key = (matrix_name or what, str(cd))
        lossy = not np.array_equal(out.astype(cd), arr, equal_nan=cd.kind in "fc")
        if key not in _MIXED_REPORTED or lossy:
            _MIXED_REPORTED.add(key)
            warnings.warn(
                f"{matrix_name or what}: chunks stored as {cd} under the matrix dtype "
                f"{meta_dtype} (a merge of inputs of different dtypes by cytome 0.3.8 or "
                f"earlier) are decoded by their own dtype"
                + (" and some values do not fit the matrix dtype" if lossy else "")
                + "; ds.repair() rewrites the matrix in one dtype.",
                UserWarning, stacklevel=5)
    return out


def _decode_block(n_major: int, n_minor: int, data_blob, indices_blob, indptr_blob, compression,
                  chunk_dtype, meta_dtype, what: str, fmt: str = "csr", matrix_name: str = ""):
    """One stored chunk as a sparse block: ``(n_major, n_minor)`` CSR rows, or
    ``(n_minor, n_major)`` CSC columns with ``fmt="csc"``."""
    indptr = _frombuffer_index(decompress_blob(indptr_blob, compression), n_major + 1,
                               f"{what} indptr")
    nnz = int(indptr[-1]) if indptr.size else 0
    indices = _frombuffer_index(decompress_blob(indices_blob, compression), nnz,
                                f"{what} indices")
    data = _frombuffer_data(decompress_blob(data_blob, compression), chunk_dtype,
                            meta_dtype, nnz, f"{what} data", matrix_name)
    if fmt == "csc":
        return sp.csc_matrix((data, indices, indptr), shape=(n_minor, n_major))
    return sp.csr_matrix((data, indices, indptr), shape=(n_major, n_minor))


class _Plan:
    """The chunks of one row-major matrix, in row order: their ids, row spans
    and recorded sizes.

    Only columns stored before the blobs are read (``dtype`` follows them, and
    reading it walks each blob's overflow pages: 70 times slower, measured),
    from the covering index ``idx_matrix_chunks_span`` where the store has it.
    """

    __slots__ = ("ids", "starts", "ends", "nnz", "in_order")

    def __init__(self, conn, matrix_name: str):
        rows = conn.execute(
            "SELECT chunk_idx, row_start, row_end, COALESCE(n_nonzero, 0) FROM matrix_chunks "
            "WHERE matrix_name = ? ORDER BY chunk_idx", (matrix_name,)).fetchall()
        arr = np.asarray(rows, dtype=np.int64).reshape(-1, 4) if rows else \
            np.zeros((0, 4), dtype=np.int64)
        ids, starts, ends, nnz = (np.ascontiguousarray(arr[:, i]) for i in range(4))
        # chunk order is row order for every writer of this package; a store
        # whose chunks were written out of order is read in row order anyway
        self.in_order = bool(len(rows) < 2 or np.all(starts[1:] >= ends[:-1]))
        if not self.in_order:
            order = np.lexsort((ids, starts))
            ids, starts, ends, nnz = ids[order], starts[order], ends[order], nnz[order]
            bad = np.flatnonzero(starts[1:] < ends[:-1])
            if bad.size:
                j = int(bad[0])
                raise ValueError(
                    f"{matrix_name}: chunks {int(ids[j])} and {int(ids[j + 1])} hold the same "
                    f"rows ({int(starts[j + 1])} to {int(ends[j]) - 1}); the matrix is damaged")
        self.ids, self.starts, self.ends, self.nnz = ids, starts, ends, nnz

    def __len__(self) -> int:
        return int(self.ids.size)

    def touching(self, wanted_sorted: np.ndarray) -> np.ndarray:
        """Positions (in row order) of the chunks holding any of ``wanted_sorted``."""
        lo = np.searchsorted(wanted_sorted, self.starts, side="left")
        hi = np.searchsorted(wanted_sorted, self.ends, side="left")
        return np.flatnonzero(hi > lo)


def _blob_rows(conn, matrix_name: str, plan: _Plan, positions=None):
    """``(row_start, row_end, data_blob, indices_blob, indptr_blob, compression, dtype)``
    for the chunks at ``positions`` of ``plan`` (all when None), in row order.

    A full read of a matrix in chunk order streams one cursor; anything else
    asks for the chunks by id, a bounded batch per statement."""
    if positions is None and plan.in_order:
        cur = conn.execute(
            "SELECT row_start, row_end, data_blob, indices_blob, indptr_blob, compression, dtype "
            "FROM matrix_chunks WHERE matrix_name = ? ORDER BY chunk_idx", (matrix_name,))
        yield from cur
        return
    pos = np.arange(len(plan)) if positions is None else np.asarray(positions, dtype=np.int64)
    step = _in_batch(conn)
    for b in range(0, pos.size, step):
        part = pos[b:b + step]
        ids = [int(plan.ids[p]) for p in part]
        got = {}
        for row in conn.execute(
                "SELECT chunk_idx, row_start, row_end, data_blob, indices_blob, indptr_blob, "
                f"compression, dtype FROM matrix_chunks WHERE matrix_name = ? AND chunk_idx IN "
                f"({','.join('?' * len(ids))})", (matrix_name, *ids)):
            got[int(row[0])] = row[1:]
        for cid in ids:
            if cid not in got:
                raise KeyError(f"{matrix_name}: chunk {cid} vanished during the read")
            yield got[cid]


def _in_batch(conn) -> int:
    """Chunk ids per ``IN (...)``: at most ``_IN_BATCH``, and under the
    connection's limit on bound parameters where Python can read it."""
    try:
        limit = int(conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
    except (AttributeError, sqlite3.Error):        # Python < 3.11: SQLite's old default
        limit = 999
    return max(1, min(_IN_BATCH, limit - 1))


def _matrix_meta(conn, matrix_name: str):
    meta = conn.execute(
        "SELECT n_rows, n_cols, dtype, n_nonzero FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,)).fetchone()
    if meta is None:
        raise KeyError(f"Matrix not found: {matrix_name}")
    return (int(meta[0]), int(meta[1]), _np_dtype(str(meta[2])),
            int(meta[3]) if meta[3] is not None else None)


def _empty_rows(n_rows: int, n_cols: int, dtype) -> sp.csr_matrix:
    return sp.csr_matrix((max(0, int(n_rows)), int(n_cols)), dtype=dtype)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

class ChunkedLayerWriter:
    """Write a sparse layer incrementally, chunk by chunk.

    Each call to :meth:`write_chunk` splits the compute-sized chunk into
    storage-sized sub-chunks, compresses them with zstd, and writes them
    to the ``matrix_chunks`` table immediately via ``executemany``.

    The layer is written as one transaction on ``conn``: the old version of
    the matrix is deleted at construction and the new one is committed by
    :meth:`finalize`. The write lock is held for the whole time, so a second
    process writing the same store waits (up to the 60 s busy timeout) and
    then fails; one writer per store at a time.

    Parameters
    ----------
    conn
        Open SQLite connection (WAL mode recommended).
    matrix_name
        Layer identifier (e.g. ``"RNA_infog"``).
    n_rows
        Total number of rows that will be written.
    n_cols
        Number of columns (features / genes).
    dtype
        Numpy dtype for values (default ``float64``). Values are converted to
        it; a float value written into an integer layer must be integral.
    compression
        Blob compression method (default ``"zstd"``).
    row_entity
        Row entity table name.
    col_entity
        Column entity table name.
    overwrite
        If *True*, delete any existing data for *matrix_name* first (its row
        chunks and any column chunks).
    storage_chunk_size
        Rows per storage blob.  Default 128 gives 8x fewer blobs than
        the previous default of 16.

    Chunks are written in row order: a chunk may not start before the end of
    the previous one (rows a writer skips are zero rows), nor pass ``n_rows``.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        matrix_name: str,
        n_rows: int,
        n_cols: int,
        dtype: np.dtype | str = np.float64,
        compression: str = "zstd",
        row_entity: str = "cells",
        col_entity: str = "genes",
        overwrite: bool = True,
        storage_chunk_size: int = 128,
    ) -> None:
        self._conn = conn
        self._matrix_name = matrix_name
        self._n_rows = int(n_rows)
        self._n_cols = int(n_cols)
        self._dtype = storage_dtype(dtype)
        self._dtype_str = str(self._dtype)
        self._compression = compression
        self._row_entity = row_entity
        self._col_entity = col_entity
        self._storage_chunk_size = max(1, int(storage_chunk_size))

        self._chunk_idx = 0
        self._total_nnz = 0
        self._next_row = 0
        self._is_integer = None      # 1 / 0 / None, settled as chunks arrive
        self._finalized = False

        from cytome.io.sqlite_engine import ensure_chunk_span_indexes
        ensure_chunk_span_indexes(conn)
        if overwrite:
            conn.execute(
                "DELETE FROM matrix_chunks WHERE matrix_name = ?",
                (matrix_name,),
            )
            conn.execute(
                "DELETE FROM matrix_csc_chunks WHERE matrix_name = ?",
                (matrix_name,),
            )
            conn.execute(
                "DELETE FROM matrix_meta WHERE matrix_name = ?",
                (matrix_name,),
            )

    @property
    def finalized(self) -> bool:
        return self._finalized

    def write_chunk(self, sparse_chunk: sp.spmatrix, row_offset: int) -> None:
        """Write a compute-sized chunk, splitting into storage sub-chunks.

        All sub-chunks from one compute batch are inserted via a single
        ``executemany`` call.

        Parameters
        ----------
        sparse_chunk
            CSR (or convertible) sparse matrix of shape
            ``(chunk_rows, n_cols)``.
        row_offset
            Global row index of the first row in *sparse_chunk*: at or after
            the end of the previous chunk.
        """
        if self._finalized:
            raise RuntimeError("Cannot write after finalize()")

        name = self._matrix_name
        data, indices, indptr = _canonical_csr(sparse_chunk, self._dtype, name)
        n_chunk_rows = indptr.size - 1
        n_cols = sparse_chunk.shape[1]
        row_offset = int(row_offset)
        if n_cols != self._n_cols:
            raise ValueError(f"{name}: a chunk of {n_cols} columns for a layer of {self._n_cols}")
        if row_offset < self._next_row:
            raise ValueError(
                f"{name}: a chunk at row {row_offset} after one ending at row {self._next_row}; "
                f"chunks are written in row order (rows read back by their position)")
        if row_offset + n_chunk_rows > self._n_rows:
            raise ValueError(
                f"{name}: rows {row_offset} to {row_offset + n_chunk_rows - 1} pass the "
                f"layer's {self._n_rows} rows")

        scs = self._storage_chunk_size
        compression = self._compression
        dtype_str = self._dtype_str

        # Build all sub-chunk rows, then insert in one executemany call
        rows = []
        for sub_start in range(0, n_chunk_rows, scs):
            sub_end = min(sub_start + scs, n_chunk_rows)
            nnz, d_blob, i_blob, p_blob = _chunk_payload(
                data, indices, indptr, sub_start, sub_end, compression, name)
            rows.append((
                name,
                self._chunk_idx,
                row_offset + sub_start,
                row_offset + sub_end,
                int(nnz),
                d_blob, i_blob, p_blob,
                dtype_str,
                compression,
            ))
            if self._is_integer != 0:
                # one non-integral chunk settles it: the matrix is
                # integral only if every chunk is
                lo = int(indptr[sub_start])
                _v = _values_are_integer(dtype_str, data[lo:lo + min(int(nnz), 20000)])
                if _v == 0:
                    self._is_integer = 0
                elif self._is_integer is None:
                    self._is_integer = _v
            self._total_nnz += int(nnz)
            self._chunk_idx += 1

        self._conn.executemany(_INSERT_CHUNK_SQL, rows)
        self._next_row = row_offset + n_chunk_rows

    def finalize(self, provenance_id: int | None = None) -> None:
        """Write matrix metadata, commit, and fold what the WAL can spare.

        The checkpoint is PASSIVE: a TRUNCATE checkpoint waits for every other
        connection's read to finish, which is the whole busy timeout (60 s)
        whenever another process has the store open.
        """
        if self._finalized:
            raise RuntimeError("Already finalized")
        self._finalized = True

        self._conn.execute(
            """
            INSERT INTO matrix_meta(
                matrix_name, n_rows, n_cols, n_nonzero, dtype,
                row_entity, col_entity, chunk_size, n_chunks,
                created_at, provenance_id, is_integer
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._matrix_name,
                int(self._n_rows),
                int(self._n_cols),
                int(self._total_nnz),
                self._dtype_str,
                self._row_entity,
                self._col_entity,
                int(self._storage_chunk_size),
                int(self._chunk_idx),
                _now_iso(),
                provenance_id,
                self._is_integer,
            ),
        )
        self._conn.commit()
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.OperationalError:
            pass                  # a busy store keeps its WAL; nothing is lost


def write_sparse_chunked(
    conn: sqlite3.Connection,
    matrix_name: str,
    csr_matrix: sp.csr_matrix,
    chunk_size: int,
    compression: str,
    row_entity: str = "cells",
    col_entity: str = "genes",
    provenance_id: int | None = None,
    dtype=None,
) -> None:
    """Write a CSR matrix into chunked compressed storage.

    Parameters
    ----------
    conn
        Database connection.
    matrix_name
        Matrix identifier.
    csr_matrix
        Sparse matrix (CSR, or any scipy sparse matrix or array).
    chunk_size
        Number of rows per chunk.
    compression
        Compression method.
    row_entity
        Row entity table name.
    col_entity
        Column entity table name.
    provenance_id
        Optional provenance row id.
    dtype
        Value dtype to store; ``None`` keeps the matrix's.
    """
    target = storage_dtype(csr_matrix.dtype if dtype is None else dtype)
    data, indices, indptr = _canonical_csr(csr_matrix, target, matrix_name)
    from cytome.io.sqlite_engine import ensure_chunk_span_indexes
    ensure_chunk_span_indexes(conn)
    conn.execute("DELETE FROM matrix_chunks WHERE matrix_name = ?", (matrix_name,))
    conn.execute("DELETE FROM matrix_csc_chunks WHERE matrix_name = ?", (matrix_name,))
    conn.execute("DELETE FROM matrix_meta WHERE matrix_name = ?", (matrix_name,))

    n_rows, n_cols = csr_matrix.shape
    chunk_size = max(1, int(chunk_size))
    chunk_idx = 0
    dtype_str = str(target)
    batch: list[tuple] = []
    for row_start in range(0, n_rows, chunk_size):
        row_end = min(row_start + chunk_size, n_rows)
        nnz, d_blob, i_blob, p_blob = _chunk_payload(
            data, indices, indptr, row_start, row_end, compression, matrix_name)
        batch.append((
            matrix_name,
            chunk_idx,
            row_start,
            row_end,
            int(nnz),
            d_blob, i_blob, p_blob,
            dtype_str,
            compression,
        ))
        chunk_idx += 1

    conn.executemany(_INSERT_CHUNK_SQL, batch)

    conn.execute(
        """
        INSERT INTO matrix_meta(
            matrix_name, n_rows, n_cols, n_nonzero, dtype,
            row_entity, col_entity, chunk_size, n_chunks, created_at,
            provenance_id, is_integer
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            matrix_name,
            int(n_rows),
            int(n_cols),
            int(data.size),
            dtype_str,
            row_entity,
            col_entity,
            int(chunk_size),
            int(chunk_idx),
            _now_iso(),
            provenance_id,
            _values_are_integer(dtype_str, data[:20000]),
        ),
    )


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def read_sparse_chunked(conn: sqlite3.Connection, matrix_name: str) -> sp.csr_matrix:
    """Read a full sparse matrix from chunk storage.

    Rows are placed by each chunk's ``row_start``; rows no chunk holds are
    zero rows. The destination is sized from the chunks' recorded
    ``n_nonzero`` before any blob is read, so a full read holds the matrix
    once, not twice; past 2**31 - 1 entries its index arrays are int64 (an
    int32 index pointer wrapped there and returned a corrupt matrix).
    """
    n_rows, n_cols, meta_dtype, n_nonzero = _matrix_meta(conn, matrix_name)
    plan = _Plan(conn, matrix_name)
    if not len(plan):
        return sp.csr_matrix((n_rows, n_cols), dtype=meta_dtype)
    out_dtype = meta_dtype

    recorded = int(plan.nnz.sum())
    n_out = max(n_rows, int(plan.ends.max()))
    cap = max(recorded, 1)
    idx_dtype = _index_dtype(max(cap, n_nonzero or 0))
    data = np.empty(cap, dtype=out_dtype)
    indices = np.empty(cap, dtype=idx_dtype)
    indptr = np.zeros(n_out + 1, dtype=idx_dtype)
    at = 0
    row_at = 0
    covered = 0
    for rs, re_, d_blob, i_blob, p_blob, comp, cdt in _blob_rows(conn, matrix_name, plan):
        rs, re_ = int(rs), int(re_)
        block = _decode_block(re_ - rs, n_cols, d_blob, i_blob, p_blob, comp, cdt,
                              meta_dtype, f"{matrix_name} rows {rs}-{re_ - 1}",
                              matrix_name=matrix_name)
        k = int(block.nnz)
        if at + k > data.size:               # the recorded sizes were short
            new_cap = max(2 * data.size, at + k)
            grown = np.empty(new_cap, dtype=out_dtype)
            grown[:at] = data[:at]
            data = grown
            new_idx = np.dtype(_index_dtype(new_cap))
            if new_idx.itemsize > indices.dtype.itemsize:
                indptr = indptr.astype(new_idx)
            grown = np.empty(new_cap, dtype=new_idx if new_idx.itemsize > indices.dtype.itemsize
                             else indices.dtype)
            grown[:at] = indices[:at]
            indices = grown
        data[at:at + k] = block.data
        indices[at:at + k] = block.indices
        _place_rows(indptr, row_at, rs, re_, block.indptr, at)
        at += k
        row_at = re_
        covered += re_ - rs
    indptr[row_at + 1:] = at
    if at != recorded or (n_nonzero is not None and at != n_nonzero) or covered != n_rows \
            or n_out != n_rows:
        # The recorded totals disagree with what is on disk. Trust the blobs,
        # not the metadata, and say so rather than returning a matrix padded
        # with uninitialised memory.
        warnings.warn(
            f"{matrix_name}: matrix_meta says {n_nonzero} nonzeros over "
            f"{n_rows} rows but the chunks hold {at} over {covered} (rows "
            f"to {n_out}). Using the chunks; rows no chunk holds are zero.",
            stacklevel=2)
    return sp.csr_matrix((data[:at], indices[:at], indptr), shape=(n_out, n_cols))


def _place_rows(indptr: np.ndarray, row_at: int, rs: int, re_: int, block_indptr, at: int) -> None:
    """Set the full index pointer for the rows ``[rs, re_)`` of one chunk
    whose entries start at ``at``; the rows from ``row_at`` (the end of the
    previous chunk) to ``rs`` are empty. Computed in ``indptr``'s dtype, which
    is int64 past 2**31 - 1 entries (int32 arithmetic wrapped there)."""
    indptr[row_at + 1:rs + 1] = at
    indptr[rs + 1:re_ + 1] = np.asarray(block_indptr[1:], dtype=indptr.dtype) + indptr.dtype.type(at)


def _rows_from_blocks(blocks, n_rows: int, n_cols: int, dtype) -> sp.csr_matrix:
    """Stack ``[(local_row, csr_block), ...]`` (ascending, non-overlapping) into
    ``n_rows`` rows; rows between blocks are zero rows."""
    parts = []
    at = 0
    for r0, blk in blocks:
        if r0 > at:
            parts.append(_empty_rows(r0 - at, n_cols, dtype))
        parts.append(blk)
        at = r0 + blk.shape[0]
    if at < n_rows:
        parts.append(_empty_rows(n_rows - at, n_cols, dtype))
    if not parts:
        return _empty_rows(n_rows, n_cols, dtype)
    if len(parts) == 1:
        return parts[0].tocsr()
    return sp.vstack(parts, format="csr")


def read_sparse_slice(
    conn: sqlite3.Connection,
    matrix_name: str,
    row_start: int,
    row_end: int,
) -> sp.csr_matrix:
    """Read the rows ``[row_start, row_end)`` of a chunked sparse matrix:
    exactly ``row_end - row_start`` rows, those no chunk holds zero."""
    _n, n_cols, meta_dtype, _nnz = _matrix_meta(conn, matrix_name)
    row_start, row_end = int(row_start), int(row_end)
    n_want = max(0, row_end - row_start)
    plan = _Plan(conn, matrix_name)
    pos = np.flatnonzero((plan.ends > row_start) & (plan.starts < row_end))
    blocks = []
    for c_start, c_end, d_blob, i_blob, p_blob, comp, cdt in _blob_rows(conn, matrix_name, plan, pos):
        c_start, c_end = int(c_start), int(c_end)
        chunk = _decode_block(c_end - c_start, n_cols, d_blob, i_blob, p_blob, comp, cdt,
                              meta_dtype, f"{matrix_name} rows {c_start}-{c_end - 1}",
                              matrix_name=matrix_name)
        local_start = max(0, row_start - c_start)
        local_end = min(c_end - c_start, row_end - c_start)
        blocks.append((c_start + local_start - row_start, chunk[local_start:local_end]))
    return _rows_from_blocks(blocks, n_want, n_cols, meta_dtype)


def read_sparse_rows(
    conn: sqlite3.Connection,
    matrix_name: str,
    row_indices: np.ndarray,
) -> sp.csr_matrix:
    """Read specific rows from a chunked sparse matrix, in the order given.

    Only decompresses chunks that contain at least one requested row.
    Returns a CSR matrix with shape ``(len(row_indices), n_cols)``: rows in any
    order and repeated rows are allowed (they needed sorted unique indices
    before, and raised otherwise), a row no chunk holds is a zero row, and an
    index outside the matrix raises ``IndexError``.

    Parameters
    ----------
    conn
        Database connection.
    matrix_name
        Matrix identifier.
    row_indices
        0-based row indices to read.
    """
    row_indices = np.asarray(row_indices)
    if row_indices.dtype == bool:
        row_indices = np.flatnonzero(row_indices)
    row_indices = np.asarray(row_indices, dtype=np.int64).ravel()
    n_rows, n_cols, meta_dtype, _nnz = _matrix_meta(conn, matrix_name)

    if row_indices.size == 0:
        return sp.csr_matrix((0, n_cols), dtype=meta_dtype)

    plan = _Plan(conn, matrix_name)
    limit = max(n_rows, int(plan.ends.max()) if len(plan) else 0)
    bad = (row_indices < 0) | (row_indices >= limit)
    if bad.any():
        raise IndexError(f"{matrix_name}: row {int(row_indices[bad][0])} is outside its "
                         f"{limit} rows")
    order = np.argsort(row_indices, kind="stable")
    wanted = row_indices[order]
    pos = plan.touching(wanted)

    blocks = []
    for c_start, c_end, d_blob, i_blob, p_blob, comp, cdt in _blob_rows(conn, matrix_name, plan, pos):
        c_start, c_end = int(c_start), int(c_end)
        lo = int(np.searchsorted(wanted, c_start, side="left"))
        hi = int(np.searchsorted(wanted, c_end, side="left"))
        chunk = _decode_block(c_end - c_start, n_cols, d_blob, i_blob, p_blob, comp, cdt,
                              meta_dtype, f"{matrix_name} rows {c_start}-{c_end - 1}",
                              matrix_name=matrix_name)
        blocks.append((lo, chunk[(wanted[lo:hi] - c_start).astype(np.intp)]))
    # the wanted rows in ascending order (rows of no chunk zero), then back to
    # the order asked for
    sorted_rows = _rows_from_blocks(blocks, wanted.size, n_cols, meta_dtype)
    if np.all(order[1:] > order[:-1]):
        return sorted_rows
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return sorted_rows[inverse]


def read_dense_rows(
    conn: sqlite3.Connection,
    array_name: str,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Read specific rows from a chunked dense array, in the order given.

    Only decompresses chunks that contain at least one requested row; rows in
    any order and repeated rows are allowed, an index outside the array raises
    ``IndexError``.

    Parameters
    ----------
    conn
        Database connection.
    array_name
        Array identifier.
    row_indices
        0-based row indices to read.
    """
    row_indices = np.asarray(row_indices)
    if row_indices.dtype == bool:
        row_indices = np.flatnonzero(row_indices)
    row_indices = np.asarray(row_indices, dtype=np.int64).ravel()
    meta = conn.execute(
        "SELECT n_rows, n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Embedding not found: {array_name}")
    n_rows, n_cols, dtype = int(meta[0]), int(meta[1]), np.dtype(meta[2])

    if row_indices.size == 0:
        return np.empty((0, n_cols), dtype=dtype)
    bad = (row_indices < 0) | (row_indices >= n_rows)
    if bad.any():
        raise IndexError(f"{array_name}: row {int(row_indices[bad][0])} is outside its "
                         f"{n_rows} rows")

    order = np.argsort(row_indices, kind="stable")
    wanted = row_indices[order]
    out = np.zeros((wanted.size, n_cols), dtype=dtype)
    chunks = conn.execute(
        """
        SELECT chunk_idx, row_start, row_end
        FROM dense_chunks
        WHERE array_name = ?
        ORDER BY chunk_idx
        """,
        (array_name,),
    ).fetchall()

    for chunk_idx, c_start, c_end in chunks:
        lo = int(np.searchsorted(wanted, c_start, side="left"))
        hi = int(np.searchsorted(wanted, c_end, side="left"))
        if lo >= hi:
            continue
        row = conn.execute(
            """
            SELECT data_blob, compression
            FROM dense_chunks
            WHERE array_name = ? AND chunk_idx = ?
            """,
            (array_name, chunk_idx),
        ).fetchone()
        blob, compression = row
        chunk = np.frombuffer(decompress_blob(blob, compression), dtype=dtype).reshape(
            (c_end - c_start, n_cols)
        )
        out[lo:hi] = chunk[(wanted[lo:hi] - c_start).astype(np.intp)]

    if np.all(order[1:] > order[:-1]):
        return out
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return out[inverse]


def read_sparse_rows_iter(
    conn: sqlite3.Connection,
    matrix_name: str,
    row_filter: "np.ndarray | None" = None,
) -> Iterator[Tuple[int, int, sp.csr_matrix]]:
    """Iterate over sparse row chunks, in row order.

    ``row_filter`` is a sorted array of global row indices the caller wants.
    Chunks holding none of them are never fetched, so the blobs are neither
    read off disk nor decompressed.

    That matters more than it sounds. A per-batch masked read used to
    decompress every chunk and discard the ones with no selected rows, so a
    35-batch GDR over 200,061 cells read the whole matrix 35 times, 17 passes
    each. Measured on that file, a median batch needs 45 of 1,563 chunks: 2.9%
    of what the full scan reads. The chunks are asked for a bounded batch at a
    time: one ``IN (...)`` of every needed chunk passed SQLite's limit on
    bound parameters at about half a million cells.
    """
    _n, n_cols, meta_dtype, _nnz = _matrix_meta(conn, matrix_name)
    plan = _Plan(conn, matrix_name)
    if row_filter is not None:
        wanted = np.asarray(row_filter)
        if wanted.dtype == bool:
            wanted = np.flatnonzero(wanted)
        wanted = np.sort(np.asarray(wanted, dtype=np.int64))
        pos = plan.touching(wanted)
        if pos.size == 0:
            return
    else:
        pos = None
    for row_start, row_end, d_blob, i_blob, p_blob, comp, cdt in _blob_rows(
            conn, matrix_name, plan, pos):
        row_start, row_end = int(row_start), int(row_end)
        chunk = _decode_block(row_end - row_start, n_cols, d_blob, i_blob, p_blob, comp, cdt,
                              meta_dtype, f"{matrix_name} rows {row_start}-{row_end - 1}",
                              matrix_name=matrix_name)
        yield row_start, row_end, chunk


def write_dense_chunked(
    conn: sqlite3.Connection,
    array_name: str,
    ndarray: np.ndarray,
    chunk_size: int,
    compression: str,
    entity: str = "cells",
    provenance_id: int | None = None,
) -> None:
    """Write a dense 2D array as row chunks.

    Only numeric arrays are stored: the bytes of an object array are
    pointers, which nothing can read back.
    """
    arr = np.asarray(ndarray)
    if arr.ndim != 2:
        raise ValueError("Dense chunked write expects a 2D array")
    if arr.dtype.kind not in "biufc" or arr.dtype.fields is not None:
        raise TypeError(
            f"{array_name}: an embedding holds numbers; an array of dtype {arr.dtype} "
            f"cannot be stored (a DataFrame with text or categorical columns belongs in "
            f"the cells table)")
    arr = arr.astype(storage_dtype(arr.dtype), copy=False)

    conn.execute("DELETE FROM dense_chunks WHERE array_name = ?", (array_name,))
    conn.execute("DELETE FROM embedding_meta WHERE array_name = ?", (array_name,))

    n_rows, n_cols = arr.shape
    chunk_idx = 0
    for row_start in range(0, n_rows, chunk_size):
        row_end = min(row_start + chunk_size, n_rows)
        chunk = np.ascontiguousarray(arr[row_start:row_end])
        conn.execute(
            """
            INSERT INTO dense_chunks(
                array_name, chunk_idx, row_start, row_end, n_cols,
                data_blob, dtype, compression
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                array_name,
                chunk_idx,
                row_start,
                row_end,
                n_cols,
                compress_blob(chunk.tobytes(), method=compression),
                str(chunk.dtype),
                compression,
            ),
        )
        chunk_idx += 1

    conn.execute(
        """
        INSERT INTO embedding_meta(
            array_name, n_rows, n_cols, dtype, entity, chunk_size,
            n_chunks, created_at, provenance_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            array_name,
            int(n_rows),
            int(n_cols),
            str(arr.dtype),
            entity,
            int(chunk_size),
            int(chunk_idx),
            _now_iso(),
            provenance_id,
        ),
    )


def read_dense_chunked(conn: sqlite3.Connection, array_name: str) -> np.ndarray:
    """Read full dense array from chunk storage."""
    meta = conn.execute(
        "SELECT n_rows, n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Embedding not found: {array_name}")
    n_rows, n_cols, dtype = int(meta[0]), int(meta[1]), np.dtype(meta[2])

    rows = conn.execute(
        """
        SELECT row_start, row_end, data_blob, compression
        FROM dense_chunks WHERE array_name = ? ORDER BY chunk_idx
        """,
        (array_name,),
    ).fetchall()
    arr = np.empty((n_rows, n_cols), dtype=dtype)
    for row_start, row_end, blob, compression in rows:
        chunk = np.frombuffer(decompress_blob(blob, compression), dtype=dtype).reshape(
            (row_end - row_start, n_cols)
        )
        arr[row_start:row_end] = chunk
    return arr


def read_dense_slice(
    conn: sqlite3.Connection,
    array_name: str,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """Read row slice of a dense chunked array."""
    meta = conn.execute(
        "SELECT n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (array_name,),
    ).fetchone()
    if meta is None:
        raise KeyError(f"Embedding not found: {array_name}")
    n_cols, dtype = int(meta[0]), np.dtype(meta[1])

    rows = conn.execute(
        """
        SELECT row_start, row_end, data_blob, compression
        FROM dense_chunks
        WHERE array_name = ? AND row_end > ? AND row_start < ?
        ORDER BY chunk_idx
        """,
        (array_name, row_start, row_end),
    ).fetchall()

    parts = []
    for c_start, c_end, blob, compression in rows:
        chunk = np.frombuffer(decompress_blob(blob, compression), dtype=dtype).reshape(
            (c_end - c_start, n_cols)
        )
        local_start = max(0, row_start - c_start)
        local_end = min(c_end - c_start, row_end - c_start)
        parts.append(chunk[local_start:local_end])

    if not parts:
        return np.empty((max(0, row_end - row_start), n_cols), dtype=dtype)
    return np.vstack(parts)
