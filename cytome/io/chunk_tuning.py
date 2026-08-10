"""Chunk size tuning utilities."""

from __future__ import annotations


def compute_chunk_size(
    n_rows: int,
    n_cols: int,
    total_nnz: int,
    target_bytes: int = 8192,
) -> int:
    """Compute rows per chunk to fit sparse data in cache.

    Parameters
    ----------
    n_rows
        Number of matrix rows.
    n_cols
        Number of matrix columns (unused in v1 heuristic).
    total_nnz
        Total nonzero values.
    target_bytes
        Target bytes per chunk for nonzero payload.

    Returns
    -------
    int
        Recommended chunk size in rows.
    """
    del n_cols
    avg_nnz_per_row = total_nnz / max(n_rows, 1)
    bytes_per_row = 8 * avg_nnz_per_row
    chunk_rows = int(target_bytes / max(bytes_per_row, 1))
    chunk_rows = max(16, chunk_rows)
    chunk_rows = min(10000, chunk_rows)
    return chunk_rows
