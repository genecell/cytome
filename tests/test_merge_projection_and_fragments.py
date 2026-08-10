"""2026-06-15: cytome.merge fast/correct paths.

- gene_strategy='union' (new default) keeps every gene; absent genes → zeros.
- gene_strategy='intersection' keeps only shared genes.
- Differing gene order → vectorized sparse projection (mat @ P), not a LIL loop.
- Fragments remap cell_idx by a contiguous +offset (cells concatenated in order)
  and stream into the compressed fragment_chunks layout, start-sorted.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cytome.core.dataset import CytomeDataset
from cytome.io.merge import merge
from cytome.io.convert_fragments import _write_block_chunks, _update_fragment_meta


def _build(path, cells, genes, rna, peaks=None, atac=None, frags=None):
    """frags: dict chrom -> (starts, ends, cells) arrays."""
    ds = CytomeDataset(path, mode="w")
    ds.set_entity("cells", pd.DataFrame({"barcode": cells}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": genes, "symbol": genes}))
    if peaks is not None:
        ds.set_entity("peaks", pd.DataFrame({
            "peak_id": peaks,
            "chr": ["chr1"] * len(peaks),
            "start": [1000 * (i + 1) for i in range(len(peaks))],
            "end_": [1000 * (i + 1) + 500 for i in range(len(peaks))],
        }))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.asarray(rna, dtype=np.float32)))
    if atac is not None:
        ds.add_matrix("ATAC_counts", sp.csr_matrix(np.asarray(atac, dtype=np.float32)))
    ds.flush()
    if frags:
        for chrom, (s, e, c) in frags.items():
            _write_block_chunks(
                ds._conn, chrom,
                np.asarray(s, np.int32), np.asarray(e, np.int32), np.asarray(c, np.int32),
                chunk_size=500_000, compression="lz4", compression_level=1,
                chunk_idx_start=0, encoding=1,
            )
            _update_fragment_meta(ds._conn, chrom, len(s))
        ds._conn.commit()
    ds.close()
    return path


def _rna(ds):
    return ds.RNA.layer("counts").to_memory().tocsr().toarray()


def test_merge_union_projection(tmp_path):
    # A: genes g1,g2,g3 ; B: genes g2,g3,g4  (overlap + disjoint)
    a = _build(str(tmp_path / "a.cytome"), ["a0", "a1"], ["g1", "g2", "g3"],
               [[1, 2, 3], [4, 5, 6]])
    b = _build(str(tmp_path / "b.cytome"), ["b0", "b1", "b2"], ["g2", "g3", "g4"],
               [[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    out = merge([a, b], output=str(tmp_path / "u.cytome"), gene_strategy="union")
    genes = list(out.genes.to_pandas()["gene_id"])
    assert genes == ["g1", "g2", "g3", "g4"]          # sorted union
    assert out.n_cells == 5
    M = _rna(out)
    gi = {g: i for i, g in enumerate(genes)}
    # A's cells: g1,g2,g3 placed; g4 = 0
    assert M[0, gi["g1"]] == 1 and M[0, gi["g2"]] == 2 and M[0, gi["g3"]] == 3
    assert M[0, gi["g4"]] == 0
    # B's cells (rows 2..4): g2,g3,g4 placed; g1 = 0
    assert M[2, gi["g2"]] == 7 and M[2, gi["g3"]] == 8 and M[2, gi["g4"]] == 9
    assert M[2, gi["g1"]] == 0
    assert M[4, gi["g4"]] == 15
    out.close()


def test_merge_intersection(tmp_path):
    a = _build(str(tmp_path / "a.cytome"), ["a0", "a1"], ["g1", "g2", "g3"],
               [[1, 2, 3], [4, 5, 6]])
    b = _build(str(tmp_path / "b.cytome"), ["b0", "b1"], ["g2", "g3", "g4"],
               [[7, 8, 9], [10, 11, 12]])
    out = merge([a, b], output=str(tmp_path / "i.cytome"), gene_strategy="intersection")
    genes = list(out.genes.to_pandas()["gene_id"])
    assert genes == ["g2", "g3"]
    M = _rna(out)
    gi = {g: i for i, g in enumerate(genes)}
    assert M[0, gi["g2"]] == 2 and M[0, gi["g3"]] == 3   # A row0
    assert M[2, gi["g2"]] == 7 and M[2, gi["g3"]] == 8   # B row0
    out.close()


def test_merge_fragments_offset_remap(tmp_path):
    # A: 2 cells, fragments on chr1 with cell_idx 0,1
    a = _build(
        str(tmp_path / "a.cytome"), ["a0", "a1"], ["g1", "g2"],
        [[1, 0], [0, 1]], peaks=["p1"], atac=[[1], [1]],
        frags={"chr1": ([100, 300], [150, 350], [0, 1])},
    )
    # B: 3 cells, fragments on chr1 (cell_idx 0,1,2) + chr2
    b = _build(
        str(tmp_path / "b.cytome"), ["b0", "b1", "b2"], ["g1", "g2"],
        [[1, 0], [0, 1], [1, 1]], peaks=["p1"], atac=[[1], [1], [1]],
        frags={"chr1": ([200, 50, 400], [250, 90, 450], [0, 1, 2]),
               "chr2": ([10], [20], [2])},
    )
    out = merge([a, b], output=str(tmp_path / "m.cytome"),
                gene_strategy="union", include_fragments=True)
    assert out.n_cells == 5
    assert "ATAC" in out.modalities

    frags = out.ATAC.fragments
    s, e, c = [], [], []
    for cs, ce, cc in frags.iter_chromosome_chunks("chr1"):
        s.append(cs); e.append(ce); c.append(cc)
    s = np.concatenate(s); e = np.concatenate(e); c = np.concatenate(c)
    # start-sorted globally
    assert list(s) == sorted(s)
    # B's cells shifted by offset 2 (A contributed 2 cells). Identify by start.
    by_start = {int(st): int(ci) for st, ci in zip(s, c)}
    assert by_start[100] == 0 and by_start[300] == 1        # A unchanged
    assert by_start[200] == 0 + 2 and by_start[50] == 1 + 2 and by_start[400] == 2 + 2
    # chr2 only from B
    s2 = np.concatenate([cs for cs, _, _ in frags.iter_chromosome_chunks("chr2")])
    c2 = np.concatenate([cc for _, _, cc in frags.iter_chromosome_chunks("chr2")])
    assert list(s2) == [10] and int(c2[0]) == 2 + 2
    out.close()
