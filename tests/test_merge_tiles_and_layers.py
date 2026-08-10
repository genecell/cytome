"""Round 29: cytome.merge must carry EVERY modality's feature table (tiles, GA,
and non-counts RNA layers), not just genes + peaks — the bug behind COSG's
'0 genes' crash on a merged multiome.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import cytome
from cytome.io.merge import merge


def _build(path, cells, genes, tiles, rng):
    ds = cytome.create(path)
    n = len(cells)
    ds.set_entity("cells", pd.DataFrame({"barcode": cells}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": genes, "symbol": genes}))
    ds.set_entity("tiles", pd.DataFrame({
        "tile_id": tiles,
        "chr": ["chr1"] * len(tiles),
        "start": [1000 * i for i in range(len(tiles))],
        "end_": [1000 * i + 1000 for i in range(len(tiles))],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(rng.poisson(1.0, (n, len(genes))).astype(np.float32)))
    # a non-counts RNA layer (the path that used to fall to identity/misalign)
    ds.add_matrix("RNA_infog", sp.csr_matrix(rng.random((n, len(genes))).astype(np.float32)))
    ds.add_matrix("tiles_counts", sp.csr_matrix((rng.random((n, len(tiles))) < 0.3).astype(np.float32)))
    ds.flush()
    return ds


def test_merge_carries_tiles_and_projects_layers(tmp_path):
    rng = np.random.RandomState(0)
    # shared tile + gene axis (genome-binned tiles are identical across inputs)
    tiles = [f"chr1:{i}" for i in range(8)]
    a = _build(str(tmp_path / "a.cytome"), ["a1", "a2", "a3"], ["g1", "g2", "g3"], tiles, rng)
    b = _build(str(tmp_path / "b.cytome"), ["b1", "b2"], ["g2", "g3", "g4"], tiles, rng)

    out = merge([a, b], output=str(tmp_path / "m.cytome"),
                gene_strategy="union", tile_strategy="union")

    # tiles feature TABLE is carried over and matches the tiles matrix width
    tiles_df = pd.read_sql_query("SELECT * FROM tiles", out._conn)
    assert len(tiles_df) == 8
    assert sorted(tiles_df["tile_id"]) == sorted(tiles)
    tcols = out._conn.execute(
        "SELECT n_cols FROM matrix_meta WHERE matrix_name='tiles_counts'").fetchone()[0]
    assert tcols == 8 == len(tiles_df)

    # non-counts RNA layer is projected onto the merged (union) gene axis
    genes = list(pd.read_sql_query("SELECT * FROM genes", out._conn)["gene_id"])
    assert genes == ["g1", "g2", "g3", "g4"]
    for m in ("RNA_counts", "RNA_infog"):
        nc = out._conn.execute(
            "SELECT n_cols FROM matrix_meta WHERE matrix_name=?", (m,)).fetchone()[0]
        assert nc == 4, f"{m} not projected onto merged gene axis"

    a.close(); b.close(); out.close()


def test_merge_tiles_then_cosg_runs(tmp_path):
    """End-to-end: after the fix, COSG on the merged tiles modality runs (no
    '0 genes' broadcast crash)."""
    cosg = pytest.importorskip("cosg")
    rng = np.random.RandomState(1)
    tiles = [f"chr1:{i}" for i in range(10)]
    a = _build(str(tmp_path / "a.cytome"), [f"a{i}" for i in range(8)],
               ["g1", "g2", "g3"], tiles, rng)
    b = _build(str(tmp_path / "b.cytome"), [f"b{i}" for i in range(8)],
               ["g1", "g2", "g3"], tiles, rng)
    out = merge([a, b], output=str(tmp_path / "m.cytome"))
    out.cells["Leiden"] = [str(i % 2) for i in range(out.n_cells)]
    out.flush()
    out.close()

    res = cosg.run_cosg_cytome(
        str(tmp_path / "m.cytome"), groupby="Leiden", modality="tiles",
        n_genes_user=5, mu=1.0, expressed_pct=0.0, verbose=False)
    assert res is not None
    a.close(); b.close()


def test_cosg_guard_errors_on_empty_feature_table(tmp_path):
    """COSG raises a clear error (not a broadcast ValueError) when the var table
    count disagrees with the matrix width."""
    cosg = pytest.importorskip("cosg")
    rng = np.random.RandomState(2)
    ds = cytome.create(str(tmp_path / "x.cytome"))
    n = 10
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(n)],
                                         "Leiden": [str(i % 2) for i in range(n)]}))
    # tiles matrix has 6 columns but the tiles var table is left empty (0 rows)
    ds.set_entity("tiles", pd.DataFrame({"tile_id": pd.Series([], dtype=str)}))
    ds.add_matrix("tiles_counts", sp.csr_matrix((rng.random((n, 6)) < 0.4).astype(np.float32)))
    ds.flush()
    path = str(ds.path); ds.close()

    with pytest.raises(ValueError, match="feature/var table is missing or out of sync"):
        cosg.run_cosg_cytome(path, groupby="Leiden", modality="tiles",
                             n_genes_user=3, mu=1.0, expressed_pct=0.0, verbose=False)


# --------------------------------------------------------------------------
# Round 29 follow-up: preserve-identical-axis, genome guard, ds.features()
# --------------------------------------------------------------------------

def _build_tiles_only(path, cells, tile_ids, genome=None):
    ds = cytome.create(path)
    n = len(cells)
    ds.set_entity("cells", pd.DataFrame({
        "barcode": cells, "Leiden": [str(i % 2) for i in range(n)]}))
    # tile_ids order = genomic (chr1, chr2, ..., chr10) — NOT lexical
    ds.set_entity("tiles", pd.DataFrame({
        "tile_id": tile_ids,
        "chr": [t.split(":")[0] for t in tile_ids],
        "start": [int(t.split(":")[1].split("-")[0]) for t in tile_ids],
        "end_": [int(t.split(":")[1].split("-")[1]) for t in tile_ids],
    }))
    rng = np.random.RandomState(len(cells))
    ds.add_matrix("tiles_counts",
                  sp.csr_matrix((rng.random((n, len(tile_ids))) < 0.4).astype(np.float32)))
    if genome is not None:
        ds._write_manifest_key("genome", genome)
    ds.flush()
    return ds


def test_merge_preserves_identical_genomic_tile_order(tmp_path):
    # multi-chromosome tiles whose LEXICAL sort (chr1, chr10, chr2) differs from
    # the GENOMIC order. Both inputs share the identical axis → merge must keep
    # the genomic order (no lexical re-sort) and stay aligned with the matrix.
    tiles = ["chr1:1-500", "chr2:1-500", "chr10:1-500", "chrX:1-500"]
    a = _build_tiles_only(str(tmp_path / "a.cytome"), ["a1", "a2", "a3"], tiles)
    b = _build_tiles_only(str(tmp_path / "b.cytome"), ["b1", "b2"], tiles)
    out = merge([a, b], output=str(tmp_path / "m.cytome"))
    merged = list(pd.read_sql_query("SELECT tile_id FROM tiles ORDER BY tile_idx",
                                    out._conn)["tile_id"])
    assert merged == tiles, "identical axis should be preserved in genomic order"
    assert merged != sorted(tiles), "must NOT lexically re-sort a shared axis"
    ncols = out._conn.execute(
        "SELECT n_cols FROM matrix_meta WHERE matrix_name='tiles_counts'").fetchone()[0]
    assert ncols == len(tiles)
    a.close(); b.close(); out.close()


def test_merge_genome_mismatch_raises(tmp_path):
    a = _build_tiles_only(str(tmp_path / "a.cytome"), ["a1", "a2"],
                          ["chr1:1-500"], genome="mm10")
    b = _build_tiles_only(str(tmp_path / "b.cytome"), ["b1", "b2"],
                          ["chr1:1-500"], genome="hg38")
    with pytest.raises(ValueError, match="different genomes"):
        merge([a, b], output=str(tmp_path / "m.cytome"))
    a.close(); b.close()


def test_merge_carries_genome_to_output(tmp_path):
    a = _build_tiles_only(str(tmp_path / "a.cytome"), ["a1", "a2"],
                          ["chr1:1-500"], genome="mm10")
    b = _build_tiles_only(str(tmp_path / "b.cytome"), ["b1", "b2"],
                          ["chr1:1-500"], genome="mm10")
    out = merge([a, b], output=str(tmp_path / "m.cytome"))
    assert out._manifest.get("genome") == "mm10"
    a.close(); b.close(); out.close()


def test_features_accessor(tmp_path):
    rng = np.random.RandomState(3)
    ds = _build(str(tmp_path / "f.cytome"), ["c1", "c2"], ["g1", "g2"],
                ["chr1:1-500", "chr2:1-500"], rng)
    # ds.features('tiles') gives the tiles TABLE (ds.tiles would be a Modality)
    assert list(ds.features("tiles")["tile_id"]) == ["chr1:1-500", "chr2:1-500"]
    assert list(ds.features("RNA")["gene_id"]) == ["g1", "g2"]   # RNA -> genes
    assert ds.features("tiles").to_pandas().shape[0] == 2
    import pytest as _pt
    with _pt.raises(ValueError):
        ds.features("nope")
    ds.close()
