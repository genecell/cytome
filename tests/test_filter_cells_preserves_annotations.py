"""2026-06-04: filter_cells / subset must preserve cell-independent annotations
(imported GTF gene models, GA_genes, var embeddings) — previously silently
dropped, destroying an imported GTF on every QC pass."""
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import cytome


def _build(path, n_cells=6, n_genes=4):
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"BC{i}" for i in range(n_cells)],
        "keep": np.array([1, 0] * (n_cells // 2)),
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "gene_id": [f"G{i}" for i in range(n_genes)],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(
        np.arange(n_cells * n_genes, dtype=np.float32).reshape(n_cells, n_genes)))
    # var embedding (gene axis)
    ds.add_var_embedding("PCs", np.ones((n_genes, 3), dtype=np.float32))
    # GA_genes var table
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "gene_id": [f"GA{i}" for i in range(n_genes)],
    }))
    ds.flush()
    # imported GTF gene models (direct insert into the base-schema tables)
    ds._conn.execute(
        "INSERT INTO _gene_annotation (gene_id, chrom, start, end, strand, "
        "gene_name, gene_type, source) VALUES (?,?,?,?,?,?,?,?)",
        ("G0", "chr1", 100, 2000, "+", "Gene0", "protein_coding", "test.gtf"))
    # spatial coords (cell-indexed): x = cell_idx*10, y = cell_idx*100
    ds._conn.executemany(
        "INSERT INTO spatial_coords (cell_idx, x, y, z) VALUES (?,?,?,?)",
        [(i, i * 10.0, i * 100.0, None) for i in range(n_cells)])
    ds._conn.commit()
    return ds


def test_filter_cells_preserves_gtf_ga_and_varm(tmp_path):
    path = str(tmp_path / "x.cytome")
    ds = _build(path)
    try:
        assert ds.gene_annotation_info() is not None      # GTF present pre-filter
        n = ds.filter_cells(np.asarray(ds.cells["keep"]).astype(bool))
        assert n == 3
        # GTF survived
        info = ds.gene_annotation_info()
        assert info is not None, "imported GTF was dropped by filter_cells!"
        genes = ds.query_gene_annotation(chrom="chr1", start=0, end=5000)
        assert len(genes) == 1 and genes.iloc[0]["gene_id"] == "G0"
        # GA_genes survived
        assert ds._conn.execute("SELECT COUNT(*) FROM GA_genes").fetchone()[0] == 4
        # var embedding survived
        assert "PCs" in ds.var_embeddings.keys()
        assert ds.var_embeddings["PCs"].shape == (4, 3)
    finally:
        ds.close()


def test_filter_cells_preserves_and_remaps_spatial_coords(tmp_path):
    """Spatial coords are cell-indexed -> kept cells' coords survive and
    cell_idx is remapped to the new 0..n-1 indexing."""
    path = str(tmp_path / "sp.cytome")
    ds = _build(path)               # keep mask = [1,0,1,0,1,0] -> old cells 0,2,4
    try:
        ds.filter_cells(np.asarray(ds.cells["keep"]).astype(bool))
        rows = ds._conn.execute(
            "SELECT cell_idx, x, y FROM spatial_coords ORDER BY cell_idx"
        ).fetchall()
        # new idx 0,1,2 carry old cells 0,2,4 -> x = 0,20,40 ; y = 0,200,400
        assert rows == [(0, 0.0, 0.0), (1, 20.0, 200.0), (2, 40.0, 400.0)]
        # rtree rebuilt
        assert ds._conn.execute(
            "SELECT COUNT(*) FROM spatial_rtree").fetchone()[0] == 3
    finally:
        ds.close()


def test_filter_cells_spatial_kept_even_when_copy_annotations_false(tmp_path):
    """Spatial is cell-data, not an annotation -> kept regardless of
    copy_annotations."""
    path = str(tmp_path / "sp2.cytome")
    ds = _build(path)
    try:
        ds.filter_cells(np.asarray(ds.cells["keep"]).astype(bool),
                        copy_annotations=False)
        n = ds._conn.execute("SELECT COUNT(*) FROM spatial_coords").fetchone()[0]
        assert n == 3
        assert ds.gene_annotation_info() is None    # annotations still dropped
    finally:
        ds.close()


def test_filter_cells_copy_annotations_false_drops_gtf(tmp_path):
    path = str(tmp_path / "y.cytome")
    ds = _build(path)
    try:
        ds.filter_cells(np.asarray(ds.cells["keep"]).astype(bool),
                        copy_annotations=False)
        assert ds.gene_annotation_info() is None          # explicitly dropped
    finally:
        ds.close()
