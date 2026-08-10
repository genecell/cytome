"""Shared pytest fixtures for Cytome tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import gzip
from pathlib import Path
import scipy.io


@pytest.fixture
def small_rna_matrix():
    """500 cells × 200 genes sparse matrix."""
    np.random.seed(42)
    density = 0.05
    n_cells, n_genes = 500, 200
    data = sp.random(n_cells, n_genes, density=density, format="csr", dtype=np.float32)
    data.data = np.random.poisson(2, size=data.nnz).astype(np.float32)
    return data


@pytest.fixture
def small_atac_matrix():
    """500 cells × 1000 peaks sparse matrix."""
    np.random.seed(42)
    n_cells, n_peaks = 500, 1000
    data = sp.random(n_cells, n_peaks, density=0.02, format="csr", dtype=np.int32)
    data.data = np.ones(data.nnz, dtype=np.int32)
    return data


@pytest.fixture
def small_fragments():
    """Synthetic fragment data."""
    np.random.seed(42)
    n_frags = 50000
    chroms = np.random.choice(["chr1", "chr2", "chr3"], n_frags, p=[0.5, 0.3, 0.2])
    starts = np.random.randint(0, 100_000_000, n_frags)
    lengths = np.random.geometric(0.005, n_frags) + 50
    ends = starts + lengths
    cell_idxs = np.random.randint(0, 500, n_frags)
    return {"chrom": chroms, "start": starts, "end": ends, "cell_idx": cell_idxs}


@pytest.fixture
def sample_cell_metadata():
    """Cell metadata for 500 cells."""
    np.random.seed(42)
    return {
        "barcode": [f"ACGT{i:04d}-1" for i in range(500)],
        "sample_id": np.random.choice(["S1", "S2"], 500),
        "n_genes": np.random.randint(200, 5000, 500),
        "total_counts": np.random.randint(1000, 50000, 500),
        "pct_mito": np.random.uniform(0, 20, 500),
        "cell_type": np.random.choice(["CD8 T", "B cell", "Monocyte", "NK"], 500),
    }


@pytest.fixture
def sample_gene_metadata():
    """Gene metadata for 200 genes."""
    genes = []
    for i in range(200):
        chrom = f"chr{(i % 22) + 1}"
        start = i * 100000
        genes.append(
            {
                "gene_id": f"ENSG{i:011d}",
                "symbol": f"GENE{i}",
                "chr": chrom,
                "start": start,
                "end_": start + 5000,
                "biotype": "protein_coding",
            }
        )
    return genes


@pytest.fixture
def sample_anndata(small_rna_matrix, sample_cell_metadata, sample_gene_metadata):
    """Complete AnnData object for conversion tests."""
    anndata = pytest.importorskip("anndata")
    pandas = pytest.importorskip("pandas")
    obs = pandas.DataFrame(sample_cell_metadata)
    obs.index = obs["barcode"]
    var = pandas.DataFrame(sample_gene_metadata)
    var.index = var["gene_id"]
    return anndata.AnnData(X=small_rna_matrix, obs=obs, var=var)


@pytest.fixture
def tmp_cytome(tmp_path):
    """Temporary Cytome path."""
    return tmp_path / "test.cytome"


@pytest.fixture
def synthetic_cellranger_dir(tmp_path, small_rna_matrix, sample_cell_metadata, sample_gene_metadata):
    """Create a synthetic Cell Ranger-like output directory."""
    outs = tmp_path / "outs"
    outs.mkdir()
    mtx_dir = outs / "filtered_feature_bc_matrix"
    mtx_dir.mkdir()

    scipy.io.mmwrite(str(mtx_dir / "matrix.mtx"), small_rna_matrix.T)
    with open(mtx_dir / "matrix.mtx", "rb") as f_in, gzip.open(mtx_dir / "matrix.mtx.gz", "wb") as f_out:
        f_out.write(f_in.read())
    (mtx_dir / "matrix.mtx").unlink()

    with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as f:
        for bc in sample_cell_metadata["barcode"]:
            f.write(f"{bc}\n")

    with gzip.open(mtx_dir / "features.tsv.gz", "wt") as f:
        for gene in sample_gene_metadata:
            f.write(f"{gene['gene_id']}\t{gene['symbol']}\tGene Expression\n")

    return outs


@pytest.fixture
def synthetic_fragments_file(tmp_path, small_fragments):
    """Create a synthetic fragments.tsv.gz file."""
    path = tmp_path / "fragments.tsv.gz"
    order = np.lexsort((small_fragments["start"], small_fragments["chrom"]))
    with gzip.open(path, "wt") as f:
        for i in order:
            chrom = small_fragments["chrom"][i]
            start = int(small_fragments["start"][i])
            end = int(small_fragments["end"][i])
            cell_idx = int(small_fragments["cell_idx"][i])
            barcode = f"ACGT{cell_idx:04d}-1"
            f.write(f"{chrom}\t{start}\t{end}\t{barcode}\t1\n")
    return path
