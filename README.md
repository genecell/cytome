# cytome

**A single-file format for single-cell multi-omics data.**

`cytome` replaces ad-hoc file stacks with one SQLite-backed `.cytome` file that keeps matrices, metadata, embeddings, fragments, and provenance together.

`cytome` stores expression matrices, cell metadata, genomic fragments, embeddings, graphs, and computational provenance in a single SQLite file. It opens from manifest metadata, supports SQL-style metadata filtering, and provides a tested CLI for conversion, merge, subset, export, and validation.

`cytome` is part of the [PIASO](https://piaso.org) toolkit for single-cell analysis. PIASO natively supports cytome datasets for streaming single-cell ATAC-seq and RNA-seq workflows.

## Why cytome?

| Capability | AnnData (`.h5ad`) | BPCells | TileDB-SOMA | cytome (`.cytome`) |
|---|---:|---:|---:|---:|
| Single portable file | Yes | No (directory layout) | No (array store) | Yes |
| Python stdlib core storage dependency | No (`h5py`) | No | No | Yes (`sqlite3`) |
| SQL metadata queries | No | No | Limited via APIs | Yes |
| Fragment genomic range queries | No native index | Limited | Yes | Yes (SQLite R-tree) |
| Native provenance table | No | No | Partial workflow metadata | Yes |
| Streaming merge API | Partial | Strong compute streaming | Distributed workflows | Yes, chunk-aware |
| Full matrix load speed | Fast | Fast | Varies | Fast |
| Cloud-native multi-writer scale-out | No | No | Yes | No (single-writer SQLite) |

## Quick start

No optional dependencies — this runs with `pip install cytome` alone.

```python
import cytome
import numpy as np
import pandas as pd
import scipy.sparse as sp

rng = np.random.default_rng(0)
counts = sp.random(1000, 500, density=0.05, format="csr", dtype=np.float32,
                   random_state=rng)

ds = cytome.create("example.cytome")
ds.set_entity("cells", pd.DataFrame({
    "barcode": [f"cell_{i}" for i in range(1000)],
    "cell_type": rng.choice(["A", "B"], 1000),
}))
ds.set_entity("genes", pd.DataFrame({"gene_id": [f"G{i}" for i in range(500)]}))
ds.add_matrix("RNA_counts", counts)
ds.flush()

print(ds)                                    # metadata-first summary
b_cells = ds.cells.query("cell_type == 'B'")  # SQL-style metadata query
block = ds.RNA.counts[:100, :50]              # random access
for start, end, chunk in ds.RNA.counts.iter_rows():
    ...                                       # bounded-memory streaming
ds.close()
```

### Marker genes, straight off the file

[COSG](https://github.com/genecell/COSG) reads a `.cytome` in chunks, so peak
memory does not scale with the number of cells and nothing is converted to
AnnData first.

```python
# pip install cosg
import cosg

markers = cosg.cosg("example.cytome", groupby="cell_type",
                    modality="RNA", layer="log1p", n_genes_user=10)
markers["names"]   # per-cell-type marker genes
```

### Coming from AnnData

```python
# pip install "cytome[anndata]"
ds = cytome.from_anndata(adata, modality="RNA", output="rna.cytome")
```

## Key features

- Instant metadata-first open (`CytomeDataset` initialization reads manifest and table metadata)
- SQL-queryable entity tables (`cells`, `genes`, `peaks`, `samples`)
- Merge, subset, and downsample APIs with chunk-aware implementations
- Fragment storage (chunked, compressed) with genomic range queries and export
- Analysis results live on the file: embeddings (`ds.embeddings`), neighbor
  graphs (`ds.graphs`), per-cell / per-feature columns, and arbitrary analysis
  artifacts (`ds.metadata`) — PIASO writes results back onto the cytome
- Provenance logging with parameters, dependency versions, and methods-text export
- ACID write transactions via SQLite
- Multi-modal naming convention (`RNA_counts`, `ATAC_counts`, etc.)
- Optional CSC feature index (`build_feature_index`) for faster column iteration
- JSON metadata store (`ds.metadata`) for arbitrary nested analysis metadata
- CLI (`cytome`) for convert, info, merge, subset, downsample, export, validate, provenance, and copy

## Installation

```bash
pip install cytome
```

Optional extras:

```bash
pip install "cytome[full]"
pip install "cytome[dev]"
```

## Usage examples

### Convert from AnnData

```python
import cytome
ds = cytome.from_anndata(adata, modality="RNA", output="rna.cytome")
ds.close()
```

### Open and query

```python
import cytome
ds = cytome.open("rna.cytome")
cd8 = ds.cells.query("cell_type == 'CD8 T'")
x = ds.RNA.counts[:200, :200]
ds.close()
```

### Merge datasets

```python
import cytome
merged = cytome.merge(["s1.cytome", "s2.cytome"], output="merged.cytome")
merged.close()
```

### Command line

```bash
cytome info merged.cytome
cytome merge s1.cytome s2.cytome -o merged.cytome
cytome subset merged.cytome -o cd8.cytome --query "cell_type == 'CD8 T'"
```

### ATAC fragments

```python
ds = cytome.open("multiome.cytome")
fr = ds.ATAC.fragments.query_region("chr1", 1_000_000, 2_000_000)
ds.ATAC.fragments.export("fragments.tsv.gz")
ds.close()
```

### With PIASO

`cytome` integrates with [PIASO](https://piaso.org) (`pip install piaso-tools`) for
single-cell analysis. PIASO supports cytome datasets directly — streaming normalization,
dimensionality reduction, clustering, and visualization without loading full matrices
into memory.

PIASO functions are **self-contained on a cytome**: they stream from the file
and write their results back onto it (graphs, embeddings, cluster labels,
per-cell/per-feature metrics, markers) — they don't return large objects, so you
read results from the dataset afterwards.

```python
import cytome
import piaso

# Open a cytome dataset and analyse it with PIASO (results persist on ds)
ds = cytome.open("atac.cytome")

piaso.pp.calculateCellMetrics(ds)         # per-cell QC -> ds.cells
piaso.tl.runSVDLazy(ds)                   # X_svd embedding -> ds.embeddings
piaso.tl.neighbors(ds)                    # connectivities/distances -> ds.graphs
piaso.tl.leiden(ds, key_added="leiden")   # labels -> ds.cells["leiden"]
piaso.tl.umap(ds)                         # X_umap -> ds.embeddings
piaso.pl.plotUMAP(ds, color="leiden")

# Marker genes can persist to the cytome too
import cosg
cosg.run_cosg_cytome(ds, groupby="leiden", modality="ATAC",
                     write_to_cytome=True)        # -> ds.metadata["cosg"]

ds.close()
```

See [piaso.org](https://piaso.org) for documentation and tutorials.

## File format

A `.cytome` file is a SQLite database with:

- Entity tables (`cells`, `genes`, `peaks`, ...)
- Chunked compressed sparse matrices (`matrix_chunks` + `matrix_meta`)
- Optional CSC chunks (`matrix_csc_chunks`)
- Dense chunked embeddings (`dense_chunks` + `embedding_meta`)
- Fragment storage (chunked compressed BLOBs) + genomic R-tree indices
- Provenance (`_provenance`) and metadata (`_metadata`)

## Comparison with existing tools

Trade-offs in the current implementation:

- Full matrix materialization can be slower than direct HDF5 reads for some workloads.
- Merge/subset APIs are chunk-aware, but current merge implementation may still materialize intermediate matrices for gene remapping.
- SQLite provides strong local reliability and portability but is not a distributed/cloud-native storage engine.

## Citation

If you use `cytome`, please cite the upcoming manuscript.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.