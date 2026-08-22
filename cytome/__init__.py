"""Top-level Cytome API."""

from __future__ import annotations

from pathlib import Path

from cytome.core.dataset import CytomeDataset

# ---------------------------------------------------------------- modality API
# Promoted to the top level as a stability commitment, not for convenience.
#
# Downstream packages (PIASO's feature resolver) need to answer "which modality
# holds this feature, and how do I read its column" without reimplementing the
# routing. They were reaching into ``cytome.utils.modality`` -- an internal
# path, on which cytome made no promises -- so any reshaping here would have
# broken them silently.
#
# These names are public API. ``MODALITY_REGISTRY`` entries are 4-tuples
# ``(modality, entity_table, idx_col, id_columns)`` and RNA comes first, because
# callers auto-detecting which modality holds a feature iterate in order.
# Changing that shape, or the signatures below, requires a major version bump.
# tests/test_public_api_contract.py enforces this.
from cytome.utils.modality import (
    MODALITY_REGISTRY,
    MODALITY_VAR_ENTITY,
    modality_var_entity,
    modality_feature_table_info,
    modality_has_feature,
    read_feature_column,
    read_feature_columns,
    modality_cell_depth,
)


__all__ = [
    "CytomeDataset",
    "open",
    "create",
    "from_anndata",
    "from_h5ad",
    "to_anndata",
    "from_mudata",
    "to_mudata",
    "from_cellranger",
    "from_cellranger_arc",
    "from_10x_h5",
    "from_barcodes",
    "merge",
    "import_gtf",
    # modality routing -- see the note above before changing any of these
    "MODALITY_REGISTRY",
    "MODALITY_VAR_ENTITY",
    "modality_var_entity",
    "modality_feature_table_info",
    "modality_has_feature",
    "read_feature_column",
    "read_feature_columns",
    "modality_cell_depth",
]
__version__ = "0.2.5"


def import_gtf(ds, gtf_path, **kwargs):
    """Convenience: forward to :func:`cytome.io.gtf_import.import_gtf`."""
    from cytome.io.gtf_import import import_gtf as _impl
    return _impl(ds, gtf_path, **kwargs)


def open(path: str | Path) -> CytomeDataset:
    """Open an existing Cytome dataset."""
    return CytomeDataset(path, mode="r+")


def create(path: str | Path, force: bool = False) -> CytomeDataset:
    """Create a new Cytome dataset.

    ``force=False`` (default) raises :class:`FileExistsError` if ``path``
    already exists; pass ``force=True`` to overwrite.
    """
    return CytomeDataset(path, mode="w", force=force)


def from_anndata(adata, modality: str = "RNA", output: str | Path | None = None,
                 force: bool = False) -> CytomeDataset:
    """Create a Cytome dataset from an in-memory AnnData object.

    Use this when you already have an ``AnnData`` instance (e.g. just
    loaded with ``scanpy.read``, or built in-memory). For h5ad files
    on disk, prefer :func:`cytome.from_h5ad` — it can stream large
    files chunk-by-chunk via ``backed=True``, keeping peak RAM bounded.

    Quick guide:

    +-------------------------------------------+--------------------------------+
    | Input                                     | Recommended entry              |
    +===========================================+================================+
    | ``AnnData`` object in memory              | ``cytome.from_anndata(adata)`` |
    +-------------------------------------------+--------------------------------+
    | ``.h5ad`` path, file fits comfortably     | ``cytome.from_h5ad(path)``     |
    | in RAM                                    | (default ``backed=False``)     |
    +-------------------------------------------+--------------------------------+
    | ``.h5ad`` path, file is large             | ``cytome.from_h5ad(path,       |
    | (>1 GB sparse, multi-million cells)       | backed=True)`` — streams via   |
    |                                           | h5py, bounded RAM              |
    +-------------------------------------------+--------------------------------+

    Both functions write structurally identical cytomes, with the same
    ``_anndata_*`` metadata for lossless round-trip via
    :func:`cytome.to_anndata`.

    Parameters
    ----------
    adata
        In-memory AnnData. ``adata.X``, ``adata.layers``, ``adata.raw``,
        ``adata.obsm`` / ``varm`` / ``obsp`` / ``varp`` / ``uns`` are all
        preserved per the cytome modality conventions.
    modality
        ``'RNA'`` (default), ``'GA'`` (gene activity), ``'ATAC'``, or
        ``'tiles'``. Routes the var-entity table via cytome's modality
        registry (``cytome.utils.modality.MODALITY_REGISTRY``).
    output
        Output path. If ``None``, a tempfile is created.
    force
        If ``False`` (default) and ``output`` exists, raise
        :class:`FileExistsError`; ``force=True`` overwrites.
    """
    from cytome.io.convert_anndata import from_anndata as _impl
    return _impl(adata=adata, modality=modality, output=output, force=force)


def from_h5ad(
    h5ad_path: str | Path,
    output: str | Path,
    modality: str = "RNA",
    backed: bool = False,
    chunk_size: int = 2048,
    storage_chunk_size: int = 128,
    compression: str = "zstd",
    verbose: bool = True,
    # Round 10 (2026-05-24) — per-category opt-outs + per-key skip lists.
    write_raw: bool = True,
    write_layers: bool = True,
    write_obsm: bool = True,
    write_obsp: bool = True,
    write_varm: bool = True,
    write_varp: bool = True,
    write_uns: bool = True,
    skip_layers: list[str] | None = None,
    skip_obsm: list[str] | None = None,
    skip_obsp: list[str] | None = None,
    skip_varm: list[str] | None = None,
    skip_varp: list[str] | None = None,
    force: bool = False,
) -> CytomeDataset:
    """Create a Cytome dataset from a ``.h5ad`` file on disk.

    Use this when the input lives on disk as an ``.h5ad`` file. For an
    in-memory ``AnnData`` object, prefer :func:`cytome.from_anndata`
    — it skips the disk read entirely.

    Quick guide:

    +-------------------------------------------+--------------------------------+
    | Input                                     | Recommended entry              |
    +===========================================+================================+
    | ``AnnData`` object in memory              | ``cytome.from_anndata(adata)`` |
    +-------------------------------------------+--------------------------------+
    | ``.h5ad`` path, file fits comfortably     | ``cytome.from_h5ad(path)``     |
    | in RAM                                    | (default ``backed=False``)     |
    +-------------------------------------------+--------------------------------+
    | ``.h5ad`` path, file is large             | ``cytome.from_h5ad(path,       |
    | (>1 GB sparse, multi-million cells)       | backed=True)`` — streams via   |
    |                                           | h5py, bounded RAM              |
    +-------------------------------------------+--------------------------------+

    Parameters
    ----------
    h5ad_path
        Path to the ``.h5ad`` file.
    output
        Output ``.cytome`` path.
    modality
        ``'RNA'`` (default), ``'GA'``, ``'ATAC'``, or ``'tiles'``.
        Routes the var-entity table via cytome's modality registry.
    backed
        If ``False`` (default), the file is loaded fully into memory
        as an ``AnnData`` and then converted (delegates to
        :func:`from_anndata`). If ``True``, the file is streamed
        chunk-by-chunk via h5py: peak RAM is bounded by ``chunk_size``
        rows (sparse) or the largest dense obsm entry (whichever is
        bigger). Use ``backed=True`` for files that don't comfortably
        fit in RAM.

    Per-category opt-outs (apply to both branches; default = preserve):
    ``write_raw``, ``write_layers``, ``write_obsm``, ``write_obsp``,
    ``write_varm``, ``write_varp``, ``write_uns``.

    Per-key skip lists (drop named entries within a category):
    ``skip_layers=[...]``, ``skip_obsm=[...]``, etc.

    Notes
    -----
    ``backed=True`` bypasses ``anndata.read_h5ad(backed='r')``
    entirely because AnnData's backed mode lazy-loads only ``.X`` and
    ``.raw.X``; layers / obsm / obsp / varm / varp / uns are eagerly
    loaded, defeating the purpose of "backed". cytome's
    ``backed=True`` path reads directly from h5py instead, keeping
    peak RAM bounded chunk-by-chunk.
    """
    from cytome.io.convert_anndata import from_h5ad as _impl
    _common_kwargs = dict(
        h5ad_path=h5ad_path,
        output=output,
        modality=modality,
        chunk_size=chunk_size,
        storage_chunk_size=storage_chunk_size,
        compression=compression,
        verbose=verbose,
        write_raw=write_raw,
        write_layers=write_layers,
        write_obsm=write_obsm,
        write_obsp=write_obsp,
        write_varm=write_varm,
        write_varp=write_varp,
        write_uns=write_uns,
        skip_layers=skip_layers,
        skip_obsm=skip_obsm,
        skip_obsp=skip_obsp,
        skip_varm=skip_varm,
        skip_varp=skip_varp,
        force=force,
    )
    if not backed:
        return _impl(backed=False, **_common_kwargs)

    # Round 12 (2026-05-27): backed mode auto-falls-back to backed=False
    # when the file's encoding isn't supported by the streaming reader
    # (e.g. dense /X, CSC matrix, CSR without shape attr). AnnData's own
    # reader handles every encoding, so the in-memory path is a strict
    # superset for correctness — at the cost of peak RAM.
    import warnings as _warnings
    try:
        return _impl(backed=True, **_common_kwargs)
    except (KeyError, NotImplementedError) as exc:
        _warnings.warn(
            f"cytome.from_h5ad(backed=True) failed ({type(exc).__name__}: "
            f"{exc}) — falling back to backed=False (in-memory load via "
            f"anndata.read_h5ad). This requires enough RAM to hold the "
            f"file. Pass backed=False explicitly to silence this warning.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _impl(backed=False, **_common_kwargs)


def to_anndata(
    ds: CytomeDataset,
    modality: str = "RNA",
    layer: str | None = None,
    include_embeddings: bool = True,
    cell_mask=None,
):
    """Convert a Cytome dataset to AnnData (function form).

    Equivalent to ``ds.to_anndata(...)`` (the method form). Modality
    routes to its own var entity table: RNA → ``genes``,
    GA → ``GA_genes``, ATAC → ``peaks``, tiles → ``tiles``.
    """
    from cytome.io.convert_anndata import to_anndata as _impl
    return _impl(
        ds=ds, modality=modality, layer=layer,
        include_embeddings=include_embeddings, cell_mask=cell_mask,
    )


def from_mudata(mdata, output: str | Path | None = None, force: bool = False):
    """Create a Cytome dataset from MuData.

    ``force=False`` (default) raises :class:`FileExistsError` if ``output``
    exists; ``force=True`` overwrites.
    """
    from cytome.io.convert_mudata import from_mudata as _impl
    return _impl(mdata=mdata, output=output, force=force)


def to_mudata(ds: CytomeDataset):
    """Convert a Cytome dataset to MuData (function form)."""
    from cytome.io.convert_mudata import to_mudata as _impl
    return _impl(ds=ds)


def from_cellranger(path, output: str | Path, sample_name=None,
                    import_fragments: bool | None = None, build_index: bool = True,
                    keep_chroms: str = "standard", modalities: str = "both",
                    force: bool = False):
    """Create a Cytome dataset from Cell Ranger / Cell Ranger ARC output folder(s).

    Reads each folder's feature-barcode matrix from ``filtered_feature_bc_matrix.h5``
    (preferred) or the ``filtered_feature_bc_matrix/`` MTX directory, splitting **Gene
    Expression** features into the ``RNA`` modality and **Peaks** into ``ATAC``, and
    imports ``atac_fragments.tsv.gz`` if present.

    ``path`` may be a single folder or a **list of folders** — a list merges the
    libraries into one dataset (union of genes/peaks, cells concatenated, fragments
    combined). ``sample_name`` accepts a **list matching ``path``** (one per folder),
    written to ``cells.sample_id``.

    Examples
    --------
    >>> cytome.from_cellranger("run/outs", output="lib.cytome", sample_name="ctrl")
    >>> cytome.from_cellranger(["A/outs", "B/outs", "C/outs"], output="merged.cytome",
    ...                        sample_name=["ctrl", "het", "cko"])
    """
    from cytome.io.convert_cellranger import from_cellranger as _impl
    return _impl(path=path, output=output, sample_name=sample_name,
                 import_fragments=import_fragments, build_index=build_index,
                 keep_chroms=keep_chroms, modalities=modalities, force=force)


def from_10x_h5(path: str | Path, output: str | Path, sample_name: str | None = None,
                build_index: bool = True, modalities: str = "both",
                force: bool = False, batch_size: int | None = None):
    """Create a Cytome dataset directly from a CellRanger ``.h5`` (no AnnData).

    Gene Expression features → RNA ``genes`` (``gene_id`` = Ensembl id, ``symbol``
    = name); Peak features (multiome) → ATAC ``peaks``. Handles CellRanger v2/v3.
    ``force=False`` (default) raises if ``output`` exists; ``force=True`` overwrites.

    The matrix is read in cell batches; ``batch_size`` defaults to a size chosen
    from the file's own density. See
    :func:`cytome.io.convert_cellranger.from_10x_h5`.
    """
    from cytome.io.convert_cellranger import from_10x_h5 as _impl
    return _impl(path=path, output=output, sample_name=sample_name,
                 build_index=build_index, modalities=modalities, force=force,
                 batch_size=batch_size)


def from_cellranger_arc(
    path: str | Path,
    output: str | Path,
    sample_name: str | None = None,
    import_fragments: bool = True,
    build_index: bool = True,
    force: bool = False,
):
    """Create a Cytome dataset from Cell Ranger ARC output.

    ``force=False`` (default) raises if ``output`` exists; ``force=True`` overwrites.
    """
    from cytome.io.convert_cellranger import from_cellranger_arc as _impl
    return _impl(
        path=path,
        output=output,
        sample_name=sample_name,
        import_fragments=import_fragments,
        build_index=build_index,
        force=force,
    )


def merge(
    inputs,
    output,
    batch_key: str = "sample_id",
    batch_labels=None,
    gene_strategy: str = "union",
    peak_strategy: str = "union",
    obs_columns="all",
    include_embeddings: bool = False,
    include_fragments: bool = True,
    include_graphs: bool = False,
    chunk_memory_mb: int = 256,
    force: bool = False,
    **kwargs,
):
    """Merge multiple Cytome datasets into one on-disk dataset.

    Cells from every input are concatenated (each input occupies a contiguous
    block); genes and peaks are reconciled by ``gene_strategy`` / ``peak_strategy``;
    and — if present and requested — compressed fragments are remapped onto the
    merged cell axis and streamed into the output's ``fragment_chunks`` store. This
    is the standard way to combine multiple libraries / samples into one atlas.

    Parameters
    ----------
    inputs : sequence of (str | Path | CytomeDataset)
        Datasets to merge, in order. Paths are opened read-only and closed
        afterwards; already-open ``CytomeDataset`` objects are left open. The
        concatenation order defines the cell ordering in the output.
    output : str | Path
        Output ``.cytome`` path. By default (``force=False``) merge raises
        :class:`FileExistsError` if it already exists; pass ``force=True`` to
        overwrite.
    batch_key : str, default ``"sample_id"``
        Name of the ``cells`` column written to record each cell's source input.
    batch_labels : sequence of str, optional
        One label per input, written into ``cells[batch_key]``. Defaults to each
        input's file stem (``Path(ds.path).stem``).
    gene_strategy : {"union", "intersection"}, default ``"union"``
        How to reconcile gene sets. ``"union"`` keeps every gene seen in any input
        (genes missing from an input become explicit zeros for its cells —
        atlas-friendly); ``"intersection"`` keeps only genes present in *all* inputs.
    peak_strategy : {"union", "intersection"}, default ``"union"``
        Same as ``gene_strategy`` but for ATAC peaks.
    obs_columns : {"all", "shared"} | list of str, default ``"all"``
        Which ``cells`` columns to keep. ``"all"`` keeps every column seen in any
        input (NaN where absent); ``"shared"`` keeps only columns common to all
        inputs; a list keeps exactly those columns (silently ignoring missing ones).
    include_embeddings : bool, default ``False``
        Reserved — per-input embeddings are not transferred (recompute on the merge).
    include_fragments : bool, default ``True``
        Remap and stream each input's fragments into the merged ``fragment_chunks``.
        Set ``False`` to produce a counts-only merge (faster, smaller).
    include_graphs : bool, default ``False``
        Reserved — neighbor graphs are not transferred (recompute on the merge).
    chunk_memory_mb : int, default ``256``
        Reserved tuning knob for the fragment-streaming chunk size.
    **kwargs
        Forwarded verbatim to :func:`cytome.io.merge.merge` for forward
        compatibility with newer options.

    Returns
    -------
    CytomeDataset
        The merged dataset (open).

    Examples
    --------
    >>> # union of three libraries, fragments combined, labelled by sample
    >>> ds = cytome.merge(
    ...     ["ctrl.cytome", "het.cytome", "cko.cytome"],
    ...     output="atlas.cytome",
    ...     batch_labels=["ctrl", "het", "cko"],
    ... )
    >>> # counts-only intersection merge (no fragments)
    >>> ds = cytome.merge(parts, "merged.cytome",
    ...                    gene_strategy="intersection", include_fragments=False)
    """
    from cytome.io.merge import merge as _impl
    return _impl(
        inputs=inputs,
        output=output,
        batch_key=batch_key,
        batch_labels=batch_labels,
        gene_strategy=gene_strategy,
        peak_strategy=peak_strategy,
        obs_columns=obs_columns,
        include_embeddings=include_embeddings,
        include_fragments=include_fragments,
        include_graphs=include_graphs,
        chunk_memory_mb=chunk_memory_mb,
        force=force,
        **kwargs,
    )


def from_barcodes(
    barcodes,
    output: str | Path,
    sample_id: str | None = None,
    force: bool = False,
) -> CytomeDataset:
    """Create a minimal Cytome dataset from barcodes.

    Parameters
    ----------
    barcodes
        List of barcode strings, file path (one per line/TSV), or DataFrame.
    output
        Output .cytome file path.
    sample_id
        Optional sample identifier for cells table.
    force
        If ``False`` (default) and ``output`` exists, raise
        :class:`FileExistsError`; ``force=True`` overwrites.
    """
    import pandas as pd

    if isinstance(barcodes, (str, Path)):
        bc_df = pd.read_csv(
            barcodes, sep="\t", header=None, comment="#", dtype=str,
        )
        bc_df = bc_df.iloc[:, 0].to_frame("barcode").drop_duplicates()
    elif isinstance(barcodes, pd.DataFrame):
        bc_df = barcodes
    else:
        bc_df = pd.DataFrame({"barcode": list(barcodes)})

    if sample_id:
        bc_df["sample_id"] = sample_id

    ds = create(output, force=force)
    ds.set_entity("cells", bc_df)
    ds.flush()
    return ds
