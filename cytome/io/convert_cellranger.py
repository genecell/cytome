"""Import Cell Ranger outputs into Cytome datasets."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import cytome
from cytome.index.builder import build_peak_index
from cytome.io.convert_fragments import import_fragments as _import_fragments
from cytome.io.convert_fragments import import_fragments_streaming as _import_fragments_streaming


def from_cellranger(
    path,
    output: str | Path,
    sample_name=None,
    import_fragments: bool | None = None,
    build_index: bool = True,
    keep_chroms: str = "standard",
    modalities: str = "both",
    force: bool = False,
):
    """Create a Cytome dataset from Cell Ranger / Cell Ranger ARC output folder(s).

    .. note::
       **Where each modality comes from.** The per-cell counts are read straight from
       Cell Ranger's ``filtered_feature_bc_matrix.h5`` (or the MTX directory): a
       multiome / ARC matrix carries **both** ``"Gene Expression"`` rows → the ``RNA``
       modality and ``"Peaks"`` rows → the ``ATAC`` modality. So the **ATAC peaks and
       peak counts are Cell Ranger's own peak calls** — *not* PICCO, and *not* derived
       from the fragment file. The ``atac_fragments.tsv.gz`` is a **separate** input,
       imported into ``fragment_chunks`` only when ``import_fragments`` is true. For the
       fast Rust fragment import, prefer :func:`piaso.pp.importCellRanger`.

    Reads the per-cell feature-barcode matrix from each folder and writes a Cytome
    dataset, splitting **Gene Expression** features into the ``RNA`` modality and
    **Peaks** (multiome / ARC) into the ``ATAC`` modality. For each folder the matrix
    is read from ``filtered_feature_bc_matrix.h5`` if present, otherwise from the
    ``filtered_feature_bc_matrix/`` MTX directory. If an ``atac_fragments.tsv.gz`` is
    present (multiome) it is imported into the ``fragment_chunks`` store.

    Parameters
    ----------
    path : str | Path | list of (str | Path)
        A single Cell Ranger output folder, OR a list of folders. When a list is
        given the libraries are **merged into one dataset** (union of genes/peaks,
        cells concatenated, fragments combined) — convenient for multi-library runs.
    output : str | Path
        Output ``.cytome`` path.
    sample_name : str | list of str, optional
        Sample identifier written to the ``cells.sample_id`` column. For multiple
        folders pass a **list of names matching ``path``** (one per folder); cells
        from each library carry their library's name. A single string is only valid
        with a single folder.
    import_fragments : bool, optional
        Whether to import ``atac_fragments.tsv.gz`` into ``fragment_chunks``. Default
        ``None`` → **True when ATAC is requested** (``modalities`` in ``{"both","atac"}``)
        and **False for** ``modalities="rna"``. Set explicitly to override.

        .. deprecated::
            Fragment import *via this function* uses the slow pure-Python streaming path.
            Prefer :func:`piaso.pp.importCellRanger` (or :func:`piaso.pp.importFragments`),
            which use the fast Rust importer (k-way merge + inline tile quantification).
            When fragments are actually imported here a ``DeprecationWarning`` is emitted.
    build_index : bool, default True
        Build the peak / fragment spatial index.
    modalities : {"both", "rna", "atac"}, default "both"
        Which feature types to write. ``"both"`` writes Gene Expression → ``RNA`` and
        Peaks → ``ATAC``. ``"rna"`` writes **only** the RNA genes/counts (no ATAC peaks).
        ``"atac"`` writes **only** the ATAC peaks (no RNA). Used by
        :func:`piaso.pp.importCellRanger`'s ``modality`` switch.
    keep_chroms : {"standard", "all"}, default "standard"
        ``"standard"`` drops ATAC peaks **and** fragments on chromosomes not in
        :data:`cytome.utils.genome.CHROM_ORDER` (chr1–22, X, Y, M) — i.e. unplaced
        scaffolds / contigs like ``GL456233.1`` that otherwise crash the spatial index.
        RNA genes are never chromosome-filtered. ``"all"`` keeps everything (peaks on
        non-standard chromosomes are still skipped by the index, with a warning).

    Returns
    -------
    CytomeDataset

    Examples
    --------
    >>> # single library (RNA-only or multiome auto-detected)
    >>> ds = cytome.from_cellranger("run/outs", output="lib.cytome", sample_name="ctrl")
    >>> # multiple libraries merged into one dataset
    >>> ds = cytome.from_cellranger(
    ...     ["run_A/outs", "run_B/outs", "run_C/outs"],
    ...     output="merged.cytome",
    ...     sample_name=["ctrl", "het", "cko"],
    ... )
    """
    if modalities not in ("both", "rna", "atac"):
        raise ValueError(f"modalities must be 'both', 'rna' or 'atac', got {modalities!r}.")
    # Default: import fragments whenever ATAC is requested; never for RNA-only.
    if import_fragments is None:
        import_fragments = modalities != "rna"
    import_fragments = bool(import_fragments) and modalities != "rna"
    if import_fragments:
        import warnings
        warnings.warn(
            "from_cellranger imports ATAC fragments via the slow pure-Python streaming path. "
            "For the fast Rust importer (k-way merge + inline tile quantification) use "
            "piaso.pp.importCellRanger (or piaso.pp.importFragments) instead.",
            DeprecationWarning, stacklevel=2)

    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    if sample_name is None:
        names: List[Optional[str]] = [None] * len(paths)
    elif isinstance(sample_name, (str, Path)):
        if len(paths) > 1:
            raise ValueError(
                "Multiple folders given but a single sample_name; pass a list of "
                f"sample names of length {len(paths)} (one per folder).")
        names = [str(sample_name)]
    else:
        names = [None if s is None else str(s) for s in sample_name]
        if len(names) != len(paths):
            raise ValueError(
                f"sample_name list length ({len(names)}) does not match the number "
                f"of folders ({len(paths)}).")

    if len(paths) == 1:
        return _from_one_cellranger(paths[0], output, names[0], import_fragments, build_index, keep_chroms, modalities, force=force)

    # Multiple folders → build each to a temp dataset, then merge (union features + fragments).
    import shutil
    import tempfile

    out = Path(output)
    tmpdir = Path(tempfile.mkdtemp(prefix="cr_merge_", dir=str(out.parent or ".")))
    parts: List[str] = []
    labels: List[str] = []
    try:
        for i, (p, nm) in enumerate(zip(paths, names)):
            label = nm if nm is not None else f"sample{i + 1}"
            part = tmpdir / f"part_{i}.cytome"
            ds_i = _from_one_cellranger(p, part, label, import_fragments, build_index, keep_chroms, modalities, force=True)
            ds_i.close()
            parts.append(str(part)); labels.append(label)
        # batch_labels → cells.sample_id carries each library's name (not the temp path stem)
        cytome.merge(parts, str(out), batch_labels=labels, include_fragments=import_fragments, force=force)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return cytome.open(str(out))


def _from_one_cellranger(folder: Path, output, sample_name, import_fragments, build_index,
                         keep_chroms="standard", modalities="both", force=False):
    """Build a Cytome dataset from one Cell Ranger folder (h5 or MTX; RNA ± ATAC)."""
    from cytome.io.convert_anndata import make_unique_ids, warn_duplicate_symbols

    folder = Path(folder)
    h5 = folder / "filtered_feature_bc_matrix.h5"
    mtx = folder / "filtered_feature_bc_matrix"
    if h5.exists():
        matrix, barcodes, ids, names, ftypes = _read_10x_h5(h5)
    elif mtx.exists():
        matrix, barcodes, features = _read_matrix_dir(mtx)
        ids = np.array([f[0] for f in features])
        names = np.array([f[1] if len(f) > 1 else f[0] for f in features])
        ftypes = np.array([f[2] if len(f) >= 3 else "Gene Expression" for f in features])
    else:
        raise FileNotFoundError(
            f"No 'filtered_feature_bc_matrix.h5' or 'filtered_feature_bc_matrix/' in {folder}")
    matrix = matrix.tocsr() if not sp.isspmatrix_csr(matrix) else matrix   # features × cells

    ds = cytome.create(output, force=force)
    cells_df = pd.DataFrame({"barcode": list(barcodes)})
    if sample_name is not None:
        cells_df["sample_id"] = sample_name
    ds.set_entity("cells", cells_df)

    ftypes = np.asarray([str(t) for t in ftypes])
    ids = np.asarray(ids); names = np.asarray(names)
    gene_mask = ftypes == "Gene Expression"
    peak_mask = np.array(["Peak" in t for t in ftypes])
    if not peak_mask.any():   # fallback: peak feature ids look like 'chr:start-end'
        peak_mask = np.array(
            [(":" in str(x) and "-" in str(x).split(":", 1)[-1]) for x in ids]) & ~gene_mask
    if not gene_mask.any() and not peak_mask.any():
        gene_mask = np.ones(len(ids), dtype=bool)   # single-modality file → all RNA
    if modalities == "rna":
        peak_mask = np.zeros(len(ids), dtype=bool)   # RNA-only: drop ATAC peaks entirely
    elif modalities == "atac":
        gene_mask = np.zeros(len(ids), dtype=bool)   # ATAC-only: drop RNA genes
    elif modalities == "both" and not peak_mask.any():
        import warnings
        warnings.warn(
            f"from_cellranger(modalities='both'): no ATAC/Peak features found in {folder} "
            f"— imported RNA only.", stacklevel=2)

    if gene_mask.any():
        warn_duplicate_symbols(names[gene_mask])
        ds.set_entity("genes", pd.DataFrame({
            "gene_id": make_unique_ids(ids[gene_mask]), "symbol": names[gene_mask]}))
        ds.add_matrix("RNA_counts", matrix[gene_mask, :].T.tocsr())

    atac_needed = False
    if peak_mask.any():
        peaks_df = _peaks_from_feature_ids(ids[peak_mask])
        atac = matrix[peak_mask, :]
        if peaks_df is None:
            peaks_df = pd.DataFrame({"peak_id": make_unique_ids(ids[peak_mask])})
        elif keep_chroms == "standard" and "chr" in peaks_df.columns:
            from cytome.utils.genome import CHROM_ORDER
            keep = peaks_df["chr"].isin(CHROM_ORDER).values
            n_drop = int((~keep).sum())
            if n_drop:
                import warnings
                warnings.warn(
                    f"from_cellranger: dropped {n_drop} peak(s) on non-standard chromosomes "
                    f"(keep_chroms='standard').", stacklevel=2)
                peaks_df = peaks_df[keep].reset_index(drop=True)
                atac = atac[keep, :]
        ds.set_entity("peaks", peaks_df)
        ds.add_matrix("ATAC_counts", atac.T.tocsr())
        ds.flush()
        if build_index:
            build_peak_index(ds._conn)
        atac_needed = True

    ds.flush()
    if import_fragments:
        frag_path = folder / "atac_fragments.tsv.gz"
        if frag_path.exists():
            mapping = {bc: i for i, bc in enumerate(barcodes)}
            from cytome.utils.genome import CHROM_ORDER
            chr_filter = set(CHROM_ORDER) if keep_chroms == "standard" else None
            ds._conn.commit()   # streaming importer sets PRAGMAs → must not be mid-transaction
            _import_fragments_streaming(ds._conn, frag_path, mapping, chr_filter=chr_filter)
            atac_needed = True

    if atac_needed:
        mods = set(ds.modalities); mods.add("ATAC")
        ds._write_manifest_key("modalities", sorted(mods))
        ds._manifest = ds._read_manifest()
    ds.flush()
    return ds


def from_cellranger_arc(
    path: str | Path,
    output: str | Path,
    sample_name: Optional[str] = None,
    import_fragments: bool = True,
    build_index: bool = True,
    keep_chroms: str = "standard",
    force: bool = False,
):
    """Create Cytome dataset from Cell Ranger ARC outputs.

    Parameters
    ----------
    keep_chroms : {"standard", "all"}, default "standard"
        ``"standard"`` drops ATAC peaks **and** fragments on chromosomes not in
        :data:`cytome.utils.genome.CHROM_ORDER` (chr1–22, X, Y, M) — unplaced
        scaffolds / contigs that otherwise crash the spatial index. RNA genes are
        never chromosome-filtered. ``"all"`` keeps everything.
    """
    outs = Path(path)
    mtx_dir = outs / "filtered_feature_bc_matrix"
    matrix, barcodes, features = _read_matrix_dir(mtx_dir)

    feature_types = np.array([f[2] if len(f) >= 3 else "Gene Expression" for f in features])
    ids = np.array([f[0] for f in features])
    names = np.array([f[1] if len(f) > 1 else f[0] for f in features])

    gene_mask = feature_types == "Gene Expression"
    peak_mask = np.array([("Peak" in t) or (":" in feat_id and "-" in feat_id) for t, feat_id in zip(feature_types, ids)])

    ds = cytome.create(output, force=force)
    cells_df = pd.DataFrame({"barcode": barcodes})
    if sample_name is not None:
        cells_df["sample_id"] = sample_name

    metrics_path = outs / "per_barcode_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        barcode_col = "barcode" if "barcode" in metrics.columns else metrics.columns[0]
        metrics = metrics.rename(columns={barcode_col: "barcode"})
        cells_df = cells_df.merge(metrics, on="barcode", how="left")

    ds.set_entity("cells", cells_df)

    if gene_mask.any():
        genes_df = pd.DataFrame({"gene_id": ids[gene_mask], "symbol": names[gene_mask]})
        ds.set_entity("genes", genes_df)
        ds.add_matrix("RNA_counts", matrix[gene_mask, :].T.tocsr())

    atac_needed = False

    if peak_mask.any():
        peaks_ids = ids[peak_mask]
        peaks_df = _peaks_from_feature_ids(peaks_ids)
        atac = matrix[peak_mask, :]
        if peaks_df is None:
            peaks_df = pd.DataFrame({"peak_id": peaks_ids})
        elif keep_chroms == "standard" and "chr" in peaks_df.columns:
            from cytome.utils.genome import CHROM_ORDER
            keep = peaks_df["chr"].isin(CHROM_ORDER).values
            n_drop = int((~keep).sum())
            if n_drop:
                import warnings
                warnings.warn(
                    f"from_cellranger_arc: dropped {n_drop} peak(s) on non-standard "
                    f"chromosomes (keep_chroms='standard').", stacklevel=2)
                peaks_df = peaks_df[keep].reset_index(drop=True)
                atac = atac[keep, :]
        ds.set_entity("peaks", peaks_df)
        ds.add_matrix("ATAC_counts", atac.T.tocsr())
        atac_needed = True

    bed_path = outs / "atac_peaks.bed"
    if bed_path.exists():
        bed_df = pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end_"],
            usecols=[0, 1, 2],
        )
        bed_df.insert(0, "peak_id", bed_df["chr"] + ":" + bed_df["start"].astype(str) + "-" + bed_df["end_"].astype(str))

        annot_path = outs / "peak_annotation.tsv"
        if annot_path.exists():
            annot = pd.read_csv(annot_path, sep="\t")
            join_key = "peak" if "peak" in annot.columns else annot.columns[0]
            annot = annot.rename(columns={join_key: "peak_id"})
            bed_df = bed_df.merge(annot, on="peak_id", how="left")

        if keep_chroms == "standard":
            from cytome.utils.genome import CHROM_ORDER
            keep = bed_df["chr"].isin(CHROM_ORDER).values
            n_drop = int((~keep).sum())
            if n_drop:
                import warnings
                warnings.warn(
                    f"from_cellranger_arc: dropped {n_drop} BED peak(s) on non-standard "
                    f"chromosomes (keep_chroms='standard').", stacklevel=2)
                bed_df = bed_df[keep].reset_index(drop=True)

        ds.set_entity("peaks", bed_df)
        ds.flush()
        if build_index:
            build_peak_index(ds._conn)
        atac_needed = True

    ds.flush()

    if import_fragments:
        frag_path = outs / "atac_fragments.tsv.gz"
        if frag_path.exists():
            mapping = {bc: i for i, bc in enumerate(barcodes)}
            _import_fragments(ds._conn, frag_path, mapping, build_index=build_index)
            atac_needed = True

    if atac_needed:
        current_modalities = set(ds.modalities)
        current_modalities.add("ATAC")
        ds._write_manifest_key("modalities", sorted(current_modalities))
        ds._manifest = ds._read_manifest()

    ds.flush()
    return ds


def _read_10x_h5(path: str | Path):
    """Parse a CellRanger ``.h5`` (v2 or v3) with h5py — no AnnData.

    Returns ``(matrix_features_x_cells, barcodes, ids, names, feature_types)``.
    """
    import h5py

    def _dec(arr):
        return [b.decode() if isinstance(b, (bytes, bytearray)) else str(b) for b in arr]

    with h5py.File(str(path), "r") as f:
        if "matrix" in f:                          # CellRanger >= 3.0
            g = f["matrix"]
            barcodes = _dec(g["barcodes"][:])
            feat = g["features"]
            ids = _dec(feat["id"][:])
            names = _dec(feat["name"][:]) if "name" in feat else ids
            ftypes = (_dec(feat["feature_type"][:]) if "feature_type" in feat
                      else ["Gene Expression"] * len(ids))
        else:                                      # CellRanger 2.x: single genome group
            gkey = next(iter(f.keys()))
            g = f[gkey]
            barcodes = _dec(g["barcodes"][:])
            ids = _dec(g["genes"][:])
            names = _dec(g["gene_names"][:]) if "gene_names" in g else ids
            ftypes = ["Gene Expression"] * len(ids)
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(int(x) for x in g["shape"][:])  # (n_features, n_cells)

    matrix = sp.csc_matrix((data, indices, indptr), shape=shape)  # features × cells
    return matrix, barcodes, np.asarray(ids), np.asarray(names), np.asarray(ftypes)


def from_10x_h5(path: str | Path, output: str | Path, sample_name: Optional[str] = None,
                build_index: bool = True, modalities: str = "both",
                force: bool = False):
    """Create a Cytome dataset directly from a CellRanger ``.h5`` — no AnnData.

    Gene Expression features → RNA ``genes`` (``gene_id`` = Ensembl id, unique;
    ``symbol`` = gene name). Peak features (multiome) → ATAC ``peaks``. Duplicate
    ids are de-duplicated (``-1``/``-2``); duplicate symbols emit a warning.

    Parameters
    ----------
    modalities : {"both", "rna", "atac"}, default "both"
        Which feature types to keep from a multiome file. Matches
        :func:`from_cellranger`. Single-modality files are unaffected.
    """
    if modalities not in ("both", "rna", "atac"):
        raise ValueError(
            f"modalities must be 'both', 'rna' or 'atac', got {modalities!r}."
        )
    from cytome.io.convert_anndata import make_unique_ids, warn_duplicate_symbols

    matrix, barcodes, ids, names, ftypes = _read_10x_h5(path)

    ds = cytome.create(output, force=force)
    cells_df = pd.DataFrame({"barcode": barcodes})
    if sample_name is not None:
        cells_df["sample_id"] = sample_name
    ds.set_entity("cells", cells_df)

    gene_mask = ftypes == "Gene Expression"
    peak_mask = np.array(["Peak" in str(t) for t in ftypes])
    if not gene_mask.any() and not peak_mask.any():
        gene_mask = np.ones(len(ids), dtype=bool)   # single-modality file → all RNA
    if modalities == "rna":
        peak_mask = np.zeros(len(ids), dtype=bool)   # RNA-only: drop ATAC peaks entirely
    elif modalities == "atac":
        gene_mask = np.zeros(len(ids), dtype=bool)   # ATAC-only: drop RNA genes

    if gene_mask.any():
        warn_duplicate_symbols(names[gene_mask])
        genes_df = pd.DataFrame({
            "gene_id": make_unique_ids(ids[gene_mask]),
            "symbol": names[gene_mask],
        })
        ds.set_entity("genes", genes_df)
        ds.add_matrix("RNA_counts", matrix[gene_mask, :].T.tocsr())

    if peak_mask.any():
        peaks_df = _peaks_from_feature_ids(ids[peak_mask])
        if peaks_df is None:
            peaks_df = pd.DataFrame({"peak_id": make_unique_ids(ids[peak_mask])})
        ds.set_entity("peaks", peaks_df)
        ds.add_matrix("ATAC_counts", matrix[peak_mask, :].T.tocsr())
        ds.flush()
        if build_index:
            build_peak_index(ds._conn)
        current_modalities = set(ds.modalities)
        current_modalities.add("ATAC")
        ds._write_manifest_key("modalities", sorted(current_modalities))
        ds._manifest = ds._read_manifest()

    ds.flush()
    return ds


def _read_matrix_dir(mtx_dir: Path):
    matrix = scipy.io.mmread(str(mtx_dir / "matrix.mtx.gz"))
    matrix = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)

    with gzip.open(mtx_dir / "barcodes.tsv.gz", "rt") as handle:
        barcodes = [line.strip() for line in handle if line.strip()]

    with gzip.open(mtx_dir / "features.tsv.gz", "rt") as handle:
        features = [line.strip().split("\t") for line in handle if line.strip()]

    return matrix, barcodes, features


def _peaks_from_feature_ids(feature_ids: np.ndarray) -> Optional[pd.DataFrame]:
    chroms: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    for feat in feature_ids:
        try:
            chrom, coords = str(feat).split(":", 1)
            start_s, end_s = coords.split("-", 1)
            chroms.append(chrom)
            starts.append(int(start_s))
            ends.append(int(end_s))
        except Exception:
            return None

    return pd.DataFrame(
        {
            "peak_id": feature_ids,
            "chr": chroms,
            "start": starts,
            "end_": ends,
        }
    )
