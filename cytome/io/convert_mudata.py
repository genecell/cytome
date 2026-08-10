"""MuData conversion utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import cytome


_MODALITY_MAP = {
    "rna": "RNA",
    "atac": "ATAC",
    "protein": "protein",
}


def from_mudata(mdata, output: str | Path | None = None, force: bool = False):
    """Convert MuData into Cytome dataset."""
    if output is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".cytome", delete=False)
        output = tmp.name
        tmp.close()
        force = True  # the tempfile we just created is ours to overwrite

    ds = cytome.create(output, force=force)

    obs = mdata.obs.copy()
    if "barcode" not in obs.columns:
        obs = obs.reset_index().rename(columns={obs.index.name or "index": "barcode"})
    else:
        obs = obs.reset_index(drop=True)
    ds.set_entity("cells", obs)

    for key, adata in mdata.mod.items():
        modality = _MODALITY_MAP.get(str(key).lower(), str(key))
        matrix = adata.X if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
        ds.add_matrix(f"{modality}_counts", matrix)

        var = adata.var.copy()
        if modality == "RNA":
            if "gene_id" not in var.columns:
                var = var.reset_index().rename(columns={var.index.name or "index": "gene_id"})
            else:
                var = var.reset_index(drop=True)
            ds.set_entity("genes", var)
        elif modality == "ATAC":
            if "peak_id" not in var.columns:
                var = var.reset_index().rename(columns={var.index.name or "index": "peak_id"})
            else:
                var = var.reset_index(drop=True)
            ds.set_entity("peaks", var)
        elif modality == "protein":
            if "protein_id" not in var.columns:
                var = var.reset_index().rename(columns={var.index.name or "index": "protein_id"})
            else:
                var = var.reset_index(drop=True)
            ds.set_entity("proteins", var)

        for layer_name, layer_mat in adata.layers.items():
            lay = layer_mat if sp.issparse(layer_mat) else sp.csr_matrix(np.asarray(layer_mat))
            ds.add_matrix(f"{modality}_{layer_name}", lay)

        for emb_name, emb in adata.obsm.items():
            suffix = emb_name[2:] if emb_name.startswith("X_") else emb_name
            ds.add_embedding(f"{modality}_{suffix}", np.asarray(emb))

    ds.flush()
    return ds


def to_mudata(ds):
    """Convert Cytome dataset into MuData."""
    try:
        import mudata
    except ImportError as exc:
        raise ImportError("mudata is required for to_mudata().") from exc

    adatas = {}
    for modality in ds.modalities:
        adatas[modality.lower()] = ds.to_anndata(modality=modality)
    return mudata.MuData(adatas)
