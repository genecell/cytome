"""Download real single-cell datasets for Cytome validation tests."""

from __future__ import annotations

import os
import urllib.request

DATASETS = {
    "Pancreas_with_cc": {
        "url": "https://cell2fate.cog.sanger.ac.uk/Pancreas_with_cc/Pancreas_with_cc_anndata.h5ad",
        "description": "Mouse pancreas development (cell2fate)",
    },
    "DentateGyrus": {
        "url": "https://cell2fate.cog.sanger.ac.uk/DentateGyrus/DentateGyrus_anndata.h5ad",
        "description": "Mouse dentate gyrus (cell2fate)",
    },
}


def download_datasets(data_dir: str = "tests/real_data/data") -> None:
    """Download and validate test datasets."""
    os.makedirs(data_dir, exist_ok=True)

    for name, info in DATASETS.items():
        filepath = os.path.join(data_dir, f"{name}_anndata.h5ad")
        if os.path.exists(filepath):
            print(f"[SKIP] {name} already exists: {filepath}")
            continue

        print(f"[DOWNLOAD] {name}: {info['description']}")
        print(f"  URL: {info['url']}")
        print(f"  Destination: {filepath}")

        try:
            urllib.request.urlretrieve(info["url"], filepath)
            file_size = os.path.getsize(filepath)
            print(f"  Size: {file_size / 1024 / 1024:.1f} MB")

            import anndata

            adata = anndata.read_h5ad(filepath)
            print(f"  Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
            print(f"  Layers: {list(adata.layers.keys())}")
            print(f"  Obs columns: {list(adata.obs.columns[:10])}...")
            print(f"  Uns keys: {list(adata.uns.keys())[:10]}...")
            del adata
            print("  [OK] Valid AnnData file")
        except Exception as exc:
            print(f"  [FAIL] Download or validation failed: {exc}")
            if os.path.exists(filepath):
                os.remove(filepath)
            continue

    print("\nDone. Downloaded datasets are in:", data_dir)


if __name__ == "__main__":
    download_datasets()
