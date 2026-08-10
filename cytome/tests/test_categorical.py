from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cytome


anndata = pytest.importorskip("anndata")


class TestCategoricalRoundTrip:
    def test_ordered_and_unused_categories_preserved(self, tmp_path):
        obs = pd.DataFrame(
            {
                "barcode": [f"CELL{i}" for i in range(5)],
                "stage": pd.Categorical(
                    ["mid", "early", "late", "mid", "early"],
                    categories=["early", "mid", "late", "terminal"],
                    ordered=True,
                ),
            }
        )
        var = pd.DataFrame({"gene_id": ["g1", "g2", "g3"]})
        adata = anndata.AnnData(X=np.zeros((5, 3), dtype=np.float32), obs=obs, var=var)

        path = tmp_path / "cat.cytome"
        ds = cytome.from_anndata(adata, output=path)
        ds.close()
        back = cytome.open(path).to_anndata()

        assert back.obs["stage"].dtype.name == "category"
        assert back.obs["stage"].cat.ordered is True
        assert list(back.obs["stage"].cat.categories) == ["early", "mid", "late", "terminal"]

    def test_color_mapping_preserved(self, tmp_path):
        obs = pd.DataFrame(
            {
                "barcode": [f"CELL{i}" for i in range(4)],
                "clusters": pd.Categorical(
                    ["Alpha", "Beta", "Alpha", "Delta"],
                    categories=["Alpha", "Beta", "Delta"],
                ),
            }
        )
        var = pd.DataFrame({"gene_id": ["g1", "g2"]})
        adata = anndata.AnnData(
            X=np.zeros((4, 2), dtype=np.float32),
            obs=obs,
            var=var,
            uns={"clusters_colors": np.array(["#ff0000", "#00ff00", "#0000ff"])},
        )
        path = tmp_path / "colors.cytome"
        ds = cytome.from_anndata(adata, output=path)
        ds.close()
        back = cytome.open(path).to_anndata()

        assert list(back.obs["clusters"].cat.categories) == ["Alpha", "Beta", "Delta"]
        assert isinstance(back.uns["clusters_colors"], np.ndarray)
        assert list(back.uns["clusters_colors"]) == ["#ff0000", "#00ff00", "#0000ff"]

    def test_var_categorical_roundtrip(self, tmp_path):
        obs = pd.DataFrame({"barcode": [f"CELL{i}" for i in range(4)]})
        var = pd.DataFrame(
            {
                "gene_id": ["g1", "g2", "g3"],
                "biotype": pd.Categorical(
                    ["protein_coding", "lincRNA", "protein_coding"],
                    categories=["protein_coding", "lincRNA", "miRNA"],
                ),
            }
        )
        adata = anndata.AnnData(X=np.zeros((4, 3), dtype=np.float32), obs=obs, var=var)
        path = tmp_path / "varcat.cytome"
        ds = cytome.from_anndata(adata, output=path)
        ds.close()
        back = cytome.open(path).to_anndata()

        assert back.var["biotype"].dtype.name == "category"
        assert "miRNA" in back.var["biotype"].cat.categories
