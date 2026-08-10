"""PyTorch dataset adapters for Cytome."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

import numpy as np

from cytome.core.dataset import CytomeDataset


class CytomeTorchDataset:
    """Row-wise PyTorch-compatible dataset backed by Cytome."""

    def __init__(
        self,
        path: str,
        n_cells: int,
        modalities: Sequence[str],
        layers: dict[str, str],
        obs_columns: Sequence[str] | None = None,
    ) -> None:
        self.path = path
        self.n_cells = int(n_cells)
        self.modalities = list(modalities)
        self.layers = dict(layers)
        self.obs_columns = list(obs_columns) if obs_columns is not None else None
        self._connections: dict[int, CytomeDataset] = {}

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds = self._get_worker_ds()
        out: Dict[str, Any] = {}
        for mod in self.modalities:
            layer_name = self.layers.get(mod, "counts")
            mat = ds.__getattr__(mod).layer(layer_name)
            row = mat[idx, :].toarray().ravel().astype(np.float32)
            out[mod] = row

        if self.obs_columns:
            cell_df = ds.cells.to_pandas().set_index("cell_idx")
            for col in self.obs_columns:
                out[col] = cell_df.loc[idx, col]
        return out

    def _get_worker_ds(self) -> CytomeDataset:
        worker_id = os.getpid()
        if worker_id not in self._connections:
            self._connections[worker_id] = CytomeDataset(self.path, mode="r")
        return self._connections[worker_id]
