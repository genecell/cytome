"""``in``, ``list()`` and ``len()`` on the accessors return.

Found by the PIASO workflow agent with gdb (2026-10-28): ``name in ds.graphs``
never returned. ``_GraphAccessor`` defined ``__getitem__``, which makes a
GraphStore for any key, and neither ``__contains__`` nor ``__iter__``, so
Python's ``in`` and ``list()`` fell back to indexing 0, 1, 2, ... waiting for
an IndexError that never came. ``list(ds.cells)`` and ``list(ds.spatial_images)``
fell back the same way and raised on the first index.
"""
import threading

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
from cytome.core.dataset import _EmbeddingAccessor, _GraphAccessor
from cytome.core.entity import EntityTable
from cytome.core.metadata import MetadataStore
from cytome.core.spatial import _SpatialImageAccessor


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "g.cytome"
    ds = cytome.create(str(p))
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"c{i}" for i in range(4)]}))
    ds.add_graph("knn", sp.csr_matrix(np.eye(4)))
    ds.flush()
    ds.add_spatial_image("lib", "hires", np.zeros((3, 2), dtype=np.uint8))
    ds.close()
    return str(p)


def _within(path, fn, timeout=10.0):
    """``fn(ds)`` in a thread with its own connection, which must finish in
    ``timeout`` seconds (on the old code it never did)."""
    box = {}

    def target():
        ds = cytome.open(path, mode="r")
        try:
            box["value"] = fn(ds)
        except BaseException as exc:          # noqa: BLE001 - reported below
            box["error"] = exc
        finally:
            ds.close()
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"did not return in {timeout:.0f} s"
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_membership_in_graphs_returns_and_creates_nothing(store):
    assert _within(store, lambda ds: "knn" in ds.graphs) is True
    assert _within(store, lambda ds: "nope" in ds.graphs) is False
    assert _within(store, lambda ds: 0 in ds.graphs) is False
    assert _within(store, lambda ds: list(ds.graphs)) == ["knn"]
    assert _within(store, lambda ds: len(ds.graphs)) == 1
    assert _within(store, lambda ds: "knn" in ds.var_graphs) is False
    assert _within(store, lambda ds: list(ds.var_graphs)) == []
    assert _within(store, lambda ds: ds.graphs.keys()) == ["knn"]       # nothing was made


def test_entity_tables_iterate_their_columns(store):
    cols = _within(store, lambda ds: ds.cells.columns)
    assert _within(store, lambda ds: list(ds.cells)) == cols
    assert "barcode" in cols
    assert _within(store, lambda ds: 0 in ds.cells) is False


def test_spatial_images_iterate_their_keys(store):
    assert _within(store, lambda ds: list(ds.spatial_images)) == [("lib", "hires")]
    assert _within(store, lambda ds: ("lib", "hires") in ds.spatial_images) is True
    assert _within(store, lambda ds: "lib" in ds.spatial_images) is False
    assert _within(store, lambda ds: 0 in ds.spatial_images) is False


@pytest.mark.parametrize("cls", [_GraphAccessor, _EmbeddingAccessor, MetadataStore, EntityTable,
                                 _SpatialImageAccessor])
def test_keyed_accessors_define_membership_and_iteration(cls):
    # a class with __getitem__ over names needs both, or `in` and list()
    # index it with 0, 1, 2, ...
    assert "__contains__" in vars(cls) and "__iter__" in vars(cls)
