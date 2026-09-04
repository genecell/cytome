"""Main Cytome dataset interface."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sqlite3

from cytome.core.embedding import EmbeddingArray
from cytome.core.entity import EntityTable, _EntityWrite
from cytome.core.graph import GraphStore
from cytome.core.metadata import MetadataStore, _MetadataWrite
from cytome.core.measurement import MeasurementLayer
from cytome.core.provenance import ProvenanceLog
from cytome.io.chunk_tuning import compute_chunk_size
from cytome.io.chunked_io import ChunkedLayerWriter, write_dense_chunked, write_sparse_chunked
from cytome.io.sqlite_engine import close_database, create_database, open_database

logger = logging.getLogger(__name__)


# Round 12 (2026-05-27): the canonical modality names that the
# ``modalities`` property recognises when deriving from matrix names.
# A matrix called ``{X}_counts`` etc. implies modality X is present
# iff X is in this set. Restricting to known names avoids false
# positives from incidentally-prefixed user matrices.
_KNOWN_MODALITY_NAMES = frozenset({"RNA", "GA", "ATAC", "tiles"})


class Modality:
    """Per-modality accessor for matrix layers."""

    def __init__(self, dataset: "CytomeDataset", name: str) -> None:
        self._dataset = dataset
        self._conn = dataset._conn
        self._name = name

    @property
    def counts(self) -> MeasurementLayer:
        return MeasurementLayer(self._conn, f"{self._name}_counts")

    def layer(self, name: str) -> MeasurementLayer:
        return MeasurementLayer(self._conn, f"{self._name}_{name}")

    @property
    def fragments(self):
        """Return fragment store for ATAC modality."""
        if self._name != "ATAC":
            raise AttributeError("fragments is only available for ATAC modality")
        from cytome.core.fragments import FragmentStore

        return FragmentStore(self._conn, self._dataset)

    def import_fragments(self, path: str | Path, build_index: bool = True) -> None:
        """Import fragments file into ATAC fragment tables.

        .. deprecated::
            Uses legacy per-row format. Use the Rust importer or
            ``import_fragments_streaming()`` instead.
        """
        if self._name != "ATAC":
            raise AttributeError("import_fragments is only available for ATAC modality")
        from cytome.io.convert_fragments import import_fragments

        rows = self._conn.execute("SELECT cell_idx, barcode FROM cells").fetchall()
        mapping = {str(barcode): int(cell_idx) for cell_idx, barcode in rows if barcode is not None}
        import_fragments(self._conn, path, mapping, build_index=build_index)

    def export_coverage(
        self,
        groupby: str,
        output_dir: str | Path,
        format: str = "bigwig",
        normalize: str = "cpm",
        bin_size: int = 10,
        region: tuple[str, int, int] | None = None,
    ):
        """Export ATAC pseudo-bulk coverage by group."""
        if self._name != "ATAC":
            raise AttributeError("export_coverage is only available for ATAC modality")
        from cytome.io.export_bigwig import export_coverage

        return export_coverage(
            dataset=self._dataset,
            groupby=groupby,
            output_dir=output_dir,
            format=format,
            normalize=normalize,
            bin_size=bin_size,
            region=region,
        )


class _EmbeddingAccessor:
    def __init__(self, conn: sqlite3.Connection, entity: str = "cells") -> None:
        self._conn = conn
        self._entity = entity

    def __getitem__(self, name: str) -> np.ndarray:
        return EmbeddingArray(self._conn, name).to_memory()

    def keys(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT array_name FROM embedding_meta WHERE entity = ? ORDER BY array_name",
            (self._entity,),
        )
        return [r[0] for r in rows]

    def __contains__(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM embedding_meta WHERE array_name = ? AND entity = ? LIMIT 1",
            (name, self._entity),
        ).fetchone()
        return row is not None

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())


class _GraphAccessor:
    def __init__(self, conn: sqlite3.Connection, axis: str = "obs", entity: str = "cells") -> None:
        self._conn = conn
        self._axis = axis
        self._entity = entity

    def __getitem__(self, name: str) -> GraphStore:
        return GraphStore(self._conn, name, axis=self._axis, entity_table=self._entity)

    def keys(self) -> list[str]:
        names = set()
        if self._axis == "obs":
            if self._conn.execute("SELECT COUNT(*) FROM graph_knn").fetchone()[0] > 0:
                names.add("knn")
            if self._conn.execute("SELECT COUNT(*) FROM graph_peak_gene").fetchone()[0] > 0:
                names.add("peak_gene")
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT graph_name FROM graph_edges WHERE axis = ? ORDER BY graph_name",
                (self._axis,),
            )
            names.update(r[0] for r in rows)
        except sqlite3.OperationalError:
            pass
        return sorted(names)


class CytomeDataset:
    """SQLite-backed single-cell multi-omics dataset."""

    def __init__(self, path: str | Path, mode: str = "r", force: bool = False) -> None:
        self.path = Path(path)
        self.mode = mode
        self._pending_writes: dict[str, Any] = {}
        self._auto_flush = False
        # Cached connection-bound accessors. Any new attribute added here
        # that holds a reference to ``self._conn`` MUST also be reset in
        # ``_refresh_after_reopen`` — otherwise it will hold a closed
        # handle after operations like ``filter_cells`` reopen the
        # database in place.
        self._metadata_obj: MetadataStore | None = None
        if mode in {"w", "create"}:
            # force=False (default) raises FileExistsError if the path exists,
            # so an existing cytome is never silently truncated.
            self._conn = create_database(self.path, force=force)
        else:
            self._conn = open_database(self.path)
        self._manifest = self._read_manifest()

    @property
    def cells(self) -> EntityTable:
        return EntityTable(self._conn, "cells", enqueue_write=self._enqueue_write)

    @property
    def genes(self) -> EntityTable:
        return EntityTable(self._conn, "genes", enqueue_write=self._enqueue_write)

    @property
    def GA_genes(self) -> EntityTable:
        """Per-modality features table for inferred Gene Activity (GA).

        Kept separate from ``genes`` so an RNA-derived genes table and a
        GA-derived gene-activity table can coexist in the same cytome
        without colliding on ``col_entity='genes'`` validation.
        """
        return EntityTable(self._conn, "GA_genes", enqueue_write=self._enqueue_write)

    @property
    def peaks(self) -> EntityTable:
        return EntityTable(self._conn, "peaks", enqueue_write=self._enqueue_write)

    @property
    def samples(self) -> EntityTable:
        return EntityTable(self._conn, "samples", enqueue_write=self._enqueue_write)

    def features(self, modality: str) -> EntityTable:
        """Return the per-**modality** feature (var) table as an :class:`EntityTable`.

        Unlike :attr:`cells` — there is exactly one cells table, so it is a
        property — each modality has its **own** feature table, so this is a
        **method** that takes the modality and returns the matching table:

        =========  ====================  ==================================
        modality   feature table         id column (rows are features)
        =========  ====================  ==================================
        ``RNA``    ``genes``             ``gene_id``
        ``GA``     ``GA_genes``          ``gene_id``
        ``ATAC``   ``peaks``             ``peak_id``
        ``tiles``  ``tiles``             ``tile_id``
        =========  ====================  ==================================

        Use this instead of attribute access for the feature axis. In particular,
        ``ds.tiles`` and ``ds.GA`` do **not** return the feature table — they
        resolve through :meth:`__getattr__` to a :class:`Modality` (which exposes
        ``.counts`` / ``.layer()`` and has no ``__getitem__`` / ``to_pandas``),
        because the *modality* name collides with the *table* name (``tiles``) or
        differs from it (``GA`` vs ``GA_genes``). ``ds.features('tiles')`` always
        gives you the table, uniformly across modalities.

        Parameters
        ----------
        modality : str
            One of ``'RNA'``, ``'GA'``, ``'ATAC'``, ``'tiles'`` (case-insensitive
            for RNA/GA/ATAC). Unknown names raise ``ValueError``.

        Returns
        -------
        EntityTable
            The feature table. Supports ``["col"]`` column access,
            ``.to_pandas()``, ``.columns``, ``.n``, etc.

        Examples
        --------
        >>> ds.features('tiles')['tile_id']         # tile ids (np.ndarray)
        >>> ds.features('RNA').to_pandas().head()   # the genes table
        >>> ds.features('ATAC')['peak_id']          # peak ids

        See Also
        --------
        cells : the (single) cell table, accessed as a property.
        genes, peaks, GA_genes : per-table property shortcuts (kept for
            back-compat); ``features(modality)`` is the uniform accessor.
        """
        from cytome.utils.modality import modality_var_entity
        entity, _id_col = modality_var_entity(modality)
        return EntityTable(self._conn, entity, enqueue_write=self._enqueue_write)

    @property
    def embeddings(self) -> _EmbeddingAccessor:
        return _EmbeddingAccessor(self._conn, entity="cells")

    @property
    def var_embeddings(self) -> _EmbeddingAccessor:
        return _EmbeddingAccessor(self._conn, entity="genes")

    @property
    def spatial_images(self):
        """Stored tissue images + scale factors (see cytome.core.spatial)."""
        from .spatial import _SpatialImageAccessor
        return _SpatialImageAccessor(self._conn)

    def add_spatial_image(self, library_id: str, img_key: str, image,
                          scalefactors=None, replace: bool = False) -> None:
        """Store a registered tissue image (ndarray, exact) or an image FILE
        (png/jpeg/tiff path, bytes kept verbatim) with its scale factors.
        Scalefactors upsert per key. See ``cytome/core/spatial.py``."""
        from .spatial import add_spatial_image as _add
        _add(self._conn, library_id, img_key, image,
             scalefactors=scalefactors, replace=replace)

    def delete_spatial_image(self, library_id: str, img_key: str) -> None:
        from .spatial import delete_spatial_image as _del
        _del(self._conn, library_id, img_key)

    def set_spatial_coords(self, coords, cell_idx=None) -> None:
        """Index per-cell spatial coordinates (full-res pixel units) in the
        schema's ``spatial_coords`` table + R*-tree, enabling
        :meth:`cells_in_region`. The ``spatial`` embedding remains the array
        analysis/plotting consume; this is its queryable index."""
        # entity writes are buffered until flush(); the FK on cell_idx needs
        # the cells rows on disk first
        self.flush()
        from .spatial import set_spatial_coords as _set
        _set(self._conn, coords, cell_idx=cell_idx)

    def cells_in_region(self, x, y):
        """Cell indices whose spatial coordinates fall in the rectangle
        ``x=(x0,x1), y=(y0,y1)`` — indexed R*-tree lookup; pairs with
        ``spatial_images.crop(...)`` for image-plus-cells ROI work."""
        from .spatial import cells_in_region as _q
        return _q(self._conn, x, y)

    @property
    def graphs(self) -> _GraphAccessor:
        return _GraphAccessor(self._conn, axis="obs", entity="cells")

    @property
    def var_graphs(self) -> _GraphAccessor:
        return _GraphAccessor(self._conn, axis="var", entity="genes")

    @property
    def metadata(self) -> MetadataStore:
        """Access arbitrary metadata store."""
        if self._metadata_obj is None:
            self._metadata_obj = MetadataStore(self._conn, enqueue_write=self._enqueue_write)
        return self._metadata_obj

    @property
    def provenance(self) -> ProvenanceLog:
        return ProvenanceLog(self._conn)

    def set_categories(self, column, order=None, colors=None) -> None:
        """Define display *order* and/or *colors* for a categorical ``cells`` column.

        The mapping is stored in ``ds.metadata['categories']`` (a dict keyed by
        column name) and is honored by PIASO plotting — e.g.
        :func:`piaso.pl.embedding`, :func:`piaso.pl.dotplot` — which reads it to
        order the categories and color them consistently across figures. The
        underlying data column is **not** modified.

        Parameters
        ----------
        column : str
            Name of a column in ``ds.cells`` (e.g. ``"cell_type"``).
        order : sequence of str, optional
            The category levels in the desired display order. If omitted, any
            existing stored order is left unchanged.
        colors : sequence of str | dict, optional
            Hex colors (``"#4E79A7"``). As a **list** it is matched positionally
            to ``order`` (so ``order`` must be given, or already stored, and the
            lengths must match). As a **dict** it maps ``category -> hex`` directly
            (only the listed categories are colored). If omitted, any existing
            stored colors are left unchanged.

        Examples
        --------
        >>> ds.set_categories(
        ...     "cell_type",
        ...     order=["Astro", "Excit", "Inhib"],
        ...     colors=["#4E79A7", "#E69F00", "#009E73"],
        ... )
        >>> ds.set_categories("region", colors={"CTX": "#CC79A7"})
        """
        actual = self.cells.resolve_column(column, warn=True)
        if actual is None:
            raise KeyError(
                f"set_categories: '{column}' is not a column in ds.cells "
                f"(available: {list(self.cells.columns)}).")
        column = actual
        cats = dict(self.metadata.get("categories", {}) or {})
        entry = dict(cats.get(column, {}))
        if order is not None:
            entry["order"] = [str(o) for o in order]
        if colors is not None:
            if isinstance(colors, dict):
                merged = dict(entry.get("colors", {}))
                merged.update({str(k): str(v) for k, v in colors.items()})
                entry["colors"] = merged
            else:
                ref = entry.get("order")
                if ref is None:
                    raise ValueError(
                        "set_categories: `colors` as a list requires `order` "
                        "(pass order=, or set it in a prior call).")
                colors = list(colors)
                if len(colors) != len(ref):
                    raise ValueError(
                        f"set_categories: len(colors)={len(colors)} does not match "
                        f"len(order)={len(ref)}.")
                entry["colors"] = {str(c): str(h) for c, h in zip(ref, colors)}
        cats[column] = entry
        self.metadata["categories"] = cats
        self.flush()

        # Validate the order against the live column: warn (don't block —
        # forward-declaring extra categories is legitimate) when a value present
        # in the column has no order entry, which usually means a truncated /
        # stale label list (e.g. a fixed-width numpy '<U' array).
        if entry.get("order"):
            missing = self._categories_missing_values(column, entry["order"])
            if missing:
                import warnings as _w
                _w.warn(
                    f"set_categories: the order for {column!r} is missing "
                    f"{len(missing)} value(s) present in the column (e.g. "
                    f"{sorted(missing)[:3]}). Those cells fall back to default "
                    f"order/colors. If the names look truncated, rebuild the "
                    f"order from ds.cells[{column!r}] (object dtype).",
                    stacklevel=2,
                )

    def get_categories(self, column=None):
        """Return stored category order/colors set via :meth:`set_categories`.

        Parameters
        ----------
        column : str, optional
            If given, return the entry ``{"order": [...], "colors": {cat: hex}}``
            for that column (or ``None`` if none stored). If omitted, return the
            full ``{column: entry}`` dict (empty dict if nothing stored).
        """
        cats = self.metadata.get("categories", {}) or {}
        if column is None:
            return cats
        actual = self.cells.resolve_column(column)
        if actual is not None:
            column = actual
        entry = cats.get(column)
        # Self-heal on read: if the column exists but its current values are no
        # longer covered by the stored order (the column was overwritten outside
        # the tracked write paths, or this is an old cytome), drop the stale
        # entry + warn so callers fall back to a fresh default rather than
        # ordering by dead labels.
        if entry and entry.get("order"):
            missing = self._categories_missing_values(column, entry["order"])
            if missing:
                self._invalidate_stale_categories(column)
                return None
        return entry

    @property
    def modalities(self) -> list[str]:
        """Modalities present in this cytome.

        Derived from actual on-disk state (matrix_meta + fragment_chunks)
        PLUS the manifest's modalities list. The manifest is honored for
        backward-compat with explicit ``add_modality`` /
        ``register_modality`` calls, but a stale manifest no longer
        masks reality.

        Round 12 (2026-05-27): previously this returned only
        ``self._manifest['modalities']``, which silently mis-reported on
        cytomes built via the RNA-first + ATAC-append pipeline (the Rust
        importer in Mode A doesn't update the manifest). The stale value
        broke ``subset()``, ``merge()``, ``__getattr__``, etc. on every
        RNA+ATAC supervised + QC run since Round 6.
        """
        mods: set[str] = set(self._manifest.get("modalities", []) or [])

        # Matrix-name prefix scan: a matrix called "{X}_counts" etc.
        # implies modality X is present. Restricted to known names so
        # that incidental matrix names with underscores don't create
        # false positives.
        try:
            rows = self._conn.execute(
                "SELECT matrix_name FROM matrix_meta"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for (name,) in rows:
            if not name or "_" not in name:
                continue
            prefix = name.split("_", 1)[0]
            if prefix in _KNOWN_MODALITY_NAMES:
                mods.add(prefix)

        # fragment_chunks: any non-empty implies ATAC modality regardless
        # of whether peaks have been called or a matrix exists yet.
        try:
            row = self._conn.execute(
                "SELECT 1 FROM fragment_chunks LIMIT 1"
            ).fetchone()
            if row is not None:
                mods.add("ATAC")
        except sqlite3.OperationalError:
            pass

        return sorted(mods)

    @property
    def n_cells(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0])

    @property
    def n_genes(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM genes").fetchone()[0])

    @property
    def n_peaks(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0])

    def __getattr__(self, name: str) -> Any:
        if name in self.modalities:
            return Modality(self, name)
        raise AttributeError(name)

    def add_matrix(
        self,
        name: str,
        sparse_matrix: sp.spmatrix,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Buffer a sparse matrix write."""
        self._pending_writes[f"matrix:{name}"] = {
            "name": name,
            "matrix": sparse_matrix.tocsr(),
            "provenance": provenance,
        }
        if "_" in name:
            modality = name.split("_", 1)[0]
            if modality:
                modalities = set(self.modalities)
                modalities.add(modality)
                self._write_manifest_key("modalities", sorted(modalities))
                self._manifest = self._read_manifest()

    def create_layer_writer(
        self,
        layer_name: str,
        n_rows: int,
        n_cols: int,
        dtype: np.dtype | str = np.float64,
        compression: str = "zstd",
        row_entity: str = "cells",
        col_entity: str | None = None,
        overwrite: bool = True,
        storage_chunk_size: int = 128,
    ) -> ChunkedLayerWriter:
        """Return a :class:`ChunkedLayerWriter` for incremental sparse writes.

        Parameters
        ----------
        layer_name
            Matrix identifier (e.g. ``"RNA_infog"``).
        n_rows
            Total row count that will be written.
        n_cols
            Number of columns (genes / features).
        dtype
            Value dtype (default ``float64``).
        compression
            Compression method (default ``"zstd"``).
        row_entity
            Row entity table name.
        col_entity
            Column entity table name. If *None*, inferred from
            *layer_name*.
        overwrite
            Delete existing data for *layer_name* before writing.
        storage_chunk_size
            Rows per storage blob (default 128).
        """
        if col_entity is None:
            col_entity = _infer_col_entity(layer_name)

        # Register modality
        if "_" in layer_name:
            modality = layer_name.split("_", 1)[0]
            if modality:
                modalities = set(self.modalities)
                modalities.add(modality)
                self._write_manifest_key("modalities", sorted(modalities))
                self._manifest = self._read_manifest()

        return ChunkedLayerWriter(
            conn=self._conn,
            matrix_name=layer_name,
            n_rows=n_rows,
            n_cols=n_cols,
            dtype=dtype,
            compression=compression,
            row_entity=row_entity,
            col_entity=col_entity,
            overwrite=overwrite,
            storage_chunk_size=storage_chunk_size,
        )

    def add_embedding(
        self,
        name: str,
        ndarray: np.ndarray,
        provenance: dict[str, Any] | None = None,
        flush: bool = True,
        dtype: "np.dtype | str | None" = None,
    ) -> None:
        """Add a dense cell embedding (e.g. ``X_umap``, ``X_gdr``).

        Persisted to disk immediately by default (``flush=True``) — the natural
        expectation. Pass ``flush=False`` to batch several writes and call
        :meth:`flush` once at the end (cheaper when adding many at once).
        """
        arr = np.asarray(ndarray)
        # An embedding written here was float64 while the same embedding
        # converted from an h5ad was float32, in the same file. `dtype` lets a
        # caller state the width; None keeps whatever the array already has,
        # so a conversion stays lossless.
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)
        self._pending_writes[f"embedding:{name}"] = {
            "name": name,
            "array": arr,
            "entity": "cells",
            "provenance": provenance,
        }
        if flush:
            self.flush()

    def add_var_embedding(
        self,
        name: str,
        ndarray: np.ndarray,
        entity: str = "genes",
        provenance: dict[str, Any] | None = None,
        flush: bool = True,
    ) -> None:
        """Add a dense embedding aligned to the variable axis (flushes by default)."""
        self._pending_writes[f"embedding:{name}"] = {
            "name": name,
            "array": np.asarray(ndarray),
            "entity": entity,
            "provenance": provenance,
        }
        if flush:
            self.flush()

    def add_entity_column(
        self, table_name: str, name: str, values, flush: bool = True,
    ) -> None:
        """Add (or overwrite) a single column on an entity table.

        Convenience over the read-modify-``set_entity`` dance::

            # before:
            cells = ds.cells.to_pandas(); cells["leiden"] = labels
            ds.set_entity("cells", cells); ds.flush()
            # after:
            ds.add_cells_column("leiden", labels)

        Validates ``len(values) == n_rows`` and flushes by default
        (``flush=False`` to batch). Reads + rewrites the (small) entity table;
        the heavy measurement matrices are untouched.
        """
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self._conn)
        values = np.asarray(values)
        if len(values) != len(df):
            raise ValueError(
                f"len(values)={len(values)} does not match {table_name} "
                f"row count ({len(df)})."
            )
        df[name] = values
        self.set_entity(table_name, df)
        if flush:
            self.flush()

    def add_cells_column(self, name: str, values, flush: bool = True) -> None:
        """Add/overwrite a column on the ``cells`` table (e.g. cluster labels)."""
        self.add_entity_column("cells", name, values, flush=flush)

    def add_genes_column(self, name: str, values, flush: bool = True) -> None:
        """Add/overwrite a column on the ``genes`` table (e.g. backfill ``symbol``)."""
        self.add_entity_column("genes", name, values, flush=flush)

    def add_graph(
        self,
        name: str,
        sparse_matrix: sp.spmatrix,
        axis: str = "obs",
        entity_table: str = "cells",
    ) -> None:
        """Buffer a graph write."""
        self._pending_writes[f"graph:{name}"] = {
            "name": name,
            "matrix": sparse_matrix,
            "axis": axis,
            "entity_table": entity_table,
        }

    def add_var_graph(self, name: str, sparse_matrix: sp.spmatrix, entity_table: str = "genes") -> None:
        """Buffer a variable-axis sparse graph write."""
        self.add_graph(name, sparse_matrix, axis="var", entity_table=entity_table)

    def add_modality(self, name: str, anndata_or_dict: Any) -> None:
        """Add modality from AnnData-like object or dict payload."""
        if hasattr(anndata_or_dict, "X"):
            self.add_matrix(f"{name}_counts", anndata_or_dict.X)
            for layer_name, mat in anndata_or_dict.layers.items():
                self.add_matrix(f"{name}_{layer_name}", mat)
        elif isinstance(anndata_or_dict, dict):
            for key, mat in anndata_or_dict.items():
                layer = "counts" if key == "X" else key
                self.add_matrix(f"{name}_{layer}", mat)
        else:
            raise TypeError("anndata_or_dict must be AnnData-like or dict")
        modalities = set(self.modalities)
        modalities.add(name)
        self._write_manifest_key("modalities", sorted(modalities))

    def set_entity(self, table_name: str, data_dict_or_df: dict[str, Any] | pd.DataFrame) -> None:
        """Buffer full entity table replacement."""
        if isinstance(data_dict_or_df, pd.DataFrame):
            payload = data_dict_or_df.copy()
        else:
            payload = pd.DataFrame(data_dict_or_df)
        # Replacing the cells table can overwrite a labelled column — drop any
        # category order that the new values no longer match (covers
        # add_cells_column, which routes through here).
        if table_name == "cells":
            cats = self.metadata.get("categories") or {}
            for col in list(cats):
                if col in payload.columns:
                    self._invalidate_stale_categories(col, payload[col].to_numpy())
        self._pending_writes[f"entity_table:{table_name}"] = payload

    def flush(self) -> None:
        """Atomically persist all buffered writes."""
        if not self._pending_writes:
            return
        with self._conn:
            for key, payload in list(self._pending_writes.items()):
                if key.startswith("entity:"):
                    assert isinstance(payload, _EntityWrite)
                    EntityTable(self._conn, payload.table_name)._apply_column_write(
                        payload.column_name, payload.values
                    )
                elif key.startswith("entity_table:"):
                    table = key.split(":", 1)[1]
                    self._write_entity_table(table, payload)
                elif key.startswith("matrix:"):
                    self._write_matrix_payload(payload)
                elif key.startswith("embedding:"):
                    self._write_embedding_payload(payload)
                elif key.startswith("graph:"):
                    GraphStore(
                        self._conn,
                        payload["name"],
                        axis=payload.get("axis", "obs"),
                        entity_table=payload.get("entity_table", "cells"),
                    ).write_sparse(payload["matrix"])
                elif key.startswith("metadata:"):
                    assert isinstance(payload, _MetadataWrite)
                    MetadataStore(self._conn)._apply_write(payload)
                else:
                    logger.warning("Unknown pending write key ignored: %s", key)
            self._pending_writes.clear()
            self._sync_manifest_counts()
            self._manifest = self._read_manifest()

    def close(self) -> None:
        """Flush pending writes then close the database."""
        self.flush()
        close_database(self._conn)

    def reopen(self) -> "CytomeDataset":
        """Flush + close + reopen the underlying connection in place.

        Releases the SQLite file (connection, locks, WAL, mmap) and opens a
        fresh handle on the same ``path``, refreshing all connection-bound
        caches. The ``Dataset`` object stays valid and now sees any changes
        another exclusive writer made to the file while it was released.

        The intended use is bracketing an out-of-process / different-SQLite
        writer that needs exclusive access — e.g. the Rust quantify backend:
        ``ds.close()`` → run the exclusive writer on ``ds.path`` →
        ``ds.reopen()``. Tolerant of an already-closed connection (skips the
        flush/close in that case). Holding this Dataset open across such a
        writer can corrupt the file (two SQLite libraries on one WAL DB);
        reopening around it is the safe pattern. Returns ``self`` for chaining.
        """
        if not self.is_closed:
            self.flush()
            close_database(self._conn)
        self._conn = open_database(self.path)
        self._refresh_after_reopen()
        self._manifest = self._read_manifest()
        return self

    @property
    def is_closed(self) -> bool:
        """Whether the underlying SQLite connection has been closed.

        Probes with a no-op ``SELECT 1``. Re-opening a closed Dataset is
        not done automatically — callers should re-open via
        ``cytome.open(ds.path)`` (or pass a path string directly to the
        function that needs it).
        """
        try:
            self._conn.execute("SELECT 1")
            return False
        except sqlite3.ProgrammingError:
            return True

    def _check_open(self) -> None:
        """Raise an actionable error if the Dataset has been closed.

        Library functions that accept a cytome Dataset should call this
        at entry. The error message points the user at the two safe
        ways to fix the problem (re-open the Dataset, or pass a path
        string instead).
        """
        if self.is_closed:
            raise RuntimeError(
                f"cytome Dataset is closed (path: {self.path}). "
                f"Two ways to fix:\n"
                f"  1. Re-open the dataset and retry: "
                f"ds = cytome.open(r'{self.path}')\n"
                f"  2. Pass the path string directly to the function "
                f"instead of the closed Dataset object."
            )

    def validate(self):
        """Run dataset validation checks."""
        from cytome.utils.validation import validate

        return validate(self)

    def repair(self) -> None:
        """Attempt repair of recoverable integrity issues."""
        from cytome.utils.validation import repair

        repair(self)

    def iter_chunks(
        self,
        modality: str = "RNA",
        layer: str = "counts",
        cell_mask: np.ndarray | None = None,
        col_mask: np.ndarray | None = None,
        batch_size: int | None = None,
    ):
        """Iterate over stored matrix chunks without full materialization.

        Yields one CSR chunk at a time from disk. Peak RAM stays at
        O(chunk_size × n_genes) regardless of dataset size.

        Parameters
        ----------
        modality
            Modality name (e.g. ``"RNA"``).
        layer
            Layer name within the modality (default ``"counts"``).
        cell_mask
            Optional boolean mask (length ``n_cells``) or sorted integer
            indices to restrict iteration to specific cells. Chunks with
            no matching cells are skipped entirely.
        col_mask
            Optional boolean mask or integer indices for column subsetting.
            Each yielded chunk is column-sliced after loading. Does not
            reduce disk I/O but reduces downstream memory.
        batch_size
            If provided, vstack multiple on-disk chunks to yield larger
            batches. Controls the trade-off between RAM usage and compute
            efficiency. ``None`` yields raw on-disk chunks (typically 16
            rows each). Suggested values: 512, 1024, 2048, 4096 rows.

        Yields
        ------
        chunk_csr : scipy.sparse.csr_matrix
            Expression data for this chunk, shape ``(chunk_rows, n_genes)``.
        row_indices : np.ndarray
            Global cell indices corresponding to the rows of *chunk_csr*.
        """
        matrix_name = f"{modality}_{layer}"
        ml = MeasurementLayer(self._conn, matrix_name)

        if cell_mask is not None:
            cell_mask = np.asarray(cell_mask)
            if cell_mask.dtype == bool:
                keep_idx = np.where(cell_mask)[0]
            else:
                keep_idx = np.sort(cell_mask)
        else:
            keep_idx = None

        if col_mask is not None:
            col_mask = np.asarray(col_mask)

        # Get entity count to cap yielded indices (prevents IndexError on
        # mismatched matrix_meta.n_rows vs entity table after subset/delete).
        row_entity = self._conn.execute(
            "SELECT row_entity FROM matrix_meta WHERE matrix_name = ?",
            (matrix_name,),
        ).fetchone()
        entity_count = None
        if row_entity is not None:
            entity_name = row_entity[0]
            if entity_name and entity_name in ("cells", "genes", "peaks", "proteins"):
                entity_count = int(
                    self._conn.execute(f"SELECT COUNT(*) FROM {entity_name}").fetchone()[0]
                )

        raw_iter = self._iter_raw_chunks(ml, keep_idx, entity_count)

        if batch_size is None:
            for chunk_csr, row_indices in raw_iter:
                if col_mask is not None:
                    chunk_csr = chunk_csr[:, col_mask]
                yield chunk_csr, row_indices
        else:
            buffer_chunks: list[sp.spmatrix] = []
            buffer_indices: list[np.ndarray] = []
            buffer_rows = 0

            for chunk_csr, row_indices in raw_iter:
                buffer_chunks.append(chunk_csr)
                buffer_indices.append(row_indices)
                buffer_rows += chunk_csr.shape[0]

                if buffer_rows >= batch_size:
                    merged = sp.vstack(buffer_chunks, format="csr")
                    if col_mask is not None:
                        merged = merged[:, col_mask]
                    yield (merged, np.concatenate(buffer_indices))
                    buffer_chunks = []
                    buffer_indices = []
                    buffer_rows = 0

            if buffer_chunks:
                merged = sp.vstack(buffer_chunks, format="csr")
                if col_mask is not None:
                    merged = merged[:, col_mask]
                yield (merged, np.concatenate(buffer_indices))

    @staticmethod
    def _iter_raw_chunks(ml: MeasurementLayer, keep_idx: np.ndarray | None,
                         entity_count: int | None = None):
        """Yield raw on-disk chunks, optionally filtered by keep_idx.

        Parameters
        ----------
        entity_count
            If provided, cap yielded row indices at this value. Chunks
            referencing rows beyond ``entity_count`` are truncated or
            skipped entirely.  This prevents IndexError when
            ``matrix_meta.n_rows`` exceeds the entity table count.
        """
        # Pass the wanted rows down so chunks holding none of them are never
        # fetched. Filtering after iter_rows() meant every chunk was
        # decompressed and then thrown away.
        for row_start, row_end, chunk_csr in ml.iter_rows(row_filter=keep_idx):
            # Cap at entity count if provided
            if entity_count is not None:
                if row_start >= entity_count:
                    continue  # entirely beyond valid range
                if row_end > entity_count:
                    local_end = entity_count - row_start
                    chunk_csr = chunk_csr[:local_end]
                    row_end = entity_count

            if keep_idx is not None:
                lo = int(np.searchsorted(keep_idx, row_start, side="left"))
                hi = int(np.searchsorted(keep_idx, row_end, side="left"))
                if lo >= hi:
                    continue
                local_rows = keep_idx[lo:hi]
                local_idx = (local_rows - row_start).astype(np.intp)
                yield chunk_csr[local_idx], local_rows
            else:
                row_indices = np.arange(row_start, row_end, dtype=np.intp)
                yield chunk_csr, row_indices

    def to_anndata(self, modality: str = "RNA", layer: str | None = None, include_embeddings: bool = True,
                   include_cross_modality_embeddings: bool = True, cell_mask=None):
        """Convert one modality to AnnData.

        Parameters
        ----------
        include_cross_modality_embeddings
            If True (default), also attach cell embeddings that belong to OTHER
            modalities (they share the same cells), named by their cytome key.
        cell_mask
            Optional boolean mask or sorted integer indices for
            chunk-selective cell subsetting.
        """
        from cytome.io.convert_anndata import to_anndata

        return to_anndata(self, modality=modality, layer=layer, include_embeddings=include_embeddings,
                          include_cross_modality_embeddings=include_cross_modality_embeddings, cell_mask=cell_mask)

    def to_mudata(self):
        """Convert dataset to MuData."""
        from cytome.io.convert_mudata import to_mudata

        return to_mudata(self)

    @staticmethod
    def merge(inputs, output, **kwargs):
        """Merge multiple datasets to one output file."""
        from cytome.io.merge import merge

        return merge(inputs=inputs, output=output, **kwargs)

    def subset(self, mask, output=None, **kwargs):
        """Subset cells and write to a new dataset."""
        from cytome.io.subset import subset

        return subset(self, mask=mask, output=output, **kwargs)

    def matrix_meta(self, name: str) -> dict[str, Any] | None:
        """Return matrix metadata as a dict, or *None* if not found.

        Wraps the ``SELECT ... FROM matrix_meta WHERE matrix_name = ?`` pattern.

        Parameters
        ----------
        name
            Matrix name (e.g. ``"RNA_counts"``, ``"ATAC_counts"``).

        Returns
        -------
        dict or None
            Keys: ``matrix_name``, ``n_rows``, ``n_cols``, ``dtype``,
            ``compression``, ``row_entity``, ``col_entity``.
        """
        cols = [
            r[1]
            for r in self._conn.execute("PRAGMA table_info(matrix_meta)").fetchall()
        ]
        col_list = ", ".join(cols)
        row = self._conn.execute(
            f"SELECT {col_list} FROM matrix_meta WHERE matrix_name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return dict(zip(cols, row))

    def list_matrices(self) -> list[str]:
        """Return all matrix names registered in the dataset."""
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT matrix_name FROM matrix_meta ORDER BY rowid"
            ).fetchall()
        ]

    def list_embeddings(self, pattern: str | None = None) -> list[str]:
        """Return embedding array names, optionally filtered by SQL LIKE pattern.

        Parameters
        ----------
        pattern
            SQL LIKE pattern (e.g. ``"%svd%"``, ``"%umap%"``). If None,
            returns all embeddings.

        Returns
        -------
        list[str]
            Embedding array names in registration order.
        """
        if pattern is None:
            sql = "SELECT array_name FROM embedding_meta ORDER BY rowid"
            params: tuple[Any, ...] = ()
        else:
            sql = (
                "SELECT array_name FROM embedding_meta "
                "WHERE array_name LIKE ? ORDER BY rowid"
            )
            params = (pattern,)
        return [r[0] for r in self._conn.execute(sql, params).fetchall()]

    def delete_embedding(self, name: str) -> None:
        """Remove an embedding (data + metadata) from the dataset.

        Parameters
        ----------
        name
            Embedding array name to remove.
        """
        self.flush()
        self._conn.execute("DELETE FROM dense_chunks WHERE array_name = ?", (name,))
        self._conn.execute("DELETE FROM embedding_meta WHERE array_name = ?", (name,))
        self._conn.commit()

    def delete_matrix(self, name_or_pattern: str, like: bool = False) -> list[str]:
        """Remove one or more matrices (data + metadata) from the dataset.

        Parameters
        ----------
        name_or_pattern
            Exact matrix name, or a SQL LIKE pattern when ``like=True``.
        like
            If True, treat ``name_or_pattern`` as a SQL LIKE pattern
            (e.g. ``"ATAC_%"`` to delete all ATAC matrices).

        Returns
        -------
        list[str]
            Names of matrices that were deleted.
        """
        self.flush()
        if like:
            names = [
                r[0]
                for r in self._conn.execute(
                    "SELECT matrix_name FROM matrix_meta WHERE matrix_name LIKE ?",
                    (name_or_pattern,),
                ).fetchall()
            ]
        else:
            names = (
                [name_or_pattern]
                if self.matrix_meta(name_or_pattern) is not None
                else []
            )
        for name in names:
            self._conn.execute(
                "DELETE FROM matrix_chunks WHERE matrix_name = ?", (name,)
            )
            self._conn.execute(
                "DELETE FROM matrix_meta WHERE matrix_name = ?", (name,)
            )
        self._conn.commit()
        return names

    def filter_cells(
        self,
        mask,
        include_fragments: bool = True,
        include_embeddings: bool = True,
        copy_annotations: bool = True,
    ) -> int:
        """Filter cells in place: keep cells where ``mask`` is True.

        Implemented atomically by writing a subset to a sibling temp file
        then replacing the original. The dataset is reopened in place
        and ``n_cells`` reflects the new count on return.

        Parameters
        ----------
        mask
            Boolean array of length ``n_cells`` (or integer indices).
            True keeps the cell.
        include_fragments
            Carry fragment data over (ATAC datasets). Default True.
        include_embeddings
            Carry embeddings over. Default True.
        copy_annotations
            Carry cell-independent annotations over: the imported GTF gene models
            (``_gene_annotation`` / ``_exon_annotation``), the ``GA_genes`` var
            table, and var (gene-axis) embeddings. Default True. Set False to
            drop them (e.g. to shrink the output). Spatial coordinates are
            cell-indexed and always carried over (subset + remapped). Cell×cell
            graphs are intentionally dropped (with a warning) — incomplete once
            cells are removed.

        Returns
        -------
        int
            Number of cells remaining after filtering.

        Notes
        -----
        For very large datasets, prefer ``subset(mask, output=...)`` which
        explicitly writes to a chosen path without touching the original.
        """
        from cytome.io.subset import subset as _subset
        from cytome.io.subset import _resolve_keep_indices

        n_before = self.n_cells

        # Short-circuit: if the mask keeps EVERY cell, filtering is a no-op —
        # skip the expensive subset + atomic replace entirely. The streaming
        # rebuild still walks every matrix + fragment chunk, so on a large
        # cytome an all-pass filter wastes hours rewriting an unchanged file
        # (ADVIS QC: FRiP dropped 0 of 199,628 cells → ~2.3 h rewriting 71 GB).
        keep_idx = _resolve_keep_indices(n_before, mask)
        if keep_idx.size == n_before:
            return int(n_before)

        tmp_path = Path(str(self.path) + ".filter_tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        # Flush pending writes so they make it into the subset
        self.flush()
        out = _subset(
            self,
            mask=mask,
            output=tmp_path,
            include_fragments=include_fragments,
            include_embeddings=include_embeddings,
            copy_annotations=copy_annotations,
        )
        n_after = out.n_cells
        out.close()

        # Atomically replace original.
        # `tmp_path` is a complete, self-contained subset db. The original is
        # being DISCARDED, so its WAL/SHM sidecars are dead weight. We must NOT
        # fold (checkpoint) the original's WAL: it can be large and reader-pinned,
        # the fold may fail, and a surviving `-wal`/`-shm` left next to the
        # replacement file is a DIFFERENT db generation → SQLite replays it on the
        # next open → "database disk image is malformed" (and a stale pre-filter
        # read first). So: close the handle, UNLINK the dead sidecars (instant,
        # never fails on a busy/huge WAL), then replace.
        original = Path(self.path)
        close_database(self._conn)
        for ext in ("-wal", "-shm"):
            Path(str(original) + ext).unlink(missing_ok=True)
        try:
            tmp_path.replace(original)
        except Exception:
            # Restore connection on failure
            self._conn = open_database(self.path)
            self._refresh_after_reopen()
            raise
        # Any sidecars the temp left behind are stale relative to the moved file —
        # drop them too (the data is already in the main file).
        for ext in ("-wal", "-shm"):
            Path(str(tmp_path) + ext).unlink(missing_ok=True)

        # Reopen the underlying connection on the same path
        self._conn = open_database(self.path)
        self._refresh_after_reopen()
        del n_before
        return int(n_after)

    def compact(self) -> dict:
        """Fold the write-ahead log into the main file and shrink the sidecars.

        After heavy writes the WAL (``<path>-wal``) can grow large (it isn't
        truncated while a reader pins an old snapshot), leaving the ``.cytome``
        as three files — awkward to copy or share. ``compact()`` runs a
        ``wal_checkpoint(TRUNCATE)``: it folds all committed WAL frames into the
        main db file and zeroes the ``-wal``, so the cytome is effectively a
        single self-contained file again.

        Cheap and safe: SQLite **streams** pages during the checkpoint (no extra
        RAM, no data copy in Python), and there is **no per-operation cost** to
        normal reads/writes — this is an explicit, on-demand call. On a busy file
        the checkpoint folds as much as it can without blocking other readers;
        whatever it can't fold stays in the WAL (no error, just not fully
        compacted).

        Returns
        -------
        dict
            ``{'busy': int, 'log_frames': int, 'checkpointed': int}`` from the
            ``wal_checkpoint`` result (``busy=1`` means a reader prevented a full
            truncate). Returns ``{'busy': -1, ...}`` if the pragma is unavailable.
        """
        self.flush()
        try:
            row = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            self._conn.commit()
            busy, log_frames, checkpointed = (row or (0, 0, 0))
            return {"busy": int(busy), "log_frames": int(log_frames),
                    "checkpointed": int(checkpointed)}
        except Exception:
            return {"busy": -1, "log_frames": 0, "checkpointed": 0}

    def query_gene_annotation(
        self,
        chrom=None,
        start=None,
        end=None,
        gene_names=None,
    ):
        """Return overlapping genes from the imported GTF annotation.

        Parameters
        ----------
        chrom : str, optional
            Filter to a single chromosome.
        start, end : int, optional
            Filter to genes overlapping ``[start, end)``. Both
            required together (or both None for no positional filter).
            Coordinates are 0-based half-open, matching cytome's
            internal convention.
        gene_names : list[str], optional
            Filter to a specific list of gene names.

        Returns
        -------
        pd.DataFrame
            Columns: ``[gene_id, chrom, start, end, strand,
            gene_name, gene_type, source]``. Empty DataFrame if no
            matches.
        """
        import pandas as pd

        clauses = []
        params = []
        if chrom is not None:
            clauses.append("chrom = ?")
            params.append(chrom)
        if start is not None and end is not None:
            # Overlap test: gene's [start, end) overlaps [start, end)
            clauses.append("start < ? AND end > ?")
            params.extend([end, start])
        if gene_names is not None:
            if len(gene_names) == 0:
                # Empty filter returns empty DataFrame
                return pd.DataFrame(columns=[
                    "gene_id", "chrom", "start", "end", "strand",
                    "gene_name", "gene_type", "source",
                ])
            placeholders = ",".join("?" * len(gene_names))
            clauses.append(f"gene_name IN ({placeholders})")
            params.extend(list(gene_names))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT gene_id, chrom, start, end, strand, gene_name, "
            f"gene_type, source FROM _gene_annotation{where} "
            f"ORDER BY chrom, start",
            params,
        ).fetchall()
        return pd.DataFrame(rows, columns=[
            "gene_id", "chrom", "start", "end", "strand",
            "gene_name", "gene_type", "source",
        ])

    def query_exon_annotation(
        self,
        chrom=None,
        start=None,
        end=None,
        gene_ids=None,
        features=None,
    ):
        """Return overlapping non-gene features (exons by default).

        Parameters
        ----------
        chrom : str, optional
            Filter to a single chromosome.
        start, end : int, optional
            Filter to features overlapping ``[start, end)``.
        gene_ids : list[str], optional
            Filter to features belonging to specific genes
            (by ``gene_id``).
        features : list[str], optional
            Filter to specific feature types (e.g.
            ``['exon', 'CDS']``). Defaults to all imported types.

        Returns
        -------
        pd.DataFrame
            Columns: ``[gene_id, transcript_id, exon_number,
            chrom, start, end, strand, feature]``.
        """
        import pandas as pd

        clauses = []
        params = []
        if chrom is not None:
            clauses.append("chrom = ?")
            params.append(chrom)
        if start is not None and end is not None:
            clauses.append("start < ? AND end > ?")
            params.extend([end, start])
        if gene_ids is not None:
            if len(gene_ids) == 0:
                return pd.DataFrame(columns=[
                    "gene_id", "transcript_id", "exon_number",
                    "chrom", "start", "end", "strand", "feature",
                ])
            placeholders = ",".join("?" * len(gene_ids))
            clauses.append(f"gene_id IN ({placeholders})")
            params.extend(list(gene_ids))
        if features is not None:
            placeholders = ",".join("?" * len(features))
            clauses.append(f"feature IN ({placeholders})")
            params.extend(list(features))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT gene_id, transcript_id, exon_number, chrom, "
            f"start, end, strand, feature, transcript_type "
            f"FROM _exon_annotation{where} "
            f"ORDER BY chrom, start",
            params,
        ).fetchall()
        return pd.DataFrame(rows, columns=[
            "gene_id", "transcript_id", "exon_number",
            "chrom", "start", "end", "strand", "feature",
            "transcript_type",
        ])

    def gene_annotation_info(self):
        """Return a summary of the imported gene annotation.

        Returns
        -------
        dict or None
            ``{'source': str, 'n_genes': int, 'n_exons': int}`` if
            annotation has been imported via
            :func:`cytome.import_gtf`. ``None`` if the annotation
            tables are empty.
        """
        n_genes = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM _gene_annotation"
            ).fetchone()[0]
        )
        if n_genes == 0:
            return None
        n_exons = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM _exon_annotation"
            ).fetchone()[0]
        )
        sources = [
            r[0] for r in self._conn.execute(
                "SELECT DISTINCT source FROM _gene_annotation"
            ).fetchall() if r[0]
        ]
        return {
            "source": sources[0] if len(sources) == 1
                else sources or None,
            "n_genes": n_genes,
            "n_exons": n_exons,
        }

    def downsample(self, n_cells=None, fraction=None, **kwargs):
        """Downsample cells with optional stratification."""
        from cytome.io.subset import downsample

        return downsample(self, n_cells=n_cells, fraction=fraction, **kwargs)

    def checkpoint(self, mode: str = "TRUNCATE") -> None:
        """Fold the write-ahead log back into the ``.cytome`` file.

        A cytome is three files: ``x.cytome``, ``x.cytome-wal`` and
        ``x.cytome-shm``. ``flush()`` COMMITS, which in WAL mode writes to the
        ``-wal``, not to the main database. So anything committed but not yet
        checkpointed lives only in the sidecar, and a plain file copy of the
        ``.cytome`` alone silently leaves it behind.

        Call this before copying the file with an external tool (``cp``,
        ``rsync``, an upload), or use :meth:`copy` / :meth:`backup`, which no
        longer need it.
        """
        self.flush()
        self._conn.commit()
        self._conn.execute(f"PRAGMA wal_checkpoint({mode})")

    def _snapshot_to(self, out: "Path") -> None:
        """Write a consistent copy of this database to ``out``.

        Uses SQLite's online backup API rather than copying the file. Copying
        the ``.cytome`` alone loses every committed-but-not-checkpointed page,
        with no error: measured on a live dataset, ``copy()`` dropped an
        embedding and ``backup()`` produced a file with no ``embedding_meta``
        table at all. The backup API reads through the WAL and is safe on a
        database that is still open.
        """
        import sqlite3
        self.flush()
        self._conn.commit()
        dest = sqlite3.connect(str(out))
        try:
            self._conn.backup(dest)
            dest.commit()
        finally:
            dest.close()

    def copy(self, output, force: bool = False):
        """Copy the full dataset to ``output`` and return a handle to the copy.

        Raises ``FileExistsError`` if ``output`` already exists unless
        ``force=True``. The live dataset (``self``) keeps pointing at the
        original path.
        """
        out = Path(output)
        if out.exists() and not force:
            raise FileExistsError(
                f"{out} already exists; pass force=True to overwrite.")
        self._snapshot_to(out)
        return CytomeDataset(out, mode="r")

    def backup(self, output, force: bool = False):
        """Snapshot this cytome to ``output`` (e.g. before a destructive
        :meth:`filter_cells`). Unlike :meth:`copy`, the live ``self`` stays open
        on the **original** and the backup path is returned (not a handle), so the
        intent — "save a safety copy, keep working on the original" — is explicit.

        Raises ``FileExistsError`` if ``output`` exists unless ``force=True``.
        """
        out = Path(output)
        if out.exists() and not force:
            raise FileExistsError(
                f"{out} already exists; pass force=True to overwrite.")
        self._snapshot_to(out)
        return out

    def to_pytorch(
        self,
        modalities=None,
        layers=None,
        obs_columns=None,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ):
        """Create PyTorch DataLoader backed by this dataset."""
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyTorch is required for to_pytorch(). Install torch.") from exc
        from cytome.dataloader.pytorch import CytomeTorchDataset

        modalities = modalities or self.modalities
        layers = layers or {m: "counts" for m in modalities}
        dataset = CytomeTorchDataset(
            path=str(self.path),
            n_cells=self.n_cells,
            modalities=modalities,
            layers=layers,
            obs_columns=obs_columns,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    def to_info_dict(self) -> dict[str, Any]:
        """Return dataset summary as dictionary."""
        matrices = self._conn.execute(
            "SELECT matrix_name, n_rows, n_cols, dtype, chunk_size, has_csc FROM matrix_meta ORDER BY matrix_name"
        ).fetchall()
        embeddings = self._conn.execute(
            "SELECT array_name, n_rows, n_cols, dtype FROM embedding_meta ORDER BY array_name"
        ).fetchall()
        n_frag = int(self._conn.execute("SELECT COALESCE(SUM(n_fragments), 0) FROM fragment_meta").fetchone()[0])
        return {
            "path": str(self.path),
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_peaks": self.n_peaks,
            "modalities": self.modalities,
            "metadata_keys": self.metadata.keys(),
            "matrices": [
                {
                    "name": m[0],
                    "shape": [int(m[1]), int(m[2])],
                    "dtype": m[3],
                    "chunk_size": int(m[4]),
                    "has_csc": bool(m[5]),
                }
                for m in matrices
            ],
            "embeddings": [
                {"name": e[0], "shape": [int(e[1]), int(e[2])], "dtype": e[3]} for e in embeddings
            ],
            "fragments_total": n_frag,
            "manifest": self._manifest,
        }

    def __enter__(self) -> "CytomeDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        mods = ", ".join(self.modalities) if self.modalities else "none"
        lines = [
            f"CytomeDataset: {self.path}",
            f"{self.n_cells:,} cells × {len(self.modalities)} modalities [{mods}]",
        ]

        # Cell columns (truncate to first 8)
        try:
            cols = self.cells.columns
            skip = {"cell_idx", "barcode", "sample_id"}
            shown = [c for c in cols if c not in skip]
            if len(shown) > 8:
                lines.append(f"  cells: barcode + {len(shown)} columns [{', '.join(shown[:8])}, ...]")
            elif shown:
                lines.append(f"  cells: barcode + {len(shown)} columns [{', '.join(shown)}]")
        except Exception:
            pass

        # Matrices
        try:
            mat_rows = self._conn.execute(
                "SELECT matrix_name, n_rows, n_cols FROM matrix_meta"
            ).fetchall()
            if mat_rows:
                parts = [f"{n}({r:,}×{c:,})" for n, r, c in mat_rows]
                lines.append(f"  matrices: {', '.join(parts)}")
        except Exception:
            pass

        # Embeddings
        try:
            emb_keys = self.embeddings.keys()
            if emb_keys:
                # Get dimensions for first few
                emb_rows = self._conn.execute(
                    "SELECT array_name, n_cols FROM embedding_meta"
                ).fetchall()
                emb_map = {n: c for n, c in emb_rows}
                parts = [f"{k}({emb_map.get(k, '?')}d)" for k in emb_keys[:6]]
                suffix = f", ... +{len(emb_keys)-6} more" if len(emb_keys) > 6 else ""
                lines.append(f"  embeddings: {', '.join(parts)}{suffix}")
        except Exception:
            pass

        # Graphs
        try:
            g_keys = self.graphs.keys()
            if g_keys:
                lines.append(f"  graphs: {', '.join(g_keys)}")
        except Exception:
            pass

        # ATAC fragments
        if "ATAC" in self.modalities:
            try:
                n_frag = int(
                    self._conn.execute(
                        "SELECT COALESCE(SUM(n_fragments), 0) FROM fragment_meta"
                    ).fetchone()[0]
                )
                lines.append(f"  ATAC fragments: {n_frag:,}")
            except Exception:
                pass

        return "\n".join(lines)

    def _enqueue_write(self, key: str, payload: Any) -> None:
        # A full-column overwrite drops that column's stored categories
        # unconditionally, with a message. The subset-is-valid tolerance the
        # stale check gives is right for cell filtering and whole-table
        # replacement, where the labels keep their meaning — but a full
        # rewrite is a different event: after a re-run at a new resolution,
        # cluster "3" is a different set of cells, and keeping its old colour
        # is exactly the "categories not updated" bug. The stale-only check
        # also never fired when the new labels were a subset of the old order,
        # so the old colours silently mapped to different cells.
        if key.startswith("entity:cells:") and isinstance(payload, _EntityWrite):
            self._drop_categories_on_overwrite(payload.column_name)
        self._pending_writes[key] = payload
        if self._auto_flush:
            self.flush()

    def _categories_missing_values(self, column, order, values=None) -> set:
        """Live distinct values of ``column`` that are absent from ``order``.

        A stored ``order`` is allowed to be a *superset* of the present values
        (e.g. zero-cell categories after ``filter_cells``); only a live value
        with NO order entry signals staleness. ``values`` overrides a DB read.
        """
        if order is None:
            return set()
        if values is None:
            try:
                if column not in self.cells.columns:
                    return set()
                values = self.cells[column]
            except Exception:
                return set()
        uniques = {
            str(v) for v in values
            if v is not None and str(v) != "" and str(v).lower() != "nan"
        }
        return uniques - {str(o) for o in order}

    def _drop_categories_on_overwrite(self, column) -> None:
        """Drop ``categories[column]`` because the column is being rewritten.

        Unconditional where :meth:`_invalidate_stale_categories` is
        conditional: the caller knows every value in the column is being
        replaced, so any stored order/colours describe a labelling that no
        longer exists — even when the new labels happen to be a subset of the
        old ones. Writers that want an order re-persist it afterwards (as
        ``piaso.tl.leiden`` does), which is also the sequence that makes the
        message below almost never user-facing.
        """
        cats = self.metadata.get("categories")
        if not cats or column not in cats:
            return
        import warnings as _w
        self.metadata["categories"] = {
            k: v for k, v in cats.items() if k != column}
        try:
            self.flush()
        except Exception:
            pass
        _w.warn(
            f"cytome: cells column {column!r} is being overwritten — its "
            f"stored category order/colors were dropped. Re-run "
            f"set_categories({column!r}, ...) if you want an explicit order.",
            stacklevel=4,
        )

    def _invalidate_stale_categories(self, column, values=None) -> None:
        """Drop a stored ``categories[column]`` entry when the column's current
        values are no longer covered by its saved order (the column was
        overwritten). Emits a warning. No-op when nothing is stored / nothing
        is stale."""
        cats = self.metadata.get("categories")
        if not cats or column not in cats:
            return
        order = cats[column].get("order")
        if not order:
            return
        missing = self._categories_missing_values(column, order, values=values)
        if missing:
            import warnings as _w
            new = {k: v for k, v in cats.items() if k != column}
            self.metadata["categories"] = new
            # Persist immediately so direct readers of ds.metadata['categories']
            # (e.g. specificity_hotspot) see the drop, not just get_categories.
            try:
                self.flush()
            except Exception:
                pass
            _w.warn(
                f"cytome: dropped stale categories for column {column!r} — "
                f"{len(missing)} current value(s) (e.g. "
                f"{sorted(missing)[:3]}) are not in the stored category order, "
                f"so the column was overwritten. Re-run set_categories to "
                f"define a new order/colors.",
                stacklevel=3,
            )

    def _write_matrix_payload(self, payload: dict[str, Any]) -> None:
        matrix = payload["matrix"].tocsr()
        chunk_size = compute_chunk_size(matrix.shape[0], matrix.shape[1], int(matrix.nnz))
        row_entity = "cells"
        col_entity = _infer_col_entity(payload["name"])
        write_sparse_chunked(
            self._conn,
            payload["name"],
            matrix,
            chunk_size=chunk_size,
            compression="zstd",
            row_entity=row_entity,
            col_entity=col_entity,
        )

    def _write_embedding_payload(self, payload: dict[str, Any]) -> None:
        arr = np.asarray(payload["array"])
        n_rows = arr.shape[0]
        chunk_size = min(max(16, n_rows // 16 if n_rows else 16), 10000)
        write_dense_chunked(
            self._conn,
            payload["name"],
            arr,
            chunk_size=chunk_size,
            compression="zstd",
            entity=payload.get("entity", "cells"),
        )

        # add_embedding has always taken a `provenance` dict and embedding_meta
        # has always had a provenance_id column, and nothing connected them:
        # every embedding written through here had provenance_id NULL, so the
        # file could not answer "which modality / function produced this".
        prov = payload.get("provenance")
        if prov:
            try:
                pid = self.provenance.log(
                    operation="embedding",
                    function_name=str(prov.get("function", "add_embedding")),
                    parameters={k: v for k, v in prov.items() if k != "function"},
                    package_name=str(prov.get("package", "cytome")),
                    package_version=str(prov.get("package_version", "unknown")),
                    input_objects=list(prov.get("input_objects", [])),
                    output_objects=[payload["name"]],
                )
                self._conn.execute(
                    "UPDATE embedding_meta SET provenance_id = ? WHERE array_name = ?",
                    (pid, payload["name"]),
                )
            except Exception as exc:      # provenance must never lose the data
                logger.warning("could not record provenance for embedding %s: %s",
                               payload["name"], exc)

    def _write_entity_table(self, table_name: str, frame: pd.DataFrame) -> None:
        key_col = _primary_key_for_table(table_name)
        if key_col not in frame.columns:
            frame = frame.copy()
            frame.insert(0, key_col, np.arange(frame.shape[0], dtype=np.int64))

        # Resolve case-insensitive column collisions (SQLite column names
        # are case-insensitive, but Python's ``in`` check is case-sensitive).
        existing_cols = [c[1] for c in self._conn.execute(f"PRAGMA table_info({table_name})")]
        existing_lower = {c.lower(): c for c in existing_cols}

        # Remap DataFrame columns that differ only by case from schema cols,
        # and deduplicate self-collisions (e.g. 'batch' vs 'Batch').
        seen_lower: dict[str, str] = {}  # lower -> first occurrence
        new_col_names = []
        for col in frame.columns:
            cl = col.lower()
            if cl in existing_lower and col != existing_lower[cl]:
                # Case-insensitive match with schema column — use schema casing
                new_col_names.append(existing_lower[cl])
                seen_lower[cl] = existing_lower[cl]
            elif cl in seen_lower and col != seen_lower[cl]:
                # Self-collision — suffix to deduplicate
                deduped = f"{col}_alt"
                new_col_names.append(deduped)
            else:
                new_col_names.append(col)
                if cl not in seen_lower:
                    seen_lower[cl] = col

        if new_col_names != list(frame.columns):
            frame = frame.copy()
            frame.columns = new_col_names

        # Add new columns not yet in the schema (case-insensitive check)
        for col in frame.columns:
            if col.lower() in existing_lower:
                continue
            sql_type = "INTEGER"
            if pd.api.types.is_float_dtype(frame[col]):
                sql_type = "REAL"
            elif pd.api.types.is_object_dtype(frame[col]):
                sql_type = "TEXT"
            elif isinstance(frame[col].dtype, pd.CategoricalDtype):
                sql_type = "TEXT"
            self._conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {_quote_ident(col)} {sql_type}"
            )
            existing_lower[col.lower()] = col

        self._conn.execute(f"DELETE FROM {table_name}")
        cols = frame.columns.tolist()
        placeholders = ",".join(["?"] * len(cols))
        quoted_cols = ", ".join(_quote_ident(c) for c in cols)
        sql = f"INSERT INTO {table_name} ({quoted_cols}) VALUES ({placeholders})"
        rows = [tuple(_py_scalar(v) for v in row) for row in frame.to_numpy()]
        self._conn.executemany(sql, rows)

    def _sync_manifest_counts(self) -> None:
        n_cells = int(self._conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0])
        self._write_manifest_key("n_cells", n_cells)

    def _refresh_after_reopen(self) -> None:
        """Reset connection-bound caches after ``self._conn`` reopens.

        ``MetadataStore`` (exposed via ``self.metadata``) caches the
        ``sqlite3.Connection`` at construction time. After we close and
        reopen the underlying handle (e.g. inside ``filter_cells``),
        the cached store holds a closed connection — any subsequent
        ``ds.metadata.get(...)`` raises
        ``ProgrammingError: cannot operate on a closed database``.

        All other accessors (``ds.cells``, ``ds.embeddings``,
        ``ds.provenance``, ``ds.graphs``, modalities, fragments,
        measurement layers) construct fresh on each property access
        and don't need refreshing.

        New cached attributes registered in ``__init__`` MUST be
        added here too — otherwise they'll be stale-cache vectors
        after the next reopen.
        """
        self._metadata_obj = None
        self._manifest = self._read_manifest()

    def _read_manifest(self) -> dict[str, Any]:
        rows = self._conn.execute("SELECT key, value FROM _manifest").fetchall()
        return {k: json.loads(v) for k, v in rows}

    def _write_manifest_key(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO _manifest(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )


def _primary_key_for_table(table_name: str) -> str:
    mapping = {
        "cells": "cell_idx",
        "genes": "gene_idx",
        "GA_genes": "gene_idx",
        "peaks": "peak_idx",
        "tiles": "tile_idx",
        "samples": "sample_idx",
        "proteins": "protein_idx",
    }
    if table_name not in mapping:
        raise ValueError(f"Unsupported entity table: {table_name}")
    return mapping[table_name]


def _py_scalar(value: Any) -> Any:
    """One DataFrame cell as something ``sqlite3`` can bind.

    ``pandas.NA`` and ``NaT`` are the cases worth naming: they carry no
    ``.item()``, and sqlite3 has no adapter for them, so a nullable column
    (``string``, ``Int64``, ``boolean``) or a categorical with missing values —
    which is what a cell-type deconvolution writes by construction — used to
    fail the whole conversion with "Error binding parameter N: type 'NAType' is
    not supported". They are missing values; SQL spells that NULL.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass          # arrays and other non-scalars: not a missing value
    return value.item() if hasattr(value, "item") else value


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _infer_col_entity(matrix_name: str) -> str:
    if "ATAC" in matrix_name or "peak" in matrix_name.lower():
        return "peaks"
    if "tile" in matrix_name.lower():
        return "tiles"
    if "protein" in matrix_name.lower():
        return "proteins"
    if matrix_name.startswith("GA_") or matrix_name == "GA":
        return "GA_genes"
    return "genes"
