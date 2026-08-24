"""SQLite engine helpers for Cytome."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_connection(conn: sqlite3.Connection) -> None:
    """Apply Cytome SQLite pragmas.

    If ``PRAGMA journal_mode=WAL`` fails (typically because another
    process is mid-write on an NFS/SquashFS/FUSE mount, or — much more
    rarely — because the filesystem genuinely cannot support WAL), the
    error is re-raised with an actionable diagnostic message so the
    caller can decide whether to wait, retry, or copy the file to a
    local disk.

    Parameters
    ----------
    conn
        Open SQLite connection.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:
        # Don't auto-fall-back to DELETE: switching journal modes
        # requires an exclusive lock that the contending writer holds,
        # so the fallback would also fail and produce a more confusing
        # traceback. Instead, raise with a checklist the user can act on.
        try:
            db_path = conn.execute("PRAGMA database_list").fetchall()[0][2]
        except Exception:
            db_path = "<your_cytome_file>"
        raise sqlite3.OperationalError(
            f"{exc}\n\n"
            f"Cytome could not enable WAL journal mode on this file. "
            f"Common causes:\n"
            f"  1. Another process is writing to the cytome (e.g. a "
            f"Snakemake pipeline still running). Wait for it to "
            f"finish — NFS write contention typically clears in "
            f"1-10 minutes — and retry cytome.open(...).\n"
            f"  2. Stale -wal/-shm sidecar files from a crashed "
            f"writer. If no other process has the file open, removing "
            f"'{db_path}-wal' and '{db_path}-shm' and retrying may help.\n"
            f"  3. The filesystem genuinely doesn't support WAL "
            f"(NFS without shared-mmap, SquashFS, some FUSE mounts). "
            f"Copy the cytome to local disk and open from there:\n"
            f"      cp '{db_path}' /tmp/local.cytome  # or /dev/shm/local.cytome\n"
        ) from exc
    conn.execute("PRAGMA busy_timeout=60000")  # 60s retry for concurrent access
    conn.execute("PRAGMA page_size=65536")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    # mmap_size: SQLite normally degrades to plain read()/write() if
    # mmap fails, but hostile NFS clients can raise OperationalError
    # here. Swallow it — SQLite's own fallback path handles it.
    try:
        conn.execute("PRAGMA mmap_size=268435456")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA foreign_keys=ON")


def create_database(path: PathLike, force: bool = False) -> sqlite3.Connection:
    """Create a new Cytome database with the full schema.

    Parameters
    ----------
    path
        Output `.cytome` file path.
    force
        If ``False`` (default), raise :class:`FileExistsError` when ``path``
        already exists, so an expensive cytome is never silently truncated by
        a typo'd output path. Pass ``force=True`` to overwrite an existing
        file on purpose. (Matches the ``force`` semantics of
        :meth:`CytomeDataset.copy` / :meth:`CytomeDataset.backup`.)

    Returns
    -------
    sqlite3.Connection
        Open configured connection.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``force`` is ``False``.
    """
    db_path = Path(path)
    if db_path.exists() and not force:
        raise FileExistsError(
            f"Cytome already exists: {db_path}. Pass force=True to overwrite "
            f"(guards against accidentally truncating an existing cytome)."
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        # force=True: drop the old DB (+ WAL/SHM sidecars) so we start from a
        # clean schema rather than reopening a partial/foreign file.
        for sfx in ("", "-wal", "-shm"):
            p = Path(str(db_path) + sfx)
            if p.exists():
                p.unlink()
    conn = sqlite3.connect(str(db_path))
    _configure_connection(conn)
    _create_schema(conn)
    _init_manifest(conn, db_path.stem)
    conn.commit()
    return conn


def open_database(path: PathLike) -> sqlite3.Connection:
    """Open an existing Cytome database.

    Parameters
    ----------
    path
        Existing `.cytome` file path.

    Returns
    -------
    sqlite3.Connection
        Open configured connection.
    """
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"Cytome file not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    _configure_connection(conn)
    _create_schema(conn)  # ensures new tables (e.g. tiles) exist in older files
    return conn


def close_database(conn: sqlite3.Connection) -> None:
    """Flush and close SQLite connection.

    Parameters
    ----------
    conn
        Open SQLite connection.
    """
    conn.commit()
    conn.close()


def _package_version() -> str:
    """The version of the code that is running, resolved lazily (sqlite_engine
    is imported from cytome/__init__, so this cannot import at module level).

    Deliberately ``cytome.__version__`` and not the installed distribution
    metadata: a checkout on PYTHONPATH beside a stale editable install would
    report the installed version and stamp every manifest with a version that
    did not write the file -- the exact claim ``writer_version`` exists to
    make. Metadata is only a fallback for the pathological case where the
    attribute is missing."""
    try:
        import cytome
        v = getattr(cytome, "__version__", None)
        if v:
            return v
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("cytome")
    except Exception:
        return "unknown"


def _init_manifest(conn: sqlite3.Connection, dataset_name: str) -> None:
    manifest = {
        "format_version": "1.0.0",
        "min_reader_version": "1.0.0",
        # Read the real version rather than a literal. This said "cytome
        # 0.1.0" for every file the package has ever written, including ones
        # written by 0.2.4, so the manifest could not be used to answer "which
        # version produced this" -- the one question it exists to answer.
        "writer_version": f"cytome {_package_version()}",
        "created_at": _now_iso(),
        "dataset_name": dataset_name,
        "modalities": [],
        "n_cells": 0,
        "genome": None,
    }
    conn.executemany(
        "INSERT OR REPLACE INTO _manifest(key, value) VALUES (?, ?)",
        [(k, json.dumps(v)) for k, v in manifest.items()],
    )


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS _manifest (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS _provenance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            operation TEXT NOT NULL,
            package_name TEXT NOT NULL,
            package_version TEXT NOT NULL,
            python_version TEXT,
            function_name TEXT NOT NULL,
            parameters TEXT NOT NULL,
            dependency_versions TEXT,
            input_objects TEXT,
            output_objects TEXT,
            random_seed INTEGER,
            hardware_info TEXT,
            duration_seconds REAL,
            peak_memory_mb REAL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS _schema_migrations (
            migration_id INTEGER PRIMARY KEY,
            description TEXT,
            applied_at TEXT,
            from_version TEXT,
            to_version TEXT
        );

        CREATE TABLE IF NOT EXISTS _metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            value_type TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS _column_meta (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            dtype TEXT NOT NULL,
            categories TEXT,
            PRIMARY KEY (table_name, column_name)
        );

        CREATE TABLE IF NOT EXISTS cells (
            cell_idx INTEGER PRIMARY KEY,
            barcode TEXT NOT NULL,
            sample_id TEXT,
            n_fragments INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_cells_sample ON cells(sample_id);

        CREATE TABLE IF NOT EXISTS genes (
            gene_idx INTEGER PRIMARY KEY,
            gene_id TEXT NOT NULL UNIQUE,
            symbol TEXT,
            chr TEXT,
            start INTEGER,
            end_ INTEGER,
            biotype TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_genes_symbol ON genes(symbol);
        CREATE INDEX IF NOT EXISTS idx_genes_coords ON genes(chr, start);

        CREATE TABLE IF NOT EXISTS GA_genes (
            gene_idx INTEGER PRIMARY KEY,
            gene_id TEXT NOT NULL UNIQUE,
            symbol TEXT,
            chr TEXT,
            start INTEGER,
            end_ INTEGER,
            biotype TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ga_genes_symbol ON GA_genes(symbol);
        CREATE INDEX IF NOT EXISTS idx_ga_genes_coords ON GA_genes(chr, start);

        CREATE TABLE IF NOT EXISTS peaks (
            peak_idx INTEGER PRIMARY KEY,
            peak_id TEXT NOT NULL,
            chr TEXT NOT NULL,
            start INTEGER NOT NULL,
            end_ INTEGER NOT NULL,
            annotation TEXT,
            nearest_gene TEXT,
            distance_to_tss INTEGER,
            -- narrowPeak / PICCO peak-calling stats (nullable; written by the
            -- quantifier when peaks come from a PICCO narrowPeak BED)
            summit INTEGER,
            score REAL,
            signal REAL,
            neg_log10_pvalue REAL,
            neg_log10_qvalue REAL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS peaks_rtree USING rtree(
            id,
            min_chr, max_chr,
            min_start, max_start,
            min_end, max_end
        );

        CREATE TABLE IF NOT EXISTS tiles (
            tile_idx INTEGER PRIMARY KEY,
            tile_id TEXT NOT NULL,
            chr TEXT NOT NULL,
            start INTEGER NOT NULL,
            end_ INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS samples (
            sample_idx INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL UNIQUE,
            batch TEXT,
            condition TEXT,
            donor TEXT,
            chemistry TEXT,
            n_cells INTEGER
        );

        CREATE TABLE IF NOT EXISTS proteins (
            protein_idx INTEGER PRIMARY KEY,
            protein_id TEXT NOT NULL UNIQUE,
            gene_mapping TEXT
        );

        CREATE TABLE IF NOT EXISTS matrix_chunks (
            id INTEGER PRIMARY KEY,
            matrix_name TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            row_start INTEGER NOT NULL,
            row_end INTEGER NOT NULL,
            n_nonzero INTEGER NOT NULL,
            data_blob BLOB NOT NULL,
            indices_blob BLOB NOT NULL,
            indptr_blob BLOB NOT NULL,
            dtype TEXT NOT NULL,
            compression TEXT DEFAULT 'zstd',
            UNIQUE(matrix_name, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_matrix_chunks ON matrix_chunks(matrix_name, chunk_idx);

        CREATE TABLE IF NOT EXISTS matrix_csc_chunks (
            id INTEGER PRIMARY KEY,
            matrix_name TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            col_start INTEGER NOT NULL,
            col_end INTEGER NOT NULL,
            n_nonzero INTEGER NOT NULL,
            data_blob BLOB NOT NULL,
            indices_blob BLOB NOT NULL,
            indptr_blob BLOB NOT NULL,
            dtype TEXT NOT NULL,
            compression TEXT DEFAULT 'zstd',
            UNIQUE(matrix_name, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_csc_chunks ON matrix_csc_chunks(matrix_name, chunk_idx);

        CREATE TABLE IF NOT EXISTS matrix_meta (
            matrix_name TEXT PRIMARY KEY,
            n_rows INTEGER NOT NULL,
            n_cols INTEGER NOT NULL,
            n_nonzero INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            row_entity TEXT NOT NULL,
            col_entity TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            n_chunks INTEGER NOT NULL,
            has_csc INTEGER DEFAULT 0,
            csc_chunk_size INTEGER,
            csc_n_chunks INTEGER,
            created_at TEXT,
            provenance_id INTEGER REFERENCES _provenance(id),
            -- Whether the stored values are integers, decided once by the
            -- writer. Consumers otherwise re-derive it from a sample, and
            -- COSG's first attempt at that was dead code that always answered
            -- "cannot tell" while looking correct. NULL = written before this
            -- column existed; probe those.
            is_integer INTEGER
        );

        CREATE TABLE IF NOT EXISTS dense_chunks (
            id INTEGER PRIMARY KEY,
            array_name TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            row_start INTEGER NOT NULL,
            row_end INTEGER NOT NULL,
            n_cols INTEGER NOT NULL,
            data_blob BLOB NOT NULL,
            dtype TEXT NOT NULL,
            compression TEXT DEFAULT 'zstd',
            UNIQUE(array_name, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_dense_chunks ON dense_chunks(array_name, chunk_idx);

        CREATE TABLE IF NOT EXISTS embedding_meta (
            array_name TEXT PRIMARY KEY,
            n_rows INTEGER NOT NULL,
            n_cols INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            entity TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            n_chunks INTEGER NOT NULL,
            created_at TEXT,
            provenance_id INTEGER REFERENCES _provenance(id)
        );

        CREATE TABLE IF NOT EXISTS fragment_meta (
            chrom TEXT PRIMARY KEY,
            n_fragments INTEGER NOT NULL,
            table_name TEXT NOT NULL,
            rtree_name TEXT NOT NULL,
            min_start INTEGER,
            max_end INTEGER
        );

        CREATE TABLE IF NOT EXISTS fragment_chunks (
            id INTEGER PRIMARY KEY,
            chrom TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            row_start INTEGER NOT NULL,
            row_end INTEGER NOT NULL,
            n_fragments INTEGER NOT NULL,
            min_start INTEGER,
            starts_blob BLOB NOT NULL,
            ends_blob BLOB NOT NULL,
            cell_idx_blob BLOB NOT NULL,
            compression TEXT DEFAULT 'zstd',
            encoding INTEGER DEFAULT 0,
            UNIQUE(chrom, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_frag_chunks_chrom
            ON fragment_chunks(chrom, chunk_idx);
        CREATE INDEX IF NOT EXISTS idx_fc_chrom_minstart
            ON fragment_chunks(chrom, min_start);

        CREATE TABLE IF NOT EXISTS graph_knn (
            cell_i INTEGER NOT NULL,
            cell_j INTEGER NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (cell_i, cell_j)
        );

        CREATE TABLE IF NOT EXISTS graph_peak_gene (
            peak_idx INTEGER NOT NULL,
            gene_idx INTEGER NOT NULL,
            distance INTEGER NOT NULL,
            correlation REAL,
            p_value REAL,
            p_adj REAL,
            provenance_id INTEGER REFERENCES _provenance(id),
            PRIMARY KEY (peak_idx, gene_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_pg_gene ON graph_peak_gene(gene_idx);

        CREATE TABLE IF NOT EXISTS graph_edges (
            graph_name TEXT NOT NULL,
            axis TEXT NOT NULL DEFAULT 'obs',
            entity_table TEXT NOT NULL DEFAULT 'cells',
            row_idx INTEGER NOT NULL,
            col_idx INTEGER NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (graph_name, axis, row_idx, col_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_edges_name_axis ON graph_edges(graph_name, axis);

        -- Genome annotation tables (added in cytome 0.2.2).
        -- Coordinates are 0-based half-open (BED convention).
        -- GTF source files are 1-based closed; the import converts.
        CREATE TABLE IF NOT EXISTS _gene_annotation (
            gene_id TEXT PRIMARY KEY,
            chrom TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            strand TEXT,
            gene_name TEXT,
            gene_type TEXT,
            source TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_gene_annotation_pos
            ON _gene_annotation(chrom, start, end);
        CREATE INDEX IF NOT EXISTS idx_gene_annotation_name
            ON _gene_annotation(gene_name);

        CREATE TABLE IF NOT EXISTS _exon_annotation (
            gene_id TEXT NOT NULL,
            transcript_id TEXT,
            exon_number INTEGER,
            chrom TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            strand TEXT,
            feature TEXT,
            transcript_type TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_exon_annotation_pos
            ON _exon_annotation(chrom, start, end);
        CREATE INDEX IF NOT EXISTS idx_exon_annotation_gene
            ON _exon_annotation(gene_id);

        CREATE TABLE IF NOT EXISTS spatial_coords (
            cell_idx INTEGER PRIMARY KEY,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL,
            FOREIGN KEY (cell_idx) REFERENCES cells(cell_idx)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS spatial_rtree USING rtree(
            id,
            min_x, max_x,
            min_y, max_y
        );

        CREATE TABLE IF NOT EXISTS lazy_layers (
            layer_name TEXT PRIMARY KEY,
            base_layer TEXT NOT NULL,
            transform_type TEXT NOT NULL,
            parameters TEXT NOT NULL,
            provenance_id INTEGER REFERENCES _provenance(id)
        );

        CREATE TABLE IF NOT EXISTS coverage_cache (
            id INTEGER PRIMARY KEY,
            group_name TEXT NOT NULL,
            groupby_key TEXT NOT NULL,
            chrom TEXT NOT NULL,
            bin_size INTEGER NOT NULL,
            normalize TEXT NOT NULL,
            values_blob BLOB NOT NULL,
            UNIQUE(group_name, groupby_key, chrom, bin_size, normalize)
        );

        CREATE TABLE IF NOT EXISTS cell_sets (
            set_name TEXT PRIMARY KEY,
            cell_indices TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS _raw_var (
            var_idx INTEGER PRIMARY KEY,
            var_name TEXT NOT NULL
        );
        """
    )

    # Round 8 (cytome 0.2.x) migration — add transcript_type
    # column to _exon_annotation if missing (older cytomes).
    _exon_cols = [
        r[1] for r in conn.execute("PRAGMA table_info(_exon_annotation)")
    ]
    if "transcript_type" not in _exon_cols:
        conn.execute(
            "ALTER TABLE _exon_annotation "
            "ADD COLUMN transcript_type TEXT"
        )

    # cytome 0.3.0 migration -- matrix_meta.is_integer. Older files leave it
    # NULL, which means "unknown, probe if you care" rather than "not
    # integer": the distinction matters because a consumer that reads NULL as
    # False would refuse to normalise a perfectly good counts matrix.
    _mm_cols = [r[1] for r in conn.execute("PRAGMA table_info(matrix_meta)")]
    if _mm_cols and "is_integer" not in _mm_cols:
        conn.execute("ALTER TABLE matrix_meta ADD COLUMN is_integer INTEGER")

    # NOTE: the narrowPeak / PICCO stat columns (summit, score, signal,
    # neg_log10_pvalue, neg_log10_qvalue) are part of the `peaks` CREATE TABLE
    # above for new files. We deliberately do NOT migrate them onto older
    # files here: opening a database should never mutate it, and the writer
    # that needs them (piaso's _write_narrowpeak_stats) adds any missing
    # columns lazily on first use. This keeps open_database side-effect-free.

    for chrom in _CHROMS:
        frag_table = f"fragments_{chrom}"
        rtree_table = f"fragments_{chrom}_rtree"
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {frag_table} (
                rowid INTEGER PRIMARY KEY,
                start INTEGER NOT NULL,
                end_ INTEGER NOT NULL,
                cell_idx INTEGER NOT NULL,
                dup_count INTEGER DEFAULT 1
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{frag_table}_cell ON {frag_table}(cell_idx)"
        )
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {rtree_table} USING rtree(
                id,
                min_start, max_start,
                min_end, max_end
            )
            """
        )
