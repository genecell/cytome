"""An older cytome whose fragment_chunks predates min_start must still open.
_create_schema indexes on min_start; without the migration the index statement
raised 'no such column' and the file was unopenable — the E18 and SAN2 result
cytomes in the field are exactly this vintage.

The migration must also stay off the path for files that are *not* old, which
is what test_a_current_file_needs_no_migration pins: the ALTER takes a write
lock, and a file a writer still holds then costs busy_timeout per open."""
import sqlite3
import numpy as np
import pandas as pd
import cytome


def test_old_fragment_chunks_without_min_start_opens(tmp_path):
    p = tmp_path / "old.cytome"
    ds = cytome.create(str(p))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(3),
                                         "barcode": list("abc")}))
    ds.flush(); ds.close()

    # Regress the schema: rebuild fragment_chunks without the two columns.
    conn = sqlite3.connect(str(p))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fragment_chunks)")]
    assert "min_start" in cols
    keep = [c for c in cols if c != "min_start"]
    conn.executescript(f"""
        DROP INDEX IF EXISTS idx_fc_chrom_minstart;
        CREATE TABLE fc_old AS SELECT {', '.join(keep)} FROM fragment_chunks;
        DROP TABLE fragment_chunks;
        ALTER TABLE fc_old RENAME TO fragment_chunks;
    """)
    conn.commit(); conn.close()

    ds = cytome.open(str(p))            # used to raise OperationalError
    cols = [r[1] for r in ds._conn.execute("PRAGMA table_info(fragment_chunks)")]
    assert "min_start" in cols
    assert list(ds.cells["barcode"]) == list("abc")
    ds.close()


def test_migration_tolerates_a_column_another_opener_already_added(tmp_path):
    """Worker processes open one older file concurrently; each reads the
    schema before any ALTER lands, and the loser's ALTER hits 'duplicate
    column'. That is success, not failure — surfaced by piaso's multi-worker
    tests the first time the migration shipped."""
    from cytome.io.sqlite_engine import _add_fragment_chunks_columns
    p = tmp_path / "t.cytome"
    ds = cytome.create(str(p)); ds.flush(); ds.close()
    conn = sqlite3.connect(str(p))
    # both columns already exist (as if another opener won the race)
    _add_fragment_chunks_columns(conn, ["min_start"])   # must not raise
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fragment_chunks)")}
    assert "min_start" in cols
    conn.close()


def test_a_current_file_needs_no_migration(tmp_path):
    """A file this version wrote must take no ALTER on open.

    The migration list carried ``max_start``, which is not a column of
    fragment_chunks and never has been. Every current file therefore looked
    like it needed migrating, and the ALTER wants a write lock -- so opening a
    cytome that another connection still held (``from_h5ad(backed=True)``
    returns one) waited out the 60 s busy_timeout and then failed to open a
    file that was never old.
    """
    from cytome.io.sqlite_engine import _migrate_fragment_chunks_columns
    p = tmp_path / "current.cytome"
    ds = cytome.create(str(p)); ds.flush(); ds.close()

    conn = sqlite3.connect(str(p))
    before = {r[1] for r in conn.execute("PRAGMA table_info(fragment_chunks)")}
    _migrate_fragment_chunks_columns(conn)
    after = {r[1] for r in conn.execute("PRAGMA table_info(fragment_chunks)")}
    assert before == after, f"migration altered a current file: {after - before}"
    conn.close()


def test_opening_a_file_a_writer_still_holds_does_not_stall(tmp_path):
    """The scenario the bug was found in, made deterministic and put on a clock.

    ``from_h5ad(backed=True)`` returns an open Dataset, and callers that ignore
    the return value leave it holding the file. A reader may legitimately have
    to wait for a writer, but not for a migration it does not need: with
    ``max_start`` on the list every open took the ALTER path, waited out
    ``busy_timeout`` (60 s) and then failed to open a file that was never old.
    """
    import time
    p = tmp_path / "held.cytome"
    ds = cytome.create(str(p))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(3),
                                         "barcode": list("abc")}))
    ds.flush(); ds.close()

    holder = sqlite3.connect(str(p))
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")          # a writer holds the file
    try:
        t0 = time.time()
        reader = cytome.open(str(p))           # used to wait 60 s, then raise
        elapsed = time.time() - t0
        assert list(reader.cells["barcode"]) == list("abc")
        reader.close()
        assert elapsed < 10, (
            f"open stalled for {elapsed:.1f}s behind a writer, on a file "
            f"that needs no migration"
        )
    finally:
        holder.rollback(); holder.close()
