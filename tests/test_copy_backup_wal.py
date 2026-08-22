"""copy() and backup() must not lose committed data.

A cytome is three files. flush() COMMITS, and in WAL mode a commit writes to
x.cytome-wal, not to x.cytome. Both methods used to flush and then
shutil.copy2 the main file alone, so everything committed since the last
checkpoint stayed behind -- silently:

    d.copy(...)   -> the embedding was simply missing, []
    d.backup(...) -> the copy had no embedding_meta table at all

backup()'s docstring offers it as "a safety copy before a destructive
filter_cells", which is exactly the case where the loss is discovered after the
original is gone.
"""
import sqlite3

import numpy as np
import pytest
import scipy.sparse as sp

import cytome

anndata = pytest.importorskip("anndata")


@pytest.fixture
def live(tmp_path):
    """A dataset with a committed-but-not-checkpointed write pending."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix((rng.random((300, 40)) < 0.3).astype(np.float32))
    d = cytome.from_anndata(anndata.AnnData(X=X), output=str(tmp_path / "src.cytome"))
    d.add_embedding("X_test", np.arange(300 * 4, dtype=np.float32).reshape(300, 4))
    d.flush()                      # commits -- into the WAL
    return d, tmp_path


def _embeddings(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [r[0] for r in c.execute("SELECT array_name FROM embedding_meta")]
    except sqlite3.OperationalError as exc:      # table absent entirely
        return f"ERROR {exc}"
    finally:
        c.close()


def test_the_wal_really_is_holding_the_write(live):
    """Guard the premise: if the WAL were empty the test would prove nothing."""
    d, _ = live
    wal = str(d.path) + "-wal"
    import os
    assert os.path.exists(wal) and os.path.getsize(wal) > 0, (
        "no pending WAL, so this fixture cannot exercise the bug")


def test_copy_preserves_a_pending_embedding(live):
    d, tmp = live
    out = tmp / "copy.cytome"
    d.copy(str(out)).close()
    assert _embeddings(out) == ["X_test"]


def test_backup_preserves_a_pending_embedding(live):
    d, tmp = live
    out = tmp / "backup.cytome"
    d.backup(str(out))
    assert _embeddings(out) == ["X_test"]


def test_copied_values_are_identical(live):
    d, tmp = live
    out = tmp / "vals.cytome"
    d.copy(str(out)).close()
    c = cytome.open(str(out))
    try:
        got = np.asarray(c.embeddings["X_test"])
    finally:
        c.close()
    np.testing.assert_array_equal(
        got, np.arange(300 * 4, dtype=np.float32).reshape(300, 4))


def test_matrix_survives_too(live):
    """Not just embeddings: the counts must match cell for cell."""
    d, tmp = live
    out = tmp / "m.cytome"
    d.copy(str(out)).close()
    a = d.RNA.counts.to_memory()
    c = cytome.open(str(out))
    try:
        b = c.RNA.counts.to_memory()
    finally:
        c.close()
    assert a.shape == b.shape and a.nnz == b.nnz
    np.testing.assert_allclose(a.toarray(), b.toarray())


def test_checkpoint_empties_the_wal(live):
    """The escape hatch for anyone copying the file with an external tool."""
    import os
    d, _ = live
    wal = str(d.path) + "-wal"
    assert os.path.getsize(wal) > 0
    d.checkpoint()
    assert os.path.getsize(wal) == 0


def test_copy_still_refuses_to_overwrite(live):
    d, tmp = live
    out = tmp / "x.cytome"
    d.copy(str(out)).close()
    with pytest.raises(FileExistsError):
        d.copy(str(out))
    d.copy(str(out), force=True).close()      # force still works


def test_source_stays_usable_after_copy(live):
    """The online backup API must not disturb the live dataset."""
    d, tmp = live
    d.copy(str(tmp / "y.cytome")).close()
    d.add_embedding("X_after", np.zeros((300, 2), dtype=np.float32))
    d.flush()
    assert "X_after" in list(d.embeddings.keys())
