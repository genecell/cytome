"""The cached numba kernels must not make ``import cytome`` fail where numba cannot write a cache.

numba picks its cache directory when a function is decorated with ``cache=True`` and raises ``RuntimeError``
when it finds none: a read-only install with no writable home, as in many containers. ``njit_cached``
(``cytome/core/_jit.py``) then compiles without a cache.
"""

import os
import shutil
import stat
import subprocess
import sys

import pytest

import cytome
import cytome.core._jit as J


def _fake_njit(calls, fail_with=None):
    def njit(cache, **options):
        def decorate(func):
            calls.append((cache, options))
            if cache and fail_with is not None:
                raise fail_with
            return func
        return decorate
    return njit


def test_no_cache_when_numba_finds_no_place_for_it(monkeypatch):
    calls = []
    err = RuntimeError("cannot cache function 'f': no locator available for file 'f.py'")
    monkeypatch.setattr(J, "_numba_njit", _fake_njit(calls, err))
    f = J.njit_cached(nogil=True)(lambda x: x + 1)
    assert f(1) == 2
    assert calls == [(True, {"nogil": True}), (False, {"nogil": True})]


def test_the_cache_is_kept_where_numba_can_write_it(monkeypatch):
    calls = []
    monkeypatch.setattr(J, "_numba_njit", _fake_njit(calls))
    f = J.njit_cached(nogil=True)(lambda x: x + 1)
    assert f(1) == 2
    assert calls == [(True, {"nogil": True})]


def test_another_error_is_not_hidden(monkeypatch):
    monkeypatch.setattr(J, "_numba_njit", _fake_njit([], ValueError("not about the cache")))
    with pytest.raises(ValueError):
        J.njit_cached(nogil=True)(lambda x: x)


@pytest.mark.skipif(not J.HAS_NUMBA, reason="numba is not installed")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes to read-only folders")
def test_import_and_kernels_in_a_read_only_install_without_a_home(tmp_path):
    """A copy of the package nobody can write to, a home nobody can write to, no NUMBA_CACHE_DIR.

    The home must exist and be read-only: numba creates a missing home's cache folder when it may, and then
    finds its place. Without the fallback this import fails with numba's RuntimeError.
    """
    site = tmp_path / "site"
    home = tmp_path / "home"
    home.mkdir()
    shutil.copytree(os.path.dirname(cytome.__file__), site / "cytome",
                    ignore=shutil.ignore_patterns("__pycache__", "tests"))
    for root, dirs, files in os.walk(site, topdown=False):
        for name in files:
            os.chmod(os.path.join(root, name), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(root, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    os.chmod(home, stat.S_IRUSR | stat.S_IXUSR)
    code = (
        "import numpy as np, cytome, cytome.core.pseudobulk as p, cytome.core.footprints as f\n"
        "assert cytome.__file__.startswith(%r), cytome.__file__\n"
        "assert p._HAS_NUMBA and f._HAS_NUMBA\n"
        "s = np.zeros((2, 3)); q = np.zeros((2, 3)); n = np.zeros((2, 3), np.int32)\n"
        "p._accum_csr_numba(np.array([0, 2, 3]), np.array([0, 2, 1], np.int32), np.array([1.0, 2.0, 3.0]),\n"
        "                   np.array([0, 1], np.int32), s, q, n)\n"
        "assert s.tolist() == [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]] and int(n.sum()) == 3\n"
        "out = np.zeros(3); p._add_row_sums_numba(np.array([0, 2, 2], np.int32), np.array([1.0, 2.0, 3.0]), out)\n"
        "assert out.tolist() == [1.0, 0.0, 5.0]\n"
        "m = np.zeros((1, 4)); f._accum_cuts_into_window_numba(np.array([10, 11, 13, 99]), np.array([0, 0, 0, 0], np.int32), 10, 14, 1, m)\n"
        "assert m.tolist() == [[1.0, 1.0, 0.0, 1.0]]\n"
        "print('ok')\n" % str(site))
    env = {k: v for k, v in os.environ.items() if k not in ("NUMBA_CACHE_DIR", "XDG_CACHE_HOME", "PYTHONPATH")}
    env.update(HOME=str(home), PYTHONPATH=str(site), PYTHONDONTWRITEBYTECODE="1")
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True, text=True)
    finally:
        os.chmod(home, stat.S_IRWXU)
        for root, dirs, files in os.walk(site):
            os.chmod(root, stat.S_IRWXU)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("ok")
    assert not any(n.endswith((".nbi", ".nbc")) for folder in (site, home) for _, _, names in os.walk(folder) for n in names)
