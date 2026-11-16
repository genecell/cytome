"""numba, when it is installed, with a cache where numba can keep one.

``njit_cached(**options)`` is ``numba.njit(cache=True, **options)``. numba looks for a place to write its cache
when a function is decorated, not when it is first called: beside the source file, else under
``NUMBA_CACHE_DIR`` or its user-wide directory. With none of them writable it raises
``RuntimeError: cannot cache function ...: no locator available``. That is the case in a container whose
``site-packages`` is read-only and whose user has no writable home (``docker run --user``, Apptainer
``--containall``, most workflow runners), and the decorators run at import, so ``import cytome`` failed there
whenever numba was in the image. The function is then compiled without a cache: once per process, the same
code.
"""

from __future__ import annotations

try:
    from numba import njit as _numba_njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    _numba_njit = None
    HAS_NUMBA = False


def njit_cached(**options):
    """``numba.njit(cache=True, **options)``; without a cache where numba finds no place to write one."""
    def decorate(func):
        try:
            return _numba_njit(cache=True, **options)(func)
        except RuntimeError:
            return _numba_njit(cache=False, **options)(func)
    return decorate
