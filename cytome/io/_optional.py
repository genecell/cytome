"""Optional third-party readers, imported where they are used.

cytome's install is deliberately small -- numpy, scipy, pandas, lz4 -- so the
HDF5 stack is not a hard dependency. That is a reasonable trade only if the
functions which *do* need it say so: a bare ``ModuleNotFoundError: No module
named 'h5py'`` out of a public reader reads like a broken install rather than
a missing extra.
"""
from __future__ import annotations


def require_h5py(what: str):
    """Import h5py, or explain which extra provides it.

    Parameters
    ----------
    what : str
        The public entry point that needs it, named in the error so the reader
        does not have to work out which call failed.
    """
    try:
        import h5py
    except ImportError as exc:                       # pragma: no cover - env
        raise ImportError(
            f"{what} reads HDF5 and needs h5py, which cytome does not install "
            "by default -- the base package is kept to numpy, scipy, pandas "
            "and lz4.\n"
            "    pip install h5py\n"
            "or take the extra that includes it:\n"
            "    pip install 'cytome[full]'"
        ) from exc
    return h5py
