"""Small readers for the atlas objects."""

from __future__ import annotations

from pathlib import Path


def read_obs(path: str | Path):
    """``obs`` of an ``.h5ad`` as anndata reads it, without loading ``X``.

    Same frame as ``anndata.read_h5ad(path).obs`` (index, categoricals, nullable
    columns), at a fraction of the memory.
    """
    import h5py
    from anndata.io import read_elem

    with h5py.File(path, "r") as h:
        return read_elem(h["obs"])
