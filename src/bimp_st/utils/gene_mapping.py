"""Xenium panel ↔ snRNA reference gene ID mapping.

Single source of truth for crosswalks between Xenium panel gene IDs
(``vNP3.1`` / ``BIMP_vNP3.1_*``), v4.3 gene IDs (``BIMP_v4.3_*``), and the
NCBI ``LOC*`` gene IDs on the 2023 genome assembly.

CSV schema (``data/panel/np31_to_v43_mapping.csv``)::

    np31_id,v43_id,loc_id,description

Usage::

    from bimp_st.utils.gene_mapping import load_gene_mapping, GENE_MAPPING_FILE

    mapping = load_gene_mapping()          # {np31_id: v43_id}
    rev = load_gene_mapping(reverse=True)  # {v43_id: np31_id}

    # Or apply a cross-namespace rename to an anndata var_names list
    new_names = map_spatial_genes_to_v43(adata.var_names, mapping)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical location of the crosswalk CSV (repo-relative). Resolved at
# import time so callers in any working directory work transparently.
_PKG_ROOT = Path(__file__).resolve().parents[3]
GENE_MAPPING_FILE: Path = _PKG_ROOT / "data" / "panel" / "np31_to_v43_mapping.csv"


def load_gene_mapping(
    path: Path | str | None = None,
    *,
    reverse: bool = False,
) -> dict[str, str]:
    """Return ``np31_id → v43_id`` (or the reverse) as a dict.

    Parameters
    ----------
    path
        Override the default crosswalk CSV path. Usually left ``None``.
    reverse
        If ``True``, return ``v43_id → np31_id`` instead.

    A missing file returns an empty dict.
    """
    csv_path = Path(path) if path is not None else GENE_MAPPING_FILE
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        np31_id = str(row.get("np31_id", "")).strip()
        v43_id = str(row.get("v43_id", "")).strip()
        if np31_id and v43_id and np31_id.lower() != "nan" and v43_id.lower() != "nan":
            if reverse:
                mapping[v43_id] = np31_id
            else:
                mapping[np31_id] = v43_id
    return mapping


def map_spatial_genes_to_v43(
    var_names,
    mapping: dict[str, str] | None = None,
) -> list[str]:
    """Rewrite spatial gene IDs (``BIMP|MGEN`` pipe-joined) to v4.3 IDs.

    Xenium panel gene IDs often look like ``BIMP_vNP3.1_01359|MGEN_v3.1_02664``
    (pipe-joined dual-namespace). This helper splits on ``|``, looks up
    each component in *mapping*, and returns the first v4.3 hit, falling
    back to the original name if nothing matched.

    Pass a preloaded *mapping* to avoid hitting disk repeatedly.
    """
    if mapping is None:
        mapping = load_gene_mapping()

    out: list[str] = []
    for g in var_names:
        hit: str | None = None
        for candidate in str(g).split("|"):
            candidate = candidate.strip()
            if candidate in mapping:
                hit = mapping[candidate]
                break
        out.append(hit if hit is not None else str(g))
    return out


__all__ = [
    "GENE_MAPPING_FILE",
    "load_gene_mapping",
    "map_spatial_genes_to_v43",
]
