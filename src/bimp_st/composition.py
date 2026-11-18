"""Per-region cell-type composition: one cell set, one contrast statistic.

Every producer of a per-region count or composition (the caste-composition
figure's panel C, its counts, the regional-against-caste contrast and
the region classifier) assigns cells to regions through :func:`atlas_regions`,
so a figure, its counts and its ratios are drawn from the same cells.

Region assignment is exclusive: a cell counts towards a region only if it lies
inside exactly one of :data:`COMPOSITION_REGIONS` (the seven statistical
regions; overlaps are dropped, see ``nine_region_common.region_of``). The
region flags are attached fresh from the nine-region crop sidecars, never read
from the ``in_*`` columns stored in an h5ad (those predate the polygon repair).
"""

from __future__ import annotations

from dataclasses import dataclass

from bimp_st.nine_region_common import (
    BASE_REGIONS,
    OUTSIDE,
    OVERLAP,
    STATISTICAL_REGIONS,
    attach_region_flags,
    region_of,
)

#: Regions a cell is resolved against; a cell in two of them is dropped.
COMPOSITION_REGIONS = list(STATISTICAL_REGIONS)

#: Regions drawn in the caste-composition figure's panel C and compared in the
#: regional-against-caste contrast. The antennal lobe is resolved (so a cell
#: inside it and a calyx compartment is dropped from both) but not compared:
#: it is traced unevenly between the two brains.
PANEL_REGIONS = list(BASE_REGIONS)

#: A class enters a contrast only if its share is at least this in BOTH groups.
#: A share near zero gives a ratio of noise (the lobula Kenyon row is a handful
#: of cells), and a one-sided floor would keep exactly those.
SHARE_FLOOR = 0.005


def atlas_regions(adata, regions=None):
    """Attach fresh region flags to ``adata.obs`` and return one region per cell.

    Returns an object array: a region name, ``OUTSIDE``, ``OVERLAP`` or None
    (untraced). ``adata`` only needs an ``obs`` with ``slice_name`` and
    ``cell_id``; its stored ``in_*`` columns are overwritten.
    """
    regions = list(regions or COMPOSITION_REGIONS)
    attach_region_flags(adata, regions=regions)
    return region_of(adata.obs, regions, overlap="drop")


def composition_table(brain, region, label, regions=None):
    """Long table of cell counts and shares per (brain, region, label).

    ``brain``, ``region`` and ``label`` are per-cell arrays of equal length;
    ``region`` comes from :func:`atlas_regions`. Cells outside ``regions``
    (default :data:`PANEL_REGIONS`), in an overlap or untraced are left out.
    Columns: brain, region, cluster, n_cells, proportion, region_total.
    """
    import numpy as np
    import pandas as pd

    regions = list(regions or PANEL_REGIONS)
    df = pd.DataFrame(
        {
            "brain": np.asarray(brain).astype(str),
            "region": np.asarray(region, dtype=object),
            "cluster": np.asarray(label).astype(str),
        }
    )
    df = df[df["region"].isin(regions)]
    n = df.groupby(["brain", "region", "cluster"]).size().rename("n_cells")
    out = n.reset_index()
    tot = out.groupby(["brain", "region"])["n_cells"].transform("sum")
    out["proportion"] = out["n_cells"] / tot
    out["region_total"] = tot
    return out.sort_values(["brain", "region", "cluster"]).reset_index(drop=True)


@dataclass(frozen=True)
class CompositionContrast:
    """Caste contrast against regional contrast, both in |log2 share ratio|."""

    caste: tuple  # per (region, class) worker-vs-queen |log2|
    regional: tuple  # per (brain, region pair, class) |log2|
    floor: float

    @property
    def caste_median(self) -> float:
        import numpy as np

        return float(np.median(self.caste))

    @property
    def regional_median(self) -> float:
        import numpy as np

        return float(np.median(self.regional))

    @property
    def ratio(self) -> float:
        return self.regional_median / self.caste_median

    @property
    def n_caste(self) -> int:
        return len(self.caste)

    @property
    def n_regional(self) -> int:
        return len(self.regional)

    @property
    def n_caste_over_regional_median(self) -> int:
        return int(sum(v > self.regional_median for v in self.caste))

    @property
    def caste_max(self) -> float:
        return float(max(self.caste))


def composition_contrast(table, floor: float = SHARE_FLOOR, regions=None):
    """Compare caste and regional composition contrasts on one table.

    ``table`` is :func:`composition_table` output (or any frame with brain,
    region, cluster, n_cells). For every class, the contrast between two groups
    of cells is |log2(share_b / share_a)|, and it is kept only when both shares
    are at least ``floor`` (symmetric, so neither arm keeps one-sided
    near-zero ratios).

      caste     worker vs queen, within one region
      regional  region vs region, within one brain

    Descriptive on both axes: one brain per caste, so no test is attached.
    """
    import numpy as np

    t = table
    if regions is not None:
        t = t[t["region"].isin(list(regions))]
    shares = {}
    for (b, r), g in t.groupby(["brain", "region"]):
        s = g.groupby("cluster")["n_cells"].sum().astype(float)
        shares[(b, r)] = s / s.sum()
    classes = sorted(t["cluster"].astype(str).unique())
    regs = sorted(t["region"].unique())

    def _l2(a, b):
        if a < floor or b < floor:
            return None
        return abs(float(np.log2(b / a)))

    caste, regional = [], []
    for r in regs:
        q, w = shares.get(("queen", r)), shares.get(("worker", r))
        if q is None or w is None:
            continue
        for c in classes:
            v = _l2(float(q.get(c, 0.0)), float(w.get(c, 0.0)))
            if v is not None:
                caste.append(v)
    for b in ("queen", "worker"):
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                s1, s2 = shares.get((b, regs[i])), shares.get((b, regs[j]))
                if s1 is None or s2 is None:
                    continue
                for c in classes:
                    v = _l2(float(s1.get(c, 0.0)), float(s2.get(c, 0.0)))
                    if v is not None:
                        regional.append(v)
    if not caste or not regional:
        raise ValueError("composition table gave no contrasts above the floor")
    return CompositionContrast(tuple(caste), tuple(regional), float(floor))


__all__ = [
    "COMPOSITION_REGIONS",
    "PANEL_REGIONS",
    "SHARE_FLOOR",
    "OUTSIDE",
    "OVERLAP",
    "atlas_regions",
    "composition_table",
    "composition_contrast",
    "CompositionContrast",
]
