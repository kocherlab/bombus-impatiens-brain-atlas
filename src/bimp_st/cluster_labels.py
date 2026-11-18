"""Canonical cluster label and color definitions.

Single source of truth for cell type names, abbreviations, and colors.
Supports both snRNA-seq cell types (12 types of the snRNA-seq reference)
and spatial Leiden clusters (32).
Edit SNRNA_CLUSTER_INFO below when biological labels change.

Usage::

    from bimp_st.cluster_labels import cluster_colors

    # Spatial cluster functions
    from bimp_st.cluster_labels import get_spatial_color, get_spatial_palette
"""

from __future__ import annotations

import colorsys

import numpy as np

from bimp_st.nine_region_common import KC_POPULATION_COLORS

# ── snRNA cell type info (union of the CB + OL annotated atlas) ───────────
# Keys match the ``cell_type`` column of the snRNA-seq reference h5ads
# (CentralBrain + OpticLobe). Central-brain types are fully annotated; the
# optic-lobe OL_1..OL_13 are placeholder subclusters pending annotation.
# These are the labels Tangram assigns to spatial cells (``tangram_cell_type``).
#
# Curated biological palette: the five named KC populations take the shared
# KC_POPULATION_COLORS (see below), projection/other neurons cool (blue/violet),
# glia green, hemocytes dark taupe; OL_* subclusters get a distinct
# golden-angle magenta/violet series.


def _norm_ct(name: str) -> str:
    """Canonicalize a cell_type label (strip embedded newlines / extra ws)."""
    return " ".join(str(name).replace("\n", " ").split())


# name -> (short, color). Publication palette: journal-style muted jewel tones,
# lineage-grouped (neurons blues, glia greens/teals, photoreceptors garnet,
# hemocytes dark taupe) with a distinct muted purple/magenta/plum family for the
# OL_* subclusters.
#
# The five named Kenyon cell populations take
# bimp_st.nine_region_common.KC_POPULATION_COLORS, the palette of
# fig:kc_subtypes and fig:atlas_overview panel E: each population in its
# namesake calyx compartment's colour (Large-type orange = KC-NC, Small-type
# mauve = KC-IC, Class II yellow = KC-OC), Middle-type warm brown, Other Kenyon
# cells mid grey.  Defining them
# here once means every figure that draws fine types follows automatically.
# Hemocytes take the dark taupe #5E5047 to stay distinct from Other Kenyon
# cells' grey.  The closest pair among the canonical-17 colours is Small-type
# vs Optic lobe (#B94FA0), dE2000 11.9, a pairing fig:atlas_overview panel E
# shows; every other pair is >= 13.
_KCP = KC_POPULATION_COLORS
_CURATED: dict[str, tuple[str, str]] = {
    "Large-type Kenyon cells": ("KC-l", _KCP["Large-type Kenyon cells"]),
    "Middle-type Kenyon cells": ("KC-m", _KCP["Middle-type Kenyon cells"]),
    "Small-type Kenyon cells": ("KC-s", _KCP["Small-type Kenyon cells"]),
    "Class II Kenyon cells": ("KC-II", _KCP["Class II Kenyon cells"]),
    "Other Kenyon cells": ("KC-o", _KCP["Other Kenyon cells"]),
    "Kenyon cells": ("KC", "#EAD9B0"),
    "Anterodorsal projection neurons": ("PN-ad", "#2F4E96"),
    "Lateral projection neurons": ("PN-lat", "#6E93D6"),
    "Other neurons": ("N-other", "#48A6C9"),
    "Photoreceptors": ("PR", "#8E1F3A"),
    "Astrocyte-like glia": ("Glia-astro", "#5F9E6E"),
    # olive, not sage: #9CC49B read as a lighter astrocyte-like glia
    "Cortex glia": ("Glia-cortex", "#A0A83C"),
    "Perineurial glia": ("Glia-peri", "#2E8B8B"),
    "Subperineurial glia": ("Glia-sub", "#1F5E52"),
    "Other glia": ("Glia-other", "#7FB3AD"),
    "Hemocytes": ("Hem", "#5E5047"),
}

_OL_SUBTYPES = [f"OL_{i}" for i in range(1, 14)]

# OL_* muted purple/magenta/plum family (lightness-spread for separation).
_OL_COLORS: dict[str, str] = {
    "OL_1": "#C539C6",
    "OL_2": "#DC85B9",
    "OL_3": "#862773",
    "OL_4": "#5932AC",
    "OL_5": "#A479D8",
    "OL_6": "#DEC3DF",
    "OL_7": "#713D56",
    "OL_8": "#C6397A",
    "OL_9": "#D15FBA",
    "OL_10": "#BA9BD6",
    "OL_11": "#5E3787",
    "OL_12": "#C28EA1",
    "OL_13": "#8C2EA0",
}


SNRNA_CLUSTER_INFO: dict[str, dict] = {}
for _n, (_s, _c) in _CURATED.items():
    SNRNA_CLUSTER_INFO[_n] = {"name": _n, "short": _s, "color": _c}
for _n in _OL_SUBTYPES:
    SNRNA_CLUSTER_INFO[_n] = {"name": _n, "short": _n, "color": _OL_COLORS[_n]}

# Backward-compatible alias


# Ordered list for consistent plotting (curated names first, then OL_* subclusters).
CELL_TYPE_ORDER = list(_CURATED.keys()) + _OL_SUBTYPES


# ── Pooled (coarse) cell-type groups ─────────────────────────────────────
# Pool the types the 101-gene Xenium panel cannot separate; keep apart what
# Tangram resolves. Evidence:
#  - The 6 Kenyon-cell subtypes are ~0.9 correlated on the panel (one expression
#    cluster) -> "Kenyon cells". (Calyx subregions are captured by in_kc_* anatomy.)
#  - The 5 glia subtypes are mutually similar/confused -> "Glia".
#  - The optic-lobe OL_* subclusters are pooled into a single "Optic lobe" group
#    (matches the reference atlas's single OL-neuron block). The fine structure
#    stays in ``tangram_cell_type``.
#  - Projection neurons, Other neurons, Photoreceptors, Hemocytes kept.
# Pools are for display; the fine ``tangram_cell_type`` is always retained.
_OL_ALL = {f"OL_{i}" for i in range(1, 14)}  # all optic-lobe subclusters -> one block

POOL_MAP: dict[str, str] = {
    "Large-type Kenyon cells": "Kenyon cells",
    "Middle-type Kenyon cells": "Kenyon cells",
    "Small-type Kenyon cells": "Kenyon cells",
    "Class II Kenyon cells": "Kenyon cells",
    "Other Kenyon cells": "Kenyon cells",
    "Kenyon cells": "Kenyon cells",
    "Anterodorsal projection neurons": "Projection neurons",
    "Lateral projection neurons": "Projection neurons",
    "Other neurons": "Other neurons",
    "Photoreceptors": "Photoreceptors",
    "Astrocyte-like glia": "Glia",
    "Cortex glia": "Glia",
    "Perineurial glia": "Glia",
    "Subperineurial glia": "Glia",
    "Other glia": "Glia",
    "Hemocytes": "Hemocytes",
    **{k: "Optic lobe" for k in _OL_ALL},
}

POOLED_ORDER = [
    "Kenyon cells",
    "Projection neurons",
    "Other neurons",
    "Photoreceptors",
    "Glia",
    "Hemocytes",
    "Optic lobe",
]

# ── the canonical 17: one vocabulary for every figure ────────────────────
# Default cell-type vocabulary for figures: the 16 annotated central-brain types
# plus OL_1..OL_13 (unannotated placeholders) collapsed into one "Optic lobe".
# Finer vocabularies (KC subclusters, OL_*) remain available where an analysis
# needs them.

CANONICAL_ORDER = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
    "Kenyon cells",
    "Anterodorsal projection neurons",
    "Lateral projection neurons",
    "Other neurons",
    "Photoreceptors",
    "Astrocyte-like glia",
    "Cortex glia",
    "Perineurial glia",
    "Subperineurial glia",
    "Other glia",
    "Hemocytes",
    "Optic lobe",
]

# fine name -> canonical name (identity for the 16, OL_* -> "Optic lobe")
CANONICAL_MAP: dict[str, str] = {
    **{n: n for n in CANONICAL_ORDER},
    **{k: "Optic lobe" for k in _OL_ALL},
}

_CANONICAL_COLORS: dict[str, str] = {
    **{
        n: SNRNA_CLUSTER_INFO[n]["color"]
        for n in CANONICAL_ORDER
        if n in SNRNA_CLUSTER_INFO
    },
    "Optic lobe": "#B94FA0",
    "Mixed": "#CBCBCB",
}


def get_canonical(name: str) -> str:
    """Map any fine cell_type to the canonical 17 (identity if already canonical)."""
    return CANONICAL_MAP.get(_norm_ct(name), _norm_ct(name))


def get_canonical_color(name: str) -> str:
    """Color for a canonical type (accepts a fine or canonical name)."""
    n = _norm_ct(name)
    return _CANONICAL_COLORS.get(n, _CANONICAL_COLORS.get(get_canonical(n), "#999999"))


def add_canonical(
    adata, src=("tangram_cell_type", "cell_type"), dst: str = "cell_type_canonical"
) -> str:
    """Add the canonical-17 cell-type column to ``adata.obs``.

    Returns the source column used.
    """
    import pandas as pd

    col = next(
        (c for c in ([src] if isinstance(src, str) else src) if c in adata.obs.columns),
        None,
    )
    if col is None:
        raise KeyError(f"none of {src} present in adata.obs")
    vals = [get_canonical(v) for v in adata.obs[col].astype(str)]
    present = [n for n in CANONICAL_ORDER if n in set(vals)]
    extra = sorted(set(vals) - set(present))
    adata.obs[dst] = pd.Categorical(vals, categories=present + extra)
    return col


# ── the canonical 15 ─────────────────────────────────────────────────────────
# The canonical 17 with two merges the 101-gene panel cannot resolve anyway:
#   "Other Kenyon cells" -> "Kenyon cells": the worst cross-modal type
#       (lowest per-cell-type concordance, and far rarer among Xenium cells than
#       among snRNA nuclei).
#   "Anterodorsal" + "Lateral projection neurons" -> "Projection neurons":
#       matches the five-class display pooling and the scCODA fit.
# The 15 is the default for figures; the 17 remains for finer analyses.

_KC_RESIDUAL = "Other Kenyon cells"
_PN_MERGED = "Projection neurons"
_PN_PARTS = ("Anterodorsal projection neurons", "Lateral projection neurons")

CANONICAL15_ORDER = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Kenyon cells",
    _PN_MERGED,
    "Other neurons",
    "Photoreceptors",
    "Astrocyte-like glia",
    "Cortex glia",
    "Perineurial glia",
    "Subperineurial glia",
    "Other glia",
    "Hemocytes",
    "Optic lobe",
]

CANONICAL15_MAP: dict[str, str] = {
    **{n: n for n in CANONICAL15_ORDER},
    **{k: "Optic lobe" for k in _OL_ALL},
    _KC_RESIDUAL: "Kenyon cells",
    **{p: _PN_MERGED for p in _PN_PARTS},
}


def get_canonical15(name: str) -> str:
    """Map any fine cell_type to the canonical 15 (see CANONICAL15_MAP)."""
    n = _norm_ct(name)
    return CANONICAL15_MAP.get(n, CANONICAL_MAP.get(n, n))


def add_canonical15(
    adata, src=("tangram_cell_type", "cell_type"), dst: str = "cell_type_canonical15"
) -> str:
    """Add the canonical-15 cell-type column to ``adata.obs``.

    Returns the source column used.
    """
    import pandas as pd

    col = next(
        (c for c in ([src] if isinstance(src, str) else src) if c in adata.obs.columns),
        None,
    )
    if col is None:
        raise KeyError(f"none of {src} present in adata.obs")
    vals = [get_canonical15(v) for v in adata.obs[col].astype(str)]
    present = [n for n in CANONICAL15_ORDER if n in set(vals)]
    extra = sorted(set(vals) - set(present))
    adata.obs[dst] = pd.Categorical(vals, categories=present + extra)
    return col


_POOLED_COLORS: dict[str, str] = {
    "Kenyon cells": "#C8781E",
    "Projection neurons": "#2F4E96",
    "Other neurons": "#48A6C9",
    "Photoreceptors": "#8E1F3A",
    "Glia": "#2E8B8B",
    "Hemocytes": _CURATED["Hemocytes"][1],
    "Optic lobe": "#B94FA0",
    "Mixed": "#CBCBCB",
}

_POOLED_SHORT: dict[str, str] = {
    "Kenyon cells": "KC",
    "Projection neurons": "PN",
    "Other neurons": "N-other",
    "Photoreceptors": "PR",
    "Glia": "Glia",
    "Hemocytes": "Hem",
    "Optic lobe": "OL",
    "Mixed": "Mixed",
}


def get_pooled(name: str) -> str:
    """Map a fine cell_type to its coarse pooled group (identity if unknown)."""
    return POOL_MAP.get(_norm_ct(name), _norm_ct(name))


def get_pooled_color(name: str) -> str:
    """Color for a pooled group (accepts a fine or pooled name)."""
    return _POOLED_COLORS.get(name, _POOLED_COLORS.get(get_pooled(name), "#999999"))


def get_pooled_short(name: str) -> str:
    g = name if name in _POOLED_SHORT else get_pooled(name)
    return _POOLED_SHORT.get(g, g)


# Display overrides for the vocabulary hierarchical_vote emits. The pooled
# "Glia" swatch collides exactly with Perineurial glia (#2E8B8B) and sits one
# step from Astrocyte-like glia (#5F9E6E), so a figure showing all three drew
# two of them identically. Kept inside the glial family -- they are all glia --
# but separated enough to tell apart in print.
DISPLAY_COLOR_OVERRIDES = {
    "Glia": "#9BC4C4",  # residual: pale teal, clearly the lightest
    "Astrocyte-like glia": "#5F9E6E",
    "Perineurial glia": "#2E8B8B",
}

# With its subtypes named beside it, the residual is not "all subtypes".
DISPLAY_NAME_OVERRIDES = {
    "Glia": "Glia (unresolved)",
}

# Glial entries adjacent, residual last of them, Unannotated/Mixed last overall,
# so the legend groups sensibly instead of wrapping a lone entry onto its own row.
DISPLAY_LEGEND_ORDER = [
    "Kenyon cells",
    "Optic lobe",
    "Optic lobe cells",
    "Astrocyte-like glia",
    "Perineurial glia",
    "Subperineurial glia",
    "Cortex glia",
    "Other glia",
    "Glia",
    "Photoreceptors",
    "Hemocytes",
    "Other neurons",
    "Projection neurons",
    "Mixed",
    "Unannotated",
    "Other",
]


def display_color(name, fallback=None):
    """Colour for a display category, honouring the overrides above."""
    if name in DISPLAY_COLOR_OVERRIDES:
        return DISPLAY_COLOR_OVERRIDES[name]
    return fallback(name) if fallback is not None else get_pooled_color(name)


def display_legend_sort(cats):
    """Order display categories so the glial block is contiguous."""
    idx = {n: i for i, n in enumerate(DISPLAY_LEGEND_ORDER)}
    return sorted(cats, key=lambda c: (idx.get(str(c), len(idx)), str(c)))


def _ranked(labels):
    """Label shares, largest first; ties broken by label name, missing dropped.

    ``value_counts`` alone breaks ties by first occurrence, which makes a vote
    depend on row order, and a missing label (NaN, "nan") would compete as a
    category of its own.
    """
    import pandas as _pd

    s = _pd.Series(labels, dtype=object)
    s = s[s.notna() & ~s.isin(["nan", "None", ""])]
    if s.empty:
        return _pd.Series(dtype=float)
    vc = s.value_counts(normalize=True)
    order = sorted(vc.index, key=lambda k: (-vc[k], str(k)))
    return vc.reindex(order)


def _labels(obs, label_col, pool):
    """Per-cell labels as an object array, NaN kept as NaN (not "nan")."""
    import numpy as _np

    raw = obs[label_col]
    lab = raw.astype(str).to_numpy(dtype=object)
    lab[raw.isna().to_numpy()] = None
    if pool:
        lab = _np.array(
            [None if v is None else get_pooled(v) for v in lab], dtype=object
        )
    return lab


#: Purity threshold for the current canonical mapping (default of
#: :func:`cluster_vote`). Calibrated, not hand-set: the threshold, scanned
#: 0.20-0.86 in 0.005 steps, that leaves the same share of cells unannotated
#: on ``paths.CANONICAL_TANGRAM`` as the hand-set 0.50 leaves on the
#: default-density mapping.
#: Recalibrate it whenever the canonical mapping changes.
CALIBRATED_MIN_PURITY = 0.410


def cluster_vote(
    adata,
    cluster_col: str = "leiden",
    label_col: str = "tangram_cell_type",
    pool: bool = True,
    min_purity: float = CALIBRATED_MIN_PURITY,
    mixed_label: str = "Mixed",
    dst: str = "cluster_vote",
):
    """Majority-vote a single (pooled) cell type per spatial cluster
    (the "smart voting").

    Assigns each ``cluster_col`` group its dominant ``label_col`` value (pooled if
    ``pool``); clusters whose top fraction < ``min_purity`` become ``mixed_label``.
    Writes the per-cell cluster label to ``adata.obs[dst]`` and returns the
    cluster -> (label, confidence) table. Retains the fine label; nothing deleted.

    label_col defaults to the combined-reference ``tangram_cell_type``, which
    is scored against the region-stitched labels on the Kenyon-cell/calyx
    correspondence; cluster voting absorbs the per-cell optic leakage that
    stitching was meant to fix.
    """
    import pandas as _pd

    df = _pd.DataFrame(
        {
            "cl": adata.obs[cluster_col].astype(str).values,
            "lab": _labels(adata.obs, label_col, pool),
        }
    )

    def _top(s):
        vc = _ranked(s.values)
        if vc.empty:
            return _pd.Series({"label": mixed_label, "conf": 0.0})
        return _pd.Series({"label": vc.index[0], "conf": float(vc.iloc[0])})

    tab = df.groupby("cl")["lab"].apply(_top).unstack()
    tab["voted"] = np.where(tab["conf"] >= min_purity, tab["label"], mixed_label)
    adata.obs[dst] = _pd.Categorical(df["cl"].map(tab["voted"]).values)
    return tab[["label", "conf", "voted"]]


# Lineages whose fine subtypes are never named in a display vocabulary because
# the panel cannot support the distinction: the Kenyon subclusters are defined
# by 150 marker genes, of which the 101-gene panel carries 7. Glia are not
# capped (perineurial and astrocyte-like clusters vote at 0.83 and 0.71).
CAP_TO_POOLED = ("Kenyon cells",)


def hierarchical_vote(
    adata,
    cluster_col: str = "leiden",
    label_col: str = "tangram_cell_type",
    min_purity: float = 0.5,
    lead_ratio: float = 1.5,
    cap_to_pooled=CAP_TO_POOLED,
    mixed_label: str = "Mixed",
    dst: str = "cluster_vote",
):
    """Per-cluster vote that names a fine type only where the data decides it.

    A flat purity floor alone is too strict for a 17-way vocabulary (chance is
    1/17 against 1/7 for the 7-way pool) and cannot tell a clear winner under
    half from a two-way tie.

    The rule, in order:
      1. Fine label if purity >= ``min_purity`` (and strictly ahead of the
         runner-up), or the winner leads the runner-up by >= ``lead_ratio``.
      2. Else the pooled label, if its purity >= ``min_purity`` (a cluster split
         between two glial subtypes is still glia).
      3. Else ``mixed_label``.

    Fine labels whose pooled group is in ``cap_to_pooled`` are then replaced by
    the group name (see CAP_TO_POOLED).

    Writes ``adata.obs[dst]`` and returns the per-cluster decision table,
    including the reason for each call.
    """
    import numpy as _np
    import pandas as _pd

    # Positional indexing, never .loc: the canonical object has 562,028
    # duplicate obs_names, so label-based lookups silently pull in unrelated rows.
    cl = adata.obs[cluster_col].astype(str)
    cl_v = cl.values
    raw = _labels(adata.obs, label_col, pool=False)
    fine_v = _np.array(
        [None if v is None else get_canonical(v) for v in raw], dtype=object
    )
    pooled_v = _np.array(
        [None if v is None else get_pooled(v) for v in fine_v], dtype=object
    )

    rows = []
    for c in _pd.unique(cl_v):
        m = cl_v == c
        f = _ranked(fine_v[m])
        p = _ranked(pooled_v[m])
        if f.empty:
            f, p = _pd.Series({mixed_label: 0.0}), _pd.Series({mixed_label: 0.0})
        f_top, f_frac = f.index[0], float(f.iloc[0])
        f_run = float(f.iloc[1]) if len(f) > 1 else 0.0
        p_top, p_frac = p.index[0], float(p.iloc[0])

        # A fine label must lead outright: an exact tie is undecided at that level.
        if f_frac >= min_purity and f_frac > f_run:
            lab, why = f_top, "fine purity %.3f" % f_frac
        elif f_run > 0 and f_frac / f_run >= lead_ratio:
            lab, why = f_top, "fine lead %.2fx" % (f_frac / f_run)
        elif p_frac >= min_purity:
            lab, why = p_top, "pooled purity %.3f (fine undecided)" % p_frac
        else:
            lab, why = (
                mixed_label,
                ("fine %.3f and pooled %.3f both undecided" % (f_frac, p_frac)),
            )

        if lab != mixed_label and get_pooled(lab) in cap_to_pooled:
            why += "; capped to %s" % get_pooled(lab)
            lab = get_pooled(lab)

        rows.append(
            {
                "cl": c,
                "n": int(m.sum()),
                "label": lab,
                "fine_top": f_top,
                "fine_purity": f_frac,
                "pooled_top": p_top,
                "pooled_purity": p_frac,
                "why": why,
            }
        )

    tab = _pd.DataFrame(rows).set_index("cl")
    adata.obs[dst] = _pd.Categorical(cl.map(tab["label"]).values)
    return tab


_OPTIC_FLAGS = ("in_lamina", "in_medulla", "in_lobula")
_CALYX_FLAGS = ("in_kc_ic", "in_kc_nc", "in_kc_oc")


def anatomy_guided_vote(
    adata,
    cluster_col: str = "leiden",
    label_col: str = "tangram_cell_type",
    pool: bool = True,
    min_purity: float = 0.5,
    region_thresh: float = 0.5,
    mixed_label: str = "Mixed",
    dst: str = "cluster_vote",
    base: str = "flat",
):
    """Anatomy-guided per-cluster vote, for method comparison only.

    Never use it for a registered figure: it reads the same ``in_*`` region flags
    those figures test against.

    For each spatial cluster:
      * if >``region_thresh`` of its cells fall in optic anatomy
        (``in_lamina|in_medulla|in_lobula``) -> "Optic lobe";
      * elif >``region_thresh`` fall in calyx anatomy
        (``in_kc_ic|in_kc_nc|in_kc_oc``) -> "Kenyon cells";
      * else the combined expression majority vote (``cluster_vote``), with
        clusters below ``min_purity`` -> ``mixed_label``.

    Uses the anatomical priors (mushroom-body calyx = Kenyon cells, optic lobe =
    OL) and lets expression decide elsewhere. In a comparison of voting variants
    this scored best (KC_F1 0.92, cleanest optic, lowest glia sink). Missing
    region flags degrade to a plain ``cluster_vote``. Writes ``adata.obs[dst]``;
    returns the cluster -> voted-label mapping (pandas Series).
    """
    import numpy as _np
    import pandas as _pd

    obs = adata.obs
    cl = obs[cluster_col].astype(str).values
    lab = _labels(obs, label_col, pool)

    def _mask(flags):
        m = _np.zeros(adata.n_obs, bool)
        for f in flags:
            if f in obs.columns:
                # untraced (NA) counts as outside the region, never inside
                m |= obs[f].astype("boolean").fillna(False).to_numpy(bool)
        return m

    optic = _mask(_OPTIC_FLAGS)
    calyx = _mask(_CALYX_FLAGS)
    if base == "hierarchical":
        # The canonical expression-only vote, pooled, as the base: the override
        # is then the only difference from the canonical labels.
        tab = hierarchical_vote(
            adata,
            cluster_col=cluster_col,
            label_col=label_col,
            min_purity=min_purity,
            mixed_label=mixed_label,
            dst="_cv_tmp",
        )
        base_map = tab["label"].map(lambda v: v if v == mixed_label else get_pooled(v))
    elif base == "flat":
        base_map = cluster_vote(
            adata,
            cluster_col=cluster_col,
            label_col=label_col,
            pool=pool,
            min_purity=min_purity,
            mixed_label=mixed_label,
            dst="_cv_tmp",
        )["voted"]
    else:
        raise ValueError(f"base must be 'flat' or 'hierarchical', not {base!r}")
    df = _pd.DataFrame({"cl": cl, "optic": optic, "calyx": calyx, "lab": lab})
    cl_optic = df.groupby("cl")["optic"].mean()
    cl_calyx = df.groupby("cl")["calyx"].mean()

    voted = {}
    for c in cl_optic.index:
        if cl_optic[c] > region_thresh:
            voted[c] = "Optic lobe"
        elif cl_calyx[c] > region_thresh:
            voted[c] = "Kenyon cells"
        else:
            voted[c] = base_map.get(c, mixed_label)
    del adata.obs["_cv_tmp"]
    adata.obs[dst] = _pd.Categorical(_pd.Series(cl).map(voted).values)
    return _pd.Series(voted, name="voted")


def _generate_n_colors(n: int) -> list[str]:
    """Generate n maximally-separated colors via golden-angle hue spacing."""
    colors = []
    golden_angle = 137.508 / 360.0
    for i in range(n):
        hue = (i * golden_angle) % 1.0
        lit = 0.45 if i % 2 == 0 else 0.55
        sat = 0.75 if i % 3 != 0 else 0.85
        r, g, b = colorsys.hls_to_rgb(hue, lit, sat)
        colors.append(f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    return colors


# ── Spatial cluster info (32 Leiden clusters) ──────────────────────────────
# Placeholder — names/shorts are None until annotation.
# Colors generated independently via golden-angle spacing: a palette of
# SPATIAL_N_CLUSTERS colours, cycled by cluster id (so cluster 31 shares
# cluster 0's colour).

SPATIAL_N_CLUSTERS = 31

_SPATIAL_PALETTE = _generate_n_colors(SPATIAL_N_CLUSTERS)


# Per-cell labels come from the ``tangram_cell_type`` column directly.


# ── snRNA accessors ──────────────────────────────────────────────────────


def get_short(cluster_id) -> str:
    """Short label, e.g. "KC-l" or "Astro". Tolerates already-short strings."""
    if isinstance(cluster_id, str):
        info = SNRNA_CLUSTER_INFO.get(cluster_id)
        if info:
            return info["short"]
        # Maybe it's already a short label
        for v in SNRNA_CLUSTER_INFO.values():
            if v["short"] == cluster_id:
                return cluster_id
        return cluster_id
    # Integer cluster id
    return f"C{cluster_id}"


def get_color(cluster_id) -> str:
    """Hex color for *cluster_id*. Accepts cell type string or short label."""
    if isinstance(cluster_id, str):
        info = SNRNA_CLUSTER_INFO.get(cluster_id)
        if info:
            return info["color"]
        # Reverse lookup by short name
        for v in SNRNA_CLUSTER_INFO.values():
            if v["short"] == cluster_id:
                return v["color"]
        return "#999999"
    # Integer cluster id
    return "#999999"


# ── Figure display names ───────────────────────────────────────────────────
# Display labels keep each term's meaning consistent across figures; stored
# values in obs/CSVs are unchanged.
#  1. Pooled labels that reuse a fine type's words (e.g. "Kenyon cells", both a
#     pool of six and one fine type) carry an "(all subtypes)" marker.
#  2. The fine "Kenyon cells" entry is a residual (662 unresolved optic-lobe
#     nuclei) that Tangram assigns to many spatial cells, so it is shown as
#     "Kenyon cells (unclassified)".
# Use get_pooled_display() for get_pooled()/cluster_vote values and
# get_canonical_display() for canonical-17 values; they differ on "Kenyon cells".

_POOLED_DISPLAY: dict[str, str] = {
    "Kenyon cells": "Kenyon cells (all subtypes)",
    "Glia": "Glia (all subtypes)",
    "Projection neurons": "Projection neurons (all subtypes)",
    "Optic lobe": "Optic lobe cells",
}

_CANONICAL_DISPLAY: dict[str, str] = {
    "Kenyon cells": "Kenyon cells (unclassified)",
    "Optic lobe": "Optic lobe cells",
}

# "Optic lobe cells" distinguishes the pooled OL_* cell-type label from the
# anatomical optic lobe (lamina + medulla + lobula); a cell can carry one
# without being in the other. Sentence case matches the sibling display labels.
# Anatomical region names do not pass through these display functions.

# "Mixed" (the internal value in obs['cluster_vote'], cached CSVs/h5ads and
# `== "Mixed"` comparisons) means no confident majority label for the spatial
# cluster, mostly because such cells lie outside every hand-drawn region.
# It is not a cell type and not a quality problem (these cells carry
# more transcripts and genes than labelled ones). Only the rendered label is
# set here, so every figure uses the same wording. "unassigned" is avoided
# because si_regional_degs.py already uses it as a region label.
MIXED_DISPLAY = "Other"


def get_pooled_display(name: str) -> str:
    """Figure label for a pooled group (get_pooled / cluster_vote output)."""
    n = _norm_ct(name)
    if n == "Mixed":
        return MIXED_DISPLAY
    return _POOLED_DISPLAY.get(n, n)


def get_canonical_display(name: str) -> str:
    """Figure label for a canonical-17 type."""
    n = _norm_ct(name)
    if n == "Mixed":
        return MIXED_DISPLAY
    return _CANONICAL_DISPLAY.get(n, n)


# ── snRNA bulk helpers ─────────────────────────────────────────────────────


def cluster_colors(clusters) -> np.ndarray:
    """Map an array of cluster labels (str) to hex color strings."""
    return np.array([get_color(c) for c in clusters])


# ── Spatial accessors ──────────────────────────────────────────────────────


def get_spatial_short(cluster_id) -> str:
    """Short label for a spatial cluster. Tolerates already-named strings."""
    try:
        cid = int(cluster_id)
    except (ValueError, TypeError):
        return str(cluster_id)
    return f"S{cid}"


def get_spatial_color(cluster_id) -> str:
    """Hex color for a spatial cluster. Tolerates already-named strings."""
    try:
        cid = int(cluster_id)
    except (ValueError, TypeError):
        return "#999999"
    return _SPATIAL_PALETTE[cid % SPATIAL_N_CLUSTERS]
