"""Per-region cell-type diversity: how transcriptomically mixed is each
anatomical neighbourhood, and does that depend on which label vocabulary
or how many cells you look at?

Regions are STATISTICAL_REGIONS in ``nine_region_common.py``: the three
optic-lobe layers (lamina, medulla, lobula), the three Kenyon-cell calyx
compartments (kc_ic, kc_nc, kc_oc) and the antennal lobe (al). The
central-complex regions (DESCRIPTIVE_ONLY_REGIONS, very rare) would
degrade every other estimate and are excluded. Only cells with exactly one
``in_*`` flag are used (the rest fall in no annotated region and are excluded,
not folded into "unassigned"). Every table states its own N.

Label circularity. The regions were drawn from DAPI morphology, independent
of the transcriptome, but ``anatomy_guided_vote`` (kept here only as a
labelled comparison) assigns
"Kenyon cells" / "Optic lobe" to a cluster when >50% of its cells lie
in calyx / optic anatomy, i.e. from the same flags this script conditions on.
Diversity is therefore computed under several label schemes:
  1. ``anatomy``      -- anatomy_guided_vote, pooled groups. Comparison only.
  2. ``expr``         -- hierarchical_vote (the figures' rule, same
                         arguments as caste_comparison_sccoda.py), pooled to the
                         POOLED_ORDER groups (+ Mixed), no anatomy step.
                         Primary.
  3. ``leiden``       -- raw Leiden cluster id, no cell-type semantics. Primary.
  4. ``expr_nomixed`` -- ``expr`` with "Mixed" cells dropped and the remaining
                         categories re-rarefied to their own (smaller) target.
Scheme 1 does not collapse richness but suppresses Shannon unevenly, mostly
in the calyx; read headline claims off schemes 2 and 3.

Mixed-excluded scheme. Under ``hierarchical_vote``, Mixed means a cluster
decided at neither the fine nor the pooled level (a statement about panel
resolution) -- the same Mixed fig:caste_comparison shows. ``expr``
(Mixed included) is primary because the within-region Mixed share is small
next to the atlas-wide share; ``expr_nomixed`` shows which parts of the
region ordering survive excluding it. Where the two schemes order the least
diverse regions differently, read them as "order not resolved".

Label-free check: mean pairwise Euclidean distance between cells' first 20
PCA coordinates, independent of any vocabulary.

Abundance. Regions range from 5,227 (lobula) to 144,802 (KC-NC) cells. Every
compositional metric (Shannon, richness, evenness) is rarefied to a common N
(the smallest region) and averaged over ``N_BOOT`` subsamples with a 95% CI.
Unrarefied values and a richness-vs-depth curve are also written.

Caste. One queen and one worker brain, so any caste split is descriptive only
(no p-values). Per-section spread (30 queen / 35 worker) is shown; sections
are not biological replicates.

Labels come from the canonical Tangram run (paths.CANONICAL_TANGRAM). Only the
``expr``/``anatomy`` schemes read the cell-type labels; region flags, Leiden
and PCA do not depend on them.

Hill numbers. ``qD = (sum p_i^q)^(1/(1-q))`` for q != 1 and ``exp(Shannon)``
at q=1: effective numbers of equally common types (q=0 richness, q=2 inverse
Simpson). The q-profile shows whether caste orderings are stable across q;
see ``hill_diversity_profile.csv``.

Beta diversity + PERMANOVA. Region composition vectors (``label_expr`` pooled
types, naive N, castes pooled) give Bray-Curtis and the Hill multiplicative
decomposition (beta = gamma/alpha, Chiu/Jost/Chao 2014) at q=0,1,2.
PERMANOVA uses Aitchison distance (Euclidean on CLR-transformed composition,
as in ``pipelines/wholebrain/queen/compositional_analysis.py::clr``) with
section (region x brain x slice_name) as the replicate unit. Region is
replicated; caste is aliased with brain identity, so ``permanova_caste.csv``
reports brain differences and carries a caveat column on every row.

Run:
    PYTHONPATH=src .venv/bin/python scripts/regions/region_diversity.py

Outputs -> results/region_diversity/:
    region_counts.csv          denominators at every level, plus
                               n_mixed_expr/pct_mixed_expr per region
    diversity_rarefied.csv     region x caste x scheme, rarefied
                               Shannon/richness/evenness (+CI)
    diversity_unrarefied.csv   same cells, naive (uncorrected) values
    rarefaction_curves.csv     richness vs subsample depth (scheme=expr, caste=both)
    heterogeneity_rarefied.csv region x caste, rarefied mean-pairwise-PCA-distance
                               (label-free)
    per_section_diversity.csv  region x brain x slice_name, unrarefied
    hill_diversity_profile.csv region x caste x scheme, rarefied Hill numbers over q in
                               [0,3] step 0.25 (+CI at every q)
    per_section_composition.csv region x brain x slice_name x label_expr category
                               proportions (the PERMANOVA input)
    beta_diversity_matrix.csv  region x region long-form: bray_curtis +
                               hill_beta_q{0,1,2}
    beta_diversity_overall.csv gamma/alpha/beta per q=0,1,2 for all regions together
    permanova_region.csv       one-way PERMANOVA, predictor=region, replicate=section
    permanova_caste.csv        one-way PERMANOVA, predictor=caste (overall + within each
                               region), confounded with brain identity (caveat column)
    region_diversity_main.png  full 6-panel methods exhibit (not registered)
    region_diversity_caste.png registered single-panel figure: rarefied Shannon
                               diversity per region, queen vs worker
    fig_s_regional_diversity.png  identical copy of region_diversity_caste.png under
                               the registered SI basename. The other analyses are in
                               region_diversity_main.png and the CSVs.
    hill_diversity_profile.png Hill-number profile figure: (A) q-profile per region,
                               castes pooled; (B) queen vs worker profile per region;
                               (C) region x region Bray-Curtis heatmap. Exploratory.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import (  # noqa: E402
    POOLED_ORDER,
    anatomy_guided_vote,
    get_pooled,
    hierarchical_vote,
)
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (  # noqa: E402
    DESCRIPTIVE_ONLY_REGIONS,
    STATISTICAL_REGIONS,
    attach_region_flags,
)
from bimp_st.si_common import (  # noqa: E402
    LEGEND_KW,
    QUEEN_COLOR,
    WORKER_COLOR,
    set_style,
)
from bimp_st.si_common import (
    save as _save,
)

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
# One file carries leiden, tangram_cell_type, the original six in_* flags,
# brain, slice_name and X_pca, so no cross-file joins are needed. in_al is not
# stored in it and is attached at load time via
# nine_region_common.attach_region_flags (see main()). This is the corrected
# Tangram object; see the module docstring.
H5AD = CANONICAL_TANGRAM
OUTPUT_DIR = REPO / "results" / "region_diversity"

# The statistical vocabulary (nine_region_common.STATISTICAL_REGIONS). The
# descriptive-only central-complex regions must never appear in these three
# structures; that is what keeps them out of every computation below. The
# display order (optic, calyx, antennal lobe) differs from STATISTICAL_REGIONS,
# so it is an explicit list and the asserts check membership only.
REGION_FLAGS = [
    "in_lamina",
    "in_medulla",
    "in_lobula",
    "in_kc_ic",
    "in_kc_nc",
    "in_kc_oc",
    "in_al",
]
REGION_NICE = {
    "in_lamina": "Lamina",
    "in_medulla": "Medulla",
    "in_lobula": "Lobula",
    "in_kc_ic": "Small-type",
    "in_kc_nc": "Large-type",
    "in_kc_oc": "Class II",
    "in_al": "Antennal lobe",
}
REGION_ORDER = [
    "Lamina",
    "Medulla",
    "Lobula",
    "Small-type",
    "Large-type",
    "Class II",
    "Antennal lobe",
]
assert set(REGION_FLAGS) == {f"in_{r}" for r in STATISTICAL_REGIONS}, (
    "REGION_FLAGS has drifted from nine_region_common.STATISTICAL_REGIONS"
)
assert not (
    set(DESCRIPTIVE_ONLY_REGIONS) & {f.removeprefix("in_") for f in REGION_FLAGS}
), (
    "ellipsoid_body/fan_shaped_body are descriptive-only and must never enter "
    "the statistical region set"
)
# Region colours match the fig:atlas_overview region palette (also
# nine_region_common.REGION_COLORS), so a region has the same colour wherever
# it appears (the main figures and fig_s_hill_diversity read this dict). They
# are distinct from the cell-type colours. The antennal lobe uses Tableau-10
# teal (#76B7B2), the one swatch in that family not already taken.
REGION_GROUP_COLOR = {
    "Lamina": "#59A14F",
    "Medulla": "#4E79A7",
    "Lobula": "#E15759",
    "Small-type": "#B07AA1",
    "Large-type": "#F28E2B",
    "Class II": "#EDC948",
    "Antennal lobe": "#76B7B2",
}

N_BOOT = 200  # rarefaction reps for compositional metrics (cheap: counting)
N_BOOT_CURVE = 50  # reps per depth for the rarefaction curve
N_BOOT_HET = 50  # reps for the pairwise-distance heterogeneity metric (O(n^2))
N_HET = 2000  # subsample size for heterogeneity (< smallest region, 5227)
N_PCS = 20  # PCA dims used for the expression-space distance
RICHNESS_FLOOR = 0.01  # "common-type" richness: relative abundance > 1%
MIN_SECTION_CELLS = 30  # per-section diversity only computed above this N
SEED = 0

# Hill-number order grid: q=0 (richness), 1 (exp-Shannon), 2 (inverse Simpson)
# are exact points on this grid (0.25 is exactly representable in binary).
Q_GRID = tuple(round(q, 2) for q in np.arange(0.0, 3.01, 0.25))
N_PERM = 999  # PERMANOVA permutations

# ── Diversity primitives ─────────────────────────────────────────────────────


def shannon(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot == 0:
        return np.nan
    p = counts[counts > 0].astype(float) / tot
    return float(-(p * np.log(p)).sum())


def richness(counts: np.ndarray, floor: float = 0.0) -> int:
    tot = counts.sum()
    if tot == 0:
        return 0
    if floor <= 0:
        return int((counts > 0).sum())
    return int((counts / tot > floor).sum())


def pielou(h: float, s: int) -> float:
    if s <= 1 or not np.isfinite(h):
        return np.nan
    return h / np.log(s)


def _summarize(vals) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan, np.nan, np.nan
    return (
        float(np.nanmean(arr)),
        float(np.nanpercentile(arr, 2.5)),
        float(np.nanpercentile(arr, 97.5)),
    )


def rarefied_diversity(
    codes: np.ndarray,
    n_cats: int,
    n_target: int,
    rng: np.random.Generator,
    b: int = N_BOOT,
) -> dict:
    """Subsample ``n_target`` cells (no replacement) ``b`` times; mean+95% CI."""
    n = len(codes)
    if n < n_target:
        return {
            k: (np.nan, np.nan, np.nan)
            for k in ("shannon", "richness", "richness_1pct", "evenness")
        }
    h_l, r0_l, r1_l, ev_l = [], [], [], []
    for _ in range(b):
        sub = rng.choice(n, size=n_target, replace=False)
        counts = np.bincount(codes[sub], minlength=n_cats)
        h = shannon(counts)
        r0 = richness(counts, 0.0)
        h_l.append(h)
        r0_l.append(r0)
        r1_l.append(richness(counts, RICHNESS_FLOOR))
        ev_l.append(pielou(h, r0))
    return {
        "shannon": _summarize(h_l),
        "richness": _summarize(r0_l),
        "richness_1pct": _summarize(r1_l),
        "evenness": _summarize(ev_l),
    }


def hill_number(p_or_counts: np.ndarray, q: float, from_counts: bool = True) -> float:
    """Hill diversity of order q: qD = (sum p_i^q)^(1/(1-q)), qD = exp(Shannon) at q=1.

    "True diversity" in units of an effective number of equally-common
    categories -- q=0 is richness, q=1 weights each type by its entropy
    contribution (exp of Shannon entropy), q=2 is the inverse Simpson
    concentration index. Unlike raw Shannon/Simpson, all q sit on one
    directly-comparable numeric scale.
    """
    if from_counts:
        tot = p_or_counts.sum()
        if tot == 0:
            return np.nan
        p = p_or_counts[p_or_counts > 0].astype(float) / tot
    else:
        p = np.asarray(p_or_counts, dtype=float)
        p = p[p > 0]
    if p.size == 0:
        return np.nan
    if abs(q - 1.0) < 1e-9:
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p**q).sum() ** (1.0 / (1.0 - q)))


def rarefied_hill_profile(
    codes: np.ndarray,
    n_cats: int,
    n_target: int,
    rng: np.random.Generator,
    q_grid=Q_GRID,
    b: int = N_BOOT,
) -> dict:
    """Rarefied Hill-number profile over ``q_grid``; mean + 95% CI per q.

    Reuses one bootstrap draw per rep for every q (cheap: a bincount + a
    handful of power sums), mirroring ``rarefied_diversity``'s subsampling
    so the two are computed on matched resamples if called with the same
    rng state.
    """
    n = len(codes)
    if n < n_target:
        return {q: (np.nan, np.nan, np.nan) for q in q_grid}
    vals = {q: [] for q in q_grid}
    for _ in range(b):
        sub = rng.choice(n, size=n_target, replace=False)
        counts = np.bincount(codes[sub], minlength=n_cats)
        for q in q_grid:
            vals[q].append(hill_number(counts, q))
    return {q: _summarize(v) for q, v in vals.items()}


def naive_diversity(codes: np.ndarray, n_cats: int) -> dict:
    counts = np.bincount(codes, minlength=n_cats)
    h = shannon(counts)
    r0 = richness(counts, 0.0)
    return {
        "n_cells": int(counts.sum()),
        "shannon": h,
        "richness": r0,
        "richness_1pct": richness(counts, RICHNESS_FLOOR),
        "evenness": pielou(h, r0),
    }


def heterogeneity(
    x_pca: np.ndarray, rng: np.random.Generator, n_het: int = N_HET, b: int = N_BOOT_HET
) -> dict:
    """Label-free within-region dispersion: mean pairwise Euclidean distance
    and centroid dispersion (mean squared distance to centroid) over the
    first N_PCS PCA dims, rarefied to a common N."""
    n = x_pca.shape[0]
    n_use = min(n_het, n)
    mpd, disp = [], []
    for _ in range(b):
        idx = rng.choice(n, size=n_use, replace=False)
        xs = x_pca[idx]
        d = pdist(xs)
        mpd.append(d.mean())
        c = xs.mean(axis=0)
        disp.append(float(np.mean(np.sum((xs - c) ** 2, axis=1))))
    return {
        "n_used": n_use,
        "mean_pairwise_dist": _summarize(mpd),
        "centroid_dispersion": _summarize(disp),
    }


# ── Compositional geometry + beta diversity + PERMANOVA ─────────────────────
# CLR/multiplicative-replacement match this project's existing CoDA convention
# verbatim (delta = 0.65 * smallest nonzero value; Martin-Fernandez et al.
# 2003) -- see pipelines/wholebrain/queen/compositional_analysis.py::clr.
# Reimplemented
# here (not imported) so this script has no dependency on either pipeline dir.


def _multiplicative_replacement(
    x: np.ndarray, delta: float | None = None
) -> np.ndarray:
    x = np.atleast_2d(x).astype(float)
    if delta is None:
        nz = x[x > 0]
        delta = 0.65 * nz.min() if len(nz) > 0 else 1e-10
    out = x.copy()
    for i in range(x.shape[0]):
        row = x[i]
        n_zeros = (row == 0).sum()
        if n_zeros > 0:
            out[i, row == 0] = delta
            out[i, row > 0] = row[row > 0] * (1 - n_zeros * delta / row.sum())
    return out


def clr(x: np.ndarray) -> np.ndarray:
    """Centered log-ratio: CLR(x)_i = log(x_i / geometric_mean(x))."""
    x = np.atleast_2d(x)
    if (x <= 0).any():
        x = _multiplicative_replacement(x)
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def multiplicative_hill_beta(
    prop_matrix: np.ndarray, q: float
) -> tuple[float, float, float]:
    """Chiu/Jost/Chao (2014) gamma/alpha/beta decomposition, N equally-weighted
    assemblages (rows of ``prop_matrix``, each a relative-abundance vector).

    gamma_q  = Hill_q of the row-averaged ("pooled") distribution
    alpha_q  = generalized (power) mean of the per-assemblage Hill_q values,
               geometric mean in the q=1 limit
    beta_q   = gamma_q / alpha_q  (1 = identical composition, N = fully
               distinct with no overlap; N = prop_matrix.shape[0])

    Works for N=2 (pairwise, used for the region x region matrix) or N>2
    (used once for the "all seven regions as one assemblage" overall beta).
    """
    prop_matrix = np.atleast_2d(prop_matrix)
    n_assemblages = prop_matrix.shape[0]
    p_bar = prop_matrix.mean(axis=0)
    gamma = hill_number(p_bar, q, from_counts=False)
    d = np.array(
        [
            hill_number(prop_matrix[j], q, from_counts=False)
            for j in range(n_assemblages)
        ]
    )
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0 or not np.isfinite(gamma):
        return np.nan, np.nan, np.nan
    if abs(q - 1.0) < 1e-9:
        alpha = float(np.exp(np.mean(np.log(d))))
    else:
        alpha = float(np.mean(d ** (1.0 - q)) ** (1.0 / (1.0 - q)))
    beta = gamma / alpha if alpha > 0 else np.nan
    return gamma, alpha, beta


def permanova(
    dist: np.ndarray,
    groups: np.ndarray,
    n_perm: int = N_PERM,
    rng: np.random.Generator | None = None,
) -> dict:
    """One-way PERMANOVA (Anderson 2001) on a precomputed square distance matrix.

    Exact sum-of-squares partition when ``dist`` is Euclidean, as Aitchison
    distance (Euclidean on CLR coordinates) is. F = (SS_among/df_among) /
    (SS_within/df_within); p from ``n_perm`` label permutations
    (right-tailed, +1/+1 correction).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    groups = np.asarray(groups)
    n = dist.shape[0]
    uniq = np.unique(groups)
    k = len(uniq)
    d2 = dist**2

    def _ss_within(g: np.ndarray) -> float:
        ss = 0.0
        for u in uniq:
            idx = np.flatnonzero(g == u)
            ng = len(idx)
            if ng < 2:
                continue
            ss += d2[np.ix_(idx, idx)].sum() / (2.0 * ng)
        return ss

    ss_t = d2.sum() / (2.0 * n)
    ss_w = _ss_within(groups)
    ss_a = ss_t - ss_w
    df_a, df_w = k - 1, n - k
    f_obs = (ss_a / df_a) / (ss_w / df_w) if ss_w > 0 and df_w > 0 else np.nan

    f_perm = np.empty(n_perm)
    for i in range(n_perm):
        gp = rng.permutation(groups)
        ssw_p = _ss_within(gp)
        ssa_p = ss_t - ssw_p
        f_perm[i] = (
            (ssa_p / df_a) / (ssw_p / df_w) if ssw_p > 0 and df_w > 0 else np.nan
        )
    valid = np.isfinite(f_perm)
    p = (
        (1 + np.sum(f_perm[valid] >= f_obs)) / (1 + valid.sum())
        if np.isfinite(f_obs)
        else np.nan
    )

    return {
        "F": f_obs,
        "p": p,
        "R2": ss_a / ss_t if ss_t > 0 else np.nan,
        "n": n,
        "k": k,
        "df_among": df_a,
        "df_within": df_w,
        "n_perm": n_perm,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Separate generator for the Hill-number profile so it does not shift the
    # `rng` draw sequence used afterwards (rarefaction curves, heterogeneity,
    # per-section subsampling), keeping those reproducible for a given SEED.
    rng_hill = np.random.default_rng(SEED + 1)

    print(f"Loading {H5AD} ...", flush=True)
    a = ad.read_h5ad(H5AD)
    a.obs_names_make_unique()
    print(f"  {a.n_obs:,} cells total", flush=True)

    # Attach all seven region flags via nine_region_common (joins on
    # slice_name + cell_id against the crop sidecars). H5AD has no in_al, and
    # its stored six predate the self-intersecting-polygon repair (KC-NC
    # 98,831 vs 99,270 in coverage.json), so all seven are replaced.
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    obs = a.obs

    # ── Region assignment: cells in exactly one annotated region ───────────
    # in_al is a nullable boolean: pd.NA means "no antennal-lobe tracing on
    # this section", not "not antennal lobe". It is NA for 377,619 of 610,376
    # worker cells (62%, 22 slice_names); the queen is fully traced.
    # .astype(bool) raises on NA, and .fillna(False) would invent region calls.
    #
    # Only undecidable cells are dropped. On traced sections the antennal lobe
    # never overlaps the six original regions (0/314,606 six-region cells are
    # in_al; 0/10,348 in_al cells hit the six), so a cell with a hit in the six
    # is unambiguous even on an untraced section. Only cells that hit none of
    # the six on an untraced section are ambiguous. Dropping every AL-unknown
    # cell would instead shrink worker-only single-region counts (133,695
    # cells) relative to the queen.
    six_flags = [f for f in REGION_FLAGS if f != "in_al"]
    al_idx = REGION_FLAGS.index("in_al")
    six_idx = [REGION_FLAGS.index(f) for f in six_flags]

    al_known = obs["in_al"].notna().values
    region_mat = np.zeros((a.n_obs, len(REGION_FLAGS)), dtype=bool)
    region_mat[:, six_idx] = obs[six_flags].astype(bool).values  # never null
    region_mat[al_known, al_idx] = obs.loc[al_known, "in_al"].astype(bool).values

    s6_sum = region_mat[:, six_idx].sum(axis=1)
    n_regions_hit = region_mat.sum(axis=1)  # AL column is a zero placeholder
    # for AL-unknown rows; only read
    # where `ambiguous` is False below
    ambiguous = (~al_known) & (s6_sum == 0)
    single = ~ambiguous & (n_regions_hit == 1)
    none_hit = ~ambiguous & (n_regions_hit == 0)
    multi_hit = ~ambiguous & (n_regions_hit > 1)
    assert single.sum() + none_hit.sum() + multi_hit.sum() + ambiguous.sum() == a.n_obs
    print(
        f"  exactly one region: {single.sum():,} | none: {none_hit.sum():,} "
        f"| >1 (dropped, ambiguous): {multi_hit.sum():,} | AL status "
        f"unresolvable (dropped, in_al NA and no hit in the other six): "
        f"{ambiguous.sum():,}",
        flush=True,
    )

    region_col = np.full(a.n_obs, "", dtype=object)
    region_col[single] = np.array(
        [REGION_NICE[REGION_FLAGS[i]] for i in region_mat[single].argmax(1)]
    )
    obs = obs.assign(region=region_col)

    # ── Three label schemes, computed on the full object (cluster-level
    # anatomy fractions must see all of a cluster's cells, not just the
    # region-restricted subset) ─────────────────────────────────────────────
    print("Computing label schemes (anatomy / expr / leiden) ...", flush=True)
    anatomy_guided_vote(
        a,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        pool=True,
        dst="label_anatomy",
    )
    # The canonical vote, with the arguments caste_comparison_sccoda.py uses, then
    # pooled: diversity is reported over the pooled groups,
    # so the glial subtypes the vote names count as one "Glia" here.
    hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst="label_expr_fine"
    )
    a.obs["label_expr"] = pd.Categorical(
        [get_pooled(v) for v in a.obs["label_expr_fine"].astype(str)]
    )
    _unexpected = set(a.obs["label_expr"].astype(str)) - set(POOLED_ORDER) - {"Mixed"}
    assert not _unexpected, f"labels outside the pooled vocabulary: {_unexpected}"
    print(
        "  hierarchical_vote labels (cells):\n"
        + a.obs["label_expr_fine"].astype(str).value_counts().to_string()
        + "\n  pooled (label_expr):\n"
        + a.obs["label_expr"].astype(str).value_counts().to_string(),
        flush=True,
    )
    obs = a.obs  # both votes wrote into a.obs in place
    obs["region"] = region_col
    obs["label_leiden"] = obs["leiden"].astype(str)

    schemes = {
        "anatomy": {
            "col": "label_anatomy",
            "circular": True,
            "desc": "anatomy_guided_vote (pooled) -- CIRCULAR",
        },
        "expr": {
            "col": "label_expr",
            "circular": False,
            "desc": "hierarchical_vote, pooled, expression-only -- not circular",
        },
        "expr_nomixed": {
            "col": "label_expr",
            "circular": False,
            "desc": (
                "hierarchical_vote, expression-only, MIXED "
                "EXCLUDED and the remaining categories "
                "renormalized -- see the MIXED-EXCLUDED "
                "SCHEME note in the module docstring"
            ),
        },
        "leiden": {
            "col": "label_leiden",
            "circular": False,
            "desc": "raw Leiden cluster id -- not circular, no cell-type semantics",
        },
    }
    for tag, s in schemes.items():
        cats = sorted(obs[s["col"]].astype(str).unique())
        if tag == "expr_nomixed":
            cats = [c for c in cats if c != "Mixed"]
        s["cats"] = cats
        s["codes"] = pd.Categorical(
            obs[s["col"]].astype(str), categories=cats
        ).codes.astype(np.int64)
        # Cells whose label isn't in `cats` (only possible for expr_nomixed,
        # where "Mixed" cells code -1) are excluded from this scheme
        # entirely, not folded into anything -- keep_mask records that.
        s["keep_mask"] = s["codes"] >= 0
        n_excl = int((~s["keep_mask"]).sum())
        extra = f", {n_excl:,} cells excluded" if n_excl else ""
        print(f"  [{tag}] {s['desc']}: {len(s['cats'])} categories{extra}", flush=True)

    sub = obs.loc[single]
    region_n = sub["region"].value_counts()
    n_rarefy = int(region_n.min())
    smallest_region = region_n.idxmin()
    print(
        f"\nRarefaction target N (caste='both') = {n_rarefy} "
        f"(smallest region: {smallest_region})",
        flush=True,
    )

    # Per-caste target must be smaller: the smallest single region x caste
    # cell count (worker lobula), not the combined-caste minimum -- queen and
    # worker lobula are both < n_rarefy, so reusing n_rarefy for the caste
    # split would silently return all-NaN (and an empty bar) for lobula.
    region_caste_n = sub.groupby(["region", "brain"], observed=True).size()
    n_rarefy_caste = int(region_caste_n.min())
    smallest_rc = region_caste_n.idxmin()
    print(
        f"Rarefaction target N (per-caste) = {n_rarefy_caste} "
        f"(smallest region x caste: {smallest_rc})",
        flush=True,
    )

    # expr_nomixed's own rarefaction targets: dropping Mixed cells shrinks
    # every region x caste N, so reusing n_rarefy/n_rarefy_caste here could
    # silently over-request and return an all-NaN row (see the comment above)
    # for whichever region x caste cell loses the most cells to Mixed.
    sub_nomixed = sub.loc[sub["label_expr"].astype(str) != "Mixed"]
    region_n_nomixed = sub_nomixed["region"].value_counts()
    n_rarefy_nomixed = int(region_n_nomixed.min())
    smallest_region_nomixed = region_n_nomixed.idxmin()
    print(
        f"Rarefaction target N (caste='both', expr_nomixed) = {n_rarefy_nomixed} "
        f"(smallest region: {smallest_region_nomixed})",
        flush=True,
    )
    region_caste_n_nomixed = sub_nomixed.groupby(
        ["region", "brain"], observed=True
    ).size()
    n_rarefy_caste_nomixed = int(region_caste_n_nomixed.min())
    smallest_rc_nomixed = region_caste_n_nomixed.idxmin()
    print(
        f"Rarefaction target N (per-caste, expr_nomixed) = {n_rarefy_caste_nomixed} "
        f"(smallest region x caste: {smallest_rc_nomixed})",
        flush=True,
    )
    # Mixed's share within each region (single-region cells only) -- the
    # number that decides whether excluding it can plausibly change the
    # panel D story or is numerically near-moot; printed here so a run's
    # console log carries it without opening region_counts.csv.
    mixed_mask_full = (obs["label_expr"].astype(str) == "Mixed").values
    print(
        "\nMixed share within each region (single-region cells, both castes):",
        flush=True,
    )
    for nice in REGION_ORDER:
        m_r = (region_col == nice) & single
        n_r = int(m_r.sum())
        n_mix = int((m_r & mixed_mask_full).sum())
        print(
            f"  {nice:8s} {n_mix:>7,} / {n_r:>7,} = {100 * n_mix / n_r:5.1f}% Mixed",
            flush=True,
        )

    # ── region_counts.csv: every denominator used below ─────────────────────
    rows = []
    for flag in REGION_FLAGS:
        nice = REGION_NICE[flag]
        # Same NA-safe cast as above (only in_al is ever nullable): untraced
        # cells contribute to neither the True nor the False count here,
        # never .fillna(False)'d.
        flag_known = obs[flag].notna().values
        m = np.zeros(a.n_obs, dtype=bool)
        m[flag_known] = obs.loc[flag_known, flag].astype(bool).values
        m_single = single & m
        n_mix_r = int((m_single & mixed_mask_full).sum())
        rows.append(
            {
                "region": nice,
                "n_any_flag": int(m.sum()),
                "n_single_region_only": int(m_single.sum()),
                "n_queen": int((m_single & (obs["brain"] == "queen").values).sum()),
                "n_worker": int((m_single & (obs["brain"] == "worker").values).sum()),
                "n_mixed_expr": n_mix_r,
                "pct_mixed_expr": (100 * n_mix_r / m_single.sum())
                if m_single.sum()
                else float("nan"),
            }
        )
    counts_df = pd.DataFrame(rows)
    _n_mix_tot = int((single & mixed_mask_full).sum())
    counts_df.loc["total"] = [
        "TOTAL (all cells, not sum of regions -- flags can overlap)",
        int(a.n_obs),
        int(single.sum()),
        int((single & (obs["brain"] == "queen").values).sum()),
        int((single & (obs["brain"] == "worker").values).sum()),
        _n_mix_tot,
        100 * _n_mix_tot / single.sum(),
    ]
    counts_df.attrs["note"] = (
        f"{none_hit.sum():,} cells fall in no annotated region and "
        f"{multi_hit.sum()} in >1 (dropped as ambiguous); "
        f"{ambiguous.sum():,} more (all worker) are dropped because they hit "
        f"none of the six original regions AND their antennal-lobe status is "
        f"unknown (in_al is NA -- not False -- on an untraced section, so "
        f"they could be unannotated or antennal lobe; see the region-"
        f"assignment comment above); all diversity tables below use the "
        f"n_single_region_only denominator, split further by caste."
    )
    counts_df.to_csv(OUTPUT_DIR / "region_counts.csv", index=False)
    print(
        f"Denominators: {a.n_obs:,} total, {single.sum():,} single-region "
        f"({100 * single.mean():.1f}%), {none_hit.sum():,} unassigned "
        f"({100 * none_hit.mean():.1f}%)",
        flush=True,
    )

    # ── diversity_rarefied.csv + diversity_unrarefied.csv ───────────────────
    brain_arr = obs["brain"].astype(str).values
    region_arr = obs["region"].values

    rare_rows, naive_rows, hill_rows = [], [], []
    for nice in REGION_ORDER:
        for caste in ("both", "queen", "worker"):
            m = (region_arr == nice) & single
            if caste != "both":
                m &= brain_arr == caste
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            n_target = n_rarefy if caste == "both" else n_rarefy_caste
            for tag, s in schemes.items():
                # expr_nomixed drops Mixed cells from this region x caste's
                # index and rarefies to its own (smaller) target -- reusing
                # idx/n_target from the other schemes would either include
                # excluded cells (impossible, codes<0) or over-request the
                # subsample (returns all-NaN, see the target-computation
                # comment above).
                if tag == "expr_nomixed":
                    idx_s = idx[s["keep_mask"][idx]]
                    n_target_s = (
                        n_rarefy_nomixed if caste == "both" else n_rarefy_caste_nomixed
                    )
                else:
                    idx_s = idx
                    n_target_s = n_target
                if len(idx_s) == 0:
                    continue
                codes = s["codes"][idx_s]
                n_cats = len(s["cats"])
                d = rarefied_diversity(codes, n_cats, n_target_s, rng, b=N_BOOT)
                saturated = len(idx_s) <= n_target_s  # no actual subsampling occurred
                # -> CI is a degenerate point, not
                # biological precision
                rare_rows.append(
                    {
                        "region": nice,
                        "caste": caste,
                        "scheme": tag,
                        "circular": s["circular"],
                        "n_available": len(idx_s),
                        "n_rarefied": n_target_s,
                        "n_boot": N_BOOT,
                        "saturated": saturated,
                        "shannon_mean": d["shannon"][0],
                        "shannon_lo": d["shannon"][1],
                        "shannon_hi": d["shannon"][2],
                        "richness_mean": d["richness"][0],
                        "richness_lo": d["richness"][1],
                        "richness_hi": d["richness"][2],
                        "richness_1pct_mean": d["richness_1pct"][0],
                        "richness_1pct_lo": d["richness_1pct"][1],
                        "richness_1pct_hi": d["richness_1pct"][2],
                        "evenness_mean": d["evenness"][0],
                        "evenness_lo": d["evenness"][1],
                        "evenness_hi": d["evenness"][2],
                    }
                )
                nd = naive_diversity(codes, n_cats)
                naive_rows.append(
                    {
                        "region": nice,
                        "caste": caste,
                        "scheme": tag,
                        "circular": s["circular"],
                        **nd,
                    }
                )

                # Hill-number profile: q=0/1/2 recover
                # richness/exp-Shannon/inverse-Simpson exactly. Primary
                # reporting scheme is "expr"; the others play the same
                # comparison role as in diversity_rarefied.csv.
                hp = rarefied_hill_profile(
                    codes, n_cats, n_target_s, rng_hill, q_grid=Q_GRID, b=N_BOOT
                )
                for q in Q_GRID:
                    mean, lo, hi = hp[q]
                    hill_rows.append(
                        {
                            "region": nice,
                            "caste": caste,
                            "scheme": tag,
                            "circular": s["circular"],
                            "q": q,
                            "n_available": len(idx_s),
                            "n_rarefied": n_target_s,
                            "n_boot": N_BOOT,
                            "saturated": saturated,
                            "hill_mean": mean,
                            "hill_lo": lo,
                            "hill_hi": hi,
                        }
                    )

    rare_df = pd.DataFrame(rare_rows)
    naive_df = pd.DataFrame(naive_rows)
    hill_df = pd.DataFrame(hill_rows)
    rare_df.to_csv(OUTPUT_DIR / "diversity_rarefied.csv", index=False)
    naive_df.to_csv(OUTPUT_DIR / "diversity_unrarefied.csv", index=False)
    hill_df.to_csv(OUTPUT_DIR / "hill_diversity_profile.csv", index=False)

    print(
        "\n--- Hill-number profile (scheme=expr, caste=both, q=0/1/2) ---", flush=True
    )
    hshow = hill_df[
        (hill_df.scheme == "expr")
        & (hill_df.caste == "both")
        & (hill_df.q.isin([0.0, 1.0, 2.0]))
    ]
    print(
        hshow.pivot(index="region", columns="q", values="hill_mean")
        .loc[REGION_ORDER]
        .to_string(),
        flush=True,
    )

    print(
        "\n--- Queen vs worker Hill q=0/1/2: does the direction flip across q? "
        f"(scheme=expr, PRIMARY but coarse: {len(schemes['expr']['cats'])} "
        "categories) ---",
        flush=True,
    )
    qw_h = hill_df[
        (hill_df.scheme == "expr")
        & (hill_df.caste != "both")
        & (hill_df.q.isin([0.0, 1.0, 2.0]))
    ]
    piv = qw_h.pivot_table(index=["region", "q"], columns="caste", values="hill_mean")
    piv["worker_minus_queen"] = piv["worker"] - piv["queen"]
    print(piv.reindex(REGION_ORDER, level="region").to_string(), flush=True)

    # Cross-check against the leiden scheme (32 raw clusters, no cell-type
    # semantics, the other primary scheme). A caste direction that holds under
    # expr's 5-category pooling but not under leiden depends on the vocabulary.
    print(
        "\n--- Same check, scheme=leiden (cleanest, non-circular, real q=0 range) ---",
        flush=True,
    )
    qw_h_l = hill_df[
        (hill_df.scheme == "leiden")
        & (hill_df.caste != "both")
        & (hill_df.q.isin([0.0, 1.0, 2.0]))
    ]
    piv_l = qw_h_l.pivot_table(
        index=["region", "q"], columns="caste", values="hill_mean"
    )
    piv_l["worker_minus_queen"] = piv_l["worker"] - piv_l["queen"]
    print(piv_l.reindex(REGION_ORDER, level="region").to_string(), flush=True)

    print("\n--- rarefied Shannon diversity (scheme=expr, caste=both) ---", flush=True)
    show = rare_df[(rare_df.scheme == "expr") & (rare_df.caste == "both")]
    print(
        show[
            ["region", "n_available", "shannon_mean", "richness_mean", "evenness_mean"]
        ].to_string(index=False),
        flush=True,
    )

    print(
        "\n--- rarefied Shannon diversity (scheme=expr_nomixed, caste=both) ---",
        flush=True,
    )
    show_nm = rare_df[(rare_df.scheme == "expr_nomixed") & (rare_df.caste == "both")]
    print(
        show_nm[
            ["region", "n_available", "shannon_mean", "richness_mean", "evenness_mean"]
        ].to_string(index=False),
        flush=True,
    )
    _ord_expr = show.set_index("region").loc[REGION_ORDER, "shannon_mean"]
    _ord_nomixed = show_nm.set_index("region").loc[REGION_ORDER, "shannon_mean"]
    _rank_expr = _ord_expr.rank(ascending=False).astype(int)
    _rank_nomixed = _ord_nomixed.rank(ascending=False).astype(int)
    print(
        "\n  region ranks (1=most diverse), Mixed-included vs Mixed-excluded:",
        flush=True,
    )
    for _r in REGION_ORDER:
        _flag = "" if _rank_expr[_r] == _rank_nomixed[_r] else "  <-- RANK CHANGES"
        print(
            f"    {_r:8s} expr={_rank_expr[_r]}  "
            f"expr_nomixed={_rank_nomixed[_r]}{_flag}",
            flush=True,
        )
    _same_order = (
        _ord_expr.rank(ascending=False).values
        == _ord_nomixed.rank(ascending=False).values
    ).all()
    print(f"  Full rank order identical: {_same_order}", flush=True)

    print(
        "\n--- circularity check: anatomy scheme richness (should collapse) ---",
        flush=True,
    )
    circ = rare_df[(rare_df.scheme == "anatomy") & (rare_df.caste == "both")]
    print(
        circ[["region", "shannon_mean", "richness_mean"]].to_string(index=False),
        flush=True,
    )

    # ── rarefaction_curves.csv (scheme=expr, caste=both) ─────────────────────
    print("\nBuilding rarefaction curves ...", flush=True)
    depths = sorted(set([200, 500, 1000, 2000, 3000, n_rarefy]))
    depths = [d for d in depths if d <= n_rarefy]
    curve_rows = []
    for nice in REGION_ORDER:
        idx = np.flatnonzero((region_arr == nice) & single)
        codes = schemes["expr"]["codes"][idx]
        n_cats = len(schemes["expr"]["cats"])
        for depth in depths:
            d = rarefied_diversity(codes, n_cats, depth, rng, b=N_BOOT_CURVE)
            curve_rows.append(
                {
                    "region": nice,
                    "depth": depth,
                    "richness_mean": d["richness"][0],
                    "richness_lo": d["richness"][1],
                    "richness_hi": d["richness"][2],
                }
            )
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(OUTPUT_DIR / "rarefaction_curves.csv", index=False)

    # ── heterogeneity_rarefied.csv (label-free) ───────────────────────────────
    print("Computing PCA-space heterogeneity (label-free) ...", flush=True)
    x_pca_full = a.obsm["X_pca"][:, :N_PCS]
    het_rows = []
    for nice in REGION_ORDER:
        for caste in ("both", "queen", "worker"):
            m = (region_arr == nice) & single
            if caste != "both":
                m &= brain_arr == caste
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            h = heterogeneity(x_pca_full[idx], rng, n_het=N_HET, b=N_BOOT_HET)
            het_rows.append(
                {
                    "region": nice,
                    "caste": caste,
                    "n_available": len(idx),
                    "n_used": h["n_used"],
                    "n_boot": N_BOOT_HET,
                    "mean_pairwise_dist_mean": h["mean_pairwise_dist"][0],
                    "mean_pairwise_dist_lo": h["mean_pairwise_dist"][1],
                    "mean_pairwise_dist_hi": h["mean_pairwise_dist"][2],
                    "centroid_dispersion_mean": h["centroid_dispersion"][0],
                    "centroid_dispersion_lo": h["centroid_dispersion"][1],
                    "centroid_dispersion_hi": h["centroid_dispersion"][2],
                }
            )
    het_df = pd.DataFrame(het_rows)
    het_df.to_csv(OUTPUT_DIR / "heterogeneity_rarefied.csv", index=False)
    print(
        het_df[het_df.caste == "both"][
            ["region", "n_used", "mean_pairwise_dist_mean"]
        ].to_string(index=False),
        flush=True,
    )

    # ── beta_diversity_matrix.csv + beta_diversity_overall.csv ──────────────
    # Region composition vectors: naive (full-N, not rarefied) proportions
    # over label_expr pooled categories, castes pooled. Not rarefied because
    # compositional proportions (unlike richness/entropy) are ~depth-unbiased;
    # rarefaction sensitivity for these categories is already characterised
    # by rarefaction_curves.csv above.
    print("\nComputing beta diversity (region composition turnover) ...", flush=True)
    expr_codes_full = schemes["expr"]["codes"]
    n_expr_cats = len(schemes["expr"]["cats"])
    expr_cats = schemes["expr"]["cats"]
    region_props = {}
    for nice in REGION_ORDER:
        idx = np.flatnonzero((region_arr == nice) & single)
        counts = np.bincount(expr_codes_full[idx], minlength=n_expr_cats)
        region_props[nice] = counts / counts.sum()

    prop_stack = np.vstack([region_props[r] for r in REGION_ORDER])
    bc_dist = squareform(pdist(prop_stack, metric="braycurtis"))

    beta_rows = []
    for i, ri in enumerate(REGION_ORDER):
        for j, rj in enumerate(REGION_ORDER):
            beta_rows.append(
                {
                    "region_a": ri,
                    "region_b": rj,
                    "metric": "bray_curtis",
                    "value": float(bc_dist[i, j]),
                }
            )
    for q in (0, 1, 2):
        for i, ri in enumerate(REGION_ORDER):
            for j, rj in enumerate(REGION_ORDER):
                if i == j:
                    beta_val = 1.0  # identical-to-self by definition
                else:
                    pm = np.vstack([region_props[ri], region_props[rj]])
                    _, _, beta_val = multiplicative_hill_beta(pm, q)
                beta_rows.append(
                    {
                        "region_a": ri,
                        "region_b": rj,
                        "metric": f"hill_beta_q{q}",
                        "value": beta_val,
                    }
                )
    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(OUTPUT_DIR / "beta_diversity_matrix.csv", index=False)

    overall_rows = []
    for q in (0, 1, 2):
        gamma, alpha, beta_val = multiplicative_hill_beta(prop_stack, q)
        overall_rows.append(
            {
                "q": q,
                "gamma": gamma,
                "alpha": alpha,
                "beta_multiplicative": beta_val,
                "n_regions": len(REGION_ORDER),
            }
        )
    overall_beta_df = pd.DataFrame(overall_rows)
    overall_beta_df.to_csv(OUTPUT_DIR / "beta_diversity_overall.csv", index=False)
    print(
        "  Region x region Bray-Curtis:\n",
        pd.DataFrame(bc_dist, index=REGION_ORDER, columns=REGION_ORDER)
        .round(3)
        .to_string(),
        flush=True,
    )
    print(
        "\n  Overall (6-region) multiplicative Hill beta:\n",
        overall_beta_df.round(3).to_string(index=False),
        flush=True,
    )
    # q=0 (richness) is ceiling-limited under label_expr: only a handful of
    # categories exist, and full-N region composition has nearly all of them
    # present almost everywhere -> beta_q0 = 1.000 above reflects that every category
    # is present in every region (saturation), not "no compositional
    # turnover". Read the beta decomposition (and the Hill profile generally)
    # at q=1/q=2, or under the leiden scheme (32 categories, real q=0
    # dynamic range) in hill_diversity_profile.csv, not at q=0 here.
    print(
        "  NOTE: beta_q0 above is ceiling-limited (the "
        f"{len(expr_cats)}-category label_expr "
        "vocabulary saturates at full N) -- not a richness-turnover result; "
        "see hill_diversity_profile.csv scheme=leiden for real q=0 range.",
        flush=True,
    )

    # ── per_section_diversity.csv (caste spread; not rarefied, N kept) ───────
    print(
        "\nComputing per-section (per-slice) diversity for the caste view ...",
        flush=True,
    )
    sec_rows, comp_rows = [], []
    slice_arr = obs["slice_name"].astype(str).values
    # a section's brain is read as obs["brain"][idx[0]] below; that's only
    # valid if no slice_name is shared across brains.
    n_shared_slices = (
        obs.groupby("slice_name", observed=True)["brain"].nunique().gt(1).sum()
    )
    assert n_shared_slices == 0, (
        f"{n_shared_slices} slice_name values span >1 brain -- "
        "per-section brain lookup below would be wrong"
    )
    for nice in REGION_ORDER:
        for sl in np.unique(slice_arr[(region_arr == nice) & single]):
            m = (region_arr == nice) & single & (slice_arr == sl)
            idx = np.flatnonzero(m)
            if len(idx) < MIN_SECTION_CELLS:
                continue
            counts = np.bincount(expr_codes_full[idx], minlength=n_expr_cats)
            h = shannon(counts)
            r0 = richness(counts, 0.0)
            # cap the pairwise-distance subsample per section for speed
            n_cap = min(len(idx), 300)
            sub_idx = (
                rng.choice(idx, size=n_cap, replace=False) if len(idx) > n_cap else idx
            )
            mpd = float(pdist(x_pca_full[sub_idx]).mean()) if n_cap > 1 else np.nan
            brain_here = obs["brain"].values[idx[0]]
            sec_rows.append(
                {
                    "region": nice,
                    "brain": brain_here,
                    "slice_name": sl,
                    "n_cells": len(idx),
                    "shannon_expr": h,
                    "richness_expr": r0,
                    "evenness_expr": pielou(h, r0),
                    "mean_pairwise_dist": mpd,
                }
            )
            # Composition vector for this section -- the PERMANOVA replicate.
            props = counts / counts.sum()
            comp_rows.append(
                {
                    "region": nice,
                    "brain": brain_here,
                    "slice_name": sl,
                    "n_cells": len(idx),
                    **dict(zip(expr_cats, props, strict=False)),
                }
            )
    sec_df = pd.DataFrame(sec_rows)
    sec_df.to_csv(OUTPUT_DIR / "per_section_diversity.csv", index=False)
    print(
        f"  {len(sec_df)} region x section rows "
        f"({sec_df.brain.eq('queen').sum()} queen, "
        f"{sec_df.brain.eq('worker').sum()} worker; "
        f"min {MIN_SECTION_CELLS} cells/section)",
        flush=True,
    )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUTPUT_DIR / "per_section_composition.csv", index=False)

    # ── PERMANOVA on per-section Aitchison (CLR-Euclidean) distances ────────
    # Replicate unit = section, not cell: region is legitimately replicated
    # (many sections/region, both castes pooled) even though caste is not
    # (one brain per caste -- see the caveat attached to every caste row).
    print(
        "\nRunning PERMANOVA (replicate = section, label_expr composition, "
        "Aitchison/CLR distance) ...",
        flush=True,
    )
    clr_x = clr(comp_df[expr_cats].values)
    aitch_d = squareform(pdist(clr_x, metric="euclidean"))

    region_res = permanova(
        aitch_d,
        comp_df["region"].values,
        n_perm=N_PERM,
        rng=np.random.default_rng(SEED),
    )
    region_perm_df = pd.DataFrame(
        [
            {
                "predictor": "region",
                "region": "ALL",
                **region_res,
                "caveat": (
                    "None: sections are legitimate independent spatial "
                    "replicates within each brain, and both castes' sections "
                    "are pooled here, so region is a properly replicated "
                    "factor (not a caste comparison)."
                ),
            }
        ]
    )
    region_perm_df.to_csv(OUTPUT_DIR / "permanova_region.csv", index=False)
    print(
        f"  region: F={region_res['F']:.3f}  R2={region_res['R2']:.3f}  "
        f"p={region_res['p']:.4f}  (n={region_res['n']} sections, "
        f"k={region_res['k']} regions)",
        flush=True,
    )

    CASTE_CAVEAT = (
        "CONFOUNDED: one brain per caste. This tests whether composition "
        "differs between the TWO BRAINS (pooled across their sections), not "
        "whether it differs by caste -- brain identity and caste identity "
        "are perfectly aliased here. A significant p-value means brain A's "
        "tissue differs from brain B's; that could be caste, individual, "
        "batch, dissection, or any combination, and this test cannot "
        "separate them. Report as a brain-to-brain difference, not a "
        "caste effect."
    )
    caste_rows = []
    # The antennal lobe is excluded from every caste PERMANOVA, including this
    # pooled one: worker AL tracing is missing on most worker sections, so its
    # sections are few and a biased subset, and would add a missing-data signal.
    non_al_mask = (comp_df["region"] != "Antennal lobe").values
    non_al_idx = np.flatnonzero(non_al_mask)
    aitch_d_no_al = aitch_d[np.ix_(non_al_idx, non_al_idx)]
    caste_overall = permanova(
        aitch_d_no_al,
        comp_df.loc[non_al_mask, "brain"].values,
        n_perm=N_PERM,
        rng=np.random.default_rng(SEED),
    )
    caste_rows.append(
        {
            "predictor": "caste_overall",
            "region": "ALL_EXCEPT_AL",
            **caste_overall,
            "caveat": CASTE_CAVEAT
            + (
                " Antennal lobe sections are excluded from this "
                "'overall' pool entirely (see caste_within_* "
                "handling) -- not just uncaptioned, dropped."
            ),
        }
    )
    print(
        f"  caste (overall, CONFOUNDED with brain, AL excluded): "
        f"F={caste_overall['F']:.3f}  R2={caste_overall['R2']:.3f}  "
        f"p={caste_overall['p']:.4f}  (n={caste_overall['n']} sections)",
        flush=True,
    )

    # Caste-within-region PERMANOVAs as an interaction proxy: still
    # brain-confounded, but the per-region F/R2 values show whether the
    # brain-to-brain gap varies across regions. The antennal lobe is not tested
    # (thin, caste-uneven worker sections); its descriptive diversity is still
    # in diversity_rarefied.csv.
    for nice in [r for r in REGION_ORDER if r != "Antennal lobe"]:
        sub_mask = (comp_df["region"] == nice).values
        sub_brains = comp_df.loc[sub_mask, "brain"].values
        if sub_mask.sum() < 6 or pd.Series(sub_brains).value_counts().min() < 3:
            print(
                f"  caste within {nice}: skipped (too few sections per brain)",
                flush=True,
            )
            continue
        sub_idx = np.flatnonzero(sub_mask)
        d_sub = aitch_d[np.ix_(sub_idx, sub_idx)]
        res = permanova(
            d_sub, sub_brains, n_perm=N_PERM, rng=np.random.default_rng(SEED)
        )
        caste_rows.append(
            {
                "predictor": f"caste_within_{nice}",
                "region": nice,
                **res,
                "caveat": CASTE_CAVEAT,
            }
        )
        print(
            f"  caste within {nice}: F={res['F']:.3f}  R2={res['R2']:.3f}  "
            f"p={res['p']:.4f}  (n={res['n']} sections)",
            flush=True,
        )
    caste_perm_df = pd.DataFrame(caste_rows)
    # BH-FDR across the six within-region tests, which are read together.
    from scipy.stats import false_discovery_control

    within_mask = caste_perm_df["predictor"].str.startswith("caste_within_")
    caste_perm_df["p_bh"] = np.nan
    caste_perm_df.loc[within_mask, "p_bh"] = false_discovery_control(
        caste_perm_df.loc[within_mask, "p"].values, method="bh"
    )
    caste_perm_df.to_csv(OUTPUT_DIR / "permanova_caste.csv", index=False)
    print(
        "\n  NOTE: PERMANOVA tests location (centroid) differences and can be "
        "confounded by unequal within-group dispersion (PERMDISP) -- not "
        "checked here; a significant result could partly reflect brain A "
        "having more/less section-to-section spread than brain B, not just "
        "a shifted centroid.",
        flush=True,
    )

    # ── Figure ────────────────────────────────────────────────────────────
    print("\nRendering figure ...", flush=True)
    make_figure(rare_df, curve_df, het_df, sec_df, n_rarefy, n_rarefy_caste)
    make_caste_figure(rare_df, n_rarefy_caste)
    make_hill_figure(hill_df, bc_dist)
    print(f"\nDone. Outputs in {OUTPUT_DIR}/", flush=True)


# ── Figure ───────────────────────────────────────────────────────────────────


def _err(mean, lo, hi):
    """Asymmetric error bars from bootstrap percentiles, clipped at 0.

    Near-saturated regions (n_available ~= n_rarefied, e.g. both-caste and
    worker Lobula) resample almost the same set every rep, so the bootstrap
    distribution is a near-point-mass with a short skewed tail; the 2.5/97.5
    percentiles can then land on the same side of the mean (both == mode),
    which would otherwise make matplotlib's yerr negative.
    """
    mean, lo, hi = np.asarray(mean), np.asarray(lo), np.asarray(hi)
    return np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])


def make_figure(rare_df, curve_df, het_df, sec_df, n_rarefy, n_rarefy_caste) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    # Panel A: rarefied Shannon diversity per region, expr vs leiden schemes
    # (the two non-circular label schemes; primary result panel).
    ax = axes[0, 0]
    both = rare_df[rare_df.caste == "both"]
    x = np.arange(len(REGION_ORDER))
    width = 0.35
    for off, tag, label, color in [
        (-1, "expr", "expr.-only cluster-vote", "#2F4E96"),
        (1, "leiden", "raw Leiden cluster", "#48A6C9"),
    ]:
        d = both[both.scheme == tag].set_index("region").loc[REGION_ORDER]
        ax.bar(
            x + off * width / 2,
            d["shannon_mean"],
            width,
            label=label,
            color=color,
            yerr=_err(d["shannon_mean"], d["shannon_lo"], d["shannon_hi"]),
            capsize=2,
            error_kw=dict(lw=0.8),
        )
    ax.set_xticks(x, REGION_ORDER, rotation=30, ha="right")
    ax.set_ylabel(f"Shannon diversity (rarefied to N={n_rarefy})")
    ax.set_title("A. Regional diversity, non-circular label schemes")
    ax.legend(**LEGEND_KW)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: rarefaction curves (richness vs subsample depth)
    ax = axes[0, 1]
    for nice in REGION_ORDER:
        d = curve_df[curve_df.region == nice].sort_values("depth")
        ax.plot(
            d["depth"],
            d["richness_mean"],
            marker="o",
            ms=3,
            color=REGION_GROUP_COLOR[nice],
            label=nice,
        )
        ax.fill_between(
            d["depth"],
            d["richness_lo"],
            d["richness_hi"],
            color=REGION_GROUP_COLOR[nice],
            alpha=0.15,
        )
    ax.axvline(n_rarefy, ls="--", lw=1, color="#6E6154")
    ax.set_xlabel("Subsample depth (cells)")
    ax.set_ylabel("Richness (expr.-only scheme)")
    ax.set_title(
        "B. Richness saturates with depth\n(uncorrected comparison would be biased)"
    )
    ax.legend(**{**LEGEND_KW, "fontsize": 6.5}, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: label-free heterogeneity (mean pairwise PCA distance)
    ax = axes[0, 2]
    hboth = het_df[het_df.caste == "both"].set_index("region").loc[REGION_ORDER]
    colors = [REGION_GROUP_COLOR[r] for r in REGION_ORDER]
    ax.bar(
        REGION_ORDER,
        hboth["mean_pairwise_dist_mean"],
        color=colors,
        yerr=_err(
            hboth["mean_pairwise_dist_mean"],
            hboth["mean_pairwise_dist_lo"],
            hboth["mean_pairwise_dist_hi"],
        ),
        capsize=3,
        error_kw=dict(lw=0.8),
    )
    ax.set_xticklabels(REGION_ORDER, rotation=30, ha="right")
    ax.set_ylabel(
        f"Mean pairwise PCA distance (N={hboth['n_used'].iloc[0]}, {N_PCS} PCs)"
    )
    ax.set_title("C. Label-free heterogeneity\n(no cell-type vocabulary at all)")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel D: circular scheme, shown to demonstrate the failure mode
    ax = axes[1, 0]
    circ = rare_df[(rare_df.scheme == "anatomy") & (rare_df.caste == "both")]
    circ = circ.set_index("region").loc[REGION_ORDER]
    ax.bar(
        REGION_ORDER,
        circ["shannon_mean"],
        color="#B0B0B0",
        hatch="////",
        edgecolor="#6E6154",
        yerr=_err(circ["shannon_mean"], circ["shannon_lo"], circ["shannon_hi"]),
        capsize=3,
        error_kw=dict(lw=0.8),
    )
    ax.set_xticklabels(REGION_ORDER, rotation=30, ha="right")
    ax.set_ylim(0, 1.05 * max(0.5, circ["shannon_hi"].max()))
    ax.set_ylabel("Shannon diversity (anatomy_guided_vote)")
    ax.set_title(
        "D. CIRCULAR by construction -- not a result\n"
        "(labels are anchored to this exact anatomy)",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # Panel E: caste comparison, rarefied region-level, n=1 per caste
    ax = axes[1, 1]
    qw = rare_df[(rare_df.scheme == "expr") & (rare_df.caste != "both")]
    x = np.arange(len(REGION_ORDER))
    for off, caste, color in [(-1, "queen", QUEEN_COLOR), (1, "worker", WORKER_COLOR)]:
        d = qw[qw.caste == caste].set_index("region").reindex(REGION_ORDER)
        ax.bar(
            x + off * width / 2,
            d["shannon_mean"],
            width,
            label=caste.title(),
            color=color,
            yerr=_err(d["shannon_mean"], d["shannon_lo"], d["shannon_hi"]),
            capsize=2,
            error_kw=dict(lw=0.8),
        )
    ax.set_xticks(x, REGION_ORDER, rotation=30, ha="right")
    ax.set_ylabel(f"Shannon diversity (rarefied to N={n_rarefy_caste}/caste)")
    ax.set_title(
        "E. Queen vs worker -- DESCRIPTIVE, n=1 brain/caste\n"
        "no stat. test; error bars = resampling CI, not biological error",
        fontsize=8.5,
    )
    ax.legend(**LEGEND_KW)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel F: per-section spread
    ax = axes[1, 2]
    rng_jit = np.random.default_rng(1)
    for i, nice in enumerate(REGION_ORDER):
        for caste, color, dx in [
            ("queen", QUEEN_COLOR, -0.15),
            ("worker", WORKER_COLOR, 0.15),
        ]:
            vals = sec_df.loc[
                (sec_df.region == nice) & (sec_df.brain == caste), "shannon_expr"
            ].dropna()
            if len(vals) == 0:
                continue
            jitter = rng_jit.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + dx + jitter,
                vals,
                s=10,
                color=color,
                alpha=0.7,
                edgecolor="none",
            )
    ax.set_xticks(range(len(REGION_ORDER)), REGION_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Per-section Shannon diversity (expr.-only, unrarefied)")
    ax.set_title(
        f"F. Section-level spread (n={MIN_SECTION_CELLS}+ cells/section)\n"
        "sections are NOT biological replicates",
        fontsize=9,
    )
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=QUEEN_COLOR,
            label="Queen sections",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=WORKER_COLOR,
            label="Worker sections",
            markersize=6,
        ),
    ]
    ax.legend(handles=handles, **LEGEND_KW)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, OUTPUT_DIR, "region_diversity_main")
    # This 6-panel methods exhibit is not registered; the SI figure is
    # make_caste_figure().


def make_caste_figure(rare_df, n_rarefy_caste) -> None:
    """The SI figure: queen vs worker diversity per region, single panel.

    The other analyses (label-scheme comparison, rarefaction curves, label-free
    heterogeneity, circularity check, per-section spread) stay in
    make_figure() and the CSVs. Single panel, so no panel letter; the
    n=1/descriptive/resampling-CI caveat is in the LaTeX caption.

    Uses the expr.-only (no-anatomy) label scheme, rarefied to a common N.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    x = np.arange(len(REGION_ORDER))
    width = 0.35

    # Rarefied Shannon diversity per region, queen vs worker.
    qw = rare_df[(rare_df.scheme == "expr") & (rare_df.caste != "both")]
    for off, caste, color in [(-1, "queen", QUEEN_COLOR), (1, "worker", WORKER_COLOR)]:
        d = qw[qw.caste == caste].set_index("region").reindex(REGION_ORDER)
        ax.bar(
            x + off * width / 2,
            d["shannon_mean"],
            width,
            label=caste.title(),
            color=color,
            yerr=_err(d["shannon_mean"], d["shannon_lo"], d["shannon_hi"]),
            capsize=2,
            error_kw=dict(lw=0.8),
        )
    ax.set_xticks(x, REGION_ORDER, rotation=30, ha="right")
    ax.set_ylabel(f"Shannon diversity (rarefied to N={n_rarefy_caste}/caste)")
    ax.legend(**LEGEND_KW)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, OUTPUT_DIR, "region_diversity_caste")
    # Under the registered basename so sync_manuscript_figures.py's by-stem
    # glob finds it without an ALIAS entry.
    import shutil as _shutil

    _shutil.copyfile(
        OUTPUT_DIR / "region_diversity_caste.png",
        OUTPUT_DIR / "fig_s_regional_diversity.png",
    )
    print(
        f"    Copied to {OUTPUT_DIR / 'fig_s_regional_diversity.png'} "
        "(manuscript sync target)",
        flush=True,
    )


def make_hill_figure(hill_df: pd.DataFrame, bc_dist: np.ndarray) -> None:
    """Hill-number-profile exhibit: does the caste ordering depend on q?

    Exploratory, not the registered SI figure (same status as
    region_diversity_main.png).

    Panel A: continuous q-profile per region, castes pooled (scheme=expr).
    Panel B: small multiples, one per region, queen vs worker profile, showing
             whether the caste direction is stable across q in [0,3].
    Panel C: region x region Bray-Curtis beta-diversity heatmap (saved as a
             separate figure).
    """
    expr_both = hill_df[(hill_df.scheme == "expr") & (hill_df.caste == "both")]
    expr_qw = hill_df[(hill_df.scheme == "expr") & (hill_df.caste != "both")]

    # Small multiples on a 3x3 grid (7 of 9 slots used), so Panel A, which
    # spans every row, keeps a roughly square aspect.
    fig = plt.figure(figsize=(15, 12.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.3, 1, 1, 1], height_ratios=[1, 1, 1])

    # Panel A: continuous profile per region, castes pooled.
    ax = fig.add_subplot(gs[:, 0])
    for nice in REGION_ORDER:
        d = expr_both[expr_both.region == nice].sort_values("q")
        ax.plot(
            d["q"], d["hill_mean"], color=REGION_GROUP_COLOR[nice], lw=1.8, label=nice
        )
        ax.fill_between(
            d["q"],
            d["hill_lo"],
            d["hill_hi"],
            color=REGION_GROUP_COLOR[nice],
            alpha=0.12,
        )
    for q_mark in (0, 1, 2):
        ax.axvline(q_mark, ls=":", lw=0.6, color="#B0B0B0", zorder=0)
    ax.set_xlabel("Hill order q  (0=richness, 1=exp-Shannon, 2=inv.-Simpson)")
    ax.set_ylabel("Hill diversity  qD  (effective # of equally-common types)")
    ax.set_title(
        "A. Regional diversity profile\n(scheme=expr, castes pooled, rarefied)",
        fontsize=9.5,
    )
    ax.legend(**{**LEGEND_KW, "fontsize": 7})
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: small multiples (3x3 grid, 7 used), queen vs worker profile
    # per region. The position list must have >= len(REGION_ORDER) entries,
    # or zip() below would silently drop a region; the assert enforces it.
    positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1)]
    assert len(positions) >= len(REGION_ORDER), (
        "Panel B grid has fewer slots than regions -- a region would be "
        "silently dropped by the zip() below"
    )
    last_row_per_col = {}
    for (row, col), _nice in zip(positions, REGION_ORDER, strict=False):
        last_row_per_col[col] = max(last_row_per_col.get(col, row), row)
    for (row, col), nice in zip(positions, REGION_ORDER, strict=False):
        ax = fig.add_subplot(gs[row, col])
        for caste, color in [("queen", QUEEN_COLOR), ("worker", WORKER_COLOR)]:
            d = expr_qw[
                (expr_qw.region == nice) & (expr_qw.caste == caste)
            ].sort_values("q")
            ax.plot(d["q"], d["hill_mean"], color=color, lw=1.6, label=caste.title())
            ax.fill_between(d["q"], d["hill_lo"], d["hill_hi"], color=color, alpha=0.15)
        ax.set_title(nice, fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        if row == last_row_per_col[col]:
            ax.set_xlabel("q", fontsize=7.5)
        if col == 1:
            ax.set_ylabel("qD", fontsize=7.5)
        if (row, col) == (0, 1):
            ax.legend(**{**LEGEND_KW, "fontsize": 6.5})
    fig.text(
        0.53,
        0.965,
        "B. Queen vs worker diversity profile per region -- DESCRIPTIVE, n=1 "
        "brain/caste\n"
        "no stat. test; does the caste direction hold across q, or flip like "
        "Shannon/Simpson did?",
        ha="center",
        fontsize=8.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, OUTPUT_DIR, "hill_diversity_profile")

    # Separate small figure: region x region Bray-Curtis heatmap (kept out of
    # the crowded profile figure above rather than added as a 7th panel).
    fig2, ax2 = plt.subplots(figsize=(5.4, 4.6))
    im = ax2.imshow(bc_dist, cmap="magma_r", vmin=0, vmax=1)
    ax2.set_xticks(range(len(REGION_ORDER)), REGION_ORDER, rotation=30, ha="right")
    ax2.set_yticks(range(len(REGION_ORDER)), REGION_ORDER)
    for i in range(len(REGION_ORDER)):
        for j in range(len(REGION_ORDER)):
            v = bc_dist[i, j]
            ax2.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v > 0.5 else "black",
            )
    fig2.colorbar(im, ax=ax2, shrink=0.8, label="Bray-Curtis dissimilarity")
    ax2.set_title(
        "Region x region beta diversity\n(label_expr composition, castes pooled)",
        fontsize=9.5,
    )
    fig2.tight_layout()
    _save(fig2, OUTPUT_DIR, "beta_diversity_heatmap")


if __name__ == "__main__":
    main()
