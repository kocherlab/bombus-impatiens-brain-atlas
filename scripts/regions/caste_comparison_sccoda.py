#!/usr/bin/env python3
"""fig:caste_comparison analysis step: scCODA and per-section composition.

Writes, under results/figures/fig2/ (``BIMP_FIG2_OUT`` overrides it, e.g. for a
scratch reproducibility run):
  slice_composition.csv     per-section class proportions (panel B)
  abundance_tangram_ct.csv  scCODA fit on the vote's labels (panel A), cached:
                            delete it to refit (NUTS is node-dependent; see
                            the AVX2 pin in the env)

Panel C's per-region table is scripts/regions/region_composition.py; the figure
is composed by fig_caste_comparison.py. Labels are the expression-only
``hierarchical_vote`` (``_display_ct_series``).

    PYTHONPATH=src:scripts .venv/bin/python scripts/regions/caste_comparison_sccoda.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# scanpy adds ~80s to import time; only load it when actually needed
# (i.e. when h5ads must be parsed in the slow path).
sc = None


def _ensure_scanpy():
    global sc
    if sc is None:
        import scanpy as _sc

        sc = _sc
    return sc


import re as _re

from bimp_st.cluster_labels import (
    POOLED_ORDER,
    get_pooled,
    get_short,
)
from bimp_st.paths import CANONICAL_TANGRAM
from bimp_st.si_common import (
    detect_ct_col as _detect_ct_col,
)
from bimp_st.si_common import (
    maybe_subsample_adata,
)
from bimp_st.si_common import (
    sort_cts as _sort_cts,
)

# ── Fine vocabulary, per-cell pooling, and why this is cluster-vote ────────
# The display vocabulary is cluster-vote, pool=True, expression-only (see
# _display_ct_series).  The alternatives do not work:
#
#  1. The raw fine vocabulary (canonical 17, per-cell or per-cluster),
#     measured on every atlas cell
#     (results/reframe/vote_over_17/vote_scheme_comparison.csv).
#     anatomy_guided_vote(pool=False) resolves only a few of the 17
#     categories and inflates "Mixed".  The names it does resolve are wrong
#     rather than merely coarse: the whole glial block votes one glial
#     subtype and the whole projection-neuron block one projection-neuron
#     subtype, asserting subtypes the panel cannot see.  Lowering the purity
#     threshold does not help: the categories that light up are the ones the
#     anatomical-plausibility audit marks FAIL/SUSPECT.  Raw per-cell argmax
#     nominally gives every category but puts a large share of the brain in
#     "Small-type Kenyon cells", which is depleted in the calyx (anatomically
#     impossible).
#  2. Per-cell labels + ``get_pooled`` (the 7-group POOLED_ORDER): pooling
#     hides the per-cell noise as a separate category, but the contaminated
#     cells move into the pooled bar, and region conditioning re-exposes them
#     as spurious KC signal outside the calyx (e.g. a strong worker-lean KC x
#     Lobula cell that is largely the same medulla-enriched misassignment).
#
# Cluster vote recovers Kenyon-cell<->calyx and absorbs the per-cell optic
# leakage.  Without the anatomy prior KC_F1 is 0.9148 (0.9237 with it); the
# optic-lobe F1 lost with the prior was fabricated by it
# Only ~5 of the 7 pooled groups are ever a cluster majority; photoreceptors
# and hemocytes structurally are not.
#
# "Mixed" (the low-purity residual) is a first-class category (grey,
# POOLED_ORDER + ["Mixed"]) shown as "no confident majority label for this
# cluster", not excluded and renormalized away in the display panels.
MIXED_LABEL = "Mixed"
# hierarchical_vote names a glial subtype where the cluster decides one, so
# the display vocabulary carries those alongside the pooled groups. Kenyon
# subtypes are never emitted (cluster_labels.CAP_TO_POOLED), so none appear.
FINE_GLIA = [
    "Astrocyte-like glia",
    "Perineurial glia",
    "Subperineurial glia",
    "Cortex glia",
    "Other glia",
]
DISPLAY_ORDER = [*POOLED_ORDER, *FINE_GLIA, MIXED_LABEL]


def _pooled_order(cts):
    """Return the display groups present in *cts*, in DISPLAY_ORDER.

    Raises on a category that is present but unlisted, rather than dropping
    it.  Silently filtering an unlisted category (e.g. a fine glial name from
    hierarchical_vote) removes those cells and renormalises the rest, so
    every bar shifts while the figure still looks entirely plausible.
    """
    present = {str(c) for c in cts}
    unlisted = sorted(present - set(DISPLAY_ORDER))
    if unlisted:
        raise AssertionError(
            "cell-type categories present in the data but not in "
            "DISPLAY_ORDER, so they would be dropped and the rest "
            "renormalised: %s" % unlisted
        )
    return [g for g in DISPLAY_ORDER if g in present]


def _display_ct_series(tangram, anatomy=False):
    """Cluster-voted pooled display labels (expression-only by default).

    Votes one pooled cell type per spatial Leiden cluster over the combined
    Tangram labels (not per-cell argmax), preserving the Kenyon-cell<->calyx
    block. Low-confidence clusters read as
    "Mixed" -- kept as a first-class category downstream, not dropped.

    The default is ``anatomy=False``.  The anatomy-guided path forces any
    cluster more than half inside calyx anatomy to "Kenyon cells" and more
    than half inside optic anatomy to "Optic lobe", using the same ``in_*``
    flags this figure's regional panels plot against, so it is circular
    wherever the two meet.  Plain ``cluster_vote`` is the per-cluster
    majority of the Tangram call and nothing else.

    ``anatomy=True`` is kept, unused by the figure, so the
    method-comparison scripts (``si_label_sensitivity``,
    ``vote_scheme_comparison``) can still request it.
    Falls back to per-cell pooling if the spatial cluster column is absent.
    """
    from bimp_st.cluster_labels import anatomy_guided_vote, hierarchical_vote

    # Cache under a method-specific name, never under the bare "cluster_vote".
    # The combined object can carry a persisted "cluster_vote" from another
    # run -- anatomy-guided, or from a different pooling scheme -- and the
    # cache-hit branch below would hand it back without recomputing.
    dst = "cluster_vote_anatomy" if anatomy else "cluster_vote_expr"
    if dst not in tangram.obs.columns:
        if not (
            "leiden" in tangram.obs.columns
            and "tangram_cell_type" in tangram.obs.columns
        ):
            dst = None
        elif anatomy:
            anatomy_guided_vote(
                tangram, cluster_col="leiden", label_col="tangram_cell_type", dst=dst
            )
        else:
            # hierarchical_vote, not cluster_vote: a flat pooled vote
            # collapses every glial subtype into one 34% block, while
            # perineurial and astrocyte-like clusters vote at 0.83 and 0.71
            # and can be named. Kenyon stays pooled by CAP_TO_POOLED.
            hierarchical_vote(
                tangram, cluster_col="leiden", label_col="tangram_cell_type", dst=dst
            )
    if dst is not None:
        out = tangram.obs[dst].astype(str)
        # "cluster_vote" is the canonical column the rest of this module reads
        # (Panels A/B, the slice-composition cache, the parquet scatter). Point
        # it at whichever method this call used, so there is exactly one
        # definition of the display labels per run.
        tangram.obs["cluster_vote"] = out.values
        return out
    src = (
        "tangram_cell_type"
        if "tangram_cell_type" in tangram.obs.columns
        else _detect_ct_col(tangram)
    )
    return tangram.obs[src].astype(str).map(get_pooled)


def _natural_key(s):
    """Split into text/number chunks so OL_2 sorts before OL_10."""
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", str(s))]


def _sort_cts(cts):  # noqa: F811 — natural-sort override of _si_common.sort_cts
    """Natural/numeric sort of cell-type labels.

    Overrides the imported lexicographic ``sort_cts`` so the OL_1..OL_13
    series orders numerically (OL_1, OL_2, ... OL_10) instead of
    OL_1, OL_10, OL_11, OL_2. Applies everywhere in this module (panels
    B, C, D, E) since they all route through ``_sort_cts``.
    """
    return sorted(cts, key=_natural_key)


# ── Configuration ────────────────────────────────────────────────────────────
## SPATIAL_H5AD is the canonical Tangram run (paths.CANONICAL_TANGRAM), whose
## cell-type columns are keyed by name. combined_clustered.h5ad carries its own
## non-canonical `tangram_cell_type` column, so it must not be the input.
SPATIAL_H5AD = str(CANONICAL_TANGRAM)
OUTPUT_DIR = Path(os.environ.get("BIMP_FIG2_OUT", "results/figures/fig2"))


# ── Composition: scCODA ─────────────────────────────────────────────────────

def _clr(x):
    """Centered log-ratio transform (add pseudocount for zeros)."""
    x = np.array(x, dtype=float)
    x = x + 0.5  # pseudocount
    gm = np.exp(np.mean(np.log(x)))
    return np.log(x / gm)


def compute_abundance(tangram, n_mc=128, seed=42):
    """Compositional differential abundance using scCODA (pertpy).

    Uses scCODA's Bayesian Dirichlet-Multinomial model via pertpy/numpyro.
    Falls back to ALDEx2-style (Dirichlet MC + CLR + Wilcoxon) if unavailable.
    """
    import anndata as ad

    # Cluster-voted pooled display groups (7-pool + Mixed, see
    # _display_ct_series); the compositional model runs on whatever subset
    # of those cluster-vote actually populates (the fine ``tangram_cell_type``
    # column is retained for the fine code paths). ``ct_col`` here is just the
    # column name used inside the temporary per-cell AnnData built for pertpy
    # below, not a tangram.obs cache key.
    ct_col = "cell_type_pooled"
    ct_pooled = _display_ct_series(tangram)

    # Build per-slice count table
    slice_ct = pd.crosstab(
        tangram.obs["slice_name"].astype(str),
        ct_pooled.values,
    )
    slice_brain = tangram.obs.groupby(
        tangram.obs["slice_name"].astype(str), observed=True
    )["brain"].first()

    all_ct = _pooled_order(slice_ct.columns.tolist())
    slice_ct = slice_ct[all_ct]

    queen_slices = slice_brain[slice_brain == "queen"].index
    worker_slices = slice_brain[slice_brain == "worker"].index
    slice_props = slice_ct.div(slice_ct.sum(axis=1), axis=0)

    # ── Try scCODA via pertpy ──
    try:
        # Compat shim: recent scanpy renamed scanpy._utils.check_use_raw ->
        # _check_use_raw (and sanitize_anndata -> _sanitize_anndata), which
        # breaks pertpy's mixscape import (pulled in by pertpy.tools). Re-expose
        # the old names BEFORE importing pertpy. No scanpy files are modified.
        import scanpy._utils as _su

        for _old, _new in (
            ("check_use_raw", "_check_use_raw"),
            ("sanitize_anndata", "_sanitize_anndata"),
        ):
            if not hasattr(_su, _old) and hasattr(_su, _new):
                setattr(_su, _old, getattr(_su, _new))
        from pertpy.tools import Sccoda

        print("    Using scCODA (pertpy, numpyro backend)...", flush=True)

        sccoda = Sccoda()

        # Load cell-level data -> pertpy aggregates to sample-level MuData
        # Need a minimal AnnData with ct_col + slice_name + brain in obs
        cell_obs = tangram.obs[["slice_name", "brain"]].copy()
        cell_obs[ct_col] = ct_pooled.values
        cell_obs[ct_col] = cell_obs[ct_col].astype(str)
        cell_obs["slice_name"] = cell_obs["slice_name"].astype(str)
        cell_obs["brain"] = cell_obs["brain"].astype(str)
        cell_adata = ad.AnnData(obs=cell_obs)
        cell_adata.obs_names_make_unique()

        mdata = sccoda.load(
            cell_adata,
            type="cell_level",
            generate_sample_level=True,
            cell_type_identifier=ct_col,
            sample_identifier="slice_name",
            covariate_obs=["brain"],
        )

        # Reference CT: most abundant (most stable)
        ref_ct = str(slice_ct.sum(axis=0).idxmax())
        print(f"    Reference cell type: {ref_ct}", flush=True)

        sccoda.prepare(
            mdata,
            modality_key="coda",
            formula="brain",
            reference_cell_type=ref_ct,
        )
        # Fix arviz/xarray dimension conflict: obs index name must not clash
        mdata["coda"].obs.index.name = None
        sccoda.run_nuts(
            mdata,
            modality_key="coda",
            num_warmup=500,
            num_samples=1000,
            rng_key=seed,
        )
        sccoda.credible_effects(mdata, modality_key="coda", est_fdr=0.05)

        # Extract results from the coda modality
        coda_adata = mdata["coda"]
        # Find the effect df key (named after the covariate contrast)
        effect_keys = [k for k in coda_adata.varm.keys() if "effect_df" in k]
        effects = coda_adata.varm[effect_keys[0]] if effect_keys else None

        rows = []
        for ct_name in all_ct:
            q_prop = slice_props.loc[queen_slices, ct_name].mean()
            w_prop = slice_props.loc[worker_slices, ct_name].mean()
            if effects is not None and ct_name in effects.index:
                # Final Parameter != 0 means scCODA's spike-and-slab kept it
                is_credible = bool(effects.loc[ct_name, "Final Parameter"] != 0)
                # Plot scCODA's compositional log2 change column
                if "log2-fold change" in effects.columns:
                    log2fc = float(effects.loc[ct_name, "log2-fold change"])
                else:
                    log2fc = float(effects.loc[ct_name, "Final Parameter"])
                inc_prob = (
                    float(effects.loc[ct_name, "Inclusion probability"])
                    if "Inclusion probability" in effects.columns
                    else np.nan
                )
            else:
                log2fc = float(np.log2(w_prop / q_prop)) if q_prop > 0 else 0.0
                is_credible = False
                inc_prob = np.nan
            rows.append(
                {
                    "cluster": ct_name,
                    "log2fc": log2fc,
                    "p_adj": 0.001 if is_credible else 0.5,
                    "credible": is_credible,
                    "inclusion_prob": inc_prob,
                    # scCODA's reference category by construction (fixed at 0,
                    # never tested) -- an explicit column rather than making
                    # every downstream reader re-derive it from
                    # inclusion_prob==0.0, which breaks if more than one category
                    # happens to land at exactly 0.0 inclusion probability.
                    "is_reference": ct_name == ref_ct,
                    "prop_queen": float(q_prop),
                    "prop_worker": float(w_prop),
                    "n_queen": len(queen_slices),
                    "n_worker": len(worker_slices),
                    "method": "scCODA",
                }
            )
        df = pd.DataFrame(rows)
        sig = df["credible"].sum()
        print(f"    {sig} credible effects (scCODA FDR < 0.05)", flush=True)
        return df

    except Exception as e:
        print(f"    scCODA failed ({e}), falling back to ALDEx2-style...", flush=True)

    # ── Fallback: ALDEx2-style Dirichlet MC + CLR + Wilcoxon ──
    from scipy.stats import dirichlet, mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    counts = slice_ct.values.astype(float)
    queen_idx = np.array(
        [i for i, s in enumerate(slice_ct.index) if slice_brain.get(s) == "queen"]
    )
    worker_idx = np.array(
        [i for i, s in enumerate(slice_ct.index) if slice_brain.get(s) == "worker"]
    )
    print(
        f"    ALDEx2-style: {len(queen_idx)} queen, {len(worker_idx)} worker "
        f"slices, {len(all_ct)} CTs, {n_mc} MC instances",
        flush=True,
    )

    rng = np.random.default_rng(seed)
    n_slices, n_ct = counts.shape
    pvals_mc = np.zeros((n_mc, n_ct))
    clr_diff_mc = np.zeros((n_mc, n_ct))
    for mc in range(n_mc):
        props = np.zeros_like(counts)
        for i in range(n_slices):
            alpha = counts[i] + 0.5
            props[i] = dirichlet.rvs(alpha, random_state=rng)[0]
        gm = np.exp(np.mean(np.log(props), axis=1, keepdims=True))
        clr = np.log(props / gm)
        for j in range(n_ct):
            q = clr[queen_idx, j]
            w = clr[worker_idx, j]
            clr_diff_mc[mc, j] = w.mean() - q.mean()
            _, pvals_mc[mc, j] = mannwhitneyu(q, w, alternative="two-sided")

    median_p = np.median(pvals_mc, axis=0)
    mean_effect = np.mean(clr_diff_mc, axis=0)
    _, p_adj, _, _ = multipletests(median_p, method="fdr_bh")

    rows = []
    for j, ct_name in enumerate(all_ct):
        rows.append(
            {
                "cluster": ct_name,
                "log2fc": float(mean_effect[j]),
                "p_value": float(median_p[j]),
                "p_adj": float(p_adj[j]),
                "prop_queen": float(slice_props.iloc[queen_idx][ct_name].mean()),
                "prop_worker": float(slice_props.iloc[worker_idx][ct_name].mean()),
                "n_queen": len(queen_idx),
                "n_worker": len(worker_idx),
                "method": "ALDEx2-style",
                # No reference category in this method (no CLR identification
                # constraint the way scCODA has); schema-compatible placeholder
                # for downstream readers that check this column.
                "is_reference": False,
            }
        )
    df = pd.DataFrame(rows)
    sig = (df["p_adj"] < 0.05).sum()
    print(f"    {sig} significant CTs (FDR < 0.05)", flush=True)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_scanpy()

    print(f"Loading {SPATIAL_H5AD}...", flush=True)
    tangram = sc.read_h5ad(SPATIAL_H5AD)
    print(f"  {tangram.n_obs:,} cells", flush=True)
    if "tangram_cell_type" not in tangram.obs.columns:
        raise SystemExit(f"tangram_cell_type missing from {SPATIAL_H5AD}")
    tangram.obs["cell_type"] = (
        tangram.obs["tangram_cell_type"].map(get_short).astype("category")
    )
    tangram = maybe_subsample_adata(tangram)

    # The display labels (cluster vote, Mixed kept as a category).
    _display_ct_series(tangram)  # creates tangram.obs["cluster_vote"]
    n_pool = tangram.obs["cluster_vote"].nunique()
    print(f"  cluster_vote: {n_pool} display groups (incl. Mixed)", flush=True)

    # Per-section class proportions (long form); same pooling and grouping as
    # compute_abundance's slice table, un-averaged.
    slice_comp_path = OUTPUT_DIR / "slice_composition.csv"
    slice_ct_all = pd.crosstab(
        tangram.obs["slice_name"].astype(str),
        tangram.obs["cluster_vote"].values,
    )
    slice_brain_all = tangram.obs.groupby(
        tangram.obs["slice_name"].astype(str), observed=True
    )["brain"].first()
    slice_props_all = slice_ct_all.div(slice_ct_all.sum(axis=1), axis=0)
    slice_comp_long = slice_props_all.reset_index(names="slice_name").melt(
        id_vars="slice_name", var_name="cluster", value_name="prop"
    )
    slice_comp_long["brain"] = slice_comp_long["slice_name"].map(slice_brain_all)
    slice_comp_long["n_cells"] = slice_comp_long["slice_name"].map(
        slice_ct_all.sum(axis=1)
    )
    slice_comp_long.to_csv(slice_comp_path, index=False)
    print(
        f"    Saved per-slice composition to {slice_comp_path} "
        f"({slice_brain_all.eq('queen').sum()} queen / "
        f"{slice_brain_all.eq('worker').sum()} worker sections)",
        flush=True,
    )

    # scCODA, cached: an existing table is trusted, so delete it after any
    # change to the vote.
    abund_path = OUTPUT_DIR / "abundance_tangram_ct.csv"
    if abund_path.exists():
        print(f"  Using cached abundance: {abund_path}", flush=True)
    else:
        abund = compute_abundance(tangram)
        abund.to_csv(abund_path, index=False)
        print(f"    Saved to {abund_path}", flush=True)
    print("Done.", flush=True)
