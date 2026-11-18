#!/usr/bin/env python3
"""Supplementary figure: per-cell confidence for the two Kenyon-restricted
Tangram runs of `fig:kc_subtypes`.

`fig:kc_subtypes` uses two independent, Kenyon-compartment-restricted Tangram
mappings, each summarised by one confidence number. This figure plots their
per-cell confidence distributions, as fig:tangram_confidence does for the
whole-atlas mapping:

  RESTRICTED RECLASSIFICATION (`fig:kc_subtypes` B): reference restricted to the 5 named
  Kenyon populations, spatial query restricted to Kenyon-compartment cells by
  anatomy (in_kc_ic | in_kc_nc | in_kc_oc). Source: kc5pop_spatial_allKC_dsource
  .h5ad, loaded exactly as fig_kc_subtypes.load_kc5pop() does, so this figure
  and the `fig:kc_subtypes` panel always read the same object.

  SUBCLUSTER PROJECTION (`fig:kc_subtypes` F, G): the 3 de-novo Large-type
  snRNA subclusters
  mapped by Tangram onto spatial cells of ALL THREE Kenyon compartments
  (Small-type, Large-type and Class II; 173,312 queen + 158,504 worker cells,
  the same total as the restricted reclassification above) -- not the
  Large-type compartment alone. Source: ltKC_CB_spatial_subclusters_dsource
  .h5ad, loaded exactly as fig_kc_subtypes.load_ltkc() does; no compartment
  filter is applied here.

"Confidence" here is the per-cell max (renormalized) probability across the
mapping's own restricted label set -- 5 populations for the first, 3
subclusters for the second -- not the whole-atlas 29-way posterior that
fig:tangram_confidence plots. Chance is therefore 1/5 = 0.20 for the first
panel and 1/3 = 0.333 for the second, not 1/29.

    PYTHONPATH=src .venv/bin/python scripts/kenyon/si_kc_restricted_confidence.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.gridspec import GridSpec

from bimp_st.paths import REPO  # noqa: E402
from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import (
    OUTPUT_DIR,
    QUEEN_COLOR,
    WORKER_COLOR,
    add_panel_headers,
    save,
    set_style,
)

# Same two objects fig_kc_subtypes.py reads -- see its load_kc5pop() and
# load_ltkc() for the provenance of each run.
KC5POP_H5AD = REPO / "results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad"
LTKC_H5AD = REPO / "results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad"

TYPE_ORDER = [
    "Large-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Middle-type Kenyon cells",
    "Other Kenyon cells",
]
TYPE_SHORT = {
    "Large-type Kenyon cells": "Large-type",
    "Small-type Kenyon cells": "Small-type",
    "Class II Kenyon cells": "Class II",
    "Middle-type Kenyon cells": "Middle-type",
    "Other Kenyon cells": "Other KC",
}
SUB_CLUSTERS = ["0", "1", "2"]
COMPARTMENT_ORDER = ["Small-type", "Large-type", "Class II"]

FIG_NAME = "fig_s_kc_restricted_confidence"
OUT_JSON = REPO / "results/figures/si/fig_s_kc_restricted_confidence_stats.json"


def _compartment(obs):
    ic = obs["in_kc_ic"].values.astype(bool)
    nc = obs["in_kc_nc"].values.astype(bool)
    return np.where(ic, "Small-type", np.where(nc, "Large-type", "Class II"))


def load_kc5pop():
    """Identical loading logic to fig_kc_subtypes.load_kc5pop() (panel B)."""
    ad = sc.read_h5ad(KC5POP_H5AD)
    ad.obs_names_make_unique()
    p = ad.obsm["tangram_ct_pred"]
    if not isinstance(p, pd.DataFrame):
        p = pd.DataFrame(np.asarray(p), columns=TYPE_ORDER)
    p = p[TYPE_ORDER]
    probs = p.div(p.sum(axis=1), axis=0).values.astype(float)
    o = ad.obs
    return dict(probs=probs, comp=_compartment(o), brain=o["brain"].astype(str).values)


def load_ltkc():
    """Identical loading logic to fig_kc_subtypes.load_ltkc() (panels F, G)."""
    ad = sc.read_h5ad(LTKC_H5AD)
    ad.obs_names_make_unique()
    p = ad.obsm["tangram_ct_pred"]
    if not isinstance(p, pd.DataFrame):
        p = pd.DataFrame(np.asarray(p), columns=SUB_CLUSTERS)
    p.columns = [str(c) for c in p.columns]
    p = p[SUB_CLUSTERS]
    probs = p.div(p.sum(axis=1), axis=0).fillna(1.0 / 3).values.astype(float)
    o = ad.obs
    return dict(probs=probs, comp=_compartment(o), brain=o["brain"].astype(str).values)


def _hist_panel(ax, conf, chance, title, color="#4A6C8C"):
    counts, _, _ = ax.hist(conf, bins=100, range=(0, 1), color=color)
    med = float(np.median(conf))
    ax.axvline(chance, color="0.35", ls=":", lw=1.0)
    ax.axvline(med, color="black", lw=1.0)
    # Headroom above the bars, and each line named at its top (chance to its
    # right, where no cell scores; median to its left) instead of a legend,
    # which could not clear the chance line in both panels.
    ax.set_ylim(0, counts.max() * 1.22)
    top = dict(transform=ax.get_xaxis_transform(), va="top", fontsize=6.0)
    ax.text(
        chance + 0.015, 0.98, f"chance\n{chance:.3f}", ha="left", color="0.35", **top
    )
    ax.text(med - 0.015, 0.98, f"median\n{med:.3f}", ha="right", color="black", **top)
    ax.set_xlabel("Max per-cell probability")
    ax.set_ylabel("Cells")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelbottom=True)
    # No titles (caption owns them); ``title`` is kept for call sites.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")
    )


def _group_median_bar(ax, conf, groups, group_order, title, colors=None):
    meds = [np.median(conf[groups == g]) for g in group_order]
    ns = [int((groups == g).sum()) for g in group_order]
    xs = np.arange(len(group_order))
    bar_colors = colors if colors is not None else ["#4A6C8C"] * len(group_order)
    ax.bar(xs, meds, color=bar_colors, width=0.6)
    # n inside each bar, rotated: horizontal labels above six adjacent bars
    # ran into each other.
    for x, n in zip(xs, ns, strict=False):
        ax.text(
            x,
            0.03,
            f"n = {n:,}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.0,
            color="#1A1A1A",
        )
    ax.set_xticks(xs)
    # Display only: the group keys (and the stats JSON) keep lowercase castes.
    shown = [
        g.replace("queen", "Queen").replace("worker", "Worker") for g in group_order
    ]
    if len(group_order) > 3:
        ax.set_xticklabels(shown, rotation=30, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(shown)
    ax.set_ylabel("Median max probability")
    ax.set_ylim(0, 1)
    # No titles (caption owns them); ``title`` is kept for call sites.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    set_style()

    kc5 = load_kc5pop()
    ltkc = load_ltkc()

    conf5 = kc5["probs"].max(axis=1)
    conf3 = ltkc["probs"].max(axis=1)
    chance5 = 1.0 / len(TYPE_ORDER)
    chance3 = 1.0 / len(SUB_CLUSTERS)

    # 7.2 in wide (the printed width), so text is authored near printed size.
    plt.rcParams.update(
        {
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.62,
        wspace=0.30,
        top=0.93,
        bottom=0.10,
        left=0.09,
        right=0.985,
    )

    axes = []
    ax = fig.add_subplot(gs[0, 0])
    axes.append(ax)
    _hist_panel(
        ax,
        conf5,
        chance5,
        "Restricted reclassification\n5 named Kenyon populations",
    )

    ax = fig.add_subplot(gs[0, 1])
    axes.append(ax)
    comp_brain5 = np.array(
        [f"{c}\n{b}" for c, b in zip(kc5["comp"], kc5["brain"], strict=False)]
    )
    group_order5 = [f"{c}\n{b}" for c in COMPARTMENT_ORDER for b in ("queen", "worker")]
    colors5 = [QUEEN_COLOR if "queen" in g else WORKER_COLOR for g in group_order5]
    _group_median_bar(
        ax,
        conf5,
        comp_brain5,
        group_order5,
        "Median confidence by\ncompartment x caste",
        colors5,
    )

    ax = fig.add_subplot(gs[1, 0])
    axes.append(ax)
    _hist_panel(
        ax,
        conf3,
        chance3,
        "Subcluster projection\n3 de-novo Large-type subclusters,\n"
        "all three Kenyon cell compartments",
        color="#8C4A6C",
    )

    ax = fig.add_subplot(gs[1, 1])
    axes.append(ax)
    group_order3 = ["queen", "worker"]
    colors3 = [QUEEN_COLOR, WORKER_COLOR]
    _group_median_bar(
        ax,
        conf3,
        ltkc["brain"],
        group_order3,
        "Median confidence by caste\n(all three Kenyon cell compartments)",
        colors3,
    )

    add_panel_headers(
        fig,
        list(
            zip(
                axes,
                "ABCD",
                [
                    "Five Kenyon types",
                    "By compartment",
                    # C/D cover every Kenyon-compartment cell (D's n sums to B's), not
                    # the Large-type compartment only; the tags say so.
                    "Three subclusters, all Kenyon compartments",
                    "By caste, all Kenyon compartments",
                ],
                strict=False,
            )
        ),
        width_frac=PRINTED_WIDTH_FRAC["fig_s_kc_restricted_confidence"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, OUTPUT_DIR, FIG_NAME)

    stats = {
        "kc5pop_restricted_reclassification": {
            "n_cells": int(conf5.size),
            "median": float(np.median(conf5)),
            "frac_below_0.5": float(np.mean(conf5 < 0.5)),
            "chance": chance5,
            "by_compartment_caste": {
                g: float(np.median(conf5[comp_brain5 == g])) for g in group_order5
            },
        },
        "ltkc_subcluster_projection": {
            "n_cells": int(conf3.size),
            "median": float(np.median(conf3)),
            "frac_below_0.5": float(np.mean(conf3 < 0.5)),
            "chance": chance3,
            "by_caste": {
                g: float(np.median(conf3[ltkc["brain"] == g])) for g in group_order3
            },
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
