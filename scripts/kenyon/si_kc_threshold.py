"""What counts as a Kenyon cell in the atlas, and where should the threshold sit?

The atlas calls clusters Kenyon cells via the expression-only cluster vote
(``hierarchical_vote``), which says nothing about what posterior a single cell
needs to earn the label. Per-cell Tangram posteriors over 29 classes are weak
(fig_s_tangram_confidence_stats.json), so the choice of cut-off does real
work and is stated rather than left implicit. The numbers are reported; the
figure is not registered.

Four candidate definitions of "this cell is a Kenyon cell":

  argmax     the top-1 Tangram class is any Kenyon-cell type
  summed     the posterior summed over all Kenyon-cell types exceeds t
  anatomy    the cell sits inside a hand-drawn calyx polygon (KC-OC/IC/NC)
  marker     a panel-gene Kenyon score exceeds t

We sweep t, and at each value report how many cells qualify and how well the
call agrees with calyx anatomy. Anatomy is not ground truth -- Kenyon somata sit
in the calyx cup, so it is a strong prior but a hand-drawn one, and the
comparison is reported as agreement, never as accuracy.

Outputs to results/reframe/kc_threshold/:
    threshold_sweep.csv     count / precision / recall / F1 / Jaccard vs t
    definition_overlap.csv  pairwise agreement between the four definitions
    kc_threshold.png
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Text is normalised to the house printed sizes at save time; this script
# reaches neither bimp_st.style.set_pub_style() nor scripts/_si_common,
# so the hook is installed here.
from bimp_st.style import set_pub_style  # noqa: E402

# House figure style (bold titles, no top/right spines, shared base fonts),
# matching the neighbouring figures.
set_pub_style()
from bimp_st.print_scale import install as _install_print_scale  # noqa: E402

_install_print_scale()
from bimp_st.cluster_labels import get_pooled  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
# Must be the combined run, whose 29-class argmax over every atlas cell is
# the atlas's own labels (combined_clustered.h5ad::tangram_cell_type). Do not
# use combined_clustered_central_colfix_tangram.h5ad: that is the
# central-brain-only run (15 classes, no OL_* or generic "Kenyon cells"
# class), whose fine labels differ from the combined run's.
TG = CANONICAL_TANGRAM
OUT = REPO / "results" / "reframe" / "kc_threshold"
OUT.mkdir(parents=True, exist_ok=True)

KC_FLAGS = ["in_kc_oc", "in_kc_ic", "in_kc_nc"]
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def agree(pred: np.ndarray, ref: np.ndarray) -> dict:
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
    union = int((pred | ref).sum())
    return dict(
        n_called=int(pred.sum()),
        precision=prec,
        recall=rec,
        f1=f1,
        jaccard=tp / union if union else np.nan,
    )


def main() -> None:
    print(f"loading {TG.name} ...", flush=True)
    a = ad.read_h5ad(TG)
    obs = a.obs
    print(f"  {a.n_obs:,} cells", flush=True)

    probs = a.obsm["tangram_ct_pred"]
    if not isinstance(probs, pd.DataFrame):
        raise SystemExit(
            "tangram_ct_pred is not a labelled DataFrame; cannot "
            "identify Kenyon-cell columns by name"
        )
    kc_cols = [c for c in probs.columns if get_pooled(str(c)) == "Kenyon cells"]
    print(f"  Kenyon-cell posterior columns ({len(kc_cols)}): {kc_cols}", flush=True)

    kc_sum = probs[kc_cols].sum(1).values
    top1 = probs.values.max(1)
    argmax_kc = np.array(
        [
            get_pooled(str(c)) == "Kenyon cells"
            for c in probs.columns[probs.values.argmax(1)]
        ]
    )

    in_calyx = np.zeros(a.n_obs, bool)
    for f in KC_FLAGS:
        if f in obs.columns:
            in_calyx |= obs[f].astype(bool).values
    print(
        f"  cells inside a calyx polygon: {in_calyx.sum():,} "
        f"({100 * in_calyx.mean():.1f}%)",
        flush=True,
    )

    rows = []
    for t in GRID:
        r = agree(kc_sum >= t, in_calyx)
        r.update(
            definition="summed KC posterior",
            threshold=float(t),
            frac_of_atlas=float((kc_sum >= t).mean()),
        )
        rows.append(r)
    for t in GRID:
        r = agree((top1 >= t) & argmax_kc, in_calyx)
        r.update(
            definition="argmax KC + top-1 conf",
            threshold=float(t),
            frac_of_atlas=float(((top1 >= t) & argmax_kc).mean()),
        )
        rows.append(r)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)

    best = {}
    for d, g in sweep.groupby("definition"):
        g = g.dropna(subset=["f1"])
        if len(g):
            b = g.loc[g.f1.idxmax()]
            best[d] = b
            print(
                f"\n  {d}: best agreement with calyx anatomy at t = {b.threshold:.2f}"
                f"\n    {int(b.n_called):,} cells ({100 * b.frac_of_atlas:.1f}% of "
                f"atlas), "
                f"precision {b.precision:.3f}, recall {b.recall:.3f}, F1 {b.f1:.3f}",
                flush=True,
            )

    # pairwise overlap between definitions at their own best operating points
    defs = {"anatomy (calyx polygon)": in_calyx, "argmax KC (no threshold)": argmax_kc}
    if "summed KC posterior" in best:
        t = best["summed KC posterior"].threshold
        defs[f"summed KC posterior >= {t:.2f}"] = kc_sum >= t
    names = list(defs)
    ov = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            u = int((defs[i] | defs[j]).sum())
            ov.loc[i, j] = float((defs[i] & defs[j]).sum()) / u if u else np.nan
    ov.to_csv(OUT / "definition_overlap.csv")
    print("\n  pairwise Jaccard between definitions:", flush=True)
    print(ov.round(3).to_string(), flush=True)

    # Three lettered panels matching the caption: (A) Kenyon calls per threshold,
    # (B) F1 against calyx anatomy, (C) pairwise Jaccard between definitions.
    # Precision and recall remain in threshold_sweep.csv.
    # Drawn at the printed width (~7.2 in) with fonts near their printed size,
    # so the print-scale hook has nothing to shrink or grow.
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.05]}
    )
    # The two series are methods, not castes: a neutral black / grey pair with
    # distinct markers, never the queen-gold / worker-blue pair.
    style = {
        "summed": dict(color="#1A1A1A", marker="o", ls="-"),
        "argmax": dict(color="#8C8C8C", marker="s", ls="-"),
    }
    for d, g in sweep.groupby("definition"):
        st = style["summed" if d.startswith("summed") else "argmax"]
        kw = dict(ms=2.6, lw=1.1, label=d, **st)
        axes[0].plot(g.threshold, g.n_called / 1000, **kw)
        axes[1].plot(g.threshold, g.f1, **kw)
    axes[0].axhline(
        in_calyx.sum() / 1000,
        ls=":",
        lw=1.0,
        color="#A32B2B",
        label="cells in calyx polygons",
    )
    axes[0].set_ylabel("Kenyon calls (thousands)")
    # No panel title: caption (A) says "Cells qualifying as Kenyon cells at
    # each posterior threshold".
    axes[1].set_ylabel("F1 vs calyx anatomy")
    # Headroom above the curves so the legends sit over empty space.
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.3)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # No panel title: caption (B) says "F1 against Kenyon-cell soma anatomy
    # swept across thresholds".
    for ax in axes[:2]:
        ax.set_xlabel("Threshold")
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            frameon=False,
            loc="upper right",
            handlelength=1.6,
            borderaxespad=0.1,
            labelspacing=0.3,
        )

    # C: pairwise agreement between the three definitions
    short = {
        n: n.replace(" (calyx polygon)", "").replace(" (no threshold)", "")
        for n in ov.index
    }
    lab = [
        short[n].replace("summed KC posterior >= ", "summed KC\nposterior ≥ ")
        for n in ov.index
    ]
    # Single-hue sequential map (Jaccard runs 0 -> 1).
    cmap = plt.get_cmap("Greys")
    im = axes[2].imshow(
        ov.values.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="equal"
    )
    axes[2].set_xticks(range(len(lab)), lab, rotation=30, ha="right")
    axes[2].set_yticks(range(len(lab)), lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            v = ov.values[i, j]
            axes[2].text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > 0.55 else "#1A1A1A",
            )
    # No panel title: caption (C) says "Pairwise Jaccard agreement".
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("Jaccard index")
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=6.5)

    axes[2].set_anchor("N")
    from bimp_st.print_scale import PRINTED_WIDTH_FRAC
    from bimp_st.si_common import add_panel_headers

    fig.tight_layout(w_pad=1.2)
    add_panel_headers(
        fig,
        [
            (axes[0], "A", "Kenyon calls"),
            (axes[1], "B", "F1 sweep"),
            (axes[2], "C", "Definition overlap"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["kc_threshold"],
    )
    fig.savefig(OUT / "kc_threshold.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
