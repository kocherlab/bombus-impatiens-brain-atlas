"""SI figure: composition of the snRNA-seq reference, library by library.

The reference is dissected tissue weighted by nuclei yield, not brain
composition, so it is shown here on its own rather than beside the spatial
atlas. Cell types collapse to the same six categories as the atlas bar of
``fig:caste_comparison``
(explicit map copied from ``fig_caste_comparison._SN_MAP``; OL_1..OL_13 pool into
"Optic lobe"). Panel A: the six central-brain libraries (queen t1-t3, then
worker t1-t3). Panel B: the single optic-lobe library per caste.

Inputs (obs only, no expression):
    bimp_st.paths.SN_COMBINED  (sample, caste, region, trial, cell_type)

Outputs (under REPO/results/figures/si/):
    figS_reference_composition.png (+ .pdf)
    fig_s_reference_composition_by_library.csv
        (sample, caste, region, label, n_nuclei, frac)

REPO is ``$BIMP_REPO_ROOT`` if set, else the repository containing this file.

Run:
    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/identity/si_reference_composition.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from bimp_st.cluster_labels import (  # noqa: E402
    DISPLAY_COLOR_OVERRIDES,
    DISPLAY_NAME_OVERRIDES,
    _norm_ct,
    get_pooled_color,
    get_pooled_display,
)
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import SN_COMBINED  # noqa: E402

# Importing si_common installs the print_scale savefig hook, which keys on the
# output stem (PRINTED_WIDTH_FRAC["figS_reference_composition"]).
from bimp_st.si_common import set_style  # noqa: E402

REPO = Path(os.environ.get("BIMP_REPO_ROOT", Path(__file__).resolve().parents[2]))
OUT = REPO / "results/figures/si"
STEM = "fig_s_reference_composition"

OTHER = "Mixed"  # internal key of the residual category; drawn as "Other"
CLASSES = [
    "Kenyon cells",
    "Astrocyte-like glia",
    "Perineurial glia",
    "Glia",
    "Optic lobe",
    OTHER,
]
CLASS_COL = {c: DISPLAY_COLOR_OVERRIDES.get(c, get_pooled_color(c)) for c in CLASSES}

# Same map as fig_caste_comparison._SN_MAP (snRNA cell_type -> atlas-bar category).
_SN_MAP = {
    "Large-type Kenyon cells": "Kenyon cells",
    "Middle-type Kenyon cells": "Kenyon cells",
    "Small-type Kenyon cells": "Kenyon cells",
    "Class II Kenyon cells": "Kenyon cells",
    "Other Kenyon cells": "Kenyon cells",
    "Kenyon cells": "Kenyon cells",
    "Astrocyte-like glia": "Astrocyte-like glia",
    "Perineurial glia": "Perineurial glia",
    "Cortex glia": "Glia",
    "Subperineurial glia": "Glia",
    "Other glia": "Glia",
    "Optic lobe": "Optic lobe",
    "Anterodorsal projection neurons": OTHER,
    "Lateral projection neurons": OTHER,
    "Other neurons": OTHER,
    "Photoreceptors": OTHER,
    "Hemocytes": OTHER,
}

IN_FS, CALL_FS, TICK_FS = 7.5, 7.0, 8.5
MIN_LABEL_FRAC = 0.018
BH = 0.70  # bar height (data units)
GAP = 0.17  # min horizontal spacing between callouts (axis fraction)


def _short(c):
    if c == OTHER:
        return "Other"
    n = DISPLAY_NAME_OVERRIDES.get(str(c), get_pooled_display(c))
    n = n.replace(" (all subtypes)", "").replace("Optic lobe cells", "Optic lobe")
    return n.replace("Astrocyte-like glia", "Astrocyte-like")


def _lum(h):
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _txt_on(h):
    return "white" if _lum(h) < 0.6 else "#1a1a1a"


def _ink(col, target=0.42):
    """Darken a fill colour enough to read as text on white."""
    import matplotlib.colors as mc

    r, g, b = mc.to_rgb(col)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum <= target:
        return col
    f = target / max(lum, 1e-6)
    return (r * f, g * f, b * f)


def _spread(xs, gap):
    out = list(xs)
    for _ in range(80):
        moved = False
        for i in range(len(out) - 1):
            short = gap - (out[i + 1] - out[i])
            if short > 1e-9:
                out[i] -= short / 2
                out[i + 1] += short / 2
                moved = True
        if not moved:
            break
    return [min(max(x, 0.06), 0.94) for x in out]


def load_composition() -> pd.DataFrame:
    obs = read_obs(SN_COMBINED)
    ct = obs["cell_type"].astype(str).map(_norm_ct)
    ct = ct.where(~ct.str.fullmatch(r"OL_\d+"), "Optic lobe")
    unk = sorted(set(ct) - set(_SN_MAP))
    assert not unk, f"cell types not in _SN_MAP: {unk}"
    df = pd.DataFrame(
        {
            "sample": obs["sample"].astype(str).values,
            "caste": obs["caste"].astype(str).str.lower().values,
            "region": obs["region"].astype(str).values,
            "label": ct.map(_SN_MAP).values,
        }
    )
    n = (
        df.groupby(["sample", "caste", "region", "label"])
        .size()
        .unstack("label")
        .reindex(columns=CLASSES)
        .fillna(0)
        .astype(int)
    )
    tot = n.sum(axis=1)
    rows = []
    for idx, r in n.iterrows():
        for lab in CLASSES:
            rows.append(
                dict(
                    zip(("sample", "caste", "region"), idx, strict=False),
                    label=_short(lab),
                    n_nuclei=int(r[lab]),
                    frac=r[lab] / tot[idx],
                    n_total=int(tot[idx]),
                )
            )
    return pd.DataFrame(rows)


def _text_w(ax, fig, s, fs):
    t = ax.text(0, 0, s, fontsize=fs, fontweight="bold")
    w = t.get_window_extent(fig.canvas.get_renderer()).width
    t.remove()
    return w


def stack(ax, fig, y, vals):
    """One stacked bar; 'name value' inside a segment if it fits, else the
    value alone if that fits, else nothing (the CSV carries every number)."""
    left = 0.0
    for cls, p in zip(CLASSES, vals, strict=False):
        col = CLASS_COL[cls]
        ax.barh(y, p, left=left, height=BH, color=col, edgecolor="white", linewidth=0.9)
        seg = (
            ax.transData.transform((left + p, 0))[0]
            - ax.transData.transform((left, 0))[0]
        )
        for txt in (f"{_short(cls)} {p:.3f}", f"{p:.3f}"):
            if p > 0 and _text_w(ax, fig, txt, IN_FS) + 4 <= seg:
                ax.text(
                    left + p / 2,
                    y,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=IN_FS,
                    color=_txt_on(col),
                    fontweight="bold",
                )
                break
        left += p


def main():
    set_style()
    comp = load_composition()
    OUT.mkdir(parents=True, exist_ok=True)
    comp.drop(columns=["n_total"]).to_csv(OUT / f"{STEM}_by_library.csv", index=False)

    def frac_of(sample):
        s = comp[comp["sample"] == sample]
        lut = dict(zip(s["label"], s["frac"], strict=True))
        return [lut[_short(c)] for c in CLASSES], int(s["n_total"].iloc[0])

    cb_rows = []
    for cap in ("Queen", "Worker"):
        for t in ("t1", "t2", "t3"):
            v, n = frac_of(f"BIMP_{cap}_{t}")
            cb_rows.append((f"{cap} {t[1]} (n = {n:,})", v))
    ol_rows = []
    for cap in ("Queen", "Worker"):
        v, n = frac_of(f"BIMP_{cap}_OpticLobe")
        ol_rows.append((f"{cap} (n = {n:,})", v))

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[6.4, 2.4],
        hspace=0.34,
        left=0.17,
        right=0.985,
        top=0.93,
        bottom=0.12,
    )
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    def draw(ax, rows, ys, ylim):
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        fig.canvas.draw()
        for (_label, vals), y in zip(rows, ys, strict=False):
            stack(ax, fig, y, vals)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=TICK_FS)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    ysA = [5.5, 4.5, 3.5, 2.0, 1.0, 0.0]
    draw(axA, cb_rows, ysA, (-0.5, 6.0))
    axA.axhline(2.75, color="0.75", lw=0.6)
    axA.set_xticks([])
    axA.spines["bottom"].set_visible(False)
    ysB = [1.0, 0.0]
    draw(axB, ol_rows, ysB, (-0.5, 1.5))
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axB.tick_params(axis="x", labelsize=TICK_FS)
    axB.set_xlabel("Fraction of nuclei", fontsize=9, labelpad=3)
    axB.text(
        1.0,
        1.02,
        "n = 1 library per caste",
        transform=axB.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        style="italic",
        color="0.3",
    )
    handles = [
        Patch(
            facecolor=CLASS_COL[c],
            edgecolor="none",
            label=_short(c).replace("Astrocyte-like", "Astrocyte-like glia"),
        )
        for c in CLASSES
    ]
    axA.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=7.5,
        handlelength=1.0,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [(axA, "A", "Central brain libraries"), (axB, "B", "Optic lobe libraries")],
        width_frac=1.0,
    )
    for ext in ("png", "pdf"):
        out = OUT / f"figS_reference_composition.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(
        comp.pivot_table(index="sample", columns="label", values="frac")
        .round(4)
        .to_string()
    )
    print(comp.groupby("sample")["n_total"].first().to_string())


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
