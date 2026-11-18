"""SI figure: between-region beta diversity (Bray-Curtis), castes pooled.

Counterpart of `region_diversity.py::make_figure`. Reads
results/region_diversity/beta_diversity_matrix.csv and REGION_ORDER from
region_diversity.py, so nothing is recomputed and the seven statistical
regions match that script (ellipsoid_body/fan_shaped_body are
descriptive-only and not included).

One panel: the region x region Bray-Curtis heatmap (label_expr composition,
full-N, castes pooled), the optic-vs-calyx split that the PERMANOVA numbers
in the caption quantify. The Hill diversity profile is in
fig:caste_comparison (panel D).

Run:
    PYTHONPATH=src:scripts .venv/bin/python scripts/regions/si_hill_diversity.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bimp_st.paths import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from regions.region_diversity import REGION_ORDER  # noqa: E402

from bimp_st.nine_region_common import display_region  # noqa: E402
from bimp_st.print_scale import PRINTED_WIDTH_FRAC  # noqa: E402
from bimp_st.si_common import (  # noqa: E402
    OUTPUT_DIR,
    set_style,
)
from bimp_st.si_common import (
    save as _save,
)

DIV_DIR = REPO / "results" / "region_diversity"
NAME = "figS_beta_diversity"


def main() -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    beta = pd.read_csv(DIV_DIR / "beta_diversity_matrix.csv")
    # CSVs written before the Large-type/Small-type/Class II renaming still
    # carry 'KC-NC'-style region names; map them at load.
    for _c in ("region_a", "region_b"):
        beta[_c] = beta[_c].map(display_region)

    # Authored at its printed width (PRINTED_WIDTH_FRAC of the 7.2 in text
    # block), so text is drawn near printed size.
    plt.rcParams.update(
        {
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    width = 7.2 * PRINTED_WIDTH_FRAC[NAME]
    fig, ax = plt.subplots(figsize=(width, 0.8 * width))

    # ── Region x region Bray-Curtis beta diversity ─────────────────────────
    # The Unannotated-included primary scheme, as in fig:caste_comparison D.
    bc = beta[beta.metric == "bray_curtis"]
    if "mixed_excluded" in bc.columns:
        # Some beta_diversity_matrix.csv versions carry both the included and
        # excluded variants (two rows per region pair); keep the included one.
        bc = bc[~bc.mixed_excluded]
    # One row per (region_a, region_b) is the precondition .pivot() needs;
    # assert it rather than let the reshape fail with an unclear message.
    assert not bc.duplicated(["region_a", "region_b"]).any(), (
        "beta_diversity_matrix.csv has more than one bray_curtis row per "
        "region pair; regenerate it with scripts/regions/region_diversity.py"
    )
    bc = bc.pivot(index="region_a", columns="region_b", values="value")
    bc = bc.loc[REGION_ORDER, REGION_ORDER]
    # Mask the diagonal and the upper triangle. A region's dissimilarity from
    # itself is zero by definition, and drawn in the same colormap that zero
    # reads as a finding; the mirror image adds no information either.
    vals = np.array(bc.values, dtype=float)
    # Only the strict lower triangle carries information (the diagonal is
    # zero by definition, the upper triangle mirrors the lower), so the first
    # row and the last column are dropped and the rest drawn as a triangle.
    rows, cols = REGION_ORDER[1:], REGION_ORDER[:-1]
    sub = vals[1:, :-1]
    shown = np.ma.masked_array(sub, mask=np.triu(np.ones_like(sub, bool), k=1))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(rows)
    ax.spines[["top", "right"]].set_visible(False)
    # Text colour from the cell's own luminance rather than a fixed value
    # cut-off, so labels stay readable on magma's near-black cells.
    for i in range(len(rows)):
        for j in range(i + 1):
            v = float(sub[i, j])
            r, g, b, _ = cmap(v)
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="white" if lum < 0.55 else "#1A1A1A",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Bray-Curtis dissimilarity", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5)
    # No title: the caption reads "Bray-Curtis dissimilarity between
    # regions, castes pooled."

    # One panel, so no panel letter or header: the caption carries the title.
    fig.tight_layout()

    _save(fig, OUTPUT_DIR, NAME)
    print(f"saved {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
