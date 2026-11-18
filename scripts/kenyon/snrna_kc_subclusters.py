"""`fig:kc_subtypes` panel A: the snRNA KC subclustering that defines the subtypes.

Rather than a per-cell Tangram probability simplex (which shows that spatial
assignments are soft, a property of the mapping rather than of the biology),
this panel shows the thing the rest of the figure is built on: the de-novo
subclustering of Kenyon-cell nuclei in the snRNA reference.

Three sub-panels, so the panel earns its place by defining the subtypes
rather than just asserting them:

  left   snRNA KC nuclei UMAP coloured by the three de-novo subclusters
  middle top marker genes per subcluster
  right  how many of those markers the Xenium panel can actually see

The right-hand sub-panel is the caveat: if the genes that separate
these subclusters are not on the 101-gene panel, the spatial data cannot
resolve them independently and the mapping is doing all the work. The KC
populations are highly correlated on the panel; this quantifies which genes
that correlation is hiding.

fig_kc_subtypes.py imports ``compute_kc_subcluster_data`` and the palette and
labels below, so the main figure and this standalone figure share one
definition of the subclustering; ``draw_snuc_kc_panels`` draws the three
sub-panels into caller-supplied axes.

Run standalone (writes results/reframe/fig5_snuc_kc/):
    results/figures/si/figS_snuc_kc_subclusters.png / .pdf
    kc_subcluster_markers.csv
    marker_panel_coverage.csv
"""

from __future__ import annotations

import functools
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Text is normalised to the house printed sizes at save time; this script
# reaches neither bimp_st.style.set_pub_style() nor scripts/_si_common,
# so the hook is installed here.
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import PANEL_CROSSWALK  # noqa: E402
from bimp_st.print_scale import PRINTED_WIDTH_FRAC  # noqa: E402
from bimp_st.print_scale import install as _install_print_scale  # noqa: E402
from bimp_st.style import set_pub_style  # noqa: E402

_install_print_scale()

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

REPO = Path(__file__).resolve().parents[2]
# CB reference's own Large-type Kenyon cells, not the separate ltKC dataset
# data/sn/adata_ltKC_preprocessed.h5ad. That object is pooled from five
# libraries outside the main reference, mostly worker, with all queen
# nuclei from one library. This one is the Large-type nuclei subset
# directly from BIMP_atlas_qw_CentralBrain.h5ad (the reference every
# other panel in this figure already uses) -- both castes in every
# cluster and spread across all 3 queen + 3 worker libraries. See
# scripts/kenyon/large_type_kc_subclusters.py.
LTKC = REPO / "results" / "fig5_kc" / "CB_LargeType_subclustered.h5ad"
PANEL_CSV = PANEL_CROSSWALK
# GFF-derived genome-wide gene table (Chromosome/Start/End/Strand/gene_id/
# gene_biotype/gene/description), keyed by BIMP_v4.3_* id. Not tracked in
# git, but this is the same lookup pipelines/snrna_reference/
# sn_reference.py already loads (as BIMP_LOOKUP) for marker-gene
# descriptions -- reuse it here instead of showing bare numeric ids.
GENE_LOOKUP_TSV = (
    REPO / "pipelines" / "combined_atlas" / "data" / "BIMP_v4.3_lookup.tsv"
)
OUT = Path("results/reframe/fig5_snuc_kc")
OUT.mkdir(parents=True, exist_ok=True)

CLUSTER_COL = "leiden_res0_05"
CLUSTERS = ["0", "1", "2"]
# Subcluster 2 is a deep blue, not a purple that would sit next to the
# Small-type compartment's mauve (#B07AA1) on the same `fig:kc_subtypes` canvas:
# a clean green / blue / red triad for the per-cell colour blend of panels F/G,
# darker and
# more saturated than the worker caste blue (#4A90D9) so the two do not read
# as one.
COLORS = {"0": "#0B775E", "1": "#27408B", "2": "#F2300F"}
LABELS = {"0": "Subcluster 1", "1": "Subcluster 2", "2": "Subcluster 3"}
N_MARKERS = 8

# Short display symbols for markers whose lookup entry is a protein
# description rather than a gene symbol (standalone figure, panel B, only;
# kc_subcluster_markers.csv keeps the lookup's own gene_name). Each is the
# conventional abbreviation of the description it replaces.
DISPLAY_SYMBOL = {
    "BIMP_v4.3_009415": "RasGRF1",  # ras-specific guanine nucleotide-releasing factor 1
    "BIMP_v4.3_002198": "SLC25A8",  # dicarboxylate carrier SLC25A8
    "BIMP_v4.3_004496": "PHACTR2",  # phosphatase and actin regulator 2
    # no conventional short name; its fly ortholog (high-confidence RBH)
    "BIMP_v4.3_005504": "Pp2B-14D",  # Ser/Thr phosphatase 2B catalytic subunit 2
    "BIMP_v4.3_009612": "Octβ1R",  # octopamine receptor beta-1R
}

# The shared house style rather than a local rcParams block, so type sizes and
# bold subpanel titles match the neighbouring figures.
set_pub_style()


def _shorten_description(desc: str, max_len: int = 34) -> str:
    """Collapse a full protein description into an axis-tick-sized label.

    Not a real gene symbol -- the least-bad label for genes whose lookup
    entry has a text description but no short symbol. A pure word-greedy
    wrap looks fine most of the time but degrades badly on descriptions
    that lead with one long (often slash/hyphen-joined) word, e.g.
    "serine/threonine-protein phosphatase ..." or "protein O-mannosyl-
    transferase ...": greedily keeping only whole words collapses those to
    "serine/threonine-p..." or the useless "protein...". Instead: hard-cut
    at max_len characters, then trim back to the last word boundary only if
    that doesn't throw away more than half the budget -- so short leading
    words still wrap cleanly, but a single long leading word gets a
    mid-word (but information-dense) cut instead of a near-empty one.
    """
    desc = desc.strip().strip('"')
    if len(desc) <= max_len:
        return desc
    cut = desc[:max_len]
    last_space = cut.rfind(" ")
    if last_space >= max_len * 0.5:
        cut = cut[:last_space]
    return cut.rstrip() + "..."


@functools.lru_cache(maxsize=1)
def _load_gene_names() -> dict:
    """v4.3 gene_id -> short display name, from GENE_LOOKUP_TSV.

    Prefers the real gene symbol; falls back to a shortened description;
    genes with neither (or missing from the table entirely) are left out of
    the dict, and callers fall back to the bare numeric id -- that fallback
    stays visually distinct (digits only, no letters) so a reader can tell
    which markers are annotated and which aren't.
    """
    if not GENE_LOOKUP_TSV.exists():
        print(
            f"  WARNING: gene lookup not found at {GENE_LOOKUP_TSV}; "
            "marker labels will fall back to bare numeric ids",
            flush=True,
        )
        return {}
    df = pd.read_csv(GENE_LOOKUP_TSV, sep="\t")
    names: dict[str, str] = {}
    for _, row in df.iterrows():
        gid = row.get("gene_id")
        if not isinstance(gid, str):
            continue
        sym = row.get("gene")
        if isinstance(sym, str) and sym.strip():
            names[gid] = sym.strip()
            continue
        desc = row.get("description")
        if isinstance(desc, str) and desc.strip():
            names[gid] = _shorten_description(desc)
    return names


def compute_kc_subcluster_data(seed: int = 0) -> dict:
    """Load the snRNA KC reference, subcluster labels, rank marker genes, and
    score Xenium-panel coverage. Pure computation, no plotting -- shared by
    the standalone script (below) and fig_kc_subtypes.py."""
    panel = read_panel_crosswalk()
    panel_ids = set(panel["v43_id"].dropna())

    print(f"loading {LTKC.name} ...", flush=True)
    ref = sc.read_h5ad(LTKC)
    ref.obs[CLUSTER_COL] = ref.obs[CLUSTER_COL].astype(str)
    ref = ref[ref.obs[CLUSTER_COL].isin(CLUSTERS)].copy()
    counts = ref.obs[CLUSTER_COL].value_counts().to_dict()
    print(
        f"  {ref.n_obs:,} KC nuclei -> "
        + ", ".join(f"{LABELS[c]} {counts.get(c, 0):,}" for c in CLUSTERS),
        flush=True,
    )

    if "X_umap" not in ref.obsm:
        print("  no stored UMAP; recomputing ...", flush=True)
        sc.pp.neighbors(ref, n_neighbors=15, random_state=seed)
        sc.tl.umap(ref, random_state=seed)

    print("  ranking marker genes ...", flush=True)
    sc.tl.rank_genes_groups(ref, CLUSTER_COL, method="wilcoxon", n_genes=200)
    rg = ref.uns["rank_genes_groups"]

    gene_names = _load_gene_names()
    rows = []
    for c in CLUSTERS:
        for rank, (g, lfc, p, padj) in enumerate(
            zip(
                rg["names"][c],
                rg["logfoldchanges"][c],
                rg["pvals"][c],
                rg["pvals_adj"][c],
                strict=False,
            )
        ):
            rows.append(
                dict(
                    subcluster=LABELS[c],
                    rank=rank + 1,
                    gene=g,
                    gene_name=gene_names.get(g, ""),
                    log2fc=float(lfc),
                    pval=float(p),
                    padj=float(padj),
                    on_xenium_panel=g in panel_ids,
                )
            )
    mk = pd.DataFrame(rows)

    # coverage: of the genes that actually separate the subclusters, how many
    # can the spatial panel see?
    cov = []
    for c in CLUSTERS:
        sub = mk[mk.subcluster == LABELS[c]]
        for n in (10, 25, 50, 100, 200):
            top = sub[sub["rank"] <= n]
            sig = top[top.padj < 0.05]
            cov.append(
                dict(
                    subcluster=LABELS[c],
                    top_n=n,
                    n_significant=len(sig),
                    n_on_panel=int(top.on_xenium_panel.sum()),
                    frac_on_panel=float(top.on_xenium_panel.mean()),
                )
            )
    cvd = pd.DataFrame(cov)

    return dict(ref=ref, counts=counts, markers=mk, coverage=cvd, panel_ids=panel_ids)


def draw_snuc_kc_panels(
    ax_umap,
    ax_markers,
    ax_cov,
    data,
    n_markers=N_MARKERS,
    seed=0,
    legend_below=False,
    max_label_len=None,
    spread_labels=False,
):
    """Draw the three snRNA KC subcluster sub-panels into caller-supplied
    axes. `data` is the dict returned by compute_kc_subcluster_data()."""
    ref = data["ref"]
    counts = data["counts"]
    mk = data["markers"]
    cvd = data["coverage"]
    panel_ids = data["panel_ids"]

    # left: UMAP
    xy = ref.obsm["X_umap"]
    lab = ref.obs[CLUSTER_COL].values
    order = np.random.default_rng(seed).permutation(len(xy))  # interleave draw order
    ax_umap.scatter(
        xy[order, 0],
        xy[order, 1],
        c=[COLORS[lab[i]] for i in order],
        s=6,
        linewidths=0,
        rasterized=True,
    )
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    # No title: caption (A) already gives the nuclei count and subcluster
    # colouring.
    for s in ax_umap.spines.values():
        s.set_visible(False)
    # Label the clusters in place rather than with a legend box: there is no
    # reliably empty corner in this embedding, and any fixed corner would
    # drift the next time the UMAP is recomputed. Direct labels cannot collide
    # with the cloud they name, and they save the reader a colour-matching step.
    from matplotlib import patheffects as pe

    halo = [pe.withStroke(linewidth=3.2, foreground="white")]
    if spread_labels:
        # Fixed, non-overlapping anchors: Subcluster 3 above its cloud,
        # Subclusters 1 and 2 under their clouds, split at the midpoint
        # between the two centroids (centroid placement overlaps 1 and 2 at
        # house size).
        ext = {
            c: (xy[lab == c, 0].mean(), xy[lab == c, 1].min(), xy[lab == c, 1].max())
            for c in CLUSTERS
            if (lab == c).any()
        }
        ypad = 0.02 * (xy[:, 1].max() - xy[:, 1].min())
        xmid = (ext["0"][0] + ext["1"][0]) / 2
        xgap = 0.05 * (xy[:, 0].max() - xy[:, 0].min())
        specs = {
            "2": (ext["2"][0], ext["2"][2] + ypad, "center", "bottom"),
            "1": (xmid - xgap, min(ext["1"][1], ext["0"][1]) - ypad, "right", "top"),
            "0": (xmid + xgap, min(ext["1"][1], ext["0"][1]) - ypad, "left", "top"),
        }
        for c, (tx, ty, ha, va) in specs.items():
            ax_umap.text(
                tx,
                ty,
                f"{LABELS[c]}\n{counts.get(c, 0):,}",
                ha=ha,
                va=va,
                fontsize=7.0,
                fontweight="bold",
                color=COLORS[c],
                zorder=6,
                path_effects=halo,
                linespacing=1.0,
            )
    else:
        for c in CLUSTERS:
            m = lab == c
            if not m.any():
                continue
            cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
            ax_umap.text(
                cx,
                cy,
                f"{LABELS[c]}\n{counts.get(c, 0):,}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=COLORS[c],
                zorder=6,
                # white halo so the text stays legible over its own dense points
                path_effects=halo,
            )

    # middle: marker dotplot-ish -- mean expression z-score of top markers
    picks, tick_lab, tick_col = [], [], []
    for c in CLUSTERS:
        sub = mk[(mk.subcluster == LABELS[c]) & (mk.padj < 0.05)].head(n_markers)
        for _, r in sub.iterrows():
            picks.append(r.gene)
            # real symbol / shortened description when the GFF-derived
            # lookup has one (see _load_gene_names); otherwise fall back to
            # the bare numeric id -- digits-only vs. a word-based name makes
            # the unannotated markers visually obvious rather than hiding
            # them behind a fabricated label.
            has_name = isinstance(r.gene_name, str) and r.gene_name
            label = r.gene_name if has_name else r.gene.replace("BIMP_v4.3_", "")
            label = DISPLAY_SYMBOL.get(r.gene, label)
            if max_label_len and len(label) > max_label_len:
                label = label[: max_label_len - 3].rstrip() + "..."
            tick_lab.append(label + ("*" if r.on_xenium_panel else ""))
            tick_col.append(COLORS[c])

    if picks:
        expr = sc.get.obs_df(ref, keys=picks + [CLUSTER_COL])
        m = expr.groupby(CLUSTER_COL, observed=True)[picks].mean()
        z = (m - m.mean()) / m.std().replace(0, 1)
        z = z.reindex(CLUSTERS)
        im = ax_markers.imshow(
            z.values, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6
        )
        ax_markers.set_yticks(range(len(CLUSTERS)), [LABELS[c] for c in CLUSTERS])
        ax_markers.set_xticks(range(len(tick_lab)), tick_lab, rotation=90)
        for t, col in zip(ax_markers.get_xticklabels(), tick_col, strict=False):
            t.set_color(col)
            # Gene symbols in italics; bare numeric ids (no symbol) upright.
            if not t.get_text().rstrip("*").isdigit():
                t.set_style("italic")
        # No title: caption (B) already reads "Top marker genes per
        # subcluster." The asterisk key stays -- it is not in the caption.
        ax_markers.text(
            1.0,
            1.02,
            "* on the Xenium panel",
            transform=ax_markers.transAxes,
            fontsize=6.0,
            color="#555555",
            ha="right",
            va="bottom",
        )
        cb = ax_markers.figure.colorbar(im, ax=ax_markers, fraction=0.04, pad=0.03)
        cb.set_label("Mean expression (z-score)")
        cb.set_ticks([-1.5, 0, 1.5])

    # right: panel coverage
    w = 0.26
    xs = np.arange(len(CLUSTERS))
    for i, n in enumerate((10, 25, 50)):
        vals = [
            cvd[(cvd.subcluster == LABELS[c]) & (cvd.top_n == n)].frac_on_panel.iloc[0]
            * 100
            for c in CLUSTERS
        ]
        ax_cov.bar(
            xs + (i - 1) * w,
            vals,
            w,
            label=f"top {n}",
            color=["#C8C3B8", "#8A8275", "#4A4438"][i],
        )
    n_vars = ref.n_vars
    ax_cov.axhline(
        100 * len(panel_ids) / n_vars,
        ls="--",
        lw=1,
        color="#A32B2B",
        label="panel share of all genes",
    )
    ax_cov.set_xticks(xs, [LABELS[c].replace("Subcluster ", "") for c in CLUSTERS])
    ax_cov.set_xlabel("Subcluster")
    ax_cov.set_ylabel("% of markers on the Xenium panel")
    # No title: caption (C) already reads "Overlap of those markers with the
    # Xenium panel," and draws this panel's conclusion in its own sentence.
    if legend_below:
        # Below the axes, one entry per row: the top-10 bars reach the top of
        # the y range in Sub 2 / Sub 3, so no corner inside the axes is clear.
        ax_cov.legend(
            frameon=False,
            fontsize=6.0,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(-0.3, -0.24),
            handlelength=1.4,
            labelspacing=0.3,
        )
    else:
        ax_cov.legend(frameon=False, fontsize=8)


def main() -> None:
    data = compute_kc_subcluster_data(seed=0)
    mk, cvd = data["markers"], data["coverage"]
    mk.to_csv(OUT / "kc_subcluster_markers.csv", index=False)
    cvd.to_csv(OUT / "marker_panel_coverage.csv", index=False)
    top50 = cvd[cvd.top_n == 50]
    print("\n  of the top 50 markers per subcluster, on the Xenium panel:", flush=True)
    for _, r in top50.iterrows():
        print(
            f"    {r.subcluster}: {int(r.n_on_panel)}/50 "
            f"({100 * r.frac_on_panel:.0f}%)",
            flush=True,
        )

    # 7.2 in wide (the printed width), so text is authored near printed size.
    plt.rcParams.update(
        {
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.5,
        }
    )
    fig = plt.figure(figsize=(7.2, 3.1))
    # A colorbar axes added via fig.colorbar(ax=...) is positioned once at
    # call time, so the B-C gap is set here, before that axes is created.
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.6, 0.7],
        wspace=0.68,
        left=0.03,
        right=0.99,
        top=0.90,
        bottom=0.33,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    draw_snuc_kc_panels(
        ax, ax2, ax3, data, legend_below=True, max_label_len=26, spread_labels=True
    )
    # Smaller points for the 7.2 in canvas.
    for coll in ax.collections:
        coll.set_sizes([1.2])
    # Room inside the UMAP axes so the in-place cluster labels neither run
    # off the edge nor over the axis labels.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.05 * (x1 - x0), x1 + 0.05 * (x1 - x0))
    ax.set_ylim(y0 - 0.20 * (y1 - y0), y1 + 0.14 * (y1 - y0))

    # Panels are lettered (A)/(B)/(C).
    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [(ax, "A", None), (ax2, "B", None), (ax3, "C", None)],
        width_frac=PRINTED_WIDTH_FRAC["figS_snuc_kc_subclusters"],
    )

    fig_dir = Path("results/figures/si")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            fig_dir / f"figS_snuc_kc_subclusters.{ext}", dpi=400, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
