#!/usr/bin/env python3
"""Differential subcellular localization analysis across caste, cell type, and region.

Statistical approach: aggregate LP fractions per region (not per cell) to avoid
pseudoreplication. Each Xenium region (~16 queen, ~16 worker) is one replicate.
Caste tests use Mann-Whitney U on region-level means (16 vs 16).

Requires: lp.parquet files in results/bento/{brain}/{region}/ (from bento_save_lp.py)

Usage:
    .venv/bin/python pipelines/subcellular/bento_differential.py
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from bimp_st.cluster_labels import MIXED_DISPLAY, POOLED_ORDER, get_pooled_display
from bimp_st.nine_region_common import display_region
from bimp_st.style import set_pub_style

# ── Canonical caste colors (as bimp_st.style) ────────────────────────────
QUEEN_COLOR = "#DAA520"
WORKER_COLOR = "#4A90D9"
BRAIN_COLORS = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}

# ── Palette + ordering ────────────────────────────────────────────────────
PATTERN_COLS = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]
PATTERN_COLORS = {
    "cell_edge": "#17becf",
    "cytoplasmic": "#1f77b4",
    "none": "#bdbdbd",
    "nuclear": "#ff7f0e",
    "nuclear_edge": "#d62728",
}
PATTERN_LABELS = {
    "cell_edge": "Cell edge",
    "cytoplasmic": "Cytoplasmic",
    "none": "Diffuse",
    "nuclear": "Nuclear",
    "nuclear_edge": "Nuclear edge",
}

# ── Cell-type vocabulary ─────────────────────────────────────────────────
# The parquets' own ``cell_type`` column (KC-l / KC-ic / Glia-ens / N-oct ...)
# must not be plotted: it was derived from ``tangram_cell_type``, which holds the
# numeric spatial Leiden index, not a cell-type name, and its KC-ic/KC-oc
# collide with the KC-IC/KC-OC anatomical regions plotted by
# ``region_comparison``.
#
# Cell types come from ``lp_cell_labels.parquet`` (built by build_lp_labels.py
# from the combined Tangram atlas), the same canonical spatial labelling Figs 1
# and 2 use. Display strings go through get_pooled_display() so pools carry
# "(all subtypes)" and the no-confident-majority category reads "Other"
# (cluster_labels.MIXED_DISPLAY).
CT_LABEL_COL = "cell_type_canonical"
# Display strings, not the raw pool names: the column stores what
# get_pooled_display() returns, so the order/filter list must match it.
CT_ORDER = [get_pooled_display(c) for c in POOLED_ORDER] + [MIXED_DISPLAY]

LABELS_PARQUET = "results/bento/differential/lp_cell_labels.parquet"

# These are the KEYS written to region_lp_fractions.csv (index); they are kept
# as the historical KC-IC/KC-NC/KC-OC so the CSV does not change. The figure
# maps them to the reader-facing names (Small-type / Large-type / Class II)
# at draw time through nine_region_common.display_region, as every other
# figure does.
REGION_LABELS = {
    "lobula": "Lobula",
    "lamina": "Lamina",
    "medulla": "Medulla",
    "kc_oc": "KC-OC",
    "kc_nc": "KC-NC",
    "kc_ic": "KC-IC",
}


def _setup():
    set_pub_style(savefig_dpi=300)


# region_lp_summary.png and celltype_lp_summary.png are stacked as panels A and
# B of one float. Both use one fixed figure size and axes rectangle and are
# saved without a tight bbox, so LaTeX scales them identically and the fonts
# render at the same size.
STACK_FIGSIZE = (6.8, 3.6)
# The left margin holds the region / cell-type names. The legend sits below the
# axes and long cell-type names wrap to two lines, so the bars fill the canvas
# in both panels.
STACK_AXES_RECT = (0.27, 0.35, 0.70, 0.62)  # left, bottom, width, height


def _stacked_bar_fig():
    """Figure + axes with the geometry shared by the two stacked summary panels."""
    fig = plt.figure(figsize=STACK_FIGSIZE)
    ax = fig.add_axes(STACK_AXES_RECT)
    return fig, ax


def _save_stacked(fig, path: Path, letter: str):
    """Draw the panel letter and save at the full fixed canvas.

    ``set_pub_style`` sets ``savefig.bbox = "tight"`` globally; passing the
    figure's own bbox overrides it so both land on exactly STACK_FIGSIZE x 300 dpi.

    The letter is drawn on the figure (not one axes) at a fixed inch offset
    from the top-left of STACK_AXES_RECT, so it lands at the same spot on
    both panels regardless of their (letter-free) content. Sized to print at
    8 pt, matching the SI house rule (fontsize is figure-canvas points, not
    page points -- see bimp_st.style.panel_label_size).
    """
    printed_pt = 8.0
    linewidth_in = 481.9 / 72.27
    width_frac = 0.49  # \includegraphics width in the SI .tex
    fontsize = printed_pt * STACK_FIGSIZE[0] / (width_frac * linewidth_in)
    # Canvas top-left corner, clear of the axes and of the y tick labels.
    fig.text(
        0.012, 0.99, letter, fontsize=fontsize, fontweight="bold", va="top", ha="left"
    )
    fig.savefig(path, dpi=300, bbox_inches=fig.bbox_inches)
    plt.close(fig)


def _draw_stacked(frac: pd.DataFrame, patterns, path: Path, letter: str, wrap: int = 0):
    """Stacked horizontal bars of per-category pattern fractions (shared by the
    region and cell-type panels so both render from one code path)."""
    import textwrap

    fig, ax = _stacked_bar_fig()
    bottom = np.zeros(len(frac))
    for pat in patterns:
        if pat not in frac.columns:
            continue
        vals = frac[pat].values
        ax.barh(
            range(len(frac)),
            vals,
            left=bottom,
            color=PATTERN_COLORS[pat],
            label=PATTERN_LABELS[pat],
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += vals
    ax.set_yticks(range(len(frac)))
    names = [display_region(str(i)) for i in frac.index]
    labels = [textwrap.fill(n, wrap) if wrap else n for n in names]
    ax.set_yticklabels(labels)
    ax.set_xlabel("Fraction")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handlelength=1.4,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _save_stacked(fig, path, letter)


def replot_stacked(repo: Path, out: Path):
    """Re-render both panels from the saved fraction CSVs (no LP reload)."""
    out.mkdir(parents=True, exist_ok=True)
    src = repo / "results/bento/differential"
    reg = pd.read_csv(src / "region_lp_fractions.csv", index_col=0)
    _draw_stacked(reg, PATTERN_COLS, out / "region_lp_summary.png", "A")
    ct = pd.read_csv(src / "celltype_lp_fractions.csv", index_col=0).drop(
        columns=["n"], errors="ignore"
    )
    _draw_stacked(ct, PATTERN_COLS, out / "celltype_lp_summary.png", "B", wrap=22)


def _canonical_label_maps(repo: Path) -> dict[str, dict[str, str]]:
    """brain -> {cell_id: display cell-type label} from lp_cell_labels.parquet."""
    lab_path = repo / LABELS_PARQUET
    if not lab_path.exists():
        raise FileNotFoundError(
            f"{lab_path} missing — run "
            "pipelines/subcellular/build_lp_labels.py first.",
        )
    lab = pd.read_parquet(lab_path)
    lab["display"] = lab["cluster_vote"].map(get_pooled_display)
    maps = {
        b: dict(zip(g["cell_id"], g["display"], strict=True))
        for b, g in lab.groupby("brain", observed=True)
    }
    print(
        "  canonical labels: "
        + ", ".join(f"{b} {len(m):,} cells" for b, m in sorted(maps.items())),
        flush=True,
    )
    return maps


def load_all_lp(repo: Path) -> pd.DataFrame:
    """Load and concatenate LP parquets, filter to real panel genes.

    Also attaches ``cell_type_canonical``: the canonical spatial cell-type label
    re-derived from the Tangram atlas (see CT_ORDER note above). The parquets'
    own ``cell_type`` column is kept only for the "drop cells with no Tangram
    join" filter (``!= "Unknown"``); nothing is plotted from it.
    """
    parquets = sorted(repo.glob("results/bento/*/output-XETG*/lp.parquet"))
    if not parquets:
        raise FileNotFoundError("No lp.parquet files found.")
    print(f"Loading {len(parquets)} LP parquets...", flush=True)
    label_maps = _canonical_label_maps(repo)

    dfs, n_before = [], 0
    for p in parquets:
        d = pd.read_parquet(p)
        n_before += len(d)
        # Filter out Xenium controls, then cells with no Tangram join.
        d = d[~d["feature_name"].str.startswith(("NegControl", "Unassigned"))]
        d = d[d["cell_type"] != "Unknown"]
        if not len(d):
            continue
        # Per-brain map: barcodes collide between the two instrument runs.
        brain = d["brain"].iloc[0]
        d[CT_LABEL_COL] = pd.Categorical(
            d["cell_boundaries"].map(label_maps.get(brain, {})),
        )
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df[CT_LABEL_COL] = df[CT_LABEL_COL].astype("category")

    n_queen = df[df["brain"] == "queen"]["region"].nunique()
    n_worker = df[df["brain"] == "worker"]["region"].nunique()
    n_unlabelled = int(df[CT_LABEL_COL].isna().sum())
    print(f"  {len(df):,} obs ({n_before - len(df):,} controls removed)", flush=True)
    print(f"  {n_queen} queen regions, {n_worker} worker regions", flush=True)
    print(
        f"  {CT_LABEL_COL}: {n_unlabelled:,} rows unmatched "
        f"({n_unlabelled / max(len(df), 1) * 100:.2f}%)",
        flush=True,
    )
    print(df[CT_LABEL_COL].value_counts(dropna=False), flush=True)
    return df


def caste_comparison(df: pd.DataFrame, out: Path):
    """Region-level caste comparison: aggregate per region, then Mann-Whitney U."""
    print("\n1. Caste comparison (region-level, no pseudoreplication)...", flush=True)

    # Aggregate: mean LP fraction per gene × region
    region_means = (
        df.groupby(["brain", "region", "gene_name"])[PATTERN_COLS].mean().reset_index()
    )

    results = []
    for gene, gdf in region_means.groupby("gene_name"):
        queen = gdf[gdf["brain"] == "queen"]
        worker = gdf[gdf["brain"] == "worker"]
        if len(queen) < 3 or len(worker) < 3:
            continue

        row = {
            "gene": gene,
            "n_queen_regions": len(queen),
            "n_worker_regions": len(worker),
        }
        for pat in PATTERN_COLS:
            q_vals = queen[pat].values
            w_vals = worker[pat].values
            row[f"{pat}_queen"] = q_vals.mean()
            row[f"{pat}_worker"] = w_vals.mean()
            row[f"{pat}_diff"] = q_vals.mean() - w_vals.mean()
            row[f"{pat}_queen_sd"] = q_vals.std()
            row[f"{pat}_worker_sd"] = w_vals.std()
            # Mann-Whitney U (non-parametric, works with small n)
            try:
                stat, pval = stats.mannwhitneyu(
                    q_vals,
                    w_vals,
                    alternative="two-sided",
                )
            except ValueError:
                pval = 1.0
            row[f"{pat}_pval"] = pval
            # Effect size: rank-biserial correlation
            row[f"{pat}_effect"] = 1 - (2 * stat) / (len(q_vals) * len(w_vals))

        # Also test dominant pattern consistency
        for pat in PATTERN_COLS:
            q_dom = (
                queen[pat] > queen[PATTERN_COLS].drop(columns=pat).max(axis=1)
            ).mean()
            w_dom = (
                worker[pat] > worker[PATTERN_COLS].drop(columns=pat).max(axis=1)
            ).mean()
            row[f"{pat}_queen_dominant_frac"] = q_dom
            row[f"{pat}_worker_dominant_frac"] = w_dom

        results.append(row)

    res = pd.DataFrame(results)
    if len(res) == 0:
        print("  No genes with enough regions", flush=True)
        return

    # BH FDR correction per pattern
    from statsmodels.stats.multitest import multipletests

    for pat in PATTERN_COLS:
        _, padj, _, _ = multipletests(res[f"{pat}_pval"], method="fdr_bh")
        res[f"{pat}_padj"] = padj

    res.to_csv(out / "caste_lp_comparison.csv", index=False)

    # ── Heatmap of differences with significance markers ──────────────────
    # Drop "Diffuse" (none) — it dominates >90% of cells and crowds out signal.
    INFORMATIVE = ["cell_edge", "cytoplasmic", "nuclear", "nuclear_edge"]
    diff_cols = [f"{p}_diff" for p in INFORMATIVE]
    padj_cols = [f"{p}_padj" for p in INFORMATIVE]

    heatdata = res.set_index("gene")[diff_cols].copy()
    heatdata.columns = [PATTERN_LABELS[p] for p in INFORMATIVE]

    sig_data = res.set_index("gene")[padj_cols].copy()
    sig_data.columns = [PATTERN_LABELS[p] for p in INFORMATIVE]

    # Filter: require at least one significant pattern AND a meaningful
    # region-mean shift (thresholds below).
    has_sig = (sig_data < 0.05).any(axis=1)
    has_effect = (heatdata.abs() >= 0.005).any(axis=1)
    keep = has_sig & has_effect
    heatdata = heatdata.loc[keep]
    sig_data = sig_data.loc[keep]

    # Score by signed nuclear+cytoplasmic shift so queen-enriched (positive)
    # cluster at the top and worker-enriched (negative) at the bottom.
    score = heatdata["Nuclear"].fillna(0) + heatdata["Cytoplasmic"].fillna(0)
    heatdata = (
        heatdata.assign(_score=score)
        .sort_values("_score", ascending=False)
        .drop(columns="_score")
    )
    sig_data = sig_data.loc[heatdata.index]

    # Panel-sized: top 6 queen-shifted + top 6 worker-shifted = 12 genes.
    if len(heatdata) > 12:
        heatdata = pd.concat([heatdata.head(6), heatdata.tail(6)])
        sig_data = sig_data.loc[heatdata.index]

    # Shorten long protein descriptions for readability.
    def _shorten(name, maxlen=28):
        s = str(name)
        for tag in (
            " isoform X1",
            " isoform X2",
            " isoform X3",
            " isoform X4",
            " isoform X5",
        ):
            s = s.replace(tag, "")
        s = s.replace("LOW QUALITY PROTEIN_ ", "")
        s = s.replace("protein ", "")
        return s if len(s) <= maxlen else s[: maxlen - 1] + "…"

    heatdata.index = [_shorten(g) for g in heatdata.index]
    sig_data.index = heatdata.index

    annot = sig_data.map(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    # Custom diverging colormap: worker blue ← 0 → queen gold
    from matplotlib.colors import LinearSegmentedColormap

    caste_cmap = LinearSegmentedColormap.from_list(
        "worker_queen",
        [WORKER_COLOR, "#f5f5f5", QUEEN_COLOR],
    )

    vmax = float(np.nanmax(np.abs(heatdata.values))) if len(heatdata) else 0.03
    vmax = max(vmax, 0.01)

    # Compact panel size for use inside a composite figure (~3.5" × 3.5").
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    sns.heatmap(
        heatdata,
        cmap=caste_cmap,
        center=0,
        ax=ax,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 6, "color": "#222"},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.3,
        cbar_kws={
            "label": "Queen − Worker",
            "shrink": 0.5,
            "pad": 0.02,
        },
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.savefig(out / "caste_lp_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "caste_lp_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    n_sig = {pat: (res[f"{pat}_padj"] < 0.05).sum() for pat in PATTERN_COLS}
    print(f"  Significant genes (FDR<0.05): {n_sig}", flush=True)
    any_sig = res[[f"{p}_padj" for p in PATTERN_COLS]].min(axis=1) < 0.05
    print(f"  {any_sig.sum()}/{len(res)} genes significant in any pattern", flush=True)


def celltype_comparison(df: pd.DataFrame, out: Path):
    """Compare LP patterns across cell types using dominant pattern (compositional)."""
    print("\n2. Cell type comparison...", flush=True)

    if CT_LABEL_COL not in df.columns:
        print(f"  No {CT_LABEL_COL} column — skipping", flush=True)
        return

    df_ct = df[df[CT_LABEL_COL].isin(CT_ORDER)].copy()
    if len(df_ct) == 0:
        return

    # Use dominant_pattern for compositional view (sums to 1, all 5 patterns)
    ALL_PATTERNS = PATTERN_COLS  # includes "none" → "Diffuse"

    # ── Summary stacked bar (dominant pattern, all categories) ───────────
    ct_counts = (
        df_ct.groupby([CT_LABEL_COL, "dominant_pattern"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    ct_counts = ct_counts[[p for p in ALL_PATTERNS if p in ct_counts.columns]]
    ct_frac = ct_counts.div(ct_counts.sum(axis=1), axis=0)
    ct_frac = ct_frac.reindex([c for c in CT_ORDER if c in ct_frac.index])

    ct_n = ct_counts.sum(axis=1).reindex(ct_frac.index)
    for c in ct_frac.index:
        print(
            f"    {c:<32s} n={int(ct_n[c]):>10,}  "
            f"nuclear={ct_frac.loc[c, 'nuclear'] * 100:5.2f}%  "
            f"diffuse={ct_frac.loc[c, 'none'] * 100:5.2f}%",
            flush=True,
        )
    ct_frac.assign(n=ct_n).to_csv(out / "celltype_lp_fractions.csv")

    _draw_stacked(ct_frac, ALL_PATTERNS, out / "celltype_lp_summary.png", "B", wrap=22)

    # ── Nuclear fraction heatmap by gene × cell type ─────────────────────
    nuc_by_ct = (
        df_ct.groupby([CT_LABEL_COL, "gene_name"], observed=True)["nuclear"]
        .mean()
        .reset_index()
    )
    nuc_pivot = nuc_by_ct.pivot(
        index="gene_name", columns=CT_LABEL_COL, values="nuclear"
    )
    nuc_pivot = nuc_pivot[[c for c in CT_ORDER if c in nuc_pivot.columns]]

    nuc_pivot["var"] = nuc_pivot.var(axis=1)
    top_genes = nuc_pivot.nlargest(30, "var").drop("var", axis=1)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        top_genes,
        cmap="YlOrRd",
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.3,
        cbar_kws={"label": "Fraction nuclear", "shrink": 0.6},
    )
    ax.set_title("Nuclear localization by cell type\n(top 30 most variable genes)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(out / "celltype_nuclear_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  {len(ct_frac)} cell types plotted", flush=True)


def region_comparison(df: pd.DataFrame, out: Path):
    """Compare LP patterns across anatomical regions."""
    print("\n3. Anatomical region comparison...", flush=True)

    in_cols = [c for c in df.columns if c.startswith("in_")]
    if not in_cols:
        print("  No in_* columns — skipping", flush=True)
        return

    region_data = {}
    for col in in_cols:
        inside = df[df[col].eq(True)]
        if len(inside) < 500:
            continue
        rname = col.replace("in_", "")
        label = REGION_LABELS.get(rname, rname)
        counts = inside["dominant_pattern"].value_counts()
        total = counts.sum()
        region_data[label] = {p: counts.get(p, 0) / total for p in PATTERN_COLS}

    if not region_data:
        print("  No anatomical regions with enough cells", flush=True)
        return

    region_df = pd.DataFrame(region_data).T
    for r in region_df.index:
        print(
            f"    {r:<10s} nuclear={region_df.loc[r, 'nuclear'] * 100:5.2f}%  "
            f"diffuse={region_df.loc[r, 'none'] * 100:5.2f}%",
            flush=True,
        )
    region_df.to_csv(out / "region_lp_fractions.csv")

    _draw_stacked(region_df, PATTERN_COLS, out / "region_lp_summary.png", "A")
    print(f"  {len(region_df)} regions plotted", flush=True)


def interaction_comparison(df: pd.DataFrame, out: Path):
    """Region-level caste × cell_type interaction."""
    print("\n4. Caste × cell type interaction (region-level)...", flush=True)

    if CT_LABEL_COL not in df.columns:
        print(f"  No {CT_LABEL_COL} column — skipping", flush=True)
        return

    df_ct = df[df[CT_LABEL_COL].isin(CT_ORDER)].copy()
    if len(df_ct) == 0:
        return

    # Region-level means per cell type
    region_ct = (
        df_ct.groupby(["brain", "region", CT_LABEL_COL], observed=True)[PATTERN_COLS]
        .mean()
        .reset_index()
    )

    # ── Nuclear fraction: caste × cell type with error bars ──────────────
    nuc_stats = (
        region_ct.groupby([CT_LABEL_COL, "brain"], observed=True)["nuclear"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    nuc_stats["sem"] = nuc_stats["std"] / np.sqrt(nuc_stats["count"])

    nuc_pivot_mean = nuc_stats.pivot(index=CT_LABEL_COL, columns="brain", values="mean")
    nuc_pivot_sem = nuc_stats.pivot(index=CT_LABEL_COL, columns="brain", values="sem")
    nuc_pivot_mean = nuc_pivot_mean.reindex(
        [c for c in CT_ORDER if c in nuc_pivot_mean.index]
    )
    nuc_pivot_sem = nuc_pivot_sem.reindex(nuc_pivot_mean.index)

    if nuc_pivot_mean.shape[1] >= 2:
        # Paired bar chart with error bars
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(nuc_pivot_mean))
        w = 0.35

        for i, (caste, color) in enumerate(
            [("queen", QUEEN_COLOR), ("worker", WORKER_COLOR)]
        ):
            if caste in nuc_pivot_mean.columns:
                ax.barh(
                    x + (i - 0.5) * w,
                    nuc_pivot_mean[caste],
                    height=w,
                    xerr=nuc_pivot_sem[caste],
                    color=color,
                    alpha=0.8,
                    label=caste.capitalize(),
                    edgecolor="white",
                    linewidth=0.3,
                    error_kw={"linewidth": 0.8, "capsize": 2},
                )

        ax.set_yticks(x)
        ax.set_yticklabels(nuc_pivot_mean.index)
        ax.set_xlabel("Fraction nuclear (mean ± SEM across regions)")
        ax.set_title(
            "Nuclear localization: queen vs worker by cell type\n(n=16 "
            "regions per caste)"
        )
        ax.legend(frameon=False)
        ax.invert_yaxis()

        fig.savefig(out / "interaction_nuclear_bars.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ── Difference with significance per cell type ────────────────────────
    interaction_results = []
    for ct in CT_ORDER:
        ct_data = region_ct[region_ct[CT_LABEL_COL] == ct]
        q = ct_data[ct_data["brain"] == "queen"]["nuclear"].values
        w = ct_data[ct_data["brain"] == "worker"]["nuclear"].values
        if len(q) < 3 or len(w) < 3:
            continue
        try:
            stat, pval = stats.mannwhitneyu(q, w, alternative="two-sided")
        except ValueError:
            pval = 1.0
        interaction_results.append(
            {
                CT_LABEL_COL: ct,
                "queen_mean": q.mean(),
                "worker_mean": w.mean(),
                "diff": q.mean() - w.mean(),
                "queen_sem": q.std() / np.sqrt(len(q)),
                "worker_sem": w.std() / np.sqrt(len(w)),
                "pval": pval,
            }
        )

    if interaction_results:
        idf = pd.DataFrame(interaction_results)
        _, idf["padj"], _, _ = __import__(
            "statsmodels.stats.multitest", fromlist=["multipletests"]
        ).multipletests(
            idf["pval"],
            method="fdr_bh",
        )
        idf.to_csv(out / "interaction_nuclear_stats.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [QUEEN_COLOR if d > 0 else WORKER_COLOR for d in idf["diff"]]
        ax.barh(
            range(len(idf)), idf["diff"], color=colors, edgecolor="white", linewidth=0.3
        )

        # Add significance markers
        for i, row in idf.iterrows():
            marker = (
                "***"
                if row["padj"] < 0.001
                else "**"
                if row["padj"] < 0.01
                else "*"
                if row["padj"] < 0.05
                else ""
            )
            if marker:
                xpos = row["diff"] + (0.001 if row["diff"] > 0 else -0.001)
                ax.text(
                    xpos,
                    i,
                    marker,
                    va="center",
                    ha="left" if row["diff"] > 0 else "right",
                    fontsize=10,
                )

        ax.set_yticks(range(len(idf)))
        ax.set_yticklabels(idf[CT_LABEL_COL])
        ax.axvline(0, color="#333", linewidth=0.5)
        ax.set_xlabel("Queen − Worker (fraction nuclear)")
        ax.set_title(
            "Caste × cell type: nuclear localization\n(Mann-Whitney U, "
            "BH-corrected, n≈16 regions/caste)"
        )
        ax.invert_yaxis()
        fig.savefig(out / "interaction_nuclear_diff.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        sig_ct = idf[idf["padj"] < 0.05]
        print(
            f"  {len(sig_ct)}/{len(idf)} cell types with significant caste difference",
            flush=True,
        )

    print("  Saved interaction plots", flush=True)


def main():
    REPO = Path(__file__).resolve().parents[2]  # never hardcode the main checkout
    OUT = REPO / "results/bento/differential"
    OUT.mkdir(parents=True, exist_ok=True)

    _setup()
    if "--replot" in sys.argv:
        # Re-render the two stacked summary panels from their saved CSVs.
        replot_stacked(REPO, Path("results/bento/differential"))
        return
    df = load_all_lp(REPO)

    caste_comparison(df, OUT)
    celltype_comparison(df, OUT)
    region_comparison(df, OUT)
    interaction_comparison(df, OUT)

    print(f"\nAll results → {OUT}/", flush=True)


if __name__ == "__main__":
    main()
