#!/usr/bin/env python3
"""SI figure: cross-modal concordance of caste-biased expression.

Supports the abstract's claim that "caste-biased expression is concordant
across measurement modalities" by comparing, gene by gene, the caste effect
measured in the Xenium spatial atlas against the caste effect measured in the
snRNA-seq reference.

Contrasts (all on the 101-gene Xenium panel, the only shared feature space):

  * spatial  = per-slice pseudobulk log2(CPM+1), queen minus worker, computed
    from ``combined_clustered.h5ad`` raw counts.  This is a descriptive effect
    size only: the spatial cohort is one queen brain and one worker brain, so
    the 30 queen / 35 worker slices are within-brain sections, not biological
    replicates.  No spatial caste p-values are computed or plotted (one brain
    per caste).
  * snRNA    = DESeq2 pseudobulk, ~ trial + caste (Wald, dispersion shrinkage, BH) on
    three independent queen and three independent worker bees -- the study's
    only caste-replicated differential expression.

Panels A-C are cell-type matched (whole reference, Kenyon cells, glia) so the
figure shows *where* the cross-modal agreement lives.

    PYTHONPATH=src .venv/bin/python scripts/caste_de/si_deg_concordance.py

NOTE: do not use ``results/figures/fig3/{spatial,snrna}_deg_queen_worker.csv``.
Those pre-atlas-migration (Mar 2026) tables come from cell-level t-test /
Wilcoxon tests on 1.3M cells and 10.6k nuclei (pseudoreplicated), and their
spatial log2FC is sign-inverted relative to the current pipeline.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from scipy.stats import binomtest, pearsonr, spearmanr

from bimp_st.caste_de import focal_gene_ids
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (
    CANONICAL_TANGRAM,
    CASTE_DE_COARSE,
    CASTE_DE_DIR,
    PANEL_CROSSWALK,  # noqa: E402
    REPO,
    explain_missing_inputs,
)
from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import (
    QUEEN_COLOR,
    WORKER_COLOR,
    add_panel_headers,
    set_style,
)
from bimp_st.si_common import (
    save as _save,
)

# ── Configuration ────────────────────────────────────────────────────────────
# BIMP_DEGCONC_OUT redirects the figure, its tables and the spatial pseudobulk
# cache (a fresh cache is computed there), so regenerating never overwrites the
# registered files under results/figures/si.
OUTPUT_DIR = Path(
    os.environ.get("BIMP_DEGCONC_OUT", str(CASTE_DE_DIR / "deg_concordance"))
)
CACHE_DIR = OUTPUT_DIR / "_cache"

SPATIAL_H5AD = "pipelines/combined_atlas/data/combined_clustered.h5ad"
# The spatial object carries its own tangram_cell_type, which is not the
# canonical labelling: never vote on it. Take the labels from the canonical
# Tangram object, as si_panel_concordance.py does.
# BIMP_TANGRAM_LABELS points the per-cell labels at another run (e.g. the
# generic-entry-free remap) for a sensitivity check; the default is the canonical run.
TANGRAM_LABELS = os.environ.get("BIMP_TANGRAM_LABELS", str(CANONICAL_TANGRAM))
DESEQ2_CSV = CASTE_DE_COARSE.relative_to(REPO)  # ~ trial + caste
GENE_MAP_CSV = PANEL_CROSSWALK

# (label, spatial pooled group, snRNA DESeq2 cell_type)
CONTRASTS = [
    ("Whole brain", None, "all"),
    ("Kenyon cells", "Kenyon cells", "Kenyon_cells"),
    ("Glia", "Glia", "Glia"),
]

# short display symbols for genes that get named in a panel; the caste-DE focal
# genes come from bimp_st.caste_de (looked up, not typed)
_FOCAL = focal_gene_ids()
FOCAL_V43 = {
    _FOCAL["sNPF"]: "sNPF",
    _FOCAL["Kr-h1"]: "Kr-h1",
    "BIMP_v4.3_009307": "oct-b2R",
    "BIMP_v4.3_009612": "oct-b1R",
    "BIMP_v4.3_009024": "OAMB",
    "BIMP_v4.3_001923": "Tret1",
    _FOCAL["Tk"]: "Tk",
    "BIMP_v4.3_004137": "Hsp90",
    "BIMP_v4.3_002958": "InR",
    "BIMP_v4.3_005222": "CaMKII",
    "BIMP_v4.3_008309": "ERR",
    "BIMP_v4.3_006836": "cwo",
    "BIMP_v4.3_000134": "CoREST",
    # NCBI names these two badly ("circadian locomoter output cycles protein
    # kaput" for the Met/gce ortholog; "LOW QUALITY PROTEIN_ ... futsch"), so
    # the gene_name fallback left them unlabelled. The panel key and the text
    # call 002131 Met.
    "BIMP_v4.3_002131": "Met",
    "BIMP_v4.3_010698": "futsch",
}

MIN_SLICE_COUNTS = 10_000  # per-group library size floor for a slice
SEED = 42


# ── Spatial pseudobulk (cached) ──────────────────────────────────────────────


def _read_obs(f):
    """Minimal obs frame from an open h5ad (categoricals decoded)."""
    import h5py

    def _cat(g):
        if isinstance(g, h5py.Group) and "categories" in g:
            cats = np.asarray(g["categories"]).astype(str)
            return cats[np.asarray(g["codes"])]
        return np.asarray(g)

    obs = pd.DataFrame(
        {
            "leiden": _cat(f["obs"]["leiden"]).astype(str),
            "brain": _cat(f["obs"]["brain"]).astype(str),
            "slice_name": _cat(f["obs"]["slice_name"]).astype(str),
            # tangram_cell_type is deliberately not read here; see TANGRAM_LABELS
        }
    )
    from anndata.io import read_elem

    from bimp_st.paths import keyed_obs_column

    # joined on (slice_name, cell_id), never by position; raises on any unmatched
    # or repeated cell rather than falling back to the pre-correction in-file column
    key = pd.DataFrame(
        {
            "slice_name": np.asarray(read_elem(f["obs"]["slice_name"])).astype(str),
            "cell_id": np.asarray(read_elem(f["obs"]["cell_id"])).astype(str),
        }
    )
    obs["tangram_cell_type"] = keyed_obs_column(
        key, "tangram_cell_type", TANGRAM_LABELS
    ).astype(str)
    return obs


class _ObsShim:
    """hierarchical_vote() only touches .obs."""

    def __init__(self, obs):
        self.obs, self.n_obs = obs, len(obs)


def compute_spatial_pseudobulk(cache=CACHE_DIR / "spatial_caste_log2fc.csv"):
    """Per-slice pseudobulk caste log2FC (queen - worker), overall and by group.

    Groups use the canonical expression-only cluster vote
    (``hierarchical_vote``), matching every other spatial figure.
    """
    if cache.exists():
        print(f"  Loading cached spatial pseudobulk: {cache}", flush=True)
        return pd.read_csv(cache)

    import h5py
    from scipy.sparse import csr_matrix

    from bimp_st.cluster_labels import get_pooled, hierarchical_vote

    print(f"  Computing spatial pseudobulk from {SPATIAL_H5AD}...", flush=True)
    f = h5py.File(SPATIAL_H5AD, "r")
    obs = _read_obs(f)
    hierarchical_vote(
        _ObsShim(obs), cluster_col="leiden", label_col="tangram_cell_type", dst="vote"
    )
    # hierarchical_vote emits the fine glial labels (Astrocyte-like, Perineurial,
    # Cortex ... glia) plus a residual "Glia"; CONTRASTS group on the pooled
    # vocabulary. Without this step the "Glia" group matches only the residual
    # clusters instead of all glia and the caste log2FC,
    # the rank correlation and the same-direction share all change.
    _pool = {
        v: (v if v == "Mixed" else get_pooled(v))
        for v in obs["vote"].astype(str).unique()
    }
    vote = obs["vote"].astype(str).map(_pool).values

    genes = np.asarray(f["raw"]["var"]["_index"]).astype(str)
    X = f["raw"]["X"]
    n_cells, n_genes = tuple(X.attrs["shape"])
    indptr = np.asarray(X["indptr"])

    slc = obs["slice_name"].values
    slices = pd.unique(slc)
    sidx = {s: i for i, s in enumerate(slices)}
    codes_all = np.array([sidx[s] for s in slc])

    groups = ["__all__"] + [g for _, g, _ in CONTRASTS if g]
    pb = {g: np.zeros((len(slices), n_genes)) for g in groups}

    step = 100_000
    for start in range(0, n_cells, step):
        stop = min(start + step, n_cells)
        lo, hi = indptr[start], indptr[stop]
        sub = csr_matrix(
            (
                np.asarray(X["data"][lo:hi]),
                np.asarray(X["indices"][lo:hi]),
                indptr[start : stop + 1] - lo,
            ),
            shape=(stop - start, n_genes),
        )
        codes = codes_all[start:stop]
        vv = vote[start:stop]
        for g in groups:
            gm = np.ones(len(codes), bool) if g == "__all__" else (vv == g)
            if not gm.any():
                continue
            for k in np.unique(codes[gm]):
                pb[g][k] += np.asarray(sub[gm & (codes == k)].sum(axis=0)).ravel()
        print(f"    {stop:,}/{n_cells:,}", flush=True)

    slice_brain = pd.Series(obs["brain"].values).groupby(pd.Series(slc)).first()
    sb = np.array([slice_brain[s] for s in slices])

    out = []
    for g in groups:
        M = pb[g]
        lib = M.sum(axis=1, keepdims=True)
        keep = lib.ravel() >= MIN_SLICE_COUNTS
        lcpm = np.log2(M / np.where(lib == 0, 1, lib) * 1e6 + 1)
        q = lcpm[keep & (sb == "queen")]
        w = lcpm[keep & (sb == "worker")]
        out.append(
            pd.DataFrame(
                {
                    "group": g,
                    "gene": genes,
                    "log2fc_QvW": q.mean(axis=0) - w.mean(axis=0),
                    "n_queen_slices": q.shape[0],
                    "n_worker_slices": w.shape[0],
                    "total_counts": M.sum(axis=0),
                }
            )
        )

    df = pd.concat(out, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"    Cached {cache}", flush=True)
    return df


# ── Merge ────────────────────────────────────────────────────────────────────


def build_contrast(spatial, deseq2, gene_map, spatial_group, snrna_ct):
    """One cell-type-matched cross-modal table."""
    key = "__all__" if spatial_group is None else spatial_group
    sp = spatial[spatial["group"] == key].copy()
    sp["np31_id"] = sp["gene"].str.split("|").str[0]
    sp = sp.merge(gene_map[["np31_id", "v43_id"]], on="np31_id", how="inner")

    sn = deseq2[deseq2["cell_type"] == snrna_ct][
        ["gene", "log2FoldChange", "padj", "baseMean", "gene_name"]
    ]
    m = sp.merge(sn, left_on="v43_id", right_on="gene", suffixes=("_sp", "_sn"))
    return m.dropna(subset=["log2FoldChange", "log2fc_QvW", "padj"])


def contrast_stats(m, n_boot=2000, seed=SEED):
    """Correlation / sign-agreement statistics for one contrast."""
    x = m["log2fc_QvW"].to_numpy()
    y = m["log2FoldChange"].to_numpy()
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            spearmanr(x[i], y[i]).statistic
            for i in rng.integers(0, len(x), size=(n_boot, len(x)))
        ]
    )
    lo, hi = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5])

    sig = (m["padj"] < 0.05).to_numpy()
    k = int(np.sum(np.sign(x[sig]) == np.sign(y[sig])))
    n_sig = int(sig.sum())
    return {
        "n": len(m),
        "r": r,
        "p_r": p_r,
        "rho": rho,
        "p_rho": p_rho,
        "rho_lo": lo,
        "rho_hi": hi,
        "sign_all": float(np.mean(np.sign(x) == np.sign(y))),
        "n_sig": n_sig,
        "n_sig_concordant": k,
        "sign_sig": (k / n_sig) if n_sig else np.nan,
        "p_binom": binomtest(k, n_sig, 0.5).pvalue if n_sig else np.nan,
    }


# ── Panels ───────────────────────────────────────────────────────────────────


# House print sizes (pt): the canvas is authored at its printed width, so these
# are the sizes on the page.
FS_TICK, FS_LABEL, FS_GENE, FS_STATS = 6.0, 6.5, 6.0, 6.0

# Candidate label offsets (points): nearest first, eight directions per ring,
# right-hand side preferred.  The first candidate that stays inside the axes and
# clears every other label wins; ties on point overlap go to the nearer ring.
_RINGS_PT = (5.0, 9.0, 14.0, 20.0, 27.0)
_DIRS = (
    (1, 0.35),
    (1, -0.35),
    (-1, 0.35),
    (-1, -0.35),
    (0.35, 1),
    (0.35, -1),
    (-0.6, 1),
    (-0.6, -1),
)


def _place_labels(ax, x, y, items, marker_r_pt=4.5):
    """Deterministic gene-label placement.

    items: list of (x, y, text).  Every label is tried at each candidate offset
    around its point; a candidate is rejected if it leaves the axes or touches
    an already-placed label, and among the rest the one covering the fewest
    scatter points (then the nearest) is kept.  A leader is drawn once the
    label sits more than one ring away.  No randomness, so two renders agree.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    pts = ax.transData.transform(np.column_stack([x, y]))
    rad = marker_r_pt * px_per_pt
    axbb = ax.get_window_extent(r)
    placed = []
    # the stiffest points (largest |y|, then |x|) first, so crowded centre
    # labels take whatever room is left rather than claiming it first
    order = sorted(
        range(len(items)), key=lambda i: -abs(items[i][1]) - abs(items[i][0])
    )
    for i in order:
        px, py, txt = items[i]
        best = None
        for ri, rr in enumerate(_RINGS_PT):
            for dx, dy in _DIRS:
                norm = np.hypot(dx, dy)
                ox, oy = rr * dx / norm, rr * dy / norm
                ha = "left" if dx > 0.5 else "right" if dx < -0.5 else "center"
                va = "bottom" if dy > 0.5 else "top" if dy < -0.5 else "center"
                t = ax.annotate(
                    txt,
                    (px, py),
                    xytext=(ox, oy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=FS_GENE,
                    fontstyle="italic",
                    zorder=6,
                )
                # margin for the print-scale hook, which grows text ~10% at save
                bb = t.get_window_extent(r).expanded(1.18, 1.3)
                t.remove()
                if (
                    bb.x0 < axbb.x0
                    or bb.x1 > axbb.x1
                    or bb.y0 < axbb.y0
                    or bb.y1 > axbb.y1
                ):
                    continue
                # a leader runs from the point to the label: its whole
                # footprint must clear the other labels and their leaders
                anc = ax.transData.transform((px, py))
                fb = bb if ri == 0 else Bbox.union([bb, Bbox.from_extents(*anc, *anc)])
                if any(fb.overlaps(o) for o in placed):
                    continue
                hit = int(
                    (
                        (pts[:, 0] > bb.x0 - rad)
                        & (pts[:, 0] < bb.x1 + rad)
                        & (pts[:, 1] > bb.y0 - rad)
                        & (pts[:, 1] < bb.y1 + rad)
                    ).sum()
                )
                # covering one hollow point costs about as much as moving
                # out two rings: far-flung labels read worse than a nick
                key = (hit + 0.6 * ri, ri)
                if best is None or key < best[0]:
                    best = (key, ox, oy, ha, va, fb, ri)
            if best is not None and best[0][0] <= 0.6 * ri:
                break
        if best is None:
            continue
        _, ox, oy, ha, va, fb, ri = best
        lead = (
            dict(arrowstyle="-", color="#555555", lw=0.5, shrinkA=0, shrinkB=3)
            if ri >= 1
            else None
        )
        ax.annotate(
            txt,
            (px, py),
            xytext=(ox, oy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=FS_GENE,
            fontstyle="italic",
            zorder=6,
            arrowprops=lead,
        )
        placed.append(fb)


def _scatter(ax, m, st, title, ylabel=True):
    x = m["log2fc_QvW"].to_numpy()
    y = m["log2FoldChange"].to_numpy()
    sig = (m["padj"] < 0.05).to_numpy()

    lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.15
    lim = max(lim, 0.5)

    # concordant quadrants (same caste direction in both modalities)
    for x0, y0 in ((0, 0), (-lim, -lim)):
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), lim, lim, facecolor="#EDEDED", edgecolor="none", zorder=0
            )
        )

    ax.axhline(0, color="#999999", lw=0.5, zorder=1)
    ax.axvline(0, color="#999999", lw=0.5, zorder=1)
    ax.plot(
        [-lim, lim], [-lim, lim], ls="--", lw=0.6, color="#666666", alpha=0.7, zorder=1
    )

    ax.scatter(
        x[~sig],
        y[~sig],
        s=9,
        c="#FFFFFF",
        edgecolors="#8C8C8C",
        linewidths=0.45,
        zorder=2,
    )
    if sig.any():
        cols = np.where(y[sig] > 0, QUEEN_COLOR, WORKER_COLOR)
        ax.scatter(
            x[sig], y[sig], s=24, c=cols, edgecolors="black", linewidths=0.5, zorder=4
        )

    # least-squares fit through the panel
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.linspace(-lim, lim, 50)
    ax.plot(xs, slope * xs + icpt, color="#C0392B", lw=1.0, alpha=0.85, zorder=3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # name every replicated gene (deterministic placement, see _place_labels)
    items = []
    for _, row in m[sig].iterrows():
        lbl = FOCAL_V43.get(row["v43_id"])
        if lbl is None:
            nm = str(row.get("gene_name") or "")
            lbl = nm if nm and not nm.startswith("BIMP_") else None
            if lbl and len(lbl) > 14:
                lbl = lbl[:13] + "…"
        if lbl:
            items.append((row["log2fc_QvW"], row["log2FoldChange"], lbl))
    if items:
        _place_labels(ax, x, y, items)

    ax.set_xlabel(
        "Spatial (Xenium) log$_2$FC\nqueen/worker, per-slice pseudobulk",
        fontsize=FS_LABEL,
    )
    if ylabel:
        ax.set_ylabel(
            "snRNA-seq log$_2$FC queen/worker\nDESeq2 ($n$ = 3 vs 3)",
            fontsize=FS_LABEL,
        )
    # No title: the panel tag names the cell-type group.

    star = (
        "***"
        if st["p_rho"] < 1e-3
        else "**"
        if st["p_rho"] < 1e-2
        else "*"
        if st["p_rho"] < 0.05
        else "n.s."
    )
    # Stats sit between the panel tag and the axes, never over the points.
    ax.text(
        0.0,
        1.03,
        f"$\\rho$ = {st['rho']:.2f} ({star}),  $r$ = {st['r']:.2f}\n"
        f"$n$ = {st['n']} panel genes,  same direction {st['sign_all'] * 100:.0f}%",
        transform=ax.transAxes,
        fontsize=FS_STATS,
        va="bottom",
        ha="left",
        linespacing=1.25,
    )
    ax.tick_params(labelsize=FS_TICK, length=2.5, pad=1.5)
    ax.locator_params(nbins=5)
    # drop the x tick at the lower-left corner: it collides with the y one
    ax.set_xticks([t for t in ax.get_xticks() if -0.97 * lim < t <= lim])
    ax.set_xlim(-lim, lim)


def _summary_panel(ax, stats):
    """Direction agreement among the caste-replicated genes, per contrast."""
    names = list(stats)
    ypos = np.arange(len(names))[::-1]

    ax.axvline(50, color="#999999", ls=":", lw=0.8, zorder=1)
    for y, nm in zip(ypos, names, strict=False):
        st = stats[nm]
        if not st["n_sig"]:
            continue
        pct = st["sign_sig"] * 100
        ax.barh(y, pct, height=0.55, color="#4C4C4C", zorder=2)
        ax.text(
            pct + 2,
            y,
            f"{st['n_sig_concordant']}/{st['n_sig']}   $P$ = {st['p_binom']:.2f}",
            va="center",
            fontsize=FS_TICK,
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{n}\n({stats[n]['n_sig']} replicated genes)" for n in names],
        fontsize=FS_TICK,
    )
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(
        "Genes with the same caste direction in both modalities (%)",
        fontsize=FS_LABEL,
    )
    ax.tick_params(labelsize=FS_TICK, length=2.5, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── Figure ───────────────────────────────────────────────────────────────────


def compose(tables, stats, output_dir):
    # Authored at the printed width (full text width, ~7 in) so every size
    # above is the size on the page.
    fig = plt.figure(figsize=(7.2, 4.75))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.52],
        hspace=0.44,
        wspace=0.30,
        left=0.105,
        right=0.99,
        top=0.885,
        bottom=0.085,
    )

    axes_ab = []
    for i, (name, _, _) in enumerate(CONTRASTS):
        ax = fig.add_subplot(gs[0, i])
        _scatter(ax, tables[name], stats[name], name, ylabel=(i == 0))
        axes_ab.append(ax)

    axd = fig.add_subplot(gs[1, :2])
    _summary_panel(axd, stats)

    # A-C headers sit above the two-line stats block; D at the usual lift.
    wf = PRINTED_WIDTH_FRAC["figS_deg_concordance"]
    add_panel_headers(
        fig,
        [
            (axes_ab[0], "A", "Whole brain"),
            (axes_ab[1], "B", "Kenyon cells"),
            (axes_ab[2], "C", "Glia"),
        ],
        width_frac=wf,
        dy_pts=21,
    )
    add_panel_headers(fig, [(axd, "D", "Direction agreement")], width_frac=wf)

    axl = fig.add_subplot(gs[1, 2])
    axl.axis("off")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor=QUEEN_COLOR,
            markeredgecolor="black",
            markersize=5,
            label="Queen-up, snRNA FDR < 0.05",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor=WORKER_COLOR,
            markeredgecolor="black",
            markersize=5,
            label="Worker-up, snRNA FDR < 0.05",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor="white",
            markeredgecolor="#8C8C8C",
            markersize=3.5,
            label="Not significant in snRNA-seq",
        ),
        Line2D([], [], color="#C0392B", lw=1.0, label="Least-squares fit"),
        Line2D([], [], color="#666666", lw=0.6, ls="--", label="Identity ($y = x$)"),
        Line2D(
            [],
            [],
            marker="s",
            ls="none",
            markerfacecolor="#EDEDED",
            markeredgecolor="#CCCCCC",
            markersize=6,
            label="Quadrants of concordant direction",
        ),
    ]
    axl.legend(
        handles=handles,
        loc="center left",
        fontsize=FS_TICK,
        frameon=False,
        handletextpad=0.6,
        labelspacing=0.7,
        bbox_to_anchor=(0.0, 0.5),
    )
    # No pseudoreplication block on the artwork: the caption carries it
    # ("(Descriptive, n=1/caste; no spatial P-values.) Caste-level inference
    # derives from snRNA-seq."), and the 30/35 slice counts are in Methods.

    # No figure-level title: it repeated the caption's opening line. House rule
    # across these figures -- the caption names the figure.
    # LaTeX includes the PDF (no extension in \includegraphics, so .pdf wins):
    # always write it beside the PNG, not only under BIMP_ALSO_PDF.
    # The figure goes to the shared SI directory unless BIMP_DEGCONC_OUT
    # redirects the run; the tables stay in output_dir.
    fig_dir = (
        output_dir if "BIMP_DEGCONC_OUT" in os.environ else Path("results/figures/si")
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figS_deg_concordance.pdf")
    _save(fig, fig_dir, "figS_deg_concordance")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with explain_missing_inputs():
        import matplotlib as mpl

        set_style()
        mpl.rcParams["savefig.dpi"] = 300
        mpl.rcParams["figure.dpi"] = 300
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not DESEQ2_CSV.exists():
            raise SystemExit(
                f"Missing {DESEQ2_CSV}; run "
                "scripts/caste_de/caste_de_blocked_all.py first."
            )

        print("Cross-modal DEG concordance", flush=True)
        spatial = compute_spatial_pseudobulk()
        deseq2 = pd.read_csv(DESEQ2_CSV)
        gene_map = read_panel_crosswalk()

        tables, stats = {}, {}
        for name, sp_group, sn_ct in CONTRASTS:
            m = build_contrast(spatial, deseq2, gene_map, sp_group, sn_ct)
            tables[name] = m
            stats[name] = contrast_stats(m)
            st = stats[name]
            print(
                f"  {name}: n={st['n']}  r={st['r']:.3f}  "
                f"rho={st['rho']:.3f} [{st['rho_lo']:.2f}, {st['rho_hi']:.2f}] "
                f"p={st['p_rho']:.2e}  same-direction={st['sign_all'] * 100:.1f}%  "
                f"replicated {st['n_sig_concordant']}/{st['n_sig']} concordant",
                flush=True,
            )

        pd.DataFrame(stats).T.rename_axis("contrast").to_csv(
            OUTPUT_DIR / "fig_s_deg_concordance_stats.csv"
        )
        pd.concat([t.assign(contrast=n) for n, t in tables.items()]).to_csv(
            OUTPUT_DIR / "fig_s_deg_concordance_genes.csv", index=False
        )

        compose(tables, stats, OUTPUT_DIR)
        print("Done.", flush=True)
