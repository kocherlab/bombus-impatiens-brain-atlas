"""Does the Xenium panel measure the same thing the snRNA reference does?

The bulk check (bulk.csv, Spearman correlation over 100 genes, castes pooled:
each gene's mean log1p-CPM averaged over the shared atlas types) asks whether
the modalities agree on which genes are abundant, not on which *cell types*
express them. This adds two stronger comparisons:

  per-gene       : for each panel gene, correlate its per-cell-type mean
                   expression profile across the canonical types between
                   modalities. A gene that is Kenyon-specific in snRNA should
                   be Kenyon-specific in Xenium.
  per-cell-type  : for each canonical cell type, correlate its 100-gene
                   expression vector between modalities.

Both use the canonical cell-type vocabulary, and genes and types are reported
individually, since a single pooled rho hides which genes carry it.

Outputs to results/reframe/panel_concordance/:
    per_gene.csv / per_cell_type.csv / bulk.csv
    concordance.png

The working vocabulary is the canonical 17 (src/bimp_st/cluster_labels.py
CANONICAL_ORDER). By default the figure is replotted from the canonical-17
CSVs in this directory (see REPLOT) rather than reloading both snRNA atlases
and the 1.3M-cell spatial object; set REPLOT = False to recompute. A replot
must write to the directory whose vocabulary it used; canonical-15 outputs
live in panel_concordance_c15/.

Label provenance: the CSVs in results/reframe/panel_concordance/ are the
REPLOT=False computation on the canonical Tangram labels.

SPATIAL stays combined_clustered.h5ad because it is the expression source (the
canonical object's X is a separate dense copy); its own labels are not the
canonical ones, so the REPLOT=False path overrides its tangram_cell_type
from TANGRAM_LABELS.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr

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
from bimp_st.cluster_labels import (  # noqa: E402
    add_canonical,
    get_canonical_color,
    get_canonical_display,
    get_pooled,
)
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (  # noqa: E402
    CANONICAL_TANGRAM,
    PANEL_CROSSWALK,  # noqa: E402
    SN_CB,
    SN_OL,
    SPATIAL_COMBINED,
    keyed_obs_column,
)

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
SPATIAL = SPATIAL_COMBINED
# Canonical Tangram labels; SPATIAL's own tangram_cell_type is not canonical.
TANGRAM_LABELS = CANONICAL_TANGRAM
PANEL_CSV = PANEL_CROSSWALK
OUT = REPO / "results" / "reframe" / "panel_concordance"
OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "results" / "figures" / "si" / "figS_panel_concordance.png"

# canonical-17 vocabulary. The CSVs in OUT (bulk / per_gene / per_cell_type /
# {sn,xenium}_mean_by_type) are the canonical-17 computation; main() replots
# this script's figure -- registered as figS_panel_concordance --
# straight from them, so no h5ad load is needed.
# Set REPLOT = False to recompute them from the source objects.
REPLOT = os.environ.get("BIMP_CONCORDANCE_REPLOT", "1") != "0"

MIN_CELLS = 100
LABEL = "cell_type_canonical"
# Neutral figure colours: nothing in this figure is a queen/worker contrast.
INK = "#2B2B2B"
NEUTRAL = "#9A9A9A"


def class_agreement(pg: pd.DataFrame) -> dict:
    """Do the two modalities agree on which cell CLASS a gene marks most?

    Takes each gene's top-expressing class in each modality (already in
    per_gene.csv as ``sn_class`` / ``xen_class``) and scores the agreement
    against what the class marginals alone would produce. Cohen's kappa is the
    right statistic because raw agreement is flattered by the marginals: a few
    classes take the maximum for most genes in both modalities, so a
    substantial share of "agreement" is arithmetic rather than biology.
    """
    po = float(pg["class_agrees"].mean())
    m_sn = pg["sn_class"].value_counts(normalize=True)
    m_xen = pg["xen_class"].value_counts(normalize=True)
    pe = float(
        sum(
            m_sn.get(c, 0.0) * m_xen.get(c, 0.0)
            # sorted: summing over a bare set made the last digit depend on
            # PYTHONHASHSEED
            for c in sorted(set(m_sn.index) | set(m_xen.index))
        )
    )
    return dict(
        n=len(pg),
        n_agree=int(pg["class_agrees"].sum()),
        p_observed=po,
        p_chance=pe,
        kappa=(po - pe) / (1 - pe) if pe < 1 else float("nan"),
    )


def _wrap_two_lines(label: str, maxlen: int = 99) -> str:
    """Break a long multi-word y-tick label onto two lines near its middle.

    Only the three outliers among the canonical display names clear
    ``maxlen`` -- "Anterodorsal projection neurons" (32 chars), "Kenyon
    cells (unclassified)" (28), "Lateral projection neurons" (27) -- so
    most labels stay single-line and only the true outliers wrap.
    """
    if len(label) <= maxlen or " " not in label:
        return label
    words = label.split(" ")
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def _annotate_gene_panel(ax, pg: pd.DataFrame) -> dict:
    """Label the middle panel's median line and print its summary (median rho,
    class agreement, chance baseline, kappa).

    Every number is computed from ``pg``, the table the histogram is drawn
    from, so the printed summary and the figure cannot drift apart.
    """
    a = class_agreement(pg)
    med = float(pg["spearman"].median())
    # Headroom above the tallest bar for the median label.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.16)
    # The median is printed to three places, as the caption gives it.
    ax.annotate(
        f"median ρ = {med:.3f}",
        xy=(med, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -1),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=INK,
    )
    # The agreement count and the chance baseline are in the caption.
    print(
        f"  B: median rho {med:.3f}; {a['n_agree']}/{a['n']} agree; "
        f"chance {a['p_chance']:.1%}; kappa {a['kappa']:.3f}",
        flush=True,
    )
    return a


def _plot_concordance(
    A: pd.DataFrame,
    B: pd.DataFrame,
    pg: pd.DataFrame,
    pct: pd.DataFrame,
    out_png: Path,
) -> dict:
    """Draw concordance.png (both the replot and the recompute path use this).

    Nothing here is caste-specific, so the scatter and histogram are neutral
    grey/black rather than the queen/worker gold/blue. Panel C keeps the
    canonical cell-type colours.
    """
    fig = plt.figure(figsize=(7.2, 2.75))
    # A and B are plain scatter/histogram panels; C carries long cell-type
    # tick labels. An empty spacer column holds those labels, so the A-B gap
    # stays tight instead of every gap being as wide as C's labels.
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 0.62, 1.3],
        wspace=0.45,
        left=0.075,
        right=0.97,
        bottom=0.17,
        top=0.88,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes.append(fig.add_subplot(gs[0, 3]))

    axes[0].scatter(A.mean(1), B.mean(1), s=6, c=INK, alpha=0.6, linewidths=0)
    axes[0].set_xlabel("snRNA mean log CPM")
    axes[0].set_ylabel("Xenium mean log CPM")
    # No panel title: the caption gives the bulk correlation and n.

    axes[1].hist(pg.spearman.dropna(), bins=26, color=NEUTRAL)
    axes[1].axvline(pg.spearman.median(), color=INK, lw=1)
    axes[1].set_xlabel("Spearman ρ across cell types")
    axes[1].set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
    axes[1].set_xticklabels(["−0.25", "0", "", "0.5", "", "1"])
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylabel("Genes")
    # No panel title: caption (B) states this as the per-gene median.
    agree = _annotate_gene_panel(axes[1], pg)

    p = pct.sort_values("spearman")
    # Labels go through get_canonical_display so the residual "Kenyon cells"
    # entry reads as "Kenyon cells (unclassified)" beside the named Kenyon
    # subtypes.
    ylabels = [_wrap_two_lines(get_canonical_display(c)) for c in p.cell_type]
    axes[2].barh(
        ylabels,
        p.spearman,
        height=0.75,
        color=[get_canonical_color(c) for c in p.cell_type],
    )
    axes[2].set_ylim(-0.6, len(p) - 0.4)
    axes[2].set_xlabel("Spearman ρ across genes")
    # No panel title: caption (C) states this as the per-cell-type median.
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [
            (axes[0], "A", "Bulk expression"),
            (axes[1], "B", "Per gene"),
            (axes[2], "C", "Per cell type"),
        ],
        width_frac=1.0,
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return agree


def _replot(data_dir: Path) -> None:
    """Cheap replot of concordance.png from an already-computed run
    (bulk.csv / per_gene.csv / per_cell_type.csv), no h5ad load needed.
    ``data_dir`` must be the directory whose vocabulary the CSVs were computed
    under -- it is also where the figure and class_agreement.csv are written,
    so the two can never drift onto different vocabularies.
    """
    bulk = pd.read_csv(data_dir / "bulk.csv")
    pg = pd.read_csv(data_dir / "per_gene.csv")
    pct = pd.read_csv(data_dir / "per_cell_type.csv")
    br = float(bulk["spearman"].iloc[0])

    A = pd.read_csv(data_dir / "sn_mean_by_type.csv", index_col=0)
    B = pd.read_csv(data_dir / "xenium_mean_by_type.csv", index_col=0)
    agree = _plot_concordance(A, B, pg, pct, FIG)
    pd.DataFrame([agree]).to_csv(data_dir / "class_agreement.csv", index=False)
    print(
        f"[replot] bulk rho={br:.3f}  "
        f"per-gene median={pg.spearman.median():.3f}  "
        f"per-cell-type median={pct.spearman.median():.3f} "
        f"(n={len(pct)} types)",
        flush=True,
    )
    print(
        f"  class agreement: {agree['n_agree']}/{agree['n']} "
        f"(chance {agree['p_chance']:.3f}, kappa {agree['kappa']:.3f})",
        flush=True,
    )
    print(f"wrote {FIG}", flush=True)


def _looks_log_transformed(M) -> bool:
    """True if the matrix is already log-scale rather than raw counts.

    Raw counts are non-negative integers. combined_clustered.h5ad has no
    layers and its X is already log1p-normalised (nonzeros ~4.7-6.9), so an
    `if "counts" in layers else X` guard alone would normalise it twice.
    """
    d = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    nz = d[d > 0]
    if nz.size == 0:
        return False
    return not np.allclose(nz[:100_000], np.round(nz[:100_000]))


def cpm_by_type(adata, label: str, genes: list[str], tag: str = "") -> pd.DataFrame:
    """Mean log1p-CPM per cell type, genes x types.

    Normalises from raw counts when they are available, and refuses to
    re-normalise data that is already on a log scale.
    """
    x = adata[:, genes].copy()
    if "counts" in x.layers:
        x.X = x.layers["counts"].copy()
        source = "counts layer"
    elif _looks_log_transformed(x.X):
        source = "X (ALREADY log-normalised -- not re-normalising)"
    else:
        source = "X (raw counts)"
    print(f"    [{tag or 'input'}] normalisation source: {source}", flush=True)

    if not source.startswith("X (ALREADY"):
        sc.pp.normalize_total(x, target_sum=1e6)
        sc.pp.log1p(x)
    M = x.X.toarray() if hasattr(x.X, "toarray") else np.asarray(x.X)
    lab = x.obs[label].astype(str).values
    out = {}
    for ct in np.unique(lab):
        m = lab == ct
        if m.sum() >= MIN_CELLS:
            out[ct] = M[m].mean(0)
    return pd.DataFrame(out, index=genes)


def main() -> None:
    if REPLOT and (OUT / "per_gene.csv").exists():
        _replot(OUT)
        return

    panel = read_panel_crosswalk()
    v43 = list(dict.fromkeys(panel["v43_id"].dropna()))

    print("loading snRNA atlases ...", flush=True)
    cb = sc.read_h5ad(SN_CB)
    ol = sc.read_h5ad(SN_OL)
    for a in (cb, ol):
        add_canonical(a, src="cell_type")
    shared_var = [g for g in v43 if g in cb.var_names and g in ol.var_names]
    sn = ad.concat(
        [cb[:, shared_var], ol[:, shared_var]],
        join="outer",
        label="atlas",
        keys=["cb", "ol"],
    )
    del cb, ol
    print(
        f"  snRNA combined: {sn.n_obs:,} nuclei, {len(shared_var)} panel genes",
        flush=True,
    )

    print("loading spatial ...", flush=True)
    sp = sc.read_h5ad(SPATIAL)
    # joined on (slice_name, cell_id): obs_names repeat, so they are not a key
    sp.obs["tangram_cell_type"] = keyed_obs_column(
        sp, "tangram_cell_type", TANGRAM_LABELS
    ).astype(str)
    add_canonical(sp, src=("tangram_cell_type", "cell_type"))

    # The two modalities live in different gene namespaces: the reference uses
    # BIMP_v4.3_* ids, the Xenium object uses the panel's BIMP_vNP3.1_* probe
    # names, some of which carry a "|MGEN_v3.1_*" cross-species suffix. Strip
    # the suffix and translate through the tracked mapping table.
    np31_to_v43 = dict(zip(panel["np31_id"], panel["v43_id"], strict=True))
    sp_v43, sp_keep = [], []
    for name in sp.var_names:
        v43_id = np31_to_v43.get(str(name).split("|")[0])
        if v43_id is not None and v43_id in set(shared_var):
            sp_keep.append(name)
            sp_v43.append(v43_id)
    sp = sp[:, sp_keep].copy()
    sp.var_names = sp_v43
    # Defensive only: the crosswalk maps 101 unique np31 ids to 101 unique v4.3
    # ids. The two "fatty acid synthase" probes (BIMP_vNP3.1_04108 ->
    # BIMP_v4.3_009457, BIMP_vNP3.1_08577 -> BIMP_v4.3_002618) are different genes
    # sharing a display name, which is why bento's tabulation has 100 genes;
    # Tangram's 100 is a separate matter (it drops corazonin).
    sp = sp[:, ~pd.Index(sp.var_names).duplicated()].copy()

    genes = [g for g in shared_var if g in set(sp.var_names)]
    print(
        f"  genes measurable in both namespaces: {len(genes)} "
        f"(from {len(sp_keep)} probes)",
        flush=True,
    )
    if not genes:
        raise SystemExit("no genes shared after namespace translation")

    A = cpm_by_type(sn, LABEL, genes, "snRNA")
    B = cpm_by_type(sp, LABEL, genes, "Xenium")
    shared_ct = [c for c in A.columns if c in B.columns]
    print(
        f"  cell types present in both (>= {MIN_CELLS} cells): {len(shared_ct)}",
        flush=True,
    )
    A, B = A[shared_ct], B[shared_ct]

    # bulk: Spearman over per-gene means
    br, bp = spearmanr(A.mean(1), B.mean(1))
    pd.DataFrame(
        [dict(comparison="bulk mean across genes", spearman=br, p=bp, n=len(genes))]
    ).to_csv(OUT / "bulk.csv", index=False)
    print(f"\nbulk mean-expression Spearman = {br:.3f} (p = {bp:.2g})", flush=True)

    # per gene: profile across cell types
    rows = []
    for g in genes:
        a, b = A.loc[g].values, B.loc[g].values
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rs, ps = spearmanr(a, b)
        rp, _ = pearsonr(a, b)
        sn_top, xen_top = shared_ct[int(np.argmax(a))], shared_ct[int(np.argmax(b))]
        rows.append(
            dict(
                gene=g,
                spearman=rs,
                pearson=rp,
                p=ps,
                sn_mean=float(a.mean()),
                xen_mean=float(b.mean()),
                sn_top=sn_top,
                xen_top=xen_top,
                top_agrees=sn_top == xen_top,
                # Six of the 17 canonical types are Kenyon subtypes, so
                # a gene whose maximum moves from "Small-type Kenyon
                # cells" to the generic "Kenyon cells" is scored as a
                # disagreement by top_agrees while actually agreeing on
                # the cell class. Score that separately.
                sn_class=get_pooled(sn_top),
                xen_class=get_pooled(xen_top),
                class_agrees=get_pooled(sn_top) == get_pooled(xen_top),
            )
        )
    pg = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    pg.to_csv(OUT / "per_gene.csv", index=False)
    print(
        f"per-gene profile concordance: median rho = {pg.spearman.median():.3f}",
        flush=True,
    )
    print(
        f"  maximum in the same fine type : {int(pg.top_agrees.sum())}/{len(pg)}",
        flush=True,
    )
    print(
        f"  maximum in the same cell class: {int(pg.class_agrees.sum())}/{len(pg)}",
        flush=True,
    )

    # Where a gene's per-cell-type profile is flat, argmax is meaningless. Report
    # agreement restricted to genes with real cell-type structure.
    struct = pg[pg.spearman.notna()].copy()
    strong = struct[struct.spearman >= 0.3]
    print(
        f"  among the {len(strong)} genes with rho >= 0.3: "
        f"{int(strong.class_agrees.sum())}/{len(strong)} agree on class",
        flush=True,
    )

    # class_agreement.csv (n_agree, p_chance, kappa) is written by whichever
    # path recomputed pg, together with per_gene.csv, so the two never come
    # from different runs.
    pd.DataFrame([class_agreement(pg)]).to_csv(OUT / "class_agreement.csv", index=False)

    A.to_csv(OUT / "sn_mean_by_type.csv")
    B.to_csv(OUT / "xenium_mean_by_type.csv")

    # per cell type: profile across genes
    rows = []
    for ct in shared_ct:
        rs, ps = spearmanr(A[ct].values, B[ct].values)
        rows.append(dict(cell_type=ct, spearman=rs, p=ps, n_genes=len(genes)))
    pct = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    pct.to_csv(OUT / "per_cell_type.csv", index=False)
    print(
        f"per-cell-type concordance: median rho = {pct.spearman.median():.3f}",
        flush=True,
    )
    print(pct.to_string(index=False), flush=True)

    _plot_concordance(A, B, pg, pct, FIG)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
