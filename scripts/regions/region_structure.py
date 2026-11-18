"""Not used by any registered figure.

Part 1 (diversity) uses per-cell ``cell_type_canonical`` (raw Tangram argmax),
which is unreliable for region-conditioned use: "Small-type Kenyon cells" is
a large share of the brain yet depleted in the Kenyon-cell compartments and
enriched in the medulla, so diversity computed from it is partly panel noise.
The registered diversity analysis is ``scripts/regions/region_diversity.py``
(three label schemes, a label-free heterogeneity metric, and rarefaction to a
common cell count across regions), which produces ``fig_s_regional_diversity``.

Part 2 (region prediction): the registered version is ``si_region_prediction.py``.

---

Regional structure of the atlas: cell-type diversity, and whether region is predictable.

Two questions, sharing one load of the 1.3M-cell combined object:

1. Cell-type diversity within each anatomical region (Shannon / Simpson /
   richness), whole-brain and split by caste. "Is the medulla a more mixed
   neighbourhood than the calyx?"

2. Can we predict anatomical region from cell-type composition, and which cell
   types carry the prediction (i.e. does transcriptomic identity track
   anatomy)? Two granularities:
     - per-cell    : cell type -> region (one categorical predictor, a weak
                     ceiling by construction)
     - per-neighbourhood : the composition of each cell's k nearest spatial
                     neighbours -> region (the main test).

Held-out scoring is grouped by section (slice_name), never random, so a
neighbourhood and its own neighbours cannot land on both sides of the split.

Everything uses the canonical 17-type vocabulary (bimp_st.cluster_labels).

Outputs to results/reframe/region_structure/:
    diversity_by_region.csv       Shannon/Simpson/richness per region x caste
    region_prediction.csv         per-region precision/recall/F1, both models
    region_confusion.csv          confusion matrix (neighbourhood model)
    celltype_importance.csv       which cell types carry each region
    diversity.png / prediction.png
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
from bimp_st.cluster_labels import (  # noqa: E402
    add_canonical,
    get_canonical_color,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (  # noqa: E402
    STATISTICAL_REGIONS,  # noqa: E402
    attach_region_flags,
    display,
)

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
H5 = REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
OUT = REPO / "results" / "reframe" / "region_structure"
OUT.mkdir(parents=True, exist_ok=True)

# canonical-17 vocabulary. When OUT already holds diversity_by_region.csv,
# diversity.png is replotted from it instead of reloading the 1.3M-cell h5ad.
# Neither diversity.png nor prediction.png is registered (see the
# module docstring). The canonical-15 replot lives in
# region_structure_c15/diversity_replot.png.
REPLOT_DIVERSITY = True

# STATISTICAL_REGIONS = the original six + antennal lobe. The central-complex
# regions are excluded because they would degrade the other
# estimates; see nine_region_common.DESCRIPTIVE_ONLY_REGIONS.
REGION_FLAGS = [f"in_{r}" for r in STATISTICAL_REGIONS]
REGION_NICE = {f"in_{r}": display(r) for r in STATISTICAL_REGIONS}
K_NEIGH = 20
SEED = 0
LABEL = "cell_type_canonical"


def shannon(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def simpson(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(1.0 - (p**2).sum())


def region_mask(obs: pd.DataFrame, flag: str) -> np.ndarray:
    """Boolean membership mask for one in_<region> flag.

    in_al is a nullable boolean: pd.NA means "no antennal-lobe tracing for
    this section" (377,619 worker cells, none in the queen), not "not
    antennal lobe"; the other flags are plain bool. Never .fillna(False),
    which would create a caste difference from missing data. Unknown cells
    are excluded from this region's mask but still count toward any other
    region they are annotated in.
    """
    col = obs[flag]
    known = col.notna().to_numpy()
    out = np.zeros(len(col), dtype=bool)
    out[known] = col[known].to_numpy(dtype=bool)
    return out


def diversity_table(obs: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    rows = []
    for flag in REGION_FLAGS:
        m_all = region_mask(obs, flag)
        for caste in ["both", "queen", "worker"]:
            m = m_all if caste == "both" else (m_all & (obs["brain"] == caste).values)
            if m.sum() == 0:
                continue
            vc = obs.loc[m, LABEL].value_counts().reindex(cats, fill_value=0).values
            rows.append(
                dict(
                    region=REGION_NICE[flag],
                    caste=caste,
                    n_cells=int(m.sum()),
                    richness=int((vc > 0).sum()),
                    shannon=shannon(vc),
                    simpson=simpson(vc),
                    evenness=shannon(vc) / np.log(max((vc > 0).sum(), 2)),
                    top_type=cats[int(np.argmax(vc))],
                    top_frac=float(vc.max() / vc.sum()),
                )
            )
    return pd.DataFrame(rows)


def build_neighbourhoods(obs: pd.DataFrame, xy: np.ndarray, cats: list[str]):
    """Per-cell composition of its k nearest neighbours, built within each section."""
    from sklearn.neighbors import NearestNeighbors

    code = pd.Categorical(obs[LABEL], categories=cats).codes
    X = np.zeros((len(obs), len(cats)), dtype=np.float32)
    for sl in obs["slice_name"].unique():
        idx = np.flatnonzero((obs["slice_name"] == sl).values)
        if len(idx) <= K_NEIGH:
            continue
        nn = NearestNeighbors(n_neighbors=K_NEIGH + 1).fit(xy[idx])
        nbr = nn.kneighbors(xy[idx], return_distance=False)[:, 1:]
        c = code[idx][nbr]  # (n, k) neighbour type codes
        for j in range(len(cats)):
            X[idx, j] = (c == j).sum(1)
    X /= np.maximum(X.sum(1, keepdims=True), 1)
    return X


def fit_and_score(X, y, groups, cats, tag: str):
    """Grouped-by-section held-out multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit
    from threadpoolctl import threadpool_limits

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    # multinomial is the default and the multi_class kwarg was removed in sklearn 1.7
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    # one BLAS thread: lbfgs sums in a thread-count-dependent order otherwise
    with threadpool_limits(limits=1):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])

    rep = classification_report(y[te], pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "region"})
    df.insert(0, "model", tag)
    labels = sorted(np.unique(y))
    cm = pd.DataFrame(
        confusion_matrix(y[te], pred, labels=labels, normalize="true"),
        index=labels,
        columns=labels,
    )
    coef = None
    if X.shape[1] == len(cats):
        coef = pd.DataFrame(clf.coef_, index=clf.classes_, columns=cats)
    acc = float((pred == y[te]).mean())
    print(
        f"  [{tag}] held-out accuracy = {acc:.3f} "
        f"(chance = {1 / len(labels):.3f}, n_test = {len(te):,})",
        flush=True,
    )
    return df, cm, coef, acc


def _replot_diversity(data_dir: Path) -> None:
    """Cheap replot of diversity.png from the already-computed
    diversity_by_region.csv, no h5ad load needed. Writes into ``data_dir``, so
    the figure stays with its own vocabulary. Does not regenerate
    prediction.png.
    """
    div = pd.read_csv(data_dir / "diversity_by_region.csv")
    both = div[div.caste == "both"].sort_values("shannon")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(both.region, both.shannon, color="#B06715")
    ax.set_xlabel("Shannon diversity of cell types (canonical 17)")
    ax.set_title("Regional cell-type diversity")
    for i, (_r, v) in enumerate(zip(both.region, both.shannon, strict=True)):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8.5, color="#241D14")
    ax.set_xlim(0, both.shannon.max() * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_dir / "diversity.png", dpi=200)
    plt.close(fig)
    print(
        "[replot] "
        + ", ".join(
            f"{r}={v:.2f}" for r, v in zip(both.region, both.shannon, strict=True)
        ),
        flush=True,
    )
    print(f"wrote {data_dir / 'diversity.png'}", flush=True)


def main() -> None:
    if REPLOT_DIVERSITY and (OUT / "diversity_by_region.csv").exists():
        _replot_diversity(OUT)
        return

    print(f"loading {H5.name} ...", flush=True)
    a = ad.read_h5ad(H5)
    add_canonical(a, src=("tangram_cell_type", "cell_type"))

    # combined_clustered.h5ad carries six in_<region> flags from labeling.py,
    # computed before self-intersecting polygons were repaired (KC-NC 98,831 vs
    # 99,270 in coverage.json). All seven flags are re-attached from the
    # nine-region crop sidecars so there is one definition of each region.
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    obs = a.obs
    cats = list(obs[LABEL].cat.categories)
    print(f"  {a.n_obs:,} cells, {len(cats)} canonical types", flush=True)

    # ── 1. diversity ────────────────────────────────────────────────────
    div = diversity_table(obs, cats)
    div.to_csv(OUT / "diversity_by_region.csv", index=False)
    print("\n--- cell-type diversity by region ---", flush=True)
    print(div[div.caste == "both"].to_string(index=False), flush=True)

    # No by-caste panel: the queen-vs-worker medulla direction flips between
    # Shannon and Simpson, so it is not reported. Per-caste numbers remain in
    # diversity_by_region.csv.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    both = div[div.caste == "both"].sort_values("shannon")
    ax.barh(both.region, both.shannon, color="#B06715")
    ax.set_xlabel("Shannon diversity of cell types (canonical 17)")
    ax.set_title("Regional cell-type diversity")
    for i, (_r, v) in enumerate(zip(both.region, both.shannon, strict=True)):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8.5, color="#241D14")
    ax.set_xlim(0, both.shannon.max() * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "diversity.png", dpi=200)
    plt.close(fig)

    # ── 2. region prediction ────────────────────────────────────────────
    # in_al is nullable (pd.NA = untraced, not "not antennal lobe" -- see
    # region_mask above). These cells are dropped from the fit rather than
    # filled with False.
    al_traced = obs["in_al"].notna().to_numpy()
    n_untraced = int((~al_traced).sum())
    if n_untraced:
        print(
            f"  dropping {n_untraced:,} cells with no antennal-lobe "
            "tracing yet (in_al is NA, not False)",
            flush=True,
        )
    obs_t = obs.loc[al_traced]

    in_any = obs_t[REGION_FLAGS].astype(bool).values
    keep = in_any.sum(1) == 1  # unambiguous region only
    print(
        f"\ncells in exactly one annotated region: {keep.sum():,} "
        f"({100 * keep.mean():.1f}% of the atlas)",
        flush=True,
    )

    sub = obs_t.loc[keep]
    y = np.array([REGION_NICE[REGION_FLAGS[i]] for i in in_any[keep].argmax(1)])
    groups = sub["slice_name"].astype(str).values

    onehot = pd.get_dummies(pd.Categorical(sub[LABEL], categories=cats)).values.astype(
        np.float32
    )
    d_cell, _, _, acc_cell = fit_and_score(
        onehot, y, groups, cats, "per-cell type only"
    )

    xy = (
        a.obsm["spatial"][al_traced][keep]
        if "spatial" in a.obsm
        else np.column_stack([sub["x_centroid"].values, sub["y_centroid"].values])
    )
    print("building spatial neighbourhoods ...", flush=True)
    Xn = build_neighbourhoods(sub.reset_index(drop=True), np.asarray(xy, float), cats)
    d_nb, cm, coef, acc_nb = fit_and_score(
        Xn, y, groups, cats, "neighbourhood composition"
    )

    pd.concat([d_cell, d_nb]).to_csv(OUT / "region_prediction.csv", index=False)
    cm.to_csv(OUT / "region_confusion.csv")
    if coef is not None:
        coef.T.to_csv(OUT / "celltype_importance.csv")
        print("\n--- top cell types per region (model weight) ---", flush=True)
        for r in coef.index:
            top = coef.loc[r].sort_values(ascending=False).head(3)
            print(
                f"  {r:8s}: " + ", ".join(f"{k} ({v:+.2f})" for k, v in top.items()),
                flush=True,
            )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    im = axes[0].imshow(cm.values, cmap="pink_r", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(cm)), cm.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(cm)), cm.index)
    axes[0].set_title(
        f"Region confusion (neighbourhood model)\nheld-out accuracy = {acc_nb:.2f}"
    )
    axes[0].set_ylabel("true")
    axes[0].set_xlabel("predicted")
    for i in range(len(cm)):
        for j in range(len(cm)):
            axes[0].text(
                j,
                i,
                f"{cm.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm.values[i, j] > 0.55 else "#241D14",
            )
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # Bucket regions explicitly: KC compartments, antennal lobe, else optic.
    kc_regions = {r for r in REGION_NICE.values() if r.startswith("KC-")}

    def _bar_color(r):
        if r in kc_regions:
            return "#B06715"
        if r == "Antennal lobe":
            return "#6E6154"
        return get_canonical_color("Optic lobe")

    f1 = d_nb[d_nb.region.isin(cm.index)].set_index("region")["f1-score"].sort_values()
    axes[1].barh(f1.index, f1.values, color=[_bar_color(r) for r in f1.index])
    axes[1].axvline(1 / len(cm), ls="--", lw=1, color="#6E6154", label="chance")
    axes[1].set_xlabel("F1 (held out by section)")
    axes[1].set_title("Per-region predictability")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "prediction.png", dpi=200)
    plt.close(fig)

    print(
        f"\naccuracy: cell-type-only {acc_cell:.3f} -> neighbourhood {acc_nb:.3f}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
