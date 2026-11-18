"""Antennal lobe (AL): matched anterior-posterior range and cross-brain transfer.

Restricts the AL to the anterior-posterior range sectioned in both brains and
transfers region labels across brains.

Motivation. The AL is traced on every queen section but on only 11/16 worker sections
(8/8 on slide 0068401, 3/8 on 0069041), and the 0069041 tracings are much
smaller in polygon area, so the queen:worker AL prevalence depends on whether
0069041 is included. This script asks whether that caste
difference survives restriction to the AP range that is traced in both brains,
and whether a classifier trained on the queen's AL transfers to the worker.

AP coordinate. ``obs['z_level']`` of the PASTE3-aligned atlas (identical in
``combined_clustered.h5ad`` and in ``results/nine_region/atlas/*_9region.h5ad``):
the serial-section stack index x the 10 um nominal cut, one value per crop.
queen 0..290 (30 crops), worker 0..340 (35 crops). It is a section order, not a
calibrated distance, its anterior/posterior direction is not established, and
the two brains' stacks are not co-registered (different lengths). To compare
the brains we use the fractional stack position u = (z/10 + 0.5) / n_crops.

Traced vs absent. "Traced" is read from the annotations_v4 GeoJSON itself: a
capture-region section is traced for the AL iff its file holds an AL polygon.
``in_al`` is False (not NA) on untraced sections of a slide that has the AL
elsewhere, so ``in_al`` alone cannot distinguish "traced, no AL" from "not
traced". Nothing here fills a missing tracing with False: untraced worker
sections are excluded from every prevalence and from classifier scoring.

Labels. Per-cell canonical-17 (corrected Tangram, ``paths.CANONICAL_TANGRAM``,
positionally aligned to ``combined_clustered.h5ad`` after a cell_id check) for
the existing composition classifier; ``hierarchical_vote`` (the expression-only
cluster vote the figures use) and raw Leiden clusters for step 3.

Classifier. Reused from ``region_prediction``: each cell is described
by the label frequencies of its 20 nearest neighbours within its own crop
(``neighbourhood_composition``, K_MAIN = 20), fed to the same multinomial
LogisticRegression over the seven STATISTICAL_REGIONS, restricted to cells in
exactly one region. Trained on the queen only, applied to the worker.

Outputs: results/handoff/C_antennal_lobe/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import h5py
import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata.io import read_elem

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# the existing 20-NN composition features (importing runs only its mkdir)
from regions.region_prediction import K_MAIN, neighbourhood_composition  # noqa: E402

from bimp_st import nine_region_common as nrc  # noqa: E402
from bimp_st.cluster_labels import add_canonical, hierarchical_vote  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import STATISTICAL_REGIONS, display  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402
from bimp_st.style import set_pub_style  # noqa: E402

warnings.filterwarnings("ignore")
OUT = REPO / "results" / "handoff" / "C_antennal_lobe"
OUT.mkdir(parents=True, exist_ok=True)
AL = "in_al"
N_BOOT = 2000


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_cells() -> ad.AnnData:
    """Obs-only AnnData: combined obs + corrected labels + flags + votes."""
    with h5py.File(SPATIAL_COMBINED, "r") as g:
        obs = read_elem(g["obs"])
        xy = g["obsm/spatial"][:]
    t = read_obs(CANONICAL_TANGRAM)
    if len(t) != len(obs) or not (t["cell_id"].values == obs["cell_id"].values).all():
        raise SystemExit("corrected labels do not line up with combined object")
    obs = obs.copy()
    obs["tangram_cell_type"] = t["tangram_cell_type"].astype(str).values
    obs["slice_name"] = obs["slice_name"].astype(str)
    obs["brain"] = obs["brain"].astype(str)
    obs.index = pd.RangeIndex(len(obs)).astype(str)
    a = ad.AnnData(obs=obs)
    a.obsm["spatial"] = xy
    # six original flags in the object are the pre-repair ones; take all seven
    # from the nine-region sidecars (same call as region_prediction)
    nrc.attach_region_flags(a, regions=STATISTICAL_REGIONS)
    add_canonical(a, src="tangram_cell_type")
    hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
    )
    return a


def section_tracing(obs: pd.DataFrame) -> pd.DataFrame:
    """Per capture-region section: is an AL polygon drawn? polygon area."""
    rows = []
    for brain in ("queen", "worker"):
        src = nrc.annotation_sources(brain)
        for (slide, sec), d in sorted(src.items()):
            path = d["v2"] or d["old"]
            regs = nrc.load_regions(path, vocabulary=None)
            geom = regs.get("al")
            rows.append(
                dict(
                    brain=brain,
                    slide=slide,
                    section=sec,
                    annotation_file=path.parent.name,
                    al_traced=geom is not None,
                    al_polygon_area_um2=float(geom.area) if geom is not None else 0.0,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# step 1
# ---------------------------------------------------------------------------
def crop_table(obs: pd.DataFrame, tracing: pd.DataFrame) -> pd.DataFrame:
    ss = obs["slice_name"].map(nrc.parse_slide_section)
    obs = obs.assign(slide=[s[0] for s in ss], section=[s[1] for s in ss])
    al = obs[AL].astype("object")
    g = (
        obs.assign(_al=(al == True).astype(int))  # noqa: E712
        .groupby(["brain", "slice_name", "slide", "section"], observed=True)
        .agg(
            z_level=("z_level", "first"),
            n_cells=("_al", "size"),
            n_al=("_al", "sum"),
            n_al_NA=(AL, lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    g["crop"] = g["slice_name"].str[-2:]
    g = g.merge(
        tracing[["brain", "slide", "section", "al_traced", "al_polygon_area_um2"]],
        on=["brain", "slide", "section"],
        how="left",
    )
    g["z_idx"] = (g["z_level"] / 10).round().astype(int)
    n = g.groupby("brain")["z_idx"].transform("max") + 1
    g["n_stack"] = n
    g["u"] = (g["z_idx"] + 0.5) / n
    g["al_pct"] = 100 * g["n_al"] / g["n_cells"]
    # matched windows: queen crops whose u falls in the stack interval of a
    # traced worker crop
    wt = g[(g.brain == "worker") & g.al_traced]
    nw = int(wt["n_stack"].iloc[0])
    wbins = [(i / nw, (i + 1) / nw) for i in sorted(wt["z_idx"])]
    g["in_matched_window"] = False
    q = g["brain"] == "queen"
    g.loc[q, "in_matched_window"] = [
        any(lo <= u < hi for lo, hi in wbins) for u in g.loc[q, "u"]
    ]
    g.loc[~q, "in_matched_window"] = g.loc[~q, "al_traced"]
    lo_c, hi_c = wbins[0][0], wbins[-1][1]
    g["in_matched_contiguous"] = False
    g.loc[q, "in_matched_contiguous"] = (g.loc[q, "u"] >= lo_c) & (g.loc[q, "u"] < hi_c)
    g.loc[~q, "in_matched_contiguous"] = g.loc[~q, "al_traced"]
    g.attrs["wbins"] = wbins
    g.attrs["contig"] = (lo_c, hi_c)
    return g


def scenarios(g: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    q, w = g.brain == "queen", g.brain == "worker"
    wt = g.al_traced.fillna(False).astype(bool)
    w_idx = set(g.loc[w & wt, "z_idx"])
    return {
        "full_range_as_shipped": (q, w),
        "drop_slide_0069041": (q, w & (g.slide == "0068401")),
        "traced_worker_sections_only": (q, w & wt),
        "matched_windows": (q & g.in_matched_window, w & wt),
        "matched_contiguous_range": (q & g.in_matched_contiguous, w & wt),
        "matched_same_raw_z_index": (q & g.z_idx.isin(w_idx), w & wt),
        # position within a slide rather than along the concatenated stack:
        # queen AL polygon area runs high-to-low inside each slide, so the
        # stack may be two parallel sweeps rather than one A-P axis
        "within_slide_sections_3_to_5": (
            q & g.section.isin([3, 4, 5]),
            w & wt & g.section.isin([3, 4, 5]),
        ),
    }


def boot_ratio(g, qm, wm, rng):
    """Resample capture-region sections within each brain."""

    def units(m):
        d = g[m].groupby(["slide", "section"])[["n_cells", "n_al"]].sum()
        return d["n_cells"].to_numpy(float), d["n_al"].to_numpy(float)

    (qn, qa), (wn, wa) = units(qm), units(wm)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(qn), len(qn))
        j = rng.integers(0, len(wn), len(wn))
        pw = wa[j].sum() / wn[j].sum()
        if pw > 0:
            out.append((qa[i].sum() / qn[i].sum()) / pw)
    return np.percentile(out, [2.5, 97.5])


def step1(obs, g):
    rng = np.random.default_rng(0)
    rows = []
    for name, (qm, wm) in scenarios(g).items():
        r = dict(scenario=name)
        for b, m in (("queen", qm), ("worker", wm)):
            sub = g[m]
            r[f"{b}_sections_capture_regions"] = sub.groupby(
                ["slide", "section"]
            ).ngroups
            r[f"{b}_crops"] = len(sub)
            r[f"{b}_cells"] = int(sub.n_cells.sum())
            r[f"{b}_al_cells"] = int(sub.n_al.sum())
            r[f"{b}_al_pct"] = 100 * sub.n_al.sum() / sub.n_cells.sum()
        r["queen_over_worker_prevalence"] = r["queen_al_pct"] / r["worker_al_pct"]
        lo, hi = boot_ratio(g, qm, wm, rng)
        r["ratio_ci95_lo_section_bootstrap"], r["ratio_ci95_hi_section_bootstrap"] = (
            lo,
            hi,
        )
        r["queen_z_idx_range"] = f"{g[qm].z_idx.min()}-{g[qm].z_idx.max()}"
        r["worker_z_idx_range"] = f"{g[wm].z_idx.min()}-{g[wm].z_idx.max()}"
        rows.append(r)
    sc = pd.DataFrame(rows)
    sc.to_csv(OUT / "step1_prevalence_scenarios.csv", index=False)

    cols = [
        "brain",
        "slice_name",
        "slide",
        "section",
        "crop",
        "z_level",
        "z_idx",
        "n_stack",
        "u",
        "n_cells",
        "n_al",
        "al_pct",
        "al_traced",
        "al_polygon_area_um2",
        "in_matched_window",
        "in_matched_contiguous",
    ]
    g[cols].sort_values(["brain", "z_idx"]).to_csv(
        OUT / "step1_ap_crop_table.csv", index=False
    )

    # composition of AL cells, per scenario x brain x vocabulary
    key = obs["slice_name"].to_numpy()
    scen = scenarios(g)
    comp = []
    for name, (qm, wm) in scen.items():
        for b, m in (("queen", qm), ("worker", wm)):
            names = set(g.loc[m, "slice_name"])
            sel = (obs[AL].astype("object").eq(True)).to_numpy() & np.isin(
                key, list(names)
            )
            for vocab, col in (
                ("canonical17_per_cell", "cell_type_canonical"),
                ("cluster_vote", "cluster_vote"),
            ):
                vc = obs.loc[sel, col].astype(str).value_counts()
                for lab, n in vc.items():
                    comp.append(
                        dict(
                            scenario=name,
                            brain=b,
                            vocabulary=vocab,
                            label=lab,
                            n_al_cells=int(n),
                            frac_of_al=n / vc.sum(),
                        )
                    )
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "step1_al_composition.csv", index=False)

    # queen-vs-worker agreement of the AL composition profile
    from scipy.stats import spearmanr

    agree = []
    for (name, vocab), d in comp.groupby(["scenario", "vocabulary"]):
        p = d.pivot_table(
            index="label", columns="brain", values="frac_of_al", fill_value=0
        )
        if {"queen", "worker"} <= set(p.columns):
            rho = spearmanr(p["queen"], p["worker"])[0]
            agree.append(
                dict(
                    scenario=name,
                    vocabulary=vocab,
                    spearman_rho=rho,
                    l1_distance=float((p["queen"] - p["worker"]).abs().sum()),
                    top_label_queen=p["queen"].idxmax(),
                    top_frac_queen=p["queen"].max(),
                    top_label_worker=p["worker"].idxmax(),
                    top_frac_worker=p["worker"].max(),
                )
            )
    pd.DataFrame(agree).to_csv(OUT / "step1_al_composition_agreement.csv", index=False)
    return sc, comp


def fig_ap(g):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for lo, hi in g.attrs["wbins"]:
        ax.axvspan(lo, hi, color="0.88", lw=0, zorder=0)
    cols = {"queen": "#8E4A9C", "worker": "#E8A317"}
    for b in ("queen", "worker"):
        d = g[g.brain == b].sort_values("u")
        ax.plot(d.u, d.al_pct, "-", color=cols[b], lw=1, alpha=0.7)
        tr = d.al_traced.fillna(False).astype(bool)
        ax.plot(
            d.u[tr], d.al_pct[tr], "o", color=cols[b], ms=4, label=f"{b}, AL traced"
        )
        if (~tr).any():
            ax.plot(
                d.u[~tr],
                d.al_pct[~tr],
                "o",
                mfc="white",
                mec=cols[b],
                ms=4,
                label=f"{b}, AL not traced",
            )
    ax.set_xlim(0, 1)
    ax.set_xlabel("fractional position along section stack (z_level)")
    ax.set_ylabel("AL cells (% of crop)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_al_ap_profile.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# steps 2 and 3
# ---------------------------------------------------------------------------
def classify(a, g, vocab_name, col, tag_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, classification_report
    from threadpoolctl import threadpool_limits

    obs = a.obs
    xy = np.asarray(a.obsm["spatial"], float)
    cat = pd.Categorical(obs[col].astype(str))
    X = neighbourhood_composition(
        xy, cat.codes, len(cat.categories), obs["slice_name"].to_numpy(), K_MAIN
    )
    reg = nrc.region_of(obs, STATISTICAL_REGIONS)
    one = np.isin(reg, STATISTICAL_REGIONS)
    truth = np.full(len(obs), "none", dtype=object)
    truth[one] = [display(r) for r in reg[one]]
    truth[reg == nrc.OVERLAP] = "multi"
    brain = obs["brain"].to_numpy()

    # worker crops where the AL is traced (section-level, from the GeoJSON)
    traced = g.set_index("slice_name")["al_traced"].fillna(False).astype(bool)
    w_traced_cell = obs["slice_name"].map(traced).fillna(False).to_numpy(bool)
    qwin = g.set_index("slice_name")["in_matched_window"]
    q_in_window = obs["slice_name"].map(qwin).fillna(False).to_numpy(bool)

    metrics, per_region, crop_rows = [], [], []
    for train_name, trmask in (
        ("queen_all_sections", (brain == "queen") & one),
        ("queen_matched_windows", (brain == "queen") & one & q_in_window),
    ):
        for weight in (None, "balanced"):
            clf = LogisticRegression(max_iter=2000, class_weight=weight)
            w = brain == "worker"
            pred = np.empty(len(obs), dtype=object)
            # one BLAS thread: lbfgs sums in a thread-count-dependent order otherwise
            with threadpool_limits(limits=1):
                clf.fit(X[trmask], truth[trmask])
                pred[w] = clf.predict(X[w])
            wl = "balanced" if weight else "none"
            ev = w & one & w_traced_cell
            strict = w & w_traced_cell & (truth != "multi")
            for scope, m in (
                ("single-region cells, traced sections", ev),
                ("all cells in traced sections (outside-region = non-AL)", strict),
            ):
                t, p = truth[m], pred[m]
                tp = int(((t == "Antennal lobe") & (p == "Antennal lobe")).sum())
                fp = int(((t != "Antennal lobe") & (p == "Antennal lobe")).sum())
                fn = int(((t == "Antennal lobe") & (p != "Antennal lobe")).sum())
                prec = tp / (tp + fp) if tp + fp else float("nan")
                rec = tp / (tp + fn) if tp + fn else float("nan")
                metrics.append(
                    dict(
                        vocabulary=vocab_name,
                        train=train_name,
                        class_weight=wl,
                        scope=scope,
                        n_scored=int(m.sum()),
                        n_true_al=tp + fn,
                        n_pred_al=tp + fp,
                        al_precision=prec,
                        al_recall=rec,
                        al_f1=(2 * prec * rec / (prec + rec)) if tp else 0.0,
                        prevalence_in_scope=(tp + fn) / m.sum(),
                        balanced_accuracy_7class=(
                            balanced_accuracy_score(t, p) if m is ev else np.nan
                        ),
                        accuracy_7class=(float((t == p).mean()) if m is ev else np.nan),
                    )
                )
            rep = pd.DataFrame(
                classification_report(
                    truth[ev], pred[ev], output_dict=True, zero_division=0
                )
            ).T
            rep = rep.loc[[i for i in rep.index if i in set(truth[ev])]]
            per_region.append(
                rep.reset_index()
                .rename(columns={"index": "region"})
                .assign(vocabulary=vocab_name, train=train_name, class_weight=wl)
            )
            # every worker crop: predicted vs traced AL
            df = pd.DataFrame(
                dict(
                    slice_name=obs["slice_name"].to_numpy()[w],
                    truth=truth[w],
                    pred=pred[w],
                )
            )
            df["pred_al"] = df.pred == "Antennal lobe"
            df["true_al"] = df.truth == "Antennal lobe"
            df["pred_al_outside_traced_al"] = df.pred_al & ~df.true_al
            # where the predicted AL sits: cells outside every drawn region
            # ("none"), cells in one of the six other regions, or traced AL.
            df["is_none"] = df.truth == "none"
            df["is_other"] = ~df.true_al & ~df.is_none & (df.truth != "multi")
            df["pred_al_in_none"] = df.pred_al & df.is_none
            df["pred_al_in_other"] = df.pred_al & df.is_other
            c = (
                df.groupby("slice_name")
                .agg(
                    n_cells=("pred_al", "size"),
                    n_pred_al=("pred_al", "sum"),
                    n_true_al=("true_al", "sum"),
                    n_pred_al_not_in_traced_al=("pred_al_outside_traced_al", "sum"),
                    n_cells_outside_all_regions=("is_none", "sum"),
                    n_pred_al_in_outside_all_regions=("pred_al_in_none", "sum"),
                    n_cells_in_other_six_regions=("is_other", "sum"),
                    n_pred_al_in_other_six_regions=("pred_al_in_other", "sum"),
                )
                .reset_index()
            )
            crop_rows.append(
                c.assign(vocabulary=vocab_name, train=train_name, class_weight=wl)
            )
    return (
        pd.DataFrame(metrics),
        pd.concat(per_region),
        pd.concat(crop_rows, ignore_index=True),
    )


def listing(crops, g):
    gi = g[g.brain == "worker"][
        [
            "slice_name",
            "slide",
            "section",
            "crop",
            "z_idx",
            "u",
            "al_traced",
            "al_polygon_area_um2",
        ]
    ]
    d = crops.merge(gi, on="slice_name")
    d["pred_al_pct_of_crop"] = 100 * d.n_pred_al / d.n_cells
    d["traced_al_pct_of_crop"] = 100 * d.n_true_al / d.n_cells
    d["status"] = np.where(
        ~d.al_traced.astype(bool),
        "AL section not traced (possible real AL)",
        np.where(
            d.n_true_al == 0,
            "section traced, no AL in this crop's cells",
            "traced AL present",
        ),
    )
    return d


def main():
    print("loading ...", flush=True)
    a = load_cells()
    obs = a.obs
    tracing = section_tracing(obs)
    tracing.to_csv(OUT / "tracing_by_section.csv", index=False)
    print(tracing.groupby(["brain", "slide"]).al_traced.agg(["sum", "size"]))
    g = crop_table(obs, tracing)
    assert g.n_al_NA.sum() == 0, "in_al has NA cells; handle before scoring"

    sc, comp = step1(obs, g)
    fig_ap(g)
    pd.set_option("display.width", 250)
    print(sc.T)

    for step, vocabs in (
        (2, [("canonical17_per_cell", "cell_type_canonical")]),
        (3, [("cluster_vote", "cluster_vote"), ("leiden_clusters", "leiden")]),
    ):
        ms, pr, cr = [], [], []
        for vname, col in vocabs:
            print(f"classifier: {vname}", flush=True)
            m, p, c = classify(a, g, vname, col, None)
            ms.append(m), pr.append(p), cr.append(c)
        ms, pr, cr = pd.concat(ms), pd.concat(pr), pd.concat(cr)
        ms.to_csv(OUT / f"step{step}_classifier_metrics.csv", index=False)
        pr.to_csv(OUT / f"step{step}_per_region_report.csv", index=False)
        lst = listing(cr, g)
        lst.to_csv(OUT / f"step{step}_worker_crop_predictions.csv", index=False)
        flag = lst[
            (lst.n_pred_al_not_in_traced_al >= 100)
            & (lst.status != "traced AL present")
            & (lst.train == "queen_all_sections")
        ]
        flag.sort_values(["vocabulary", "class_weight", "z_idx"]).to_csv(
            OUT / f"step{step}_predicted_AL_not_traced.csv", index=False
        )
        diag = lst[lst.class_weight == "none"].assign(
            group=lambda d: (
                d.slide.astype(str)
                + np.where(d.al_traced.astype(bool), " AL traced", " AL not traced")
            )
        )
        diag = (
            diag[diag.train == "queen_all_sections"]
            .groupby(["vocabulary", "group"])
            .agg(
                crops=("n_cells", "size"),
                cells=("n_cells", "sum"),
                traced_al_cells=("n_true_al", "sum"),
                pred_al=("n_pred_al", "sum"),
                cells_outside_all_regions=("n_cells_outside_all_regions", "sum"),
                pred_al_in_outside_all_regions=(
                    "n_pred_al_in_outside_all_regions",
                    "sum",
                ),
                cells_in_other_six_regions=("n_cells_in_other_six_regions", "sum"),
                pred_al_in_other_six_regions=("n_pred_al_in_other_six_regions", "sum"),
            )
            .reset_index()
        )
        diag["pred_al_pct_of_cells"] = 100 * diag.pred_al / diag.cells
        diag["pct_outside_regions_called_al"] = (
            100 * diag.pred_al_in_outside_all_regions / diag.cells_outside_all_regions
        )
        diag["pct_other_regions_called_al"] = (
            100 * diag.pred_al_in_other_six_regions / diag.cells_in_other_six_regions
        )
        diag.to_csv(OUT / f"step{step}_traced_vs_untraced_diagnostic.csv", index=False)
        print(diag.round(2).to_string())
        print(ms[ms.scope.str.startswith("single")].to_string())


if __name__ == "__main__":
    main()
