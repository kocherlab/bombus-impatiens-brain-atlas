"""Resolution stability of the spatial-atlas Leiden clustering under resampling
(cf. chooseR, Patterson-Cross et al. 2021).

How the atlas clustering was produced (pipelines/combined_atlas/
combined_analysis.py `cluster`, object paths.SPATIAL_COMBINED): the 1,300,192
queen+worker cells, X = log1p(normalize_total 1e4) of the 101-gene panel,
HVG flags all 101 (n_top_genes=2000 > 101), PCA 50 comps on X, kNN graph
(n_neighbors=15, scanpy 'umap' method, euclidean, random_state 0),
sc.tl.leiden(flavor='igraph', n_iterations=2, directed=False, random_state=0).
Stored sweep: picked resolution 2.8 -> 32 clusters (r=3.2 -> 35 clusters);
the stored `leiden` column has 32 clusters.

Design here: for each of 20 seeds (0..19) draw 80% of cells, rebuild PCA and
the kNN graph on that subsample (same parameters), run Leiden at every
resolution in RESOLUTIONS (random_state 0, as for the atlas), save labels. The
full-data reference at each resolution is Leiden on the stored full graph.
Metrics are computed in `aggregate`.

Usage (from repo checkout; driven by pipelines/analyses/atlas_analyses.smk
rules leiden_ref / leiden_seed / leiden_aggregate / leiden_figure):
  python si_leiden_stability.py ref          # full-data reference
  python si_leiden_stability.py seed N       # one subsample, N in 0..19
  python si_leiden_stability.py aggregate    # csvs (+ figure)
  python si_leiden_stability.py figure       # redraw from leiden_stability.csv
                                             # (also the default with no argument,
                                             # which is how scripts/paper/Snakefile runs
                                             # it as the figS_leiden_stability
                                             # generator)
Resumable: finished (seed, resolution) label files are skipped.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# cap threads before numpy/scipy load (OpenBLAS segfault on this host)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config
# Repo-relative (this file is scripts/atlas/<name>.py); BIMP_MAIN overrides it
# for an isolated test run. `results/` is the shared symlink in a worktree.
MAIN = Path(os.environ.get("BIMP_MAIN", Path(__file__).resolve().parents[2]))
H5AD = (
    MAIN / "pipelines/combined_atlas/data/combined_clustered.h5ad"
)  # paths.SPATIAL_COMBINED
OUT = Path(
    os.environ.get("BIMP_LEIDEN_OUT", MAIN / "results/handoff/A_leiden_stability")
)
# The figure goes to the shared SI directory, or beside the tables when
# BIMP_LEIDEN_OUT redirects a test run.
FIG_OUT = OUT if "BIMP_LEIDEN_OUT" in os.environ else MAIN / "results/figures/si"
RESOLUTIONS = [round(r, 1) for r in np.arange(0.1, 2.01, 0.1)] + [
    2.2,
    2.4,
    2.6,
    2.8,
    3.0,
    3.2,
    3.5,
]
COMPARISON_RES = 3.2  # a second resolution, run alongside the picked one
PICKED_RES = 2.8  # uns['leiden_sweep'] picked_resolution -> 32 clusters (`leiden`)
SEEDS = list(range(20))
FRAC = 0.8
N_NEIGHBORS, N_PCS = 15, 50
LEIDEN_SEED = 0
PURITY = 0.9
REGIONS = ["in_kc_ic", "in_kc_nc", "in_kc_oc", "in_lamina", "in_medulla", "in_lobula"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_x():
    import anndata as ad

    a = ad.read_h5ad(H5AD)
    a.obs_names_make_unique()
    return a


def leiden_labels(a, res):
    import scanpy as sc

    sc.tl.leiden(
        a,
        resolution=res,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=LEIDEN_SEED,
        key_added="_l",
    )
    return a.obs["_l"].astype(np.int32).values


def run_ref():
    import scanpy as sc

    sc.settings.n_jobs = 16
    (OUT / "ref").mkdir(parents=True, exist_ok=True)
    a = load_x()
    for res in RESOLUTIONS:
        f = OUT / "ref" / f"labels_r{res}.npy"
        if f.exists():
            continue
        t = time.time()
        lab = leiden_labels(a, res)
        np.save(f, lab.astype(np.int16))
        log(f"ref r={res}: {len(set(lab))} clusters, {time.time() - t:.0f}s")
    # validation against the stored column
    f = OUT / "ref" / "validation.txt"
    from sklearn.metrics import adjusted_rand_score as ari

    stored = a.obs["leiden"].astype(int).values
    lines = [f"stored leiden: {len(set(stored))} clusters"]
    for res in (PICKED_RES, COMPARISON_RES):
        lab = np.load(OUT / "ref" / f"labels_r{res}.npy")
        lines.append(
            f"ARI(stored, rerun r={res}) = {ari(stored, lab):.4f}; n={len(set(lab))}"
        )
    f.write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


def run_seed(seed):
    import anndata as ad
    import scanpy as sc

    sc.settings.n_jobs = int(os.environ.get("N_JOBS", 16))
    d = OUT / f"seed{seed:02d}"
    d.mkdir(parents=True, exist_ok=True)
    todo = [r for r in RESOLUTIONS if not (d / f"labels_r{r}.npy").exists()]
    if not todo:
        log(f"seed {seed} complete")
        return
    a = load_x()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(a.n_obs, int(round(FRAC * a.n_obs)), replace=False))
    np.save(d / "idx.npy", idx)
    # rebuild everything downstream of normalisation on the subsample
    b = ad.AnnData(X=a.X[idx].copy(), var=a.var[["highly_variable"]].copy())
    del a
    sc.tl.pca(b, n_comps=N_PCS, mask_var="highly_variable")
    t = time.time()
    sc.pp.neighbors(b, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=0)
    log(f"seed {seed}: pca+knn on {b.n_obs} cells ({time.time() - t:.0f}s)")
    for res in todo:
        t = time.time()
        lab = leiden_labels(b, res)
        np.save(d / f"labels_r{res}.npy", lab.astype(np.int16))
        log(f"seed {seed} r={res}: {len(set(lab))} clusters {time.time() - t:.0f}s")


def frac_pure(lab, reg):
    """Fraction of clusters with >PURITY of cells inside one region.
    reg: (n_cells, 6) bool, region membership (plain bool in this object; no
    nulls exist in the six original in_* columns). Denominator is all cells of
    the cluster, so cells outside every traced region count against purity."""
    n = lab.max() + 1
    cnt = np.bincount(lab, minlength=n).astype(float)
    ok = cnt > 0
    best = np.zeros(n)
    for j in range(reg.shape[1]):
        best = np.maximum(best, np.bincount(lab, weights=reg[:, j], minlength=n))
    return float(np.mean(best[ok] / cnt[ok] > PURITY))


def aggregate():
    import anndata as ad
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    a = ad.read_h5ad(H5AD, backed="r")
    reg = a.obs[REGIONS].astype(bool).values
    rows = []
    for res in RESOLUTIONS:
        ref = np.load(OUT / "ref" / f"labels_r{res}.npy").astype(np.int32)
        ref_pure = frac_pure(ref, reg)
        for s in SEEDS:
            d = OUT / f"seed{s:02d}"
            f = d / f"labels_r{res}.npy"
            if not f.exists():
                continue
            idx = np.load(d / "idx.npy")
            lab = np.load(f).astype(np.int32)
            rows.append(
                dict(
                    resolution=res,
                    seed=s,
                    ari=ari(ref[idx], lab),
                    n_clusters=len(np.unique(lab)),
                    n_clusters_full=len(np.unique(ref)),
                    frac_clusters_pure=frac_pure(lab, reg[idx]),
                    frac_clusters_pure_full=ref_pure,
                )
            )
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "leiden_stability_per_seed.csv", index=False)
    g = per.groupby("resolution")
    out = pd.DataFrame(
        dict(
            n_seeds=g.size(),
            ari_mean=g.ari.mean(),
            ari_sd=g.ari.std(),
            n_clusters_mean=g.n_clusters.mean(),
            n_clusters_sd=g.n_clusters.std(),
            n_clusters_full=g.n_clusters_full.first(),
            frac_pure_mean=g.frac_clusters_pure.mean(),
            frac_pure_sd=g.frac_clusters_pure.std(),
            frac_pure_full=g.frac_clusters_pure_full.first(),
        )
    ).reset_index()
    out["is_paper_stated_res"] = out.resolution == COMPARISON_RES
    out["is_paper_picked_res"] = out.resolution == PICKED_RES
    out.to_csv(OUT / "leiden_stability.csv", index=False)
    print(out.round(3).to_string())
    figure(out)


def figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bimp_st.style import set_pub_style

    set_pub_style()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = out.resolution.values
    ax[0].errorbar(
        x, out.ari_mean, out.ari_sd, marker="o", ms=3, lw=1, capsize=2, color="#333"
    )
    ax[0].set_ylabel("ARI vs full data clustering")
    ax[0].set_ylim(0, 1.02)
    ax[1].errorbar(
        x,
        out.n_clusters_mean,
        out.n_clusters_sd,
        marker="o",
        ms=3,
        lw=1,
        capsize=2,
        color="#333",
        label="80% subsamples (mean ± s.d.)",
    )
    ax[1].plot(x, out.n_clusters_full, ls=":", color="#999", lw=1, label="full data")
    ax[1].set_ylabel("Number of clusters")
    ax[1].legend(frameon=False, loc="upper left")
    for A in ax:
        A.set_xlabel("Leiden resolution")
        A.axvline(PICKED_RES, color="#C0392B", lw=1, ls="--")
        for s in ("top", "right"):
            A.spines[s].set_visible(False)
    ax[0].text(
        PICKED_RES - 0.05,
        0.04,
        "r = 2.8\n(32 clusters)",
        color="#C0392B",
        ha="right",
        fontsize=8,  # tick size: prints at the house 6 pt
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))  # headroom for the panel headers
    from bimp_st.print_scale import PRINTED_WIDTH_FRAC
    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [(ax[0], "A", "Agreement"), (ax[1], "B", "Cluster number")],
        width_frac=PRINTED_WIDTH_FRAC["figS_leiden_stability"],
    )
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_OUT / f"figS_leiden_stability.{ext}", dpi=300)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        mode = sys.argv[1] if len(sys.argv) > 1 else "figure"
        OUT.mkdir(parents=True, exist_ok=True)
        if mode == "ref":
            run_ref()
        elif mode == "seed":
            run_seed(int(sys.argv[2]))
        elif mode == "aggregate":
            aggregate()
        elif mode == "figure":
            import pandas as pd

            figure(pd.read_csv(OUT / "leiden_stability.csv"))
