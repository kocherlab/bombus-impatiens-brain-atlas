#!/usr/bin/env python
"""Is the expression-cluster region-prediction result an artifact of leakage?

``region_prediction.py`` reports that neighbourhood composition in
the panel's own Leiden clusters predicts anatomical region at 0.933 accuracy /
0.928 balanced, far above the cell-type-free geometry control (0.776 / 0.725).
Those Leiden labels were fit once over all cells, so held-out sections helped
define the vocabulary used to describe them; that result measures how
anatomically confined the clusters are, not transfer of a fixed vocabulary.

This re-runs the comparison with the vocabulary rebuilt inside each fold, so
nothing about the held-out sections informs the clustering:

    for each leave-section split:
        PCA          fit on train cells only        (X_pca in the h5ad is global)
        Leiden       fit on train cells only
        transfer     nearest train-cluster centroid in train-PCA space
        features     within-section neighbourhood composition, as in
                     region_prediction.py: neighbours drawn from every cell of
                     the section, only cells inside exactly one region scored
        model        multinomial logistic regression, scaled, train -> test

PCA is refit per fold too, since the stored X_pca was also fit over all cells
(cheap for a 101-gene panel).

The geometry control and the Tangram-17 vocabulary are re-run under identical
splits, so all three are compared on the same folds.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/regions/region_prediction_foldwise_leiden.py
"""

from __future__ import annotations

import faulthandler
import os

faulthandler.enable()

# Threading: this host crashes inside sklearn/scipy numeric kernels without a
# traceback when thread counts are wrong in either direction, so the environment
# is left stock and threadpool_limits(1) is applied around the clustering block
# and around each classifier fit (score()), which also makes the result
# independent of the node's thread count.
_ = os  # kept for path handling below

import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import add_canonical  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (  # noqa: E402
    STATISTICAL_REGIONS,  # noqa: E402
    attach_region_flags,
    display,
    region_of,
)

warnings.filterwarnings("ignore")
sc.settings.n_jobs = 1

REPO = Path(__file__).resolve().parents[2]
H5 = SPATIAL_COMBINED
TANGRAM_LABELS = CANONICAL_TANGRAM
# BIMP_REGION_PREDICTION_ROOT redirects outputs (scratch reproducibility runs).
# The _v2 directories hold the run whose neighbours come from every cell of the
# section (as region_prediction.py); foldwise_leiden{,_res2.0}/ hold a run that
# draws neighbours from in-region cells only.
OUT = Path(
    os.environ.get("BIMP_REGION_PREDICTION_ROOT", REPO / "results" / "reframe")
) / (
    "foldwise_leiden_v2"
    if os.environ.get("BIMP_LEIDEN_RES") is None
    else f"foldwise_leiden_v2_res{os.environ['BIMP_LEIDEN_RES']}"
)
OUT.mkdir(parents=True, exist_ok=True)

# STATISTICAL_REGIONS = the original six hand-drawn regions + antennal lobe
# (nine_region_common). Ellipsoid body and fan-shaped body are excluded (see
# nine_region_common.DESCRIPTIVE_ONLY_REGIONS); this matches the seven-way label
# of region_prediction.py, whose comparison this script re-runs.
REGION_FLAGS = [f"in_{r}" for r in STATISTICAL_REGIONS]
NICE = {f"in_{r}": display(r) for r in STATISTICAL_REGIONS}
# As region_prediction.MORPH: nucleus_count is empty for every worker cell, so a
# median-imputed column would act as a brain indicator.
MORPH = ["cell_area", "nucleus_area", "transcript_counts", "n_genes"]
K_MAIN = 20
SEEDS = [0, 1, 2]
N_PCS = 50
# Calibrated on a 60k training subsample so the fold-wise vocabulary has the
# same grain as the global one (28 clusters against its 29); otherwise removing
# the leak would be confounded with coarsening the vocabulary (resolution 1.0
# gives 8 clusters, 2.0 gives 19, 4.0 gives 52).
LEIDEN_RES = float(os.environ.get("BIMP_LEIDEN_RES", "3.0"))
# Set BIMP_LEIDEN_RES=2.0 for the capacity control: ~19 clusters, close to
# Tangram-17's 17, which asks whether Leiden's margin is vocabulary quality
# or simply a wider feature vector.
# Clustering is fit on a subsample of the training cells and transferred to
# every cell. A 310k-cell neighbour graph is slow and tends to hit the OpenBLAS
# crash; the subsample does not change the method, since the vocabulary is
# transferred either way.
CLUSTER_SUBSAMPLE = 60_000
TRANSFER_K = 15


def neighbourhood_composition(xy, codes, n_cat, sections, k, score):
    """kNN cluster frequency per scored cell, as in region_prediction.py.

    The tree of each section is built on every cell of the section; only the
    ``score`` cells (boolean mask) are queried, and ``X[score]`` is returned.
    """
    X = np.zeros((len(codes), n_cat), dtype=np.float32)
    for s in np.unique(sections[score]):
        idx = np.flatnonzero(sections == s)
        q = idx[score[idx]]
        if len(idx) <= k or not len(q):
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xy[idx])
        nbr = nn.kneighbors(xy[q], return_distance=False)[:, 1:]
        c = codes[idx][nbr]
        for j in range(n_cat):
            X[q, j] = (c == j).sum(1)
    X = X / np.maximum(X.sum(1, keepdims=True), 1)
    return X[score]


def geometry_features(xy, morph, sections, k, score):
    """Local packing + morphology of the neighbours; tree over all cells."""
    M = np.array(morph, dtype=np.float32, copy=True)
    for j in range(M.shape[1]):
        col = M[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0
    out = np.zeros((len(M), 2 + M.shape[1]), dtype=np.float32)
    for s in np.unique(sections[score]):
        idx = np.flatnonzero(sections == s)
        q = idx[score[idx]]
        if len(idx) <= k or not len(q):
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xy[idx])
        d, nbr = nn.kneighbors(xy[q])
        out[q, 0] = d[:, 1:].mean(1)
        out[q, 1] = d[:, 1:].max(1)
        out[q, 2:] = M[idx][nbr[:, 1:]].mean(1)
    return np.nan_to_num(out[score], nan=0.0, posinf=0.0, neginf=0.0)


def score(Xtr, ytr, Xte, yte):
    # Same call as region_prediction.evaluate(), on one BLAS thread: lbfgs
    # depends on the summation order, which a multi-threaded BLAS changes with
    # the thread count.
    sc_ = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    with threadpool_limits(limits=1):
        clf.fit(sc_.transform(Xtr), ytr)
        pred = clf.predict(sc_.transform(Xte))
    return dict(
        accuracy=accuracy_score(yte, pred),
        balanced_accuracy=balanced_accuracy_score(yte, pred),
        macro_f1=f1_score(yte, pred, average="macro"),
    )


def _centroid_transfer(ref, ref_lab, query, block=50_000):
    """Assign each query cell to the nearest training cluster centroid.

    A clustering fit on the training sections is projected onto held-out cells
    by nearest centroid in the training PCA space, so nothing about the
    held-out sections informs either the basis or the partition. Used instead
    of _knn_transfer because a k=15 vote over 60k training cells needs a 1 GB
    distance block per chunk and takes over ten minutes per fold.
    """
    uniq, inv = np.unique(ref_lab, return_inverse=True)
    cent = np.zeros((len(uniq), ref.shape[1]), dtype=np.float32)
    for j in range(len(uniq)):
        cent[j] = ref[inv == j].mean(0)
    c2 = (cent**2).sum(1)
    out = np.empty(len(query), dtype=np.int64)
    for i in range(0, len(query), block):
        q = query[i : i + block]
        d = c2[None, :] - 2.0 * (q @ cent.T)
        out[i : i + block] = d.argmin(1)
    return uniq[out]


def _knn_transfer(ref, ref_lab, query, k=15, block=4000):
    """Majority-vote kNN label transfer, in plain numpy.

    sklearn's KNeighborsClassifier.predict crashes on this host without a
    traceback inside _pairwise_distances_reduction, which threads beneath
    n_jobs and ignores threadpool_limits (kd_tree and brute both crash). This
    does the same computation with an explicit BLAS matmul.

    Squared euclidean via |a|^2 - 2ab + |b|^2; the |a|^2 term is constant
    within a row so it is dropped from the argpartition.
    """
    codes, uniq = pd.factorize(ref_lab)
    r2 = (ref**2).sum(1)
    out = np.empty(len(query), dtype=np.int64)
    for i in range(0, len(query), block):
        q = query[i : i + block]
        d = r2[None, :] - 2.0 * (q @ ref.T)
        nn = np.argpartition(d, k, axis=1)[:, :k]
        lab = codes[nn]
        # One bincount over row-offset labels (np.add.at in a loop over k is ~100x
        # slower).
        nlab = len(uniq)
        offs = lab + (np.arange(len(q))[:, None] * nlab)
        counts = np.bincount(offs.ravel(), minlength=len(q) * nlab)
        out[i : i + block] = counts.reshape(len(q), nlab).argmax(1)
    return uniq[out]


def foldwise_leiden(Xexpr, tr, rest, seed):
    """PCA + Leiden on scored train cells only; transfer to every other cell.

    ``tr`` indexes the scored training cells (the clustering is fit on a
    subsample of them); ``rest`` is every other atlas cell, the held-out
    sections included, which get labels by nearest training centroid and so
    inform neither the basis nor the partition.
    """
    rng = np.random.default_rng(seed)
    fit_idx = (
        tr
        if len(tr) <= CLUSTER_SUBSAMPLE
        else rng.choice(tr, CLUSTER_SUBSAMPLE, replace=False)
    )

    print(f"    clustering on {len(fit_idx):,} train cells", flush=True)
    # One BLAS thread so PCA and the neighbour graph sum in a fixed order (the
    # fold-wise vocabulary must not depend on the node's thread count).
    with threadpool_limits(limits=1):
        pca = PCA(n_components=N_PCS, random_state=seed).fit(Xexpr[fit_idx])
        Pfit = pca.transform(Xexpr[fit_idx])
        print("    PCA ok", flush=True)
        Ptr, Pte = pca.transform(Xexpr[tr]), pca.transform(Xexpr[rest])
        print("    PCA transform ok", flush=True)

        tmp = ad.AnnData(np.zeros((len(fit_idx), 1), dtype=np.float32))
        tmp.obsm["X_pca"] = Pfit
        sc.pp.neighbors(tmp, n_neighbors=15, use_rep="X_pca", random_state=seed)
        print("    neighbors ok", flush=True)
        sc.tl.leiden(
            tmp,
            resolution=LEIDEN_RES,
            random_state=seed,
            key_added="cl",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        lab_fit = tmp.obs["cl"].astype(str).values
        print(f"    leiden ok -> {len(set(lab_fit))} clusters", flush=True)

        lab_tr = _centroid_transfer(Pfit, lab_fit, Ptr)
        print("    transferred to train", flush=True)
        lab_te = _centroid_transfer(Pfit, lab_fit, Pte)
        print("    transferred to all other cells", flush=True)
    return lab_tr, lab_te


CACHE = OUT / "prepared.npz"


def _newest_input_mtime():
    """Newest mtime across everything the cache is derived from.

    ``y`` in the cache is the region label, so a relabel must invalidate it;
    without this check a stale cache silently reproduces results from the
    old labels. results/ is a shared symlink with no git history.
    """
    paths = [H5, TANGRAM_LABELS]
    paths += sorted((REPO / "results" / "nine_region" / "crops").glob("*/*.parquet"))
    # The cache also depends on the code that fills it: changing how flags are
    # joined changes `y` while every data mtime stays put, so this script and the
    # vocabulary module are included.
    paths += [
        Path(__file__).resolve(),
        REPO / "src" / "bimp_st" / "nine_region_common.py",
    ]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def prepare():
    """Load once, cache the arrays the comparison actually needs."""
    if CACHE.exists():
        if CACHE.stat().st_mtime >= _newest_input_mtime():
            print(f"using cached {CACHE.name}", flush=True)
            z = np.load(CACHE, allow_pickle=True)
            return (
                z["Xexpr"],
                z["keep"],
                z["y"],
                z["groups"],
                z["xy"],
                z["morph"],
                z["gleiden"],
                z["tangram"],
            )
        print(f"{CACHE.name} is older than its inputs -- rebuilding", flush=True)

    print(f"loading {H5.name} ...", flush=True)
    a = ad.read_h5ad(H5)
    from bimp_st.paths import keyed_obs_column

    # joined on (slice_name, cell_id), never by position; raises on any unmatched
    # or repeated cell rather than falling back to the in-file column
    a.obs["tangram_cell_type"] = keyed_obs_column(
        a, "tangram_cell_type", TANGRAM_LABELS
    ).astype(str)
    add_canonical(a, src="tangram_cell_type")

    # Join all seven fitted regions from the crop sidecars, not just the antennal
    # lobe. The six in_<region> flags in combined_clustered.h5ad predate polygon
    # repair, which moves cells (queen KC-NC +439; four polygons on worker
    # 0068401 R1 move 119 cells from the lobula to the antennal lobe). Mixing
    # stored and sidecar flags would mark
    # reclassified cells as both in_lobula and in_al, and the exactly-one-region
    # filter would silently drop them (610 cells).
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    # One region per cell through the shared helper (as region_prediction.py):
    # untraced (None), OVERLAP and OUTSIDE cells are not scored, but every cell
    # stays in the neighbour pool.
    region = region_of(a.obs, STATISTICAL_REGIONS, overlap="drop")
    keep = np.isin(region.astype(str), list(STATISTICAL_REGIONS))
    print(f"  {int(pd.isna(region).sum()):,} cells untraced for a region", flush=True)
    obs = a.obs.reset_index(drop=True)
    y = np.array([display(r) for r in region[keep]])
    groups = obs["slice_name"].astype(str).values
    xy = np.asarray(a.obsm["spatial"], float)
    X = a.X
    Xexpr = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    cols = [c for c in MORPH if c in obs.columns]
    morph = obs[cols].to_numpy(dtype=np.float32)
    gleiden = obs["leiden"].astype(str).values
    tangram = obs["cell_type_canonical"].astype(str).values

    np.savez_compressed(
        CACHE,
        Xexpr=Xexpr,
        keep=keep,
        y=y,
        groups=groups,
        xy=xy,
        morph=morph,
        gleiden=gleiden,
        tangram=tangram,
    )
    print(f"cached -> {CACHE.name}", flush=True)
    return Xexpr, keep, y, groups, xy, morph, gleiden, tangram


def main() -> None:
    Xexpr, keep, y, groups, xy, morph, gleiden_s, tangram_s = prepare()
    global_leiden = pd.Categorical(gleiden_s)
    tangram = pd.Categorical(tangram_s)
    kidx = np.flatnonzero(keep)  # scored cell -> atlas row
    kgroups = groups[keep]
    print(
        f"  {len(y):,} of {len(keep):,} cells in exactly one region (scored; "
        f"neighbours drawn from all cells); expression {Xexpr.shape}",
        flush=True,
    )
    print(f"  global Leiden: {len(global_leiden.categories)} clusters", flush=True)
    rows = []

    def run(codes, n_cat, label, seed, tr, te):
        Xs = neighbourhood_composition(xy, codes, n_cat, groups, K_MAIN, keep)
        m = score(Xs[tr], y[tr], Xs[te], y[te])
        m.update(vocabulary=label, seed=seed, n_clusters=n_cat)
        rows.append(m)
        print(
            f"  {label:<34} acc {m['accuracy']:.3f} "
            f"bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f}",
            flush=True,
        )

    for seed in SEEDS:
        # Splits over the scored cells, grouped by section.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr, te = next(gss.split(kidx, y, kgroups))
        print(
            f"\n=== seed {seed}: {len(tr):,} train / {len(te):,} test cells, "
            f"{len(set(kgroups[te]))} held-out sections ===",
            flush=True,
        )

        # 1. fold-wise Leiden (no leakage): fit on scored train cells, every
        # other atlas cell labelled by nearest training centroid.
        tr_atlas = kidx[tr]
        rest = np.setdiff1d(np.arange(len(keep)), tr_atlas)
        lab_tr, lab_rest = foldwise_leiden(Xexpr, tr_atlas, rest, seed)
        allc = np.empty(len(keep), dtype=object)
        allc[tr_atlas], allc[rest] = lab_tr, lab_rest
        cat = pd.Categorical(allc)
        run(
            cat.codes,
            len(cat.categories),
            "Leiden expression (fold-wise)",
            seed,
            tr,
            te,
        )

        # 2. the leaky global Leiden, same split
        run(
            global_leiden.codes,
            len(global_leiden.categories),
            "Leiden expression (global, leaky)",
            seed,
            tr,
            te,
        )

        # 3. Tangram-17, same split
        run(
            tangram.codes, len(tangram.categories), "Tangram canonical-17", seed, tr, te
        )

        # 4. geometry control, same split
        Xgeo = geometry_features(xy, morph, groups, K_MAIN, keep)
        m = score(Xgeo[tr], y[tr], Xgeo[te], y[te])
        m.update(vocabulary="geometry only", seed=seed, n_clusters=0)
        rows.append(m)
        print(
            f"  {'geometry only':<34} acc {m['accuracy']:.3f} "
            f"bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "foldwise_leiden.csv", index=False)
    summ = (
        df.groupby("vocabulary")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summ.to_csv(OUT / "summary.csv")
    print("\n=== mean over seeds ===")
    print(summ)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
