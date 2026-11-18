"""Are the three Large-type Kenyon subclusters actually three distinct groups?

Tests whether the reference's three de-novo Large-type subclusters form
distinct groups in the reference embedding (if they do, a blurred spatial
projection reflects a panel limitation rather than graded biology).
Resolution 0.05 is the only resolution in a 0.05-0.4 sweep giving exactly three
clusters (higher resolutions split the largest cluster), so the number three is a
choice and its separability is tested here.

Three tests:

1. Silhouette on the Harmony-corrected PCA the clustering used. Computed on
   the embedding the labels came from, so necessary but not sufficient.

2. Held-out classification: train on 70% of nuclei, predict subcluster on the
   remaining 30%, against a label-permuted null. Partly circular (labels were
   derived from this expression), so the null matters more than the raw
   number.

3. Split-half reproducibility: re-cluster two disjoint random halves from
   scratch at the same resolution, then ask whether a classifier trained on
   half A's own clustering recovers half B's own clustering (adjusted Rand /
   AMI). A real three-way structure should be found in both halves; an
   artifact of the resolution setting should not.

Read-only apart from its own output directory.
Writes results/reframe/kc_subcluster_separability/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]

# harmonypy is not in .venv; it lives in the main checkout's vendored prefix
# (the one pipelines/analyses/atlas_analyses.smk puts on PYTHONPATH), which
# is found by walking up from REPO so this also works from a worktree. Without
# it the split-half re-clustering falls back to uncorrected PCA.
for _anc in (REPO, *REPO.parents):
    _vp = _anc / "vendor" / "pylibs"
    if _vp.is_dir():
        sys.path.insert(0, str(_vp))
        break

SRC = REPO / "results" / "fig5_kc" / "CB_LargeType_subclustered.h5ad"
OUT = REPO / "results" / "reframe" / "kc_subcluster_separability"
COL = "leiden_res0_05"
RESOLUTION = 0.05
SEED = 0
# Silhouette on 17.5k x 17.5k distances is wasteful; a stratified subsample
# estimates the same quantity. Fixed seed so the number is reproducible.
SIL_N = 6000


def _recluster(sub, resolution, seed):
    """Cluster a subset from scratch: its own HVGs, PCA, Harmony, graph.

    Does not reuse the parent object's neighbour graph or PCA, which were
    built on every nucleus and would leak full-data structure into an
    independent half.
    """
    s = sub.copy()
    s.X = s.layers["counts"].copy()
    sc.pp.normalize_total(s, target_sum=1e4)
    sc.pp.log1p(s)
    sc.pp.highly_variable_genes(s, n_top_genes=2000, batch_key="sample")
    s = s[:, s.var.highly_variable].copy()
    sc.pp.scale(s, max_value=10)
    sc.tl.pca(s, n_comps=30, svd_solver="arpack", random_state=seed)
    # Harmony can fail here for a library-version reason: this scanpy's
    # harmony_integrate hands anndata a (n_pcs,) array where obsm wants
    # (n_obs, n_pcs); half_a_harmony_error records the message. Only the
    # re-clustering is affected (the source X_pca_harmony is fine). Falling back to
    # uncorrected PCA is conservative, since residual per-library structure would
    # reduce agreement between the halves rather than create it.
    rep, why = "X_pca", None
    try:
        import scanpy.external as sce

        sce.pp.harmony_integrate(s, key="sample", max_iter_harmony=10)
        rep = "X_pca_harmony"
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
        print(f"    harmony unavailable ({why}); using raw PCA", flush=True)
    _recluster.last_harmony_error = why
    sc.pp.neighbors(s, n_neighbors=15, use_rep=rep, random_state=seed)
    sc.tl.leiden(
        s,
        resolution=resolution,
        key_added="_k",
        random_state=seed,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    return s.obs["_k"].astype(str).values, s, rep


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("loading...", flush=True)
    a = ad.read_h5ad(SRC)
    y = a.obs[COL].astype(str).values
    Z = np.asarray(a.obsm["X_pca_harmony"], dtype=np.float32)
    print(
        f"  {a.n_obs:,} nuclei, {len(set(y))} subclusters, embedding {Z.shape}",
        flush=True,
    )

    res: dict[str, object] = {
        "n_nuclei": int(a.n_obs),
        "sizes": pd.Series(y).value_counts().to_dict(),
    }

    # ---- 1. silhouette on the embedding the clustering used ---------------
    idx = rng.choice(a.n_obs, size=min(SIL_N, a.n_obs), replace=False)
    sil = float(silhouette_score(Z[idx], y[idx]))
    print(f"\n=== 1. silhouette (n={len(idx):,}, Harmony PCA) = {sil:.4f} ===")
    print("    >0.25 would be clear separation; ~0 means the three sit in one")
    print("    continuous cloud. Computed on the SAME embedding the labels")
    print("    came from, so this is an upper bound, not a neutral test.")
    res["silhouette_harmony_pca"] = sil

    # ---- 2. held-out classification vs a permuted null ---------------------
    print("\n=== 2. held-out classification (expression -> subcluster) ===")
    Xtr, Xte, ytr, yte = train_test_split(
        Z, y, test_size=0.3, random_state=SEED, stratify=y
    )
    sc_ = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, n_jobs=1)
    # one BLAS thread: lbfgs sums in a thread-count-dependent order otherwise
    with threadpool_limits(limits=1):
        clf.fit(sc_.transform(Xtr), ytr)
        bacc = float(balanced_accuracy_score(yte, clf.predict(sc_.transform(Xte))))
    yperm = rng.permutation(ytr)
    clf2 = LogisticRegression(max_iter=3000, n_jobs=1)
    with threadpool_limits(limits=1):
        clf2.fit(sc_.transform(Xtr), yperm)
        bacc_null = float(
            balanced_accuracy_score(yte, clf2.predict(sc_.transform(Xte)))
        )
    print(
        f"    balanced accuracy {bacc:.4f}   permuted-label null "
        f"{bacc_null:.4f}   chance {1 / len(set(y)):.4f}"
    )
    res["heldout_balanced_accuracy"] = bacc
    res["heldout_balanced_accuracy_permuted_null"] = bacc_null

    # ---- 3. split-half reproducibility -- the real test --------------------
    print("\n=== 3. split-half reproducibility (the test that can fail) ===")
    print("    Re-cluster two disjoint halves FROM SCRATCH, then ask whether")
    print("    a classifier trained on half A's own clustering recovers half")
    print("    B's own clustering. Neither half sees the other's labels.", flush=True)
    perm = rng.permutation(a.n_obs)
    ha, hb = perm[: a.n_obs // 2], perm[a.n_obs // 2 :]
    print("  clustering half A...", flush=True)
    ka, sa, repa = _recluster(a[ha], RESOLUTION, SEED)
    res["half_a_embedding"] = repa
    res["half_a_harmony_error"] = _recluster.last_harmony_error
    print("  clustering half B...", flush=True)
    kb, sb, repb = _recluster(a[hb], RESOLUTION, SEED + 1)
    res["half_b_embedding"] = repb
    res["half_b_harmony_error"] = _recluster.last_harmony_error
    print(
        f"    half A found {len(set(ka))} clusters {sorted(set(ka))}; "
        f"half B found {len(set(kb))} clusters {sorted(set(kb))}"
    )
    res["half_a_n_clusters"] = int(len(set(ka)))
    res["half_b_n_clusters"] = int(len(set(kb)))

    # Train on A's own labels in A's own space, predict B. B's embedding is
    # its own, so project B through A's gene space: use the shared HVGs.
    shared = [g for g in sa.var_names if g in set(sb.var_names)]
    print(f"    shared HVGs between the two halves: {len(shared)}")
    res["shared_hvgs"] = int(len(shared))
    Xa = np.asarray(sa[:, shared].X, dtype=np.float32)
    Xb = np.asarray(sb[:, shared].X, dtype=np.float32)
    scab = StandardScaler().fit(Xa)
    clf3 = LogisticRegression(max_iter=3000, n_jobs=1)
    with threadpool_limits(limits=1):  # see above
        clf3.fit(scab.transform(Xa), ka)
        pred_b = clf3.predict(scab.transform(Xb))
    ari = float(adjusted_rand_score(kb, pred_b))
    ami = float(normalized_mutual_info_score(kb, pred_b))
    print(
        f"    A-trained prediction of B vs B's OWN clustering: "
        f"ARI {ari:.4f}  AMI {ami:.4f}"
    )
    print("    ARI ~1 = the same partition rediscovered independently;")
    print("    ARI ~0 = the two halves did not find the same structure.")
    res["split_half_ari"] = ari
    res["split_half_ami"] = ami

    # How the halves' own sizes compare to the three-subcluster solution
    for nm, k in (("half_a", ka), ("half_b", kb)):
        d = (
            pd.Series(k)
            .value_counts(normalize=True)
            .sort_values(ascending=False)
            .round(4)
            .tolist()
        )
        print(f"    {nm} densities (descending): {d}")
        res[f"{nm}_densities"] = d
    res["published_reference_densities"] = [0.496, 0.369, 0.135]

    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    pd.DataFrame({"half_b_own_cluster": kb, "half_b_predicted_from_A": pred_b}).to_csv(
        OUT / "split_half_labels.csv", index=False
    )
    print(f"\nwrote {OUT}/")

    print("\n=== VERDICT INPUTS ===")
    print(f"  silhouette                 {sil:.4f}")
    print(f"  held-out balanced acc      {bacc:.4f} (null {bacc_null:.4f})")
    print(f"  split-half ARI             {ari:.4f}")
    print(f"  half A / half B n clusters {len(set(ka))} / {len(set(kb))}")
    print("  'three distinct groups' is supportable only if the split-half")
    print("  ARI is high AND both halves independently find three.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
