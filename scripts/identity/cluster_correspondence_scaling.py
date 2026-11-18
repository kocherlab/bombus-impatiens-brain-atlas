"""Is the cluster-correspondence result an artefact of leaving centroids unscaled?

The SI atlas-quality figure reports de novo snRNA clusters matching "6-8
distinct Xenium clusters against a wrong-partition null of 1.2-1.4". A
same-genes rerun with different preprocessing found 15 distinct hits against
a null of 14.1 (not significant). The two differ in several ways at once, so
this isolates the cause.

Hypothesis: noncircular_validation.py computes centroids on unscaled
log-normalised expression and correlates them with Spearman. On unscaled data
every centroid is dominated by the global abundance profile, so a centroid
from a random partition of the nuclei is close to the global mean and
best-matches whichever Xenium cluster is nearest that mean, which would
explain a null of ~1.2 distinct hits for reasons unrelated to biology.

Design: one dataset, one clustering, one matching function, with two factors
fully crossed:

    scaling : unscaled log-norm   vs   z-scored per gene (sc.pp.scale)
    metric  : Spearman            vs   Pearson

reporting observed distinct hits and the wrong-partition null for all four
combinations. A null of ~1 only when unscaled and ~14 when scaled would mean
the effect is preprocessing rather than signal.

Outputs to results/reframe/testb_sensitivity/sensitivity.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata

from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import PANEL_CROSSWALK  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

REPO = Path(__file__).resolve().parents[2]
SN = REPO / "data" / "sn"
XEN = (
    REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
)
PANEL_CSV = PANEL_CROSSWALK
OUT = REPO / "results" / "reframe" / "testb_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

RES = 1.0
N_PERM = 300
N_XEN = 60_000
SEED = 0


def dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _looks_log(M) -> bool:
    d = dense(M[:2000])
    nz = d[d > 0]
    return bool(nz.size) and not np.allclose(nz[:50_000], np.round(nz[:50_000]))


def corr_matrix(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    if metric == "spearman":
        A = rankdata(A, axis=1).astype(np.float64)
        B = rankdata(B, axis=1).astype(np.float64)
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return np.nan_to_num(A @ B.T, nan=0.0)


def main() -> None:
    panel = read_panel_crosswalk()
    v43 = list(dict.fromkeys(panel["v43_id"].dropna()))
    np31 = dict(zip(panel["np31_id"], panel["v43_id"], strict=True))

    cb = sc.read_h5ad(SN / "BIMP_atlas_qw_CentralBrain.h5ad")
    ol = sc.read_h5ad(SN / "BIMP_atlas_qw_OpticLobe.h5ad")
    shared = [g for g in v43 if g in cb.var_names and g in ol.var_names]
    sn = ad.concat(
        [cb[:, shared], ol[:, shared]], join="outer", label="atlas", keys=["cb", "ol"]
    )
    del cb, ol
    if "counts" in sn.layers:
        sn.X = sn.layers["counts"].copy()
    if not _looks_log(sn.X):
        sc.pp.normalize_total(sn, target_sum=1e4)
        sc.pp.log1p(sn)

    xe = sc.read_h5ad(XEN)
    keep, ids = [], []
    for name in xe.var_names:
        vid = np31.get(str(name).split("|")[0])
        if vid in set(shared):
            keep.append(name)
            ids.append(vid)
    xe = xe[:, keep].copy()
    xe.var_names = ids
    xe = xe[:, ~pd.Index(xe.var_names).duplicated()].copy()
    genes = [g for g in shared if g in set(xe.var_names)]
    sn, xe = sn[:, genes].copy(), xe[:, genes].copy()

    rng = np.random.default_rng(SEED)
    if xe.n_obs > N_XEN:
        xe = xe[rng.choice(xe.n_obs, N_XEN, replace=False)].copy()
    print(
        f"{sn.n_obs:,} nuclei / {xe.n_obs:,} cells / {len(genes)} shared genes",
        flush=True,
    )

    # One clustering, shared by every cell of the design.
    def cluster(a, tag):
        b = a.copy()
        for k in ("X_pca", "X_pca_harmony", "X_umap"):
            b.obsm.pop(k, None)
        if "highly_variable" in b.var.columns:
            del b.var["highly_variable"]
        sc.pp.scale(b, max_value=10)
        sc.tl.pca(
            b, n_comps=min(50, b.n_vars - 1), svd_solver="arpack", random_state=SEED
        )
        sc.pp.neighbors(b, n_neighbors=15, random_state=SEED)
        sc.tl.leiden(
            b,
            resolution=RES,
            key_added="cl",
            random_state=SEED,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        print(f"  {tag}: {b.obs['cl'].nunique()} clusters", flush=True)
        return b.obs["cl"].astype(str).values

    sn_cl, xe_cl = cluster(sn, "snRNA"), cluster(xe, "Xenium")

    Esn_raw, Exe_raw = dense(sn.X).astype(np.float32), dense(xe.X).astype(np.float32)
    mu, sd = Esn_raw.mean(0), Esn_raw.std(0) + 1e-9
    mu2, sd2 = Exe_raw.mean(0), Exe_raw.std(0) + 1e-9
    mats = {
        "unscaled log-norm": (Esn_raw, Exe_raw),
        "z-scored per gene": ((Esn_raw - mu) / sd, (Exe_raw - mu2) / sd2),
    }

    sn_ids = sorted(pd.unique(sn_cl))
    xe_ids = sorted(pd.unique(xe_cl))
    sizes = pd.Series(sn_cl).value_counts().reindex(sn_ids).values
    rows = []
    for scal, (Esn, Exe) in mats.items():
        Cxe = np.vstack([Exe[xe_cl == c].mean(0) for c in xe_ids])
        Csn = np.vstack([Esn[sn_cl == c].mean(0) for c in sn_ids])
        for metric in ("spearman", "pearson"):
            obs_hits = len(set(corr_matrix(Csn, Cxe, metric).argmax(1).tolist()))
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                perm = rng.permutation(len(sn_cl))
                fake, s0 = [], 0
                for n in sizes:
                    fake.append(Esn[perm[s0 : s0 + n]].mean(0))
                    s0 += n
                null[i] = len(
                    set(corr_matrix(np.vstack(fake), Cxe, metric).argmax(1).tolist())
                )
            p = float((null >= obs_hits).mean())
            rows.append(
                dict(
                    scaling=scal,
                    metric=metric,
                    n_sn_clusters=len(sn_ids),
                    n_xen_clusters=len(xe_ids),
                    observed_distinct=obs_hits,
                    null_mean=float(null.mean()),
                    null_sd=float(null.std()),
                    p_emp=p,
                    significant=p < 0.05,
                )
            )
            print(
                f"  {scal:20s} {metric:9s} observed={obs_hits:3d}  "
                f"null={null.mean():5.2f}+/-{null.std():4.2f}  p={p:.4f}  "
                f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    print(f"\nwrote {OUT}", flush=True)
    ns = df.groupby("scaling")["null_mean"].mean()
    print(f"\nnull mean by scaling:\n{ns.round(2).to_string()}", flush=True)
    print(
        "\nIf the null is ~1 unscaled and ~14 scaled, the observed effect is "
        "a preprocessing artefact, not signal.",
        flush=True,
    )


if __name__ == "__main__":
    main()
