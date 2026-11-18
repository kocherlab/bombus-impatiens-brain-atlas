"""Compositional data analysis (CoDA) for ltKC subcluster Q vs W.

Properly handles the simplex constraint via:
1. CLR (centered log-ratio) transform per slice
2. Aitchison-distance PERMANOVA for Q vs W global test
3. Per-subcluster log-ratio differences (Welch t-test on CLR coords)
4. PCA on CLR-transformed proportions for visualization
5. (Optional) scCODA Bayesian compositional analysis via pertpy

Input: the same object `fig:kc_subtypes` F/G/H reads (the CB-reference,
no-cell-type-gate projection written by project_large_type_subclusters.py),
restricted to KC-NC.
Not the ltKC-reference projection
(``results/tangram_ltKC/ltKC_spatial_subclusters_nc_only.h5ad``), which is
built on the confounded ltKC dataset and gated on the per-cell
``tangram_cell_type == "Large-type Kenyon cells"`` argmax, a panel-unresolvable
call. Cells are selected by hand-drawn anatomy (``in_kc_nc``), never by
``tangram_cell_type``, so the Tangram column-name permutation cannot reach
this analysis.

Runs per section (30 queen / 35 worker). ltKC-reference cluster indices
have no established correspondence to the CB S1/S2/S3 labels.

READ THIS BEFORE INTERPRETING THE OUTPUT: a large, highly significant
PERMANOVA statistic here is EXPECTED and is not evidence of caste biology. The
queen-vs-worker difference in this composition is very likely technical --
it appears at comparable magnitude in cell types with no Kenyon-cell
content, grows rather than shrinks with call confidence, is larger than
the unrestricted 29-category Tangram run gives for the same cells, persists
in a Tangram-free centroid control, and does not replicate in the n=3-per-
caste snRNA arm.
The permuted units are also sections within one brain per caste, so the
P value is pseudoreplicated by construction.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ttest_ind

OUT = Path("results/tangram_ltKC")
# The CB-reference projection `fig:kc_subtypes` F/G/H uses (module docstring).
IN_H5AD = Path("results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad")
SEED = 0
N_BOOT_SLICE = 200

COLORS = {"0": "#1F77B4", "1": "#FF7F0E", "2": "#2CA02C"}
BRAIN_COLORS = {"queen": "#B5359B", "worker": "#168A4F"}
CLUSTERS = ["0", "1", "2"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Load Tangram preds ───────────────────────────────────────────────────
print("Loading...")
sp = sc.read_h5ad(IN_H5AD)
sp.obs_names_make_unique()
# The CB object spans all three Kenyon compartments; this analysis is KC-NC
# only (the compartment Large-type occupies), matching `fig:kc_subtypes` F/G/H.
# The gate
# is anatomical, not a per-cell tangram_cell_type call.
sp = sp[sp.obs["in_kc_nc"].astype(bool).values].copy()
print(f"  KC-NC cells: {sp.n_obs:,}", flush=True)
probs = sp.obsm["tangram_ct_pred"]
if not isinstance(probs, pd.DataFrame):
    probs = pd.DataFrame(probs, index=sp.obs_names, columns=CLUSTERS)
probs.columns = [str(c) for c in probs.columns]
prob_norm = probs.div(probs.sum(axis=1), axis=0).fillna(1.0 / probs.shape[1])
prob_arr = prob_norm.values
classes = np.array(probs.columns)

brain_arr = sp.obs["brain"].values
z_arr = sp.obs["z_level"].values


# ── Per-slice bootstrap proportions ──────────────────────────────────────
def sample_one_pass(prob_subset, rng):
    u = rng.random(len(prob_subset))
    cum = np.cumsum(prob_subset, axis=1)
    idx = (cum < u[:, None]).sum(axis=1)
    idx = np.minimum(idx, len(classes) - 1)
    return classes[idx]


def bootstrap_proportions(prob_subset, n_boot, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, 3))
    for i in range(n_boot):
        labels = sample_one_pass(prob_subset, rng)
        for j, c in enumerate(classes):
            out[i, j] = (labels == c).mean()
    return out


print("Per-slice bootstrap...")
slices = []
for brain in ["queen", "worker"]:
    bmask = brain_arr == brain
    z_unique = sorted(set(z_arr[bmask]))
    for z in z_unique:
        zmask = bmask & (z_arr == z)
        if zmask.sum() < 100:
            continue
        boots = bootstrap_proportions(
            prob_arr[zmask],
            N_BOOT_SLICE,
            int(z) + (0 if brain == "queen" else 1000),
        )
        m = boots.mean(axis=0)
        slices.append(
            {
                "brain": brain,
                "z": int(z),
                "n": int(zmask.sum()),
                "p0": m[0],
                "p1": m[1],
                "p2": m[2],
            }
        )
sdf = pd.DataFrame(slices)
print(f"  {len(sdf)} slices")

# ── 1. CLR transform ─────────────────────────────────────────────────────
print("\n=== 1. CLR transform ===")


def clr(X, eps=1e-6):
    """Centered log-ratio transform. X: (n_samples, n_classes), rows sum to 1."""
    X = np.asarray(X, dtype=float)
    X = X + eps  # zero-imputation
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    return log_X - log_X.mean(axis=1, keepdims=True)


props = sdf[["p0", "p1", "p2"]].values
clr_props = clr(props)
sdf["clr0"] = clr_props[:, 0]
sdf["clr1"] = clr_props[:, 1]
sdf["clr2"] = clr_props[:, 2]
print("CLR ranges per coord:")
for i, c in enumerate(CLUSTERS):
    print(
        f"  Sub {c}: mean={clr_props[:, i].mean():.3f}, std={clr_props[:, i].std():.3f}"
    )

# ── 2. PERMANOVA on Aitchison distance (Q vs W) ──────────────────────────
print("\n=== 2. PERMANOVA: Aitchison distance Q vs W ===")


def aitchison_distance_matrix(clr_X):
    n = len(clr_X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(clr_X[i] - clr_X[j])
            D[i, j] = D[j, i] = d
    return D


def permanova(D, groups, n_perm=9999, seed=0):
    """One-way PERMANOVA on distance matrix, groups: array of labels."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n = len(groups)

    def pseudo_F(grp):
        ssw = 0
        ssb = 0
        # Sum of squares within: for each group, sum of squared distances within / 2*n_g
        for g in unique_groups:
            mask = grp == g
            n_g = mask.sum()
            if n_g <= 1:
                continue
            sub_D = D[np.ix_(mask, mask)]
            ssw += (sub_D**2).sum() / (2 * n_g)
        # Total sum of squares: sum of all squared distances / 2n
        sst = (D**2).sum() / (2 * n)
        ssb = sst - ssw
        a = len(unique_groups)
        F = (ssb / (a - 1)) / (ssw / (n - a))
        return F

    F_obs = pseudo_F(groups)
    F_perm = np.zeros(n_perm)
    for i in range(n_perm):
        permuted = rng.permutation(groups)
        F_perm[i] = pseudo_F(permuted)
    p = ((F_perm >= F_obs).sum() + 1) / (n_perm + 1)
    return F_obs, p


D = aitchison_distance_matrix(clr_props)
brain_lbl = sdf["brain"].values
F, pval = permanova(D, brain_lbl, n_perm=9999)
print(f"  Pseudo-F = {F:.3f}, p = {pval:.4f} (n_perm=9999)")
print(
    f"  → {'SIGNIFICANT' if pval < 0.05 else 'not significant'} compositional "
    f"difference Q vs W"
)

# ── 3. Per-subcluster CLR test (Welch t-test) ────────────────────────────
print("\n=== 3. Per-subcluster CLR Welch t-test ===")
qmask_s = sdf["brain"] == "queen"
wmask_s = sdf["brain"] == "worker"

clr_results = []
for _i, c in enumerate(CLUSTERS):
    q_clr = sdf.loc[qmask_s, f"clr{c}"].values
    w_clr = sdf.loc[wmask_s, f"clr{c}"].values
    t, p = ttest_ind(q_clr, w_clr, equal_var=False)
    diff = w_clr.mean() - q_clr.mean()
    clr_results.append(
        {
            "subcluster": c,
            "queen_clr": q_clr.mean(),
            "worker_clr": w_clr.mean(),
            "diff_W_Q": diff,
            "t": t,
            "p_unadj": p,
        }
    )

# Bonferroni
n_tests = len(clr_results)
for r in clr_results:
    r["p_bonf"] = min(1, r["p_unadj"] * n_tests)
clr_df = pd.DataFrame(clr_results)
print(clr_df.to_string(index=False))
clr_df.to_csv(OUT / "ltKC_coda_clr_tests.csv", index=False)

# ── 4. PCA on CLR-transformed slice proportions ──────────────────────────
print("\n=== 4. PCA on CLR-space ===")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
clr_pca = pca.fit_transform(clr_props)
print(f"  Explained variance: {pca.explained_variance_ratio_}")
sdf["pc1"] = clr_pca[:, 0]
sdf["pc2"] = clr_pca[:, 1]

# ── 5. Visualization: 3-panel CoDA summary ───────────────────────────────
print("\n=== 5. Generating CoDA figure ===")

fig = plt.figure(figsize=(20, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.35)

# Panel A: PCA on CLR-space (each slice = 1 point)
ax = fig.add_subplot(gs[0])
for brain in ["queen", "worker"]:
    bsdf = sdf[sdf["brain"] == brain]
    ax.scatter(
        bsdf["pc1"],
        bsdf["pc2"],
        c=BRAIN_COLORS[brain],
        s=100,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.7,
        label=f"{brain.capitalize()} ({len(bsdf)})",
        zorder=3,
    )
ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
ax.set_title(
    f"PCA of CLR-transformed proportions\nPERMANOVA: F = {F:.2f}, p = {pval:.4f}",
    fontsize=11,
)
ax.legend(fontsize=10, framealpha=0.95, edgecolor="none")
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

# Panel B: CLR coord per-subcluster Welch test (boxplot Q vs W per Sub)
ax = fig.add_subplot(gs[1])
positions = []
labels = []
jitter_rng = np.random.default_rng(SEED)  # seeded strip-plot jitter
for i, c in enumerate(CLUSTERS):
    for j, brain in enumerate(["queen", "worker"]):
        x_pos = i * 3 + j
        positions.append(x_pos)
        bsdf = sdf[sdf["brain"] == brain]
        vals = bsdf[f"clr{c}"].values
        # Strip + median bar
        ax.scatter(
            np.full(len(vals), x_pos) + jitter_rng.uniform(-0.15, 0.15, len(vals)),
            vals,
            color=BRAIN_COLORS[brain],
            alpha=0.6,
            s=40,
            edgecolors="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.plot(
            [x_pos - 0.3, x_pos + 0.3],
            [vals.mean(), vals.mean()],
            color="black",
            linewidth=2.5,
            zorder=4,
        )
        labels.append(brain.capitalize())

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")

# Subcluster labels above each pair
for i, c in enumerate(CLUSTERS):
    r = clr_df[clr_df["subcluster"] == c].iloc[0]
    star = (
        "***"
        if r["p_bonf"] < 0.001
        else ("**" if r["p_bonf"] < 0.01 else ("*" if r["p_bonf"] < 0.05 else "ns"))
    )
    ax.text(
        i * 3 + 0.5,
        1.04,
        f"Sub {c}\np_bonf = {r['p_bonf']:.2e}\n{star}",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLORS[c],
        fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )

ax.set_ylabel("CLR coordinate (per slice)")
ax.set_title("Per-subcluster CLR test\n(Welch t, Bonferroni-corrected)", fontsize=11)
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

# Panel C: Q-W mean log-ratio differences (effect sizes)
ax = fig.add_subplot(gs[2])
diffs = clr_df["diff_W_Q"].values
ses = []
for c in CLUSTERS:
    q_clr = sdf.loc[qmask_s, f"clr{c}"].values
    w_clr = sdf.loc[wmask_s, f"clr{c}"].values
    se = np.sqrt(q_clr.var(ddof=1) / len(q_clr) + w_clr.var(ddof=1) / len(w_clr))
    ses.append(se)
ses = np.array(ses)
ci_lo = diffs - 1.96 * ses
ci_hi = diffs + 1.96 * ses

x = np.arange(len(CLUSTERS))
bar_colors = [COLORS[c] for c in CLUSTERS]
ax.barh(
    x,
    diffs,
    color=bar_colors,
    alpha=0.85,
    xerr=[diffs - ci_lo, ci_hi - diffs],
    capsize=5,
    edgecolor="white",
    linewidth=0.5,
)
for i, (d, lo, hi) in enumerate(zip(diffs, ci_lo, ci_hi, strict=False)):
    ax.text(
        d + 0.01 * np.sign(d),
        i,
        f"{d:+.3f}\n[{lo:+.2f}, {hi:+.2f}]",
        va="center",
        ha="left" if d > 0 else "right",
        fontsize=9,
    )
ax.set_yticks(x)
ax.set_yticklabels([f"Sub {c}" for c in CLUSTERS], fontsize=12)
for i, c in enumerate(CLUSTERS):
    ax.get_yticklabels()[i].set_color(COLORS[c])
    ax.get_yticklabels()[i].set_fontweight("bold")
ax.axvline(0, color="black", linewidth=1, zorder=0)
ax.set_xlabel("Worker − Queen CLR (log-ratio difference)")
ax.set_title("Effect sizes (CLR space)\nW > 0: enriched in worker", fontsize=11)

plt.suptitle(
    "Compositional data analysis (CoDA) of ltKC subclusters in the Large-type "
    "compartment",
    fontsize=15,
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUT / "final_coda_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved CoDA summary figure")

# ── 6. Optional: scCODA via pertpy (Bayesian compositional analysis) ─────
print("\n=== 6. scCODA via pertpy (if available) ===")
try:
    import pertpy as pt

    print(f"  pertpy version: {pt.__version__}")

    # Build counts AnnData: per-slice cell counts per subcluster
    counts_data = []
    for _, row in sdf.iterrows():
        counts_data.append(
            {
                "sample_id": f"{row['brain']}_z{row['z']}",
                "brain": row["brain"],
                "Sub_0": int(row["p0"] * row["n"]),
                "Sub_1": int(row["p1"] * row["n"]),
                "Sub_2": int(row["p2"] * row["n"]),
            }
        )
    counts_df = pd.DataFrame(counts_data).set_index("sample_id")

    # AnnData for scCODA
    import anndata as ad

    counts_X = counts_df[["Sub_0", "Sub_1", "Sub_2"]].values
    coda_obs = counts_df[["brain"]].copy()
    coda_var = pd.DataFrame(index=["Sub_0", "Sub_1", "Sub_2"])
    coda_adata = ad.AnnData(X=counts_X.astype(float), obs=coda_obs, var=coda_var)

    sccoda = pt.tl.Sccoda()
    coda_data = (
        sccoda.load(
            coda_adata,
            type="cell_level",
            generate_sample_level=False,
            cell_type_identifier=None,
            sample_identifier=None,
            covariate_obs=["brain"],
        )
        if hasattr(sccoda, "load")
        else None
    )

    # Try the modern API
    coda_model = (
        sccoda.prepare(
            coda_adata,
            formula="brain",
            reference_cell_type="automatic",
        )
        if hasattr(sccoda, "prepare")
        else None
    )

    if coda_model is not None:
        sccoda.run_nuts(coda_model, num_samples=2000, num_warmup=1000)
        results = sccoda.summary(coda_model)
        print(results)
        # Save
        with open(OUT / "ltKC_sccoda_summary.txt", "w") as f:
            f.write(str(results))
    else:
        print("  scCODA API not available in this pertpy version")
except ImportError as e:
    print(f"  pertpy not installed: {e}")
except Exception as e:
    print(f"  scCODA run failed: {type(e).__name__}: {e}")

print("\nDone!")
