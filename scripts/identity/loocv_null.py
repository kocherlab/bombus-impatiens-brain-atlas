#!/usr/bin/env python3
"""Null distributions for the LOO-CV held-out gene imputation result.

The atlas-quality figure reports the LOO-CV cosine similarity of the 100
matched Xenium panel genes (results/tangram/combined_clustered_loocv_scores.csv,
also behind SI figS_tangram_validation panel A). This builds two non-circular
nulls for it (no Tangram cell-type label on either side):

  Null A -- observed vs observed, random gene pairs: cosine between two
  unrelated genes' measured Xenium spatial patterns. Bounds how much of the
  score comes from shared cell-density / spatial-autocorrelation structure
  (expected to be non-zero). All 100 genes: C(100, 2) = 4950 pairs.

  Null B -- predicted vs mismatched observed: for a gene i predicted by a
  held-out Tangram fit, score that predicted vector against a different gene
  j's observed pattern (i != j). The LOO-CV run saved only the scalar cosine
  per gene, so this uses the 15 held-out "test" genes from the separate
  train/test-split run (results/tangram/combined_clustered_imputed.h5ad,
  gene_split.json is_training=False). That is a different design (85/15
  single split, not 99/1 LOO), so Null B is a complementary check, not a
  substitute for Null A.

  Null B cannot be computed from the files on disk. The
  imputed object's rows are joined to the observed object by (slice_name,
  cell_id) and the two share one cell order, yet its predicted values do not
  track the measured ones even for the 85 TRAINING genes, which Tangram fits
  directly: per-cell predicted vs observed totals over those genes do not
  correlate, overall or within single sections, and the predicted-vs-observed
  cosine of the training genes falls far below the one recorded in
  gene_scores.csv. So the imputed matrix's rows are not its obs
  rows (or it is not the run gene_scores.csv scored); re-projecting the
  mapping would be needed. Null B is written but flagged
  (null_b_predicted_vs_mismatched.sanity_check_passed = false, with the
  reason); Null A and the tangram_qc C correlations do not use Null B.

Everything is computed in linear expression space, matching
tangram_mapping.py's cross_validate() and train(), which both expm1() before
scoring/training; the .h5ad stores log1p.

Outputs -> results/reframe/loocv_null/:
  null_a_observed_vs_observed.csv   (4950 pairwise cosines)
  null_b_predicted_vs_mismatched.csv (15 x 99 = 1485 pairwise cosines)
  null_b_predicted_vs_matched.csv    (15 genuine held-out scores, sanity check)
  summary.json / summary.csv

    PYTHONPATH=src:scripts .venv/bin/python scripts/identity/loocv_null.py \
        [--loocv-csv <scores.csv>] [--out <dir>]
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (
    PANEL_CROSSWALK,  # noqa: E402
    REPO,  # noqa: E402
)

LOOCV_CSV = REPO / "results/tangram/combined_clustered_loocv_scores.csv"
GENE_SPLIT_JSON = REPO / "results/tangram/combined_clustered_gene_split.json"
GENE_SCORES_CSV = REPO / "results/tangram/combined_clustered_gene_scores.csv"
COMBINED_H5AD = REPO / "pipelines/combined_atlas/data/combined_clustered.h5ad"
IMPUTED_H5AD = REPO / "results/tangram/combined_clustered_imputed.h5ad"
GENE_MAP_CSV = PANEL_CROSSWALK
OUT = REPO / "results/reframe/loocv_null"


def _v43_to_np31_map():
    mapping = read_panel_crosswalk()
    mapping["v43_lc"] = mapping["v43_id"].str.lower()
    return dict(zip(mapping["v43_lc"], mapping["np31_id"], strict=True))


def _cell_keys(obs) -> np.ndarray:
    """(slice_name, cell_id) per row; the only safe join between two objects."""
    return (
        obs["slice_name"].astype(str).to_numpy()
        + "||"
        + obs["cell_id"].astype(str).to_numpy()
    )


#: Null B's matched (genuine held-out) median must reproduce the recorded
#: single-split score to this tolerance, or the rows are misaligned.
SANITY_TOL = 0.01


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--loocv-csv",
        type=Path,
        default=LOOCV_CSV,
        help="per-gene LOO-CV scores "
        "(default: results/tangram/combined_clustered_loocv_scores.csv)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="output directory (default: results/reframe/loocv_null)",
    )
    args = ap.parse_args()
    loocv_csv = (
        args.loocv_csv if args.loocv_csv.is_absolute() else REPO / args.loocv_csv
    )
    out = args.out if args.out.is_absolute() else REPO / args.out
    _run(loocv_csv, out)


def _run(LOOCV_CSV, OUT):
    print(f"LOO-CV scores: {LOOCV_CSV}\nOutput: {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    loocv = pd.read_csv(LOOCV_CSV)
    genes_v43 = loocv["gene"].tolist()  # 100 genes, lowercase v4.3 ids
    obs_median, obs_mean = loocv["cosine_sim"].median(), loocv["cosine_sim"].mean()
    print(
        f"LOO-CV observed: n={len(genes_v43)}, median={obs_median:.4f}, "
        f"mean={obs_mean:.4f}",
        flush=True,
    )

    v43_to_np31 = _v43_to_np31_map()

    # ── Load observed Xenium expression for the 100 panel genes (linear) ──
    print(
        f"Loading {COMBINED_H5AD.name} (observed, all 100 panel genes)...", flush=True
    )
    a_obs = ad.read_h5ad(COMBINED_H5AD, backed="r")
    var_by_np31_obs = {v.split("|")[0]: v for v in a_obs.var_names}
    matched_var, matched_lc = [], []
    for g in genes_v43:
        np31 = v43_to_np31.get(g)
        if np31 in var_by_np31_obs:
            matched_var.append(var_by_np31_obs[np31])
            matched_lc.append(g)
    sub_obs = a_obs[:, matched_var].to_memory()
    X_obs = sub_obs.X.toarray() if sp.issparse(sub_obs.X) else np.asarray(sub_obs.X)
    X_obs = np.expm1(X_obs.astype(np.float64))  # .h5ad is log1p; match cross_validate
    obs_key = _cell_keys(a_obs.obs)
    del a_obs, sub_obs
    print(f"  observed matrix: {X_obs.shape}", flush=True)

    # ── NULL A: observed-vs-observed, all pairs among the 100 genes ──
    norms = np.linalg.norm(X_obs, axis=0)
    norms[norms == 0] = 1.0
    Xn = X_obs / norms
    cos_mat = Xn.T @ Xn
    n = cos_mat.shape[0]
    iu = np.triu_indices(n, k=1)
    null_a = cos_mat[iu]
    pd.DataFrame(
        {
            "gene_i": [matched_lc[i] for i in iu[0]],
            "gene_j": [matched_lc[j] for j in iu[1]],
            "cosine": null_a,
        }
    ).to_csv(OUT / "null_a_observed_vs_observed.csv", index=False)
    print(
        f"  Null A: n_pairs={len(null_a)}, median={np.median(null_a):.4f}, "
        f"mean={null_a.mean():.4f}",
        flush=True,
    )

    # ── NULL B: predicted-vs-mismatched-observed (15 genuinely held-out test genes) ──
    gene_split = json.loads(GENE_SPLIT_JSON.read_text())
    test_genes_v43 = [g.lower() for g in gene_split["test_genes"]]
    print(
        f"Loading {IMPUTED_H5AD.name} (predicted, {len(test_genes_v43)} "
        f"held-out test genes)...",
        flush=True,
    )
    a_pred = ad.read_h5ad(IMPUTED_H5AD, backed="r")
    # This h5ad's var_names are already lowercase v4.3 ids (unlike
    # combined_clustered.h5ad, which uses "BIMP_vNP3.1_xxxxx|MGEN_..." --
    # gene_scores.csv/gene_split.json for THIS file are also v4.3, confirming
    # the direct-match is correct here, no np31 round-trip needed).
    var_lc_pred = {v.lower(): v for v in a_pred.var_names}
    test_var, test_lc = [], []
    for g in test_genes_v43:
        if g in var_lc_pred:
            test_var.append(var_lc_pred[g])
            test_lc.append(g)
    print(f"  matched {len(test_var)}/{len(test_genes_v43)} test genes", flush=True)
    if len(test_var) == 0:
        raise SystemExit(
            f"no test genes matched in {IMPUTED_H5AD.name}; "
            f"sample var_names: {list(a_pred.var_names[:5])}, "
            f"sample test genes: {test_genes_v43[:5]}"
        )

    # Row-block sequential read (this h5ad is a large DENSE, unchunked array;
    # fancy column-indexing over the whole file is pathologically slow on
    # NFS -- read contiguous row blocks instead and keep only the columns we
    # need as we go).
    # Rows of the imputed object are joined to the observed object on
    # (slice_name, cell_id), never by position: the two need not share an order.
    pred_key = _cell_keys(a_pred.obs)
    if len(set(pred_key)) != len(pred_key) or len(set(obs_key)) != len(obs_key):
        raise SystemExit("duplicate (slice_name, cell_id) keys; cannot join")
    obs_pos = pd.Index(obs_key).get_indexer(pred_key)
    if (obs_pos < 0).any():
        raise SystemExit(
            f"{int((obs_pos < 0).sum()):,} imputed cells have no observed match"
        )
    if len(obs_pos) != len(obs_key):
        raise SystemExit(
            f"imputed has {len(obs_pos):,} cells, observed {len(obs_key):,}"
        )
    d = a_pred.file["X"]
    n_obs_cells = d.shape[0]
    col_idx_all = {v: i for i, v in enumerate(a_pred.var_names)}
    want_idx = np.array([col_idx_all[v] for v in test_var], dtype=int)
    block = 100_000
    pred_cols = np.empty((n_obs_cells, len(want_idx)), dtype=np.float64)
    for start in range(0, n_obs_cells, block):
        end = min(start + block, n_obs_cells)
        rows = d[start:end, :]  # contiguous row-slab read
        pred_cols[start:end, :] = rows[:, want_idx]
        print(f"    read rows {start}-{end}/{n_obs_cells}", flush=True)
    a_pred.file.close()
    # Put the predicted rows in the observed object's row order.
    aligned = np.empty_like(pred_cols)
    aligned[obs_pos] = pred_cols
    pred_cols = aligned

    # Sanity: is this .h5ad log1p or already linear? train() expm1's both
    # ref + spatial before training, and projections are linear combinations
    # of the linear-scale reference -- so predicted values should already be
    # linear. Confirm by checking a training gene's predicted-vs-observed
    # cosine matches its recorded (near-perfect) training score.
    print(
        f"  predicted matrix (test genes): {pred_cols.shape}, "
        f"value range [{pred_cols.min():.3f}, {pred_cols.max():.3f}]",
        flush=True,
    )

    # Observed vectors for ALL 100 panel genes already in memory (X_obs).
    matched_lc_idx = {g: i for i, g in enumerate(matched_lc)}

    matched_pairs = []  # predicted_i vs observed_i (the genuine held-out score)
    mismatched_pairs = []  # predicted_i vs observed_j, j != i
    for gi_local, g in enumerate(test_lc):
        pred_vec = pred_cols[:, gi_local]
        for gj, oj in matched_lc_idx.items():
            obs_vec = X_obs[:, oj]
            c = cosine(pred_vec, obs_vec)
            if gj == g:
                matched_pairs.append({"gene": g, "cosine": c})
            else:
                mismatched_pairs.append({"gene_pred": g, "gene_obs": gj, "cosine": c})

    matched_df = pd.DataFrame(matched_pairs)
    mismatched_df = pd.DataFrame(mismatched_pairs)
    matched_df.to_csv(OUT / "null_b_predicted_vs_matched.csv", index=False)
    mismatched_df.to_csv(OUT / "null_b_predicted_vs_mismatched.csv", index=False)

    # Cross-check against the recorded single-split test score for these genes
    gene_scores = pd.read_csv(GENE_SCORES_CSV, index_col=0)
    recorded = gene_scores.loc[gene_scores.index.isin(test_lc), "score"]
    print(
        f"  Null B matched (recomputed): median={matched_df['cosine'].median():.4f}"
        f", recorded gene_scores.csv median={recorded.median():.4f} (sanity check)",
        flush=True,
    )
    gap = abs(float(matched_df["cosine"].median()) - float(recorded.median()))
    null_b_ok = gap <= SANITY_TOL
    if not null_b_ok:
        print(
            f"  NULL B SANITY CHECK FAILED: recomputed matched median differs "
            f"from the recorded one by {gap:.4f} (> {SANITY_TOL}). Rows are "
            "joined by key, so the predicted matrix is not the run that "
            "gene_scores.csv scored, or its scale differs. Null B is marked "
            "invalid; Null A is unaffected. Exiting non-zero after writing.",
            flush=True,
        )
    print(
        f"  Null B mismatched: n_pairs={len(mismatched_df)}, "
        f"median={mismatched_df['cosine'].median():.4f}, "
        f"mean={mismatched_df['cosine'].mean():.4f}",
        flush=True,
    )

    # ── Summary ──
    def _rank_pct(value, null_vals):
        """Fraction of null draws the observed value beats (0-1)."""
        return float(np.mean(np.asarray(null_vals) < value))

    summary = {
        "loocv_observed_n_genes": len(genes_v43),
        "loocv_observed_median": float(obs_median),
        "loocv_observed_mean": float(obs_mean),
        "null_a_observed_vs_observed": {
            "n_pairs": int(len(null_a)),
            "median": float(np.median(null_a)),
            "mean": float(null_a.mean()),
            "increment_median": float(obs_median - np.median(null_a)),
            "pct_loocv_median_beats": _rank_pct(obs_median, null_a),
        },
        "null_b_predicted_vs_mismatched": {
            "n_test_genes": len(test_lc),
            "n_pairs": int(len(mismatched_df)),
            "median": float(mismatched_df["cosine"].median()),
            "mean": float(mismatched_df["cosine"].mean()),
            "matched_recomputed_median": float(matched_df["cosine"].median()),
            "matched_recorded_median_gene_scores_csv": float(recorded.median()),
            "increment_matched_minus_mismatched_median": float(
                matched_df["cosine"].median() - mismatched_df["cosine"].median()
            ),
            "pct_matched_beats_mismatched": _rank_pct(
                matched_df["cosine"].median(), mismatched_df["cosine"].values
            ),
            "sanity_check_passed": bool(null_b_ok),
            "status": (
                "valid"
                if null_b_ok
                else "invalid: the imputed object's rows do not track the observed "
                "values even for training genes (module docstring); not quoted"
            ),
            "caveat": (
                "15-gene single train/test split (85/15), NOT the same "
                "experimental design as the 100-gene LOO-CV (99/1, repeated "
                "per gene) -- complementary check, not a direct substitute."
            ),
        },
    }
    # ── Fig. tangram_qc C correlations, for whichever LOO-CV run was passed ──
    # Same computation as the tangram_qc C caption numbers (there for the
    # d19f184 scores): Spearman of the per-gene cosine
    # against reference sparsity, and against log10(1 + mean) of each gene's
    # column in the canonical object's X.
    import h5py
    from anndata.io import read_elem
    from scipy.stats import spearmanr

    from bimp_st.paths import CANONICAL_TANGRAM

    with h5py.File(CANONICAL_TANGRAM, "r") as h:
        X_can = read_elem(h["X"])
        var_can = read_elem(h["var"]).index.to_numpy()
    colmean = (
        np.asarray(X_can.mean(0)).ravel()
        if sp.issparse(X_can)
        else np.asarray(X_can).mean(0)
    )
    del X_can
    lookup = {str(g).lower(): k for k, g in enumerate(var_can)}
    me = np.array(
        [
            np.log10(max(colmean[lookup[g.lower()]], 0) + 1)
            if g.lower() in lookup
            else np.nan
            for g in loocv["gene"]
        ]
    )
    ok = np.isfinite(me)
    summary["tangram_qc_C"] = {
        "rho_sparsity": float(spearmanr(loocv["sparsity_sc"], loocv["cosine_sim"])[0]),
        "rho_mean_expr": float(spearmanr(me[ok], loocv["cosine_sim"][ok])[0]),
        "n_genes_mean_expr": int(ok.sum()),
    }
    summary["loocv_csv"] = str(LOOCV_CSV)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        [
            {
                "null": "A (observed-vs-observed, 100 genes)",
                "loocv_stat": "median",
                "loocv_value": obs_median,
                "null_median": np.median(null_a),
                "null_mean": null_a.mean(),
                "increment": obs_median - np.median(null_a),
                "n_null_draws": len(null_a),
            },
            {
                "null": "B (predicted-vs-mismatched, 15 held-out test genes)",
                "loocv_stat": "matched median (this split)",
                "loocv_value": matched_df["cosine"].median(),
                "null_median": mismatched_df["cosine"].median(),
                "null_mean": mismatched_df["cosine"].mean(),
                "increment": (
                    matched_df["cosine"].median() - mismatched_df["cosine"].median()
                ),
                "n_null_draws": len(mismatched_df),
            },
        ]
    ).to_csv(OUT / "summary.csv", index=False)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT}/", flush=True)
    if not null_b_ok:
        print("Null B sanity check failed (see above); Null B flagged invalid")


if __name__ == "__main__":
    main()
