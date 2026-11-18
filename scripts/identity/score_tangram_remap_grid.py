"""Score the re-map grid and apply its gates, mechanically.

The re-map grid's selection criteria were fixed before mapping: every metric
and threshold used here. This script computes them and prints the gate
table; it makes no choices of its own.

Scored: only runs made through the production
`tangram_mapping.py train` path, i.e. results/tangram/<stem>_tangram.h5ad with
`<stem> = combined_clustered_grid_<cell>_s<seed>`. The standalone harness
contributes no numbers.

Metrics, all on the raw `tangram_cell_type` pooled to the 7 display groups
(cluster_labels.get_pooled); the cluster vote is not part of the criterion:

  kc_recall / kc_prec / kc_f1  Kenyon placement against the hand-drawn calyx
                               polygons (in_kc_ic | in_kc_nc | in_kc_oc).
  ol_recall                    fraction of cells inside a lamina/medulla/lobula
                               polygon labelled "Optic lobe" (a gate).
                               Optic precision is not a gate: the
                               polygons trace neuropil rather than soma layers,
                               so precision sits near the base rate.
  loo_median                   median cosine similarity of the held-out genes,
                               read from <stem>_gene_scores.csv (is_training ==
                               False). This stands in for a true
                               leave-one-out sweep, which would re-train once per
                               gene per run.
  catchall_share               share of the atlas on the unspecified 662-nucleus
                               `Kenyon cells` reference entry. Report-only as a
                               level; its across-seed SD is Gate 1.
  rho_ref                      Spearman of reference share vs assigned share.
                               Report-only, never scored: it is circular for any
                               arm that anchors to reference composition.

Usage:
    scripts/identity/score_tangram_remap_grid.py              # every grid run on disk
    scripts/identity/score_tangram_remap_grid.py --tags grid_U1_s42 grid_U1_s43
    scripts/identity/score_tangram_remap_grid.py --coherence  # + tie-break coherence

Writes results/reframe/remap_grid/{per_run,per_cell}.csv. Nothing canonical.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.cluster_labels import get_pooled  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

TANGRAM_DIR = REPO / "results/tangram"
REF_PATH = REPO / "data/sn/BIMP_atlas_qw_combined.h5ad"
OUT = REPO / "results/reframe/remap_grid"

KC_FLAGS = ["in_kc_ic", "in_kc_nc", "in_kc_oc"]
OL_FLAGS = ["in_lamina", "in_medulla", "in_lobula"]
CATCHALL = "Kenyon cells"  # the unspecified reference entry, fine label

# Gate thresholds -- fixed before mapping, not tuned here.
SD_F1_MAX = 0.03
SD_CATCHALL_MAX_PP = 2.0
KC_PREC_FLOOR = 0.70
KC_REC_FLOOR = 0.70


OBS_COLS = ["tangram_cell_type"] + KC_FLAGS + OL_FLAGS


def read_obs(path, cols=OBS_COLS, extra=()):
    """Read only the obs columns we score, straight from HDF5.

    anndata's reader materialises all 37 obs columns for 1.3M rows (~4 min per
    object on the shared filesystem); ten columns take seconds. Handles
    categorical (codes + categories), plain arrays, and nullable arrays
    (values + mask). The `in_*` flags are nullable in the nine-region objects,
    and reading them as plain would turn "never traced" into False.
    """
    out = {}
    with h5py.File(str(path), "r") as f:
        g = f["obs"]
        for c in list(cols) + list(extra):
            if c not in g:
                continue
            d = g[c]
            if isinstance(d, h5py.Group):
                keys = set(d.keys())
                if {"codes", "categories"} <= keys:
                    cats = [
                        x.decode() if isinstance(x, bytes) else str(x)
                        for x in d["categories"][:]
                    ]
                    out[c] = pd.Categorical.from_codes(d["codes"][:], cats)
                elif {"values", "mask"} <= keys:
                    vals, mask = d["values"][:], d["mask"][:]
                    out[c] = pd.array(vals, dtype="boolean", copy=True)
                    out[c][mask] = pd.NA
                else:
                    raise ValueError("unhandled obs encoding for %s: %s" % (c, keys))
            else:
                arr = d[:]
                if arr.dtype.kind == "S":
                    arr = np.array([x.decode() for x in arr])
                out[c] = arr
    return pd.DataFrame(out)


def _flag(obs, cols):
    """OR of a set of nullable-boolean region flags. NA counts as False here
    because these six regions carry no nulls (the nulls live in `al` and
    `central_complex`, which this criterion does not use)."""
    m = np.zeros(len(obs), dtype=bool)
    for c in cols:
        if c in obs:
            m |= (obs[c] == True).to_numpy()  # noqa: E712
    return m


def reference_shares():
    """Fine-label shares of the full snRNA reference, for the report-only rho."""
    if not REF_PATH.exists():
        return None
    a = ad.read_h5ad(str(REF_PATH), backed="r")
    vc = a.obs["cell_type"].astype(str).value_counts(normalize=True)
    a.file.close()
    return vc


def score_one(stem, ref_share=None, coherence=False):
    """Every metric for one production run, from its own outputs.

    Cached per run: the grid is scored repeatedly as cells land, and each read
    pulls the obs of a 1.3M-cell object off a shared filesystem. The cache is
    keyed on the h5ad's mtime, so a re-run under the same tag invalidates it.
    """
    h5 = TANGRAM_DIR / f"{stem}_tangram.h5ad"
    cache = OUT / "cache" / f"{stem}.json"
    mtime = h5.stat().st_mtime
    if cache.exists():
        c = json.load(open(cache))
        if c.get("_mtime") == mtime and (not coherence or "coh_ratio" in c):
            return {k: v for k, v in c.items() if k != "_mtime"}
    obs = read_obs(h5, extra=("slice_name",) if coherence else ())
    fine = obs["tangram_cell_type"].astype(str).to_numpy()
    pooled = pd.Series(fine).map(get_pooled).fillna(pd.Series(fine)).to_numpy()

    kcm = _flag(obs, KC_FLAGS)
    olm = _flag(obs, OL_FLAGS)
    is_kc = pooled == "Kenyon cells"
    is_ol = pooled == "Optic lobe"

    rec = float((is_kc & kcm).sum() / max(kcm.sum(), 1))
    pre = float((is_kc & kcm).sum() / max(is_kc.sum(), 1))
    f1 = 2 * rec * pre / (rec + pre) if (rec + pre) else 0.0
    ol_rec = float((is_ol & olm).sum() / max(olm.sum(), 1))
    ol_pre = float((is_ol & olm).sum() / max(is_ol.sum(), 1))

    row = {
        "stem": stem,
        "n_cells": int(len(fine)),
        "kc_recall": rec,
        "kc_prec": pre,
        "kc_f1": f1,
        "ol_recall": ol_rec,
        "ol_prec": ol_pre,
        "catchall_share": float((fine == CATCHALL).mean()),
        "kc_share": float(is_kc.mean()),
        "ol_share": float(is_ol.mean()),
    }

    # Gate 2: held-out-gene cosine, written by train() itself.
    sc_path = TANGRAM_DIR / f"{stem}_gene_scores.csv"
    if sc_path.exists():
        df = pd.read_csv(sc_path, index_col=0)
        test = df[~df["is_training"].astype(bool)]
        row["loo_median"] = float(test["score"].median())
        row["loo_n"] = int(len(test))
        row["train_median"] = float(
            df[df["is_training"].astype(bool)]["score"].median()
        )
    else:
        row["loo_median"] = np.nan
        row["loo_n"] = 0
        row["train_median"] = np.nan

    # Report-only: does assignment track reference abundance at all?
    if ref_share is not None:
        sh = pd.Series(fine).value_counts(normalize=True)
        common = [c for c in ref_share.index if c in sh.index or True]
        x = np.array([ref_share.get(c, 0.0) for c in common])
        y = np.array([sh.get(c, 0.0) for c in common])
        row["rho_ref"] = float(spearmanr(x, y).statistic)

    if coherence:
        a = ad.read_h5ad(str(h5), backed="r")
        row.update(spatial_coherence(a, pooled, obs))
        a.file.close()

    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump({**row, "_mtime": mtime}, open(cache, "w"), indent=2)
    return row


def spatial_coherence(a, pooled, obs=None, k=6, n_perm=5, per_section=20000, seed=0):
    """Neighbours sharing a label vs a within-section permutation null.

    The null holds positions and per-section composition fixed, so a section
    that is 90% one label cannot score as coherent by composition alone.
    Coordinates are the aligned frame. That is fine for a within-section
    neighbour statistic because the alignment is affine per section, but it
    would not be safe for absolute distances.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    if obs is None:
        obs = a.obs
    xy_all = np.asarray(a.obsm["spatial"])
    sec = obs["slice_name"].astype(str).to_numpy()
    obs_frac, null_frac, w = [], [], []
    for s in pd.unique(sec):
        idx = np.flatnonzero(sec == s)
        if len(idx) < k + 1:
            continue
        if len(idx) > per_section:
            idx = rng.choice(idx, per_section, replace=False)
        xy = xy_all[idx]
        lab = pooled[idx]
        _, nn = cKDTree(xy).query(xy, k=k + 1)
        nn = nn[:, 1:]
        obs_frac.append(float((lab[nn] == lab[:, None]).mean()))
        nulls = []
        for _ in range(n_perm):
            p = rng.permutation(len(lab))
            lp = lab[p]
            nulls.append(float((lp[nn] == lp[:, None]).mean()))
        null_frac.append(float(np.mean(nulls)))
        w.append(len(idx))
    w = np.array(w, dtype=float)
    o = float(np.average(obs_frac, weights=w))
    n = float(np.average(null_frac, weights=w))
    return {"coh_obs": o, "coh_null": n, "coh_ratio": o / n if n else np.nan}


def score_shipped(ref_share=None, coherence=False):
    """The comparator: raw labels of the canonical object. Cached like the rest."""
    cache = OUT / "cache" / "SHIPPED.json"
    mtime = CANONICAL_TANGRAM.stat().st_mtime
    if cache.exists():
        c = json.load(open(cache))
        if c.get("_mtime") == mtime and (not coherence or "coh_ratio" in c):
            return {k: v for k, v in c.items() if k != "_mtime"}
    obs = read_obs(CANONICAL_TANGRAM, extra=("slice_name",) if coherence else ())
    fine = obs["tangram_cell_type"].astype(str).to_numpy()
    pooled = pd.Series(fine).map(get_pooled).fillna(pd.Series(fine)).to_numpy()
    kcm = _flag(obs, KC_FLAGS)
    olm = _flag(obs, OL_FLAGS)
    is_kc = pooled == "Kenyon cells"
    is_ol = pooled == "Optic lobe"
    rec = float((is_kc & kcm).sum() / max(kcm.sum(), 1))
    pre = float((is_kc & kcm).sum() / max(is_kc.sum(), 1))
    row = {
        "stem": "SHIPPED",
        "n_cells": int(len(fine)),
        "kc_recall": rec,
        "kc_prec": pre,
        "kc_f1": 2 * rec * pre / (rec + pre) if (rec + pre) else 0.0,
        "ol_recall": float((is_ol & olm).sum() / max(olm.sum(), 1)),
        "ol_prec": float((is_ol & olm).sum() / max(is_ol.sum(), 1)),
        "catchall_share": float((fine == CATCHALL).mean()),
        "kc_share": float(is_kc.mean()),
        "ol_share": float(is_ol.mean()),
    }
    stem = CANONICAL_TANGRAM.name.replace("_tangram.h5ad", "")
    sc_path = TANGRAM_DIR / f"{stem}_gene_scores.csv"
    if sc_path.exists():
        df = pd.read_csv(sc_path, index_col=0)
        test = df[~df["is_training"].astype(bool)]
        row["loo_median"] = float(test["score"].median())
        row["loo_n"] = int(len(test))
        row["train_median"] = float(
            df[df["is_training"].astype(bool)]["score"].median()
        )
    if ref_share is not None:
        sh = pd.Series(fine).value_counts(normalize=True)
        x = np.array([ref_share.get(c, 0.0) for c in ref_share.index])
        y = np.array([sh.get(c, 0.0) for c in ref_share.index])
        row["rho_ref"] = float(spearmanr(x, y).statistic)
    if coherence:
        a = ad.read_h5ad(str(CANONICAL_TANGRAM), backed="r")
        row.update(spatial_coherence(a, pooled, obs))
        a.file.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump({**row, "_mtime": mtime}, open(cache, "w"), indent=2)
    return row


def check_holdouts(stems):
    """Every candidate at a given seed must hold out the same genes.

    `train()` draws the holdout from a sorted gene intersection; unsorted set
    iteration would make it depend on PYTHONHASHSEED rather than `--seed`.
    Nothing downstream would otherwise notice a mismatch.
    """
    by_seed = {}
    for s in stems:
        p = TANGRAM_DIR / f"{s}_gene_split.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        by_seed.setdefault(int(d["seed"]), {})[s] = frozenset(d["test_genes"])
    bad = []
    for seed, runs in sorted(by_seed.items()):
        sets = set(runs.values())
        if len(sets) > 1:
            bad.append(
                "seed %d: %d distinct held-out sets across %d runs"
                % (seed, len(sets), len(runs))
            )
    if bad:
        raise SystemExit(
            "HOLDOUT MISMATCH -- candidates are not on comparable genes, so "
            "Gate 2 is meaningless:\n  " + "\n  ".join(bad)
        )
    print(
        "holdout guard OK: one held-out gene set per seed (%s)"
        % ", ".join("s%d n=%d" % (k, len(v)) for k, v in sorted(by_seed.items()))
    )


def apply_gates(per_cell, baseline, tau):
    """The fixed decision rule, in order, with no discretion.

    `baseline` is the U0 row, not the canonical run: the canonical run holds
    out a different gene set and cannot be rescored on this one (those genes
    were in its training set), so a canonical-vs-candidate cosine comparison
    is not like-for-like.
    """
    lines, survivors = [], []
    for cell, r in per_cell.iterrows():
        why = []
        if r["sd_kc_f1"] > SD_F1_MAX:
            why.append("G1 seed SD of F1 %.3f > %.2f" % (r["sd_kc_f1"], SD_F1_MAX))
        if 100 * r["sd_catchall_share"] > SD_CATCHALL_MAX_PP:
            why.append(
                "G1 seed SD of catch-all %.2f pp > %.0f pp"
                % (100 * r["sd_catchall_share"], SD_CATCHALL_MAX_PP)
            )
        floor_loo = baseline["loo_median"] - tau["loo"]
        if r["loo_median"] < floor_loo:
            why.append(
                "G2 held-out cosine %.3f < U0-tau %.3f" % (r["loo_median"], floor_loo)
            )
        if r["kc_prec"] < KC_PREC_FLOOR:
            why.append("G3 KC precision %.3f < %.2f" % (r["kc_prec"], KC_PREC_FLOOR))
        if r["kc_recall"] < KC_REC_FLOOR:
            why.append("G3 KC recall %.3f < %.2f" % (r["kc_recall"], KC_REC_FLOOR))
        floor_ol = baseline["ol_recall"] - tau["olrec"]
        if r["ol_recall"] < floor_ol:
            why.append(
                "G3 optic recall %.3f < U0-tau %.3f" % (r["ol_recall"], floor_ol)
            )
        if why:
            lines.append("  %-6s EXCLUDED  %s" % (cell, "; ".join(why)))
        else:
            lines.append("  %-6s passes all gates" % cell)
            survivors.append(cell)
    return lines, survivors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--coherence", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ref_share = reference_shares()

    if args.tags:
        stems = [
            "combined_clustered_%s" % t if not t.startswith("combined_") else t
            for t in args.tags
        ]
    else:
        stems = sorted(
            Path(p).name.replace("_tangram.h5ad", "")
            for p in glob.glob(
                str(TANGRAM_DIR / "combined_clustered_grid_*_tangram.h5ad")
            )
        )
    if not stems:
        raise SystemExit("no production grid runs on disk yet (%s)" % TANGRAM_DIR)
    print("scoring %d run(s)" % len(stems))
    check_holdouts(stems)

    rows = [score_shipped(ref_share, args.coherence)]
    for s in stems:
        print("  %s" % s)
        rows.append(score_one(s, ref_share, args.coherence))
    per_run = pd.DataFrame(rows).set_index("stem")
    per_run.to_csv(OUT / "per_run.csv")

    # cell/seed decomposition
    grid = per_run.drop(index="SHIPPED").copy()
    cells, seeds = [], []
    for s in grid.index:
        m = re.match(r"combined_clustered_grid_(.+)_s(\d+)$", s)
        cells.append(m.group(1) if m else s)
        seeds.append(int(m.group(2)) if m else -1)
    grid["cell"] = cells
    grid["seed"] = seeds

    num = grid.select_dtypes("number")
    per_cell = num.groupby(grid["cell"]).mean()
    sd = num.groupby(grid["cell"]).std(ddof=1)
    per_cell["sd_kc_f1"] = sd["kc_f1"]
    per_cell["sd_catchall_share"] = sd["catchall_share"]
    per_cell["n_seeds"] = grid.groupby("cell").size()
    per_cell.to_csv(OUT / "per_cell.csv")

    shipped = per_run.loc["SHIPPED"]

    # tau_m = 2 x the pooled across-seed SD over all cluster-mode candidates
    def pooled_sd(col):
        v = sd[col].dropna()
        return float(np.sqrt((v**2).mean())) if len(v) else 0.0

    tau = {
        "f1": 2 * pooled_sd("kc_f1"),
        "loo": 2 * pooled_sd("loo_median"),
        "olrec": 2 * pooled_sd("ol_recall"),
    }

    print("\n=== per-cell means (seeds %s) ===" % sorted(set(seeds)))
    cols = [
        "kc_recall",
        "kc_prec",
        "kc_f1",
        "sd_kc_f1",
        "ol_recall",
        "loo_median",
        "catchall_share",
        "rho_ref",
    ]
    cols = [c for c in cols if c in per_cell.columns]
    print(per_cell[cols].round(4).to_string())
    print(
        "\nSHIPPED: "
        + "  ".join(
            "%s=%.4f" % (c, shipped[c])
            for c in cols
            if c in shipped.index and pd.notna(shipped[c])
        )
    )
    print(
        "\ntau (2 x pooled across-seed SD): "
        + "  ".join("%s=%.4f" % (k, v) for k, v in tau.items())
    )

    # Gate comparator: U0 (density off, seeded, current code), not canonical,
    # which holds out a different gene set and cannot be rescored on this one.
    if "U0" not in per_cell.index:
        raise SystemExit(
            "U0 is the gate comparator and is not present; "
            "score it before applying the gates."
        )
    baseline = per_cell.loc["U0"]
    print(
        "\nGate comparator U0: held-out cosine %.4f, optic recall %.4f  "
        "(shipped is report-only: %.4f / %.4f, different held-out genes)"
        % (
            baseline["loo_median"],
            baseline["ol_recall"],
            shipped["loo_median"],
            shipped["ol_recall"],
        )
    )

    print("\n=== gates ===")
    lines, survivors = apply_gates(per_cell, baseline, tau)
    print("\n".join(lines))

    if not survivors:
        print(
            "\nNo candidate passes. The fallback fixed before mapping applies "
            "(max F1 among Gate 1+2 survivors, naming the missed floor)."
        )
    else:
        s = per_cell.loc[survivors].sort_values("kc_f1", ascending=False)
        best = s.index[0]
        tied = [
            c
            for c in s.index
            if abs(s.loc[c, "kc_f1"] - s.loc[best, "kc_f1"]) < tau["f1"]
        ]
        print("\nprimary score (mean Kenyon F1), survivors:")
        print(s[["kc_f1", "ol_recall", "loo_median"]].round(4).to_string())
        if len(tied) > 1:
            print(
                "\ntie within tau_F1=%.4f: %s -> break on optic recall, "
                "then LOO, then coherence, then smaller lambda_d"
                % (tau["f1"], ", ".join(tied))
            )
            t = s.loc[tied].sort_values(["ol_recall", "loo_median"], ascending=False)
            best = t.index[0]
        print(
            "\nWINNER: %s   (F1 %.4f vs shipped %.4f, delta %+.4f, tau_F1 %.4f)"
            % (
                best,
                s.loc[best, "kc_f1"],
                shipped["kc_f1"],
                s.loc[best, "kc_f1"] - shipped["kc_f1"],
                tau["f1"],
            )
        )

    json.dump(
        {
            "tau": tau,
            "survivors": survivors,
            "shipped": {
                k: (float(v) if pd.notna(v) else None)
                for k, v in shipped.items()
                if isinstance(v, (int, float, np.floating))
            },
        },
        open(OUT / "gates.json", "w"),
        indent=2,
    )
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
