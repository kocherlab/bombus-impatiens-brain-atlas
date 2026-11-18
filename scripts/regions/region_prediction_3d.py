"""Does the reconstructed 3D volume improve region prediction?

region_prediction.py describes each cell by the cell-type
frequencies of its k nearest neighbours within its own section (2D). This
script finds the same neighbourhood in 3D instead and asks whether region
prediction improves.

Everything except the neighbour search is held fixed: the loader,
`neighbourhood_composition` and `evaluate` are imported from region_prediction.
As there, the neighbour tree holds every cell of the brain and only the cells
inside exactly one region are scored.

Correctness gate: the 2D arm must reproduce the registered baseline
(vocabulary "Tangram canonical-17", features "neighbourhood k=20", cv_scheme
"leave-section" seed-mean, from
results/reframe/region_prediction_v3/scheme_comparison.csv) to 1e-4, with
exactly 458,194 scored cells; otherwise the script stops before the 3D arm.

Z scale: z_level is section index x 10 um nominal thickness (microns), while
obsm['spatial'] is the PASTE3-aligned frame in Xenium pixels at 0.2125 um/px
(analysis_workflow.py divides microns by PIXEL_SCALE before registering). The
axes are matched by the unit conversion s = 1 / 0.2125 = 4.705882, not a fit;
the micron -> aligned-frame distance ratio is 4.705882 on all 12 crops checked
(per-crop CV ~1e-15).

One tree per brain: queen and worker z frames are independent (0-290 and
0-340) and not co-registered.

Caveat: under leave-section CV the 3D features cross the fold boundary (a
training cell in section S-1 draws neighbours from held-out section S), so any
3D gain is partly leakage. The `blocked` scheme (contiguous 5-section blocks)
is reported as a bound; leave-section is the headline to match the baseline.

The permuted-label and geometry-only controls, k sweep and Leiden vocabulary
remain in the 2D script.

Outputs to results/reframe/region_prediction_3d_v2/; region_prediction_v3/ is
only read for the baseline check.
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The 2D arm is imported, not reimplemented, so the two arms cannot drift.
from regions.region_prediction import (  # noqa: E402
    OUT as OUT_2D,
)
from regions.region_prediction import (  # noqa: E402
    RESULTS_ROOT,
    evaluate,
    neighbourhood_composition,
)
from regions.region_prediction import load as rp_load  # noqa: E402

from bimp_st.nine_region_common import STATISTICAL_REGIONS  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
# RESULTS_ROOT is results/reframe unless BIMP_REGION_PREDICTION_ROOT overrides it
BASELINE_CSV = OUT_2D / "scheme_comparison.csv"
OUT = RESULTS_ROOT / "region_prediction_3d_v2"

REGION_FLAGS = [f"in_{r}" for r in STATISTICAL_REGIONS]
K_MAIN = 20
SEEDS = [0, 1, 2]
SCHEMES = ["leave-section", "blocked"]

# 1 / 0.2125 um-per-pixel. Not a fit -- see the module docstring.
Z_SCALE = 4.705882
EXPECT_N_CELLS = 458_194  # scored cells (n_train + n_test in every row)


def neighbourhood_composition_3d(xyz, codes, n_cat, brains, k, score=None):
    """kNN cell-type frequency per cell, neighbours drawn in 3D within each brain.

    Structurally identical to neighbourhood_composition() in
    region_prediction (n_neighbors=k+1 with self dropped, same count
    loop and row normalisation, tree on every cell, only ``score`` cells
    returned), except the tree is built on (x, y, z * Z_SCALE) over a whole
    brain rather than on (x, y) within one section.

    Also returns diagnostics for the neighbourhoods used: the
    fraction of each cell's k neighbours drawn from a different z_level, and
    the mean |dz| in z_levels (i.e. in units of 10 um sections).
    """
    from sklearn.neighbors import NearestNeighbors

    score = np.ones(len(codes), bool) if score is None else np.asarray(score, bool)
    X = np.zeros((len(codes), n_cat), dtype=np.float32)
    cross_z = np.full(len(codes), np.nan, dtype=np.float64)
    mean_dz = np.full(len(codes), np.nan, dtype=np.float64)
    for b in np.unique(brains[score]):
        idx = np.flatnonzero(brains == b)
        q = idx[score[idx]]
        if len(idx) <= k or not len(q):
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xyz[idx])
        nbr = nn.kneighbors(xyz[q], return_distance=False)[:, 1:]
        c = codes[idx][nbr]
        for j in range(n_cat):
            X[q, j] = (c == j).sum(1)
        # z back out of the scaled column, in z_level units (10 um sections)
        zb = xyz[idx, 2] / Z_SCALE / 10.0
        zq = xyz[q, 2] / Z_SCALE / 10.0
        dz = np.abs(zb[nbr] - zq[:, None])
        cross_z[q] = (dz > 0).mean(1)
        mean_dz[q] = dz.mean(1)
    X = X / np.maximum(X.sum(1, keepdims=True), 1)
    return X[score], cross_z[score], mean_dz[score]


def load() -> dict:
    """The 2D script's loader: same objects, labels, regions and scored set."""
    d = rp_load()
    counts = pd.Series(d["y"]).value_counts()
    d["majority"] = float(counts.max() / counts.sum())
    print(
        f"  MAJORITY-CLASS baseline = {d['majority']:.3f} "
        f"(uniform 1/{len(REGION_FLAGS)} = {1 / len(REGION_FLAGS):.3f})",
        flush=True,
    )
    return d


def registered_baseline():
    if not BASELINE_CSV.exists():
        return None
    b = pd.read_csv(BASELINE_CSV)
    m = b[
        (b.vocabulary == "Tangram canonical-17")
        & (b.features == f"neighbourhood k={K_MAIN}")
        & (b.cv_scheme == "leave-section")
    ]
    if m.empty:
        return None
    print(
        "  registered per-seed: "
        + ", ".join(
            f"seed {int(r.seed)} {r.accuracy:.6f}"
            for r in m.sort_values("seed").itertuples()
        ),
        flush=True,
    )
    return float(m.accuracy.mean())


def run_arm(name, X, d, rows):
    for scheme in SCHEMES:
        for seed in SEEDS:
            m, _ = evaluate(X, d["y"], d["groups"], d["brain"], scheme, seed)
            m.update(
                vocabulary="Tangram canonical-17",
                features=name,
                cv_scheme=scheme,
                seed=seed,
                k=K_MAIN,
            )
            rows.append(m)
        got = [r for r in rows if r["features"] == name and r["cv_scheme"] == scheme]
        print(
            f"  {name:26s} {scheme:14s} "
            f"acc {np.mean([g['accuracy'] for g in got]):.4f} "
            f"+/- {np.std([g['accuracy'] for g in got]):.4f}  "
            f"balacc {np.mean([g['balanced_accuracy'] for g in got]):.3f}  "
            f"macroF1 {np.mean([g['macro_f1'] for g in got]):.3f}",
            flush=True,
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    if len(d["y"]) != EXPECT_N_CELLS:
        raise SystemExit(
            f"GATE FAILED: scored {len(d['y']):,} cells, expected "
            f"{EXPECT_N_CELLS:,} (n_train + n_test of every registered row). "
            "refusing to build a 3D comparison on it."
        )
    print(
        f"  GATE 1 OK: {len(d['y']):,} cells matches the registered n_train + n_test",
        flush=True,
    )

    # Categories over all cells, as in the 2D script: neighbours can be any cell.
    cat = pd.Categorical(d["obs"]["cell_type_canonical"])
    codes, ncat = np.asarray(cat.codes), len(cat.categories)
    keep = d["keep"]
    print(f"\n=== Tangram canonical-17 ({ncat} labels), k={K_MAIN} ===", flush=True)

    rows = []

    # --- 2D arm: the registered baseline, recomputed through the imported
    # 2D function, on the imported evaluate().
    print("\n[2D] neighbourhood within section", flush=True)
    X2 = neighbourhood_composition(
        d["xy"], codes, ncat, d["sections"], K_MAIN, keep
    )
    run_arm("neighbourhood 2D k=20", X2, d, rows)

    acc2 = [
        r["accuracy"]
        for r in rows
        if r["features"] == "neighbourhood 2D k=20"
        and r["cv_scheme"] == "leave-section"
    ]
    print(
        "\n  reproduced per-seed: "
        + ", ".join(f"seed {s} {a:.6f}" for s, a in zip(SEEDS, acc2, strict=False)),
        flush=True,
    )
    reg = registered_baseline()
    mean2 = float(np.mean(acc2))
    if reg is None or abs(mean2 - reg) > 1e-4:
        raise SystemExit(
            f"GATE FAILED: 2D leave-section reproduced {mean2:.6f}, registered "
            f"{reg} ({BASELINE_CSV}). Run region_prediction.py first; not "
            "proceeding to the 3D arm on an unverified harness."
        )
    print(
        f"  GATE 2 OK: 2D leave-section = {mean2:.6f} reproduces the "
        f"registered {reg:.6f}",
        flush=True,
    )

    # --- 3D arm: identical in every respect but the neighbour search.
    print("\n[3D] neighbourhood across the volume, one tree per brain", flush=True)
    z = d["obs"]["z_level"].to_numpy(dtype=float) * Z_SCALE
    xyz = np.column_stack([d["xy"], z])
    X3, cross_z, mean_dz = neighbourhood_composition_3d(
        xyz, codes, ncat, d["brain_all"], K_MAIN, keep
    )

    diag_rows = []
    for b in ["queen", "worker"]:
        i = d["brain"] == b
        if not i.any():
            continue
        diag_rows.append(
            dict(
                brain=b,
                n_cells=int(i.sum()),
                cross_z_fraction=float(np.nanmean(cross_z[i])),
                mean_abs_dz_zlevels=float(np.nanmean(mean_dz[i])),
            )
        )
    diag_rows.append(
        dict(
            brain="both",
            n_cells=int(len(cross_z)),
            cross_z_fraction=float(np.nanmean(cross_z)),
            mean_abs_dz_zlevels=float(np.nanmean(mean_dz)),
        )
    )
    diag = pd.DataFrame(diag_rows)
    diag["z_scale"] = Z_SCALE
    diag["k"] = K_MAIN
    for r in diag.itertuples():
        print(
            f"  cross-z fraction ({r.brain}, n={r.n_cells:,}): "
            f"{r.cross_z_fraction:.3f}   mean |dz| "
            f"{r.mean_abs_dz_zlevels:.2f} z-levels",
            flush=True,
        )
    print(
        "  (reference: the geometry stage's sanity check, run on ALL cells "
        "of 6 adjacent sections, gave 0.328 queen / 0.362 worker. A value in "
        "the 20-60% band the scale was checked against is expected; a value "
        "near 0, or above ~0.7, would mean the scale was not applied or was "
        "applied to the wrong column.)",
        flush=True,
    )

    run_arm("neighbourhood 3D k=20", X3, d, rows)

    df = pd.DataFrame(rows)
    df["majority_baseline"] = d["majority"]
    df["z_scale"] = np.where(df.features.str.contains("3D"), Z_SCALE, np.nan)
    df["n_cells"] = len(d["y"])
    df.to_csv(OUT / "scheme_comparison_3d.csv", index=False)
    diag.to_csv(OUT / "cross_z_diagnostics.csv", index=False)

    summ = (
        df.groupby(["features", "cv_scheme"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    summ["majority_baseline"] = d["majority"]
    summ.to_csv(OUT / "summary_2d_vs_3d.csv", index=False)

    print("\n=== SUMMARY ===", flush=True)
    print(summ.to_string(index=False), flush=True)
    ls = summ[summ.cv_scheme == "leave-section"].set_index("features")
    a2 = float(ls.loc["neighbourhood 2D k=20", "accuracy_mean"])
    a3 = float(ls.loc["neighbourhood 3D k=20", "accuracy_mean"])
    print(f"\nleave-section: 2D {a2:.4f} -> 3D {a3:.4f} ({a3 - a2:+.4f})", flush=True)
    print(
        "NOTE: under leave-section the 3D features read cell-type codes out "
        "of held-out sections; the 2D features cannot. Read any 3D gain "
        "against that, and against the blocked row.",
        flush=True,
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
