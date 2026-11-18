"""Does local cell-type composition identify anatomical region?

Representation: each cell is described by the cell-type frequencies of its k
nearest spatial neighbours (the cellular-neighbourhood representation of
Goltsev et al. 2018 and Schurch et al. 2020, who use k = 10); k is swept.

Validation. Serial sections are 10 um apart, so holding out one section while
training on its neighbours leaks information; blocks must exceed the
autocorrelation range (Roberts et al. 2017). Four schemes are reported:
       random          cells shuffled (deliberately leaky; quantifies inflation)
       leave-section   the main scheme
       blocked         contiguous blocks of sections held out together
       cross-brain     train on queen, test on worker, and the reverse
The cross-brain number tests generalisation.

Baselines. Regions are imbalanced (KC-NC is ~28x the lobula), so the null is
the majority-class rate, recomputed at run time. Two controls:
       geometry-only   neighbour distances, cell and nucleus area and
                       transcript and gene counts, no cell types; composition
                       must beat this. nucleus_count is left out: it is empty
                       for every worker cell, so after imputation it would
                       identify the brain.
       permuted        cell-type labels shuffled over all cells of each
                       section, which destroys composition while preserving
                       density and class balance.

Neighbourhoods. The kNN tree for a section is built on ALL of its cells, and
only cells inside exactly one region are scored, so a border cell's neighbours
are its true neighbours, in a region or not. Section order (for the blocked
scheme) is the PASTE3 stack order, z_level, cut per brain, so a block never
spans the two brains.

Metrics: accuracy, balanced accuracy and macro-F1.

Vocabulary: both the canonical-17 Tangram vocabulary and the Leiden expression
clusters (what the panel measures directly) are run.

Determinism: every fit runs on one BLAS/OpenMP thread (threadpoolctl, inside
the script, so the shell environment cannot change it). lbfgs stops at a
tolerance, and a multi-threaded BLAS sums in a thread-count-dependent order, so
reruns under different thread counts moved CV scores by up to 0.004.

Outputs to results/reframe/region_prediction_v3/ (root overridable with
BIMP_REGION_PREDICTION_ROOT, e.g. for a scratch reproducibility run):
    scheme_comparison.csv     every (vocabulary, feature set, CV scheme) cell
    per_region_f1.csv
    (the figure is scripts/regions/si_region_prediction.py -> figS_region_prediction)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

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

REPO = Path(__file__).resolve().parents[2]
H5 = SPATIAL_COMBINED
TANGRAM_LABELS = CANONICAL_TANGRAM
# Root of the region-prediction outputs; a reproducibility rerun points it at
# scratch so the shared results/ is not overwritten.
RESULTS_ROOT = Path(
    os.environ.get("BIMP_REGION_PREDICTION_ROOT", REPO / "results" / "reframe")
)
OUT = RESULTS_ROOT / "region_prediction_v3"

# STATISTICAL_REGIONS = the six original hand-drawn regions + antennal lobe.
# The central-complex regions are excluded because such rare
# classes degrade the other estimates in a multinomial fit; see
# nine_region_common.DESCRIPTIVE_ONLY_REGIONS.
REGION_FLAGS = [f"in_{r}" for r in STATISTICAL_REGIONS]
NICE = {f"in_{r}": display(r) for r in STATISTICAL_REGIONS}
K_SWEEP = [10, 20, 40]  # 10 = the Schurch/Goltsev default
K_MAIN = 20
BLOCK_SIZE = 5  # sections per held-out block
SEEDS = [0, 1, 2]
# nucleus_count is not a geometry feature: it is empty for every worker cell, so
# a median-imputed column would act as a brain indicator.
MORPH = ["cell_area", "nucleus_area", "transcript_counts", "n_genes"]


def neighbourhood_composition(xy, codes, n_cat, sections, k, score=None):
    """kNN cell-type frequency per cell, neighbours drawn within each section.

    The tree is built on every cell passed in. With ``score`` (a boolean mask),
    only those cells are scored and the rows returned are ``X[score]``; their
    neighbours are still drawn from all cells of the section.
    """
    from sklearn.neighbors import NearestNeighbors

    score = np.ones(len(codes), bool) if score is None else np.asarray(score, bool)
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


def geometry_features(xy, obs, sections, k, score=None):
    """Local packing + morphology, with no cell-type information.

    As in :func:`neighbourhood_composition`, the tree uses every cell and only
    ``score`` cells are returned.
    """
    from sklearn.neighbors import NearestNeighbors

    cols = [c for c in MORPH if c in obs.columns]
    M = obs[cols].to_numpy(dtype=np.float32)
    # Some morphology columns are undefined for a minority of cells (e.g. an
    # anucleate segment has no nucleus_area), and a NaN propagates through the
    # neighbour mean. Median-impute per column and report how much was imputed.
    for j, c in enumerate(cols):
        bad = ~np.isfinite(M[:, j])
        if bad.any():
            med = float(np.nanmedian(M[np.isfinite(M[:, j]), j]))
            print(
                f"    geometry: {c} missing in {bad.sum():,} cells "
                f"({100 * bad.mean():.2f}%) -- imputed with median {med:.3g}",
                flush=True,
            )
            M[bad, j] = med
    score = np.ones(len(obs), bool) if score is None else np.asarray(score, bool)
    out = np.zeros((len(obs), 2 + len(cols)), dtype=np.float32)
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


SECTION_ORDER: dict[str, float] = {}  # slice_name -> z_level, filled by load()


def section_blocks(groups, brain):
    """Block id per cell: BLOCK_SIZE consecutive sections of one brain.

    Sections are ordered by z_level (the PASTE3 stack, SECTION_ORDER) within
    each brain, so a block never spans the two brains.
    """
    df = pd.DataFrame({"s": groups, "b": brain}).drop_duplicates("s")
    bmap, nxt = {}, 0
    for b in sorted(df["b"].unique()):
        secs = sorted(df.loc[df["b"] == b, "s"], key=lambda x: SECTION_ORDER[x])
        for i, sname in enumerate(secs):
            bmap[sname] = nxt + i // BLOCK_SIZE
        nxt += (len(secs) + BLOCK_SIZE - 1) // BLOCK_SIZE
    return np.array([bmap[g] for g in groups])


def evaluate(X, y, groups, brain, scheme, seed, scale=False):
    """Fit and score under one CV scheme. Returns (metrics dict, per-region df).

    scale=True fits StandardScaler on the training fold only and applies it to both
    folds. Composition features are unit-scale fractions and converge unscaled;
    the geometry-only control does not converge without scaling.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        f1_score,
    )
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if scheme == "random":
        tr, te = train_test_split(
            np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y
        )
    elif scheme == "leave-section":
        tr, te = next(
            GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups)
        )
    elif scheme == "blocked":
        # contiguous runs of sections held out together, so a test section's
        # immediate serial neighbours are also unseen
        gb = section_blocks(groups, brain)
        tr, te = next(
            GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, gb)
        )
    elif scheme.startswith("cross-brain"):
        src = "queen" if scheme.endswith("q2w") else "worker"
        tr = np.flatnonzero(brain == src)
        te = np.flatnonzero(brain != src)
    else:
        raise ValueError(scheme)

    if scale:
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(X[tr])
        X = sc.transform(X)
    from threadpoolctl import threadpool_limits

    # One BLAS thread: the lbfgs path depends on summation order, which a
    # multi-threaded BLAS changes with the thread count (see module docstring).
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    with threadpool_limits(limits=1):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
    rep = pd.DataFrame(
        classification_report(y[te], pred, output_dict=True, zero_division=0)
    ).T
    per = rep.loc[[i for i in rep.index if i in set(y)]].reset_index()
    per.columns = ["region"] + list(per.columns[1:])
    return dict(
        accuracy=float((pred == y[te]).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y[te], pred)),
        macro_f1=float(f1_score(y[te], pred, average="macro")),
        n_train=len(tr),
        n_test=len(te),
    ), per


def load() -> dict:
    """All atlas cells (the neighbour pool) and the scored subset.

    Returns a dict: ``obs``/``xy``/``sections``/``brain_all`` over every cell;
    ``keep`` (bool, cells inside exactly one of the seven regions, the scored
    set); and ``sub``/``y``/``groups``/``brain``/``xy_keep`` over ``keep``.
    Also fills SECTION_ORDER from z_level and checks it against the lexical
    slice order the blocked scheme used before.
    """
    print(f"loading {H5.name} ...", flush=True)
    a = ad.read_h5ad(H5)
    # H5's own tangram_cell_type is not the canonical labelling. The
    # vocabularies are built by name, so cell-type
    # columns must be keyed by name; labels are taken from the canonical
    # object instead.
    from bimp_st.paths import keyed_obs_column

    # joined on (slice_name, cell_id), never by position; raises on any unmatched
    # or repeated cell rather than falling back to the in-file column
    a.obs["tangram_cell_type"] = keyed_obs_column(
        a, "tangram_cell_type", TANGRAM_LABELS
    ).astype(str)
    add_canonical(a, src="tangram_cell_type")

    # combined_clustered.h5ad carries six in_<region> flags from labeling.py,
    # computed before self-intersecting polygons were repaired (KC-NC 98,831 vs
    # 99,270 in coverage.json). All seven flags are re-attached from the
    # nine-region crop sidecars so there is one definition of each region.
    # One region per cell through the shared helper: a cell is scored only if
    # it lies inside exactly one of the seven regions. A cell on a section
    # with no tracing for one of them would be None (untraced) and is not
    # scored; on the current sidecars none of the seven flags is NA.
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    region = region_of(a.obs, STATISTICAL_REGIONS, overlap="drop")
    n_untraced = int(pd.isna(region).sum())
    print(f"  {n_untraced:,} cells untraced for one of the seven regions", flush=True)
    keep = np.isin(region.astype(str), list(STATISTICAL_REGIONS))

    obs = a.obs.reset_index(drop=True)
    xy_all = np.asarray(a.obsm["spatial"], float)
    sections = obs["slice_name"].astype(str).to_numpy()
    brain_all = obs["brain"].astype(str).to_numpy()

    # PASTE3 stack order. One z_level per section; within each brain it must
    # rise with the section, and it matches the lexical slice order.
    z = obs.groupby(sections)["z_level"].agg(["min", "max"])
    assert (z["min"] == z["max"]).all(), "a section spans several z_levels"
    SECTION_ORDER.clear()
    SECTION_ORDER.update(z["min"].astype(float).to_dict())
    for b in np.unique(brain_all):
        secs = list(pd.unique(sections[brain_all == b]))
        by_z = sorted(secs, key=lambda x: SECTION_ORDER[x])
        assert len({SECTION_ORDER[x] for x in secs}) == len(secs), b
        if by_z != sorted(secs):
            print(f"  {b}: z_level order differs from lexical order", flush=True)

    sub = obs.loc[keep].reset_index(drop=True)
    y = np.array([display(r) for r in region[keep]])
    groups = sections[keep]
    brain = brain_all[keep]
    print(
        f"  {keep.sum():,} of {len(obs):,} cells in exactly one region "
        "(scored; neighbours drawn from all cells)",
        flush=True,
    )
    return dict(
        obs=obs,
        xy=xy_all,
        sections=sections,
        brain_all=brain_all,
        keep=keep,
        sub=sub,
        y=y,
        groups=groups,
        brain=brain,
        xy_keep=xy_all[keep],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    obs, xy_all, sections, keep = d["obs"], d["xy"], d["sections"], d["keep"]
    y, groups, brain = d["y"], d["groups"], d["brain"]

    counts = pd.Series(y).value_counts()
    majority = counts.max() / counts.sum()
    # Uniform null is 1/len(REGION_FLAGS); the majority rate is computed from
    # the label distribution above.
    print(
        f"  MAJORITY-CLASS baseline = {majority:.3f} "
        f"(uniform 1/{len(REGION_FLAGS)} = {1 / len(REGION_FLAGS):.3f} -- "
        "the wrong null for imbalanced regions)",
        flush=True,
    )
    print("  region sizes:", counts.to_dict(), flush=True)

    # Categories over ALL cells: a scored cell's neighbours can be any cell.
    vocabs = {
        "Tangram canonical-17": pd.Categorical(obs["cell_type_canonical"]),
        "Leiden expression": pd.Categorical(obs["leiden"].astype(str)),
    }

    rows, per_rows = [], []
    rng = np.random.default_rng(0)

    for vname, cat in vocabs.items():
        codes, ncat = np.asarray(cat.codes), len(cat.categories)
        print(f"\n=== {vname} ({ncat} labels) ===", flush=True)
        for k in K_SWEEP:
            Xn = neighbourhood_composition(xy_all, codes, ncat, sections, k, keep)
            feats = {f"neighbourhood k={k}": Xn}
            if k == K_MAIN:
                onehot = np.eye(ncat, dtype=np.float32)[codes[keep]]
                feats["own cell type"] = onehot
                perm = codes.copy()
                for s in np.unique(sections):  # shuffle over all section cells
                    i = np.flatnonzero(sections == s)
                    perm[i] = rng.permutation(perm[i])
                feats[f"permuted labels k={k}"] = neighbourhood_composition(
                    xy_all, perm, ncat, sections, k, keep
                )
            for fname, X in feats.items():
                schemes = ["leave-section", "blocked"]
                if k == K_MAIN and fname.startswith("neighbourhood"):
                    schemes = [
                        "random",
                        "leave-section",
                        "blocked",
                        "cross-brain q2w",
                        "cross-brain w2q",
                    ]
                for scheme in schemes:
                    seeds = SEEDS if not scheme.startswith("cross-brain") else [0]
                    for seed in seeds:
                        m, per = evaluate(X, y, groups, brain, scheme, seed)
                        m.update(
                            vocabulary=vname,
                            features=fname,
                            cv_scheme=scheme,
                            seed=seed,
                            k=k,
                        )
                        rows.append(m)
                        if seed == 0:
                            per = per.assign(
                                vocabulary=vname, features=fname, cv_scheme=scheme, k=k
                            )
                            per_rows.append(per)
                    got = [
                        r
                        for r in rows
                        if r["features"] == fname
                        and r["cv_scheme"] == scheme
                        and r["vocabulary"] == vname
                        and r["k"] == k
                    ]
                    print(
                        f"  {fname:24s} {scheme:16s} "
                        f"acc {np.mean([g['accuracy'] for g in got]):.3f}  "
                        f"balacc {np.mean([g['balanced_accuracy'] for g in got]):.3f}  "
                        f"macroF1 {np.mean([g['macro_f1'] for g in got]):.3f}",
                        flush=True,
                    )

    # geometry-only control, vocabulary-independent; averaged over SEEDS like
    # the other feature sets.
    print("\n=== geometry only (NO cell types) ===", flush=True)
    Xg = geometry_features(xy_all, obs, sections, K_MAIN, keep)
    for scheme in ["leave-section", "blocked", "cross-brain q2w"]:
        for seed in SEEDS:
            m, per = evaluate(Xg, y, groups, brain, scheme, seed, scale=True)
            m.update(
                vocabulary="(none)",
                features="geometry only",
                cv_scheme=scheme,
                seed=seed,
                k=K_MAIN,
            )
            rows.append(m)
            per_rows.append(
                per.assign(
                    vocabulary="(none)",
                    features="geometry only",
                    cv_scheme=scheme,
                    k=K_MAIN,
                )
            )
        got = [
            r
            for r in rows
            if r["features"] == "geometry only" and r["cv_scheme"] == scheme
        ]
        print(
            f"  geometry only            {scheme:16s} "
            f"acc {np.mean([g['accuracy'] for g in got]):.3f}  "
            f"balacc {np.mean([g['balanced_accuracy'] for g in got]):.3f}  "
            f"macroF1 {np.mean([g['macro_f1'] for g in got]):.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df["majority_baseline"] = majority
    df.to_csv(OUT / "scheme_comparison.csv", index=False)
    pd.concat(per_rows).to_csv(OUT / "per_region_f1.csv", index=False)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
