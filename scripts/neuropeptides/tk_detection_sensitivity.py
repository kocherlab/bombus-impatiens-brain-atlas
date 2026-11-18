"""Panel-wide specificity and detection-sensitivity controls for the Tk shift.

Ranks every panel gene by the worker-ward shift of its Kenyon-vs-rest
enrichment, log(fold_worker / fold_queen), where fold is the fraction of
cells detecting the gene inside the Kenyon compartments over the fraction
outside them (``fig_neuropeptide_spatial._kc_enrichment``, the function behind the
queen / worker Tk folds of `fig:neuropeptide_spatial`). It then asks whether a
less sensitive worker run could produce
Tk's loss of extra-Kenyon detection: genes detected in a similar fraction of
queen non-Kenyon cells (the decile of queen non-Kenyon detection that holds
Tk) should lose detection in the worker too if it did.

Inputs
  ``paths.SPATIAL_COMBINED`` (combined_clustered.h5ad): expression (.X > 0 is
  detection), slice_name and cell_id. Rows are used positionally; no label
  column is read. The Kenyon compartments are the nine-region
  ``in_kc_{ic,nc,oc}`` flags (``bimp_st.nine_region_common.attach_region_flags``,
  joined on (slice_name, cell_id)), the flags behind the pooled per-caste
  folds. The ``in_kc_*`` columns stored in combined_clustered.h5ad predate them
  and are not read. The same
  quantities, both flag sets side by side, are in
  ``tk_shift_battery.py``; this script must agree with its
  nine-region column.

Outputs (results/handoff/tk_detection_sensitivity/)
  per_gene.csv     detection in/out of Kenyon compartments per caste, fold,
                   shift, change in non-Kenyon detection (percentage points),
                   decile of queen non-Kenyon detection
  summary.csv      one row per summary quantity (name, value)
  candidate_nulls.csv  Tk's z against several prevalence-matched gene subsets

Run
    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/neuropeptides/tk_detection_sensitivity.py
"""

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from neuropeptides.fig_neuropeptide_spatial import _kc_enrichment

from bimp_st.nine_region_common import attach_region_flags, kc_mask, traced_mask
from bimp_st.paths import RESULTS, SPATIAL_COMBINED

OUT = RESULTS / "handoff" / "tk_detection_sensitivity"
TK = "BIMP_vNP3.1_07029"
SNPF = "BIMP_vNP3.1_01359"
KC = ("kc_ic", "kc_nc", "kc_oc")


class _Shim:
    def __init__(self, obs):
        self.obs = obs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(SPATIAL_COMBINED)
    X = sp.csc_matrix(a.X)
    names = np.array(a.var_names)
    brain = a.obs["brain"].astype(str).to_numpy()
    d = a.obs[["slice_name", "cell_id"]].copy()
    attach_region_flags(_Shim(d), regions=list(KC))
    # Untraced is NA, not False: traced_mask raises if any KC flag is NA.
    kc = traced_mask(kc_mask(d))
    assert int(kc.sum()) == 332_583, int(kc.sum())
    queen = brain == "queen"
    worker = brain == "worker"

    def det(mask):
        return np.asarray((X[np.where(mask)[0]] > 0).mean(0)).ravel()

    df = pd.DataFrame(
        {
            "gene": names,
            "det_in_queen": det(queen & kc),
            "det_out_queen": det(queen & ~kc),
            "det_in_worker": det(worker & kc),
            "det_out_worker": det(worker & ~kc),
        }
    )
    df["fold_queen"] = df.det_in_queen / df.det_out_queen
    df["fold_worker"] = df.det_in_worker / df.det_out_worker
    df["shift"] = np.log(df.fold_worker / df.fold_queen)
    df["d_out_pp"] = 100 * (df.det_out_worker - df.det_out_queen)
    df["decile_out_queen"] = pd.qcut(
        df.det_out_queen.rank(method="first"), 10, labels=False
    )
    tk = int(np.flatnonzero(pd.Series(names).str.startswith(TK))[0])
    sn = int(np.flatnonzero(pd.Series(names).str.startswith(SNPF))[0])
    df["is_tk"], df["is_snpf"] = False, False
    df.loc[tk, "is_tk"], df.loc[sn, "is_snpf"] = True, True
    df.to_csv(OUT / "per_gene.csv", index=False)

    # Gate: the fold must match the fig_neuropeptide_spatial value on the
    # nine-region flags.
    e = np.asarray(X[:, tk].todense()).ravel()
    fq, fw = _kc_enrichment(e, queen, kc), _kc_enrichment(e, worker, kc)
    assert abs(fq - 2.864) < 0.01 and abs(fw - 6.670) < 0.01, (fq, fw)

    t = df.loc[tk]
    shift = df["shift"].replace([np.inf, -np.inf], np.nan)
    ev = shift.notna()
    rank_desc = shift.rank(ascending=False)
    null = df[ev & ~df.is_tk & ~df.is_snpf]

    def z_against(sub):
        v = sub["shift"].to_numpy()
        return (t["shift"] - v.mean()) / v.std(ddof=1)

    dec = df[(df.decile_out_queen == t.decile_out_queen) & ~df.is_tk]
    rows = [
        ("fold_queen", fq),
        ("fold_worker", fw),
        ("n_genes", len(df)),
        ("n_evaluable", int(ev.sum())),
        ("tk_rank_shift", float(rank_desc[tk])),
        ("snpf_rank_shift", float(rank_desc[sn])),
        ("tk_z_vs_null", z_against(null)),
        ("n_null", len(null)),
        ("tk_det_out_queen_pct", 100 * t.det_out_queen),
        ("tk_det_out_worker_pct", 100 * t.det_out_worker),
        ("tk_d_out_pp", t.d_out_pp),
        ("decile_n_other_genes", len(dec)),
        ("decile_min_det_out_queen_pct", 100 * dec.det_out_queen.min()),
        ("decile_max_det_out_queen_pct", 100 * dec.det_out_queen.max()),
        ("decile_median_d_out_pp", float(dec.d_out_pp.median())),
        ("panel_median_d_out_pp", float(df[~df.is_tk].d_out_pp.median())),
    ]
    summ = pd.DataFrame(rows, columns=["name", "value"])
    summ.to_csv(OUT / "summary.csv", index=False)

    # Prevalence-matched subsets: the 48-gene subset has no surviving
    # definition, so the candidates are written out side by side.
    cand = []
    tin, tout = t.det_in_queen, t.det_out_queen
    for label, m in [
        (
            "KC detection >= 0.8 in both castes",
            (df.det_in_queen >= 0.8) & (df.det_in_worker >= 0.8),
        ),
        ("KC detection >= 0.5 in queen", df.det_in_queen >= 0.5),
        ("queen KC detection within 0.25 of Tk", (df.det_in_queen - tin).abs() <= 0.25),
        (
            "queen non-KC detection within 0.25 of Tk",
            (df.det_out_queen - tout).abs() <= 0.25,
        ),
        ("queen non-KC detection 0.1-0.7", df.det_out_queen.between(0.1, 0.7)),
    ]:
        sub = df[m & ev & ~df.is_tk & ~df.is_snpf]
        cand.append(
            dict(
                subset=label,
                n_genes=len(sub),
                tk_z=z_against(sub) if len(sub) > 2 else np.nan,
            )
        )
    pd.DataFrame(cand).to_csv(OUT / "candidate_nulls.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(pd.DataFrame(cand).to_string(index=False))


if __name__ == "__main__":
    main()
