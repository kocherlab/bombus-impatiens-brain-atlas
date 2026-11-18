#!/usr/bin/env python
"""Step 4: replicated program x cell-type calls and their trial label-swap null.

Source: the ORIGINAL notebook
<snrna_source>/src/scripts/03_analysis/
01_QC_BIMP-Atlas_DifferentialExpression.ipynb (md5
5d120475e946d3953bc106195cad70ce), cell 113 (also cell 85 of
..._archive.ipynb, md5 452283158ddea3039e8103fd5e1f1dab), with the settings
and data lines of its cell 111 that cell 113 reads (PROGS, USE_FLOOR, usage,
ob, r50). Neither cell is in the clean notebook or the fig4_supp scripts.

A program x cell type is a replicated call (a star in panel d) when its
per-trial effects (``joint_k50_caste.csv``, cell 25) share one sign in all
three trials, |mean| >= 0.5 SD, the weakest trial >= 0.25 SD, and the program's
mean usage in the cell type is >= 2%. The null flips the sign of each trial's
effect in every pattern and recounts. With one queen and one worker per trial
there are only 2^3 = 8 within-trial relabelings, and the rule is
sign-symmetric, so the identity and the all-flipped pattern both give the
observed count: the smallest permutation P this design can reach is 2/8 = 0.25.
The notebook output: 9 observed, 0 1 0 0 1 0 under the six other patterns. The
calls are descriptive; that floor is the only P value the design supports.

Writes ``label_swap_null.csv`` (count per flip pattern) and
``label_swap_stars.csv`` (every observed star; the notebook cell printed only
the figure programs') under ``pipeline/<source>/``. For ``--source repo`` the
figure programs are the saved run's numbers mapped through ``program_map.csv`` when it
exists.

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/label_swap_null.py --source repo
"""

import argparse
import itertools
import re

import numpy as np
import pandas as pd
from cnmf import cNMF
from cnmf_gsea_common import CASTE, CT, TRIAL, load_atlas, setup
from cnmf_gsea_paths import (
    CNMF_NAME,
    FIG_PROGS,
    PROGRAM_MAP,
    SAVED_PROGRAM_COL,
    add_source_arg,
    source,
)

from bimp_st.paths import SN_CB

USE_FLOOR = 2.0  # min mean usage (%) for a cell type to be shaded (cell 111)
MEAN_MIN = 0.5  # |mean effect| in SD (cell 113)
WEAKEST_MIN = 0.25  # weakest trial |effect| in SD (cell 113)


def norm(s: str) -> str:
    """Collapse whitespace in a cell-type label (cell 111)."""
    return re.sub(r"\s+", " ", s)


def figure_programs(src_name: str) -> list[int]:
    """The figure programs (saved numbering), in this run's numbering."""
    progs = list(FIG_PROGS)
    if src_name == "repo" and PROGRAM_MAP.exists():
        m = pd.read_csv(PROGRAM_MAP).set_index(SAVED_PROGRAM_COL)["repo_program"]
        progs = [int(m[p]) for p in progs]
    print("PROGS", progs)
    return progs


def stars(em: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Cell 113: replicated calls for a (rows x trials) effect matrix."""
    same = (np.sign(em) == np.sign(em[:, [0]])).all(1)
    return (
        same
        & (np.abs(em.mean(1)) >= MEAN_MIN)
        & (np.abs(em).min(1) >= WEAKEST_MIN)
        & elig
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arg(ap)
    src = source(ap.parse_args().source)
    setup()
    adata, _, _ = load_atlas(SN_CB)
    progs = figure_programs(src.name)

    # cell 111: usage (%) per nucleus and the k = 50 caste table (cell 25)
    oj = cNMF(output_dir=str(src.cnmf_dir), name=CNMF_NAME)
    usage, sscore, _, _ = oj.load_results(K=50, density_threshold=2.0)
    if not set(usage.columns) <= set(sscore.columns):
        sscore = sscore.T
    usage = usage.div(usage.sum(1), axis=0) * 100
    ob = adata.obs.loc[usage.index, [CT, CASTE, TRIAL]].astype(str).copy()
    ob[CT] = ob[CT].map(norm)
    r50 = pd.read_csv(src.out_dir / "joint_k50_caste.csv")
    r50[CT] = r50[CT].map(norm)

    # cell 113
    tcols = [c for c in r50.columns if re.fullmatch(r"t\d+", str(c))]
    um = usage.groupby(ob[CT]).mean()
    r50["use"] = [
        um.loc[ct, p] if ct in um.index else np.nan
        for ct, p in zip(r50[CT], r50["program"], strict=False)
    ]
    e = r50[tcols].values
    elig = (r50["use"] >= USE_FLOOR).values
    in_fig = r50["program"].isin(progs).values

    flips = np.array(list(itertools.product([1, -1], repeat=len(tcols))))
    is_id = (flips == 1).all(1) | (flips == -1).all(1)  # identity and its mirror
    obs_mask = stars(e, elig)

    rows = []
    for f in flips:
        m = stars(e * f, elig)
        mirror = (f == 1).all() or (f == -1).all()
        rows.append(
            {
                "flip": "".join("+" if s > 0 else "-" for s in f),
                "n_stars_all50": int(m.sum()),
                "n_stars_figure5": int((m & in_fig).sum()),
                "type": "observed/mirror" if mirror else "label swap",
            }
        )
    nul = pd.DataFrame(rows)
    print(nul)
    nul.to_csv(src.out_dir / "label_swap_null.csv", index=False)

    sw = nul[nul["type"] == "label swap"]
    a50, f5 = sw["n_stars_all50"], sw["n_stars_figure5"]
    print(
        f"\nobserved stars, all 50 programs: {obs_mask.sum()}   | other "
        f"relabelings: {a50.min()}-{a50.max()} each"
    )
    print(
        f"observed stars, five figure programs: {(obs_mask & in_fig).sum()}   | "
        f"other relabelings: {f5.min()}-{f5.max()} each"
    )
    p = (nul["n_stars_all50"] >= obs_mask.sum()).mean()
    print(f"exact permutation P over all {len(nul)} relabelings: {p:.2f} (the floor)")

    # per-star view: which observed stars would also survive a trial swap?
    r50["star"] = obs_mask
    robust = [
        sum(stars(e * f, elig)[i] for f in flips[~is_id]) for i in np.where(obs_mask)[0]
    ]
    r50.loc[obs_mask, "n_label_swaps_also_star"] = robust
    cols = [CT, "program", *tcols, "mean", "use", "n_label_swaps_also_star"]
    st = r50[r50["star"]][cols].round(2)
    print(st.to_string())
    st.to_csv(src.out_dir / "label_swap_stars.csv", index=False)


if __name__ == "__main__":
    main()
