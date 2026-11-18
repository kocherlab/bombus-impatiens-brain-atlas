#!/usr/bin/env python
"""Match the repo's joint cNMF programs (k = 50) to the saved run.

cNMF program numbers are arbitrary unless the factorization is bit-identical, so
the figure's programs (saved 18, 20, 6, 7, 12) are located in the repo run by
gene-spectra correlation: Pearson r between every pair of saved and repo
programs over the shared genes, on the consensus gene-spectra scores
(``gene_spectra_score``, all genes) and on the consensus spectra
(``spectra``, the 3,000 overdispersed genes); one-to-one assignment
(Hungarian) on the score correlation.

Writes ``pipeline/program_map.csv`` (``cnmf_gsea_paths.SAVED_PROGRAM_COL`` =
the saved run's number, repo_program, r_gene_score, r_spectra, best_other_r,
argmax_agrees, in_figure) and prints the summary. Run after
``cnmf_caste.py --source repo``.

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/program_map.py
"""

import numpy as np
import pandas as pd
from cnmf_gsea_paths import (
    CNMF_NAME,
    FIG_PROGS,
    PROGRAM_MAP,
    REPO_CNMF,
    SAVED_CNMF,
    SAVED_PROGRAM_COL,
)
from scipy.optimize import linear_sum_assignment

REPO_J = REPO_CNMF / CNMF_NAME
SAVED_J = SAVED_CNMF / CNMF_NAME


def load(d, kind) -> pd.DataFrame:
    """Consensus k = 50 spectra (programs x genes) of one run."""
    f = d / f"joint.{kind}.k_50.dt_2_0{'.consensus' if kind == 'spectra' else ''}.txt"
    x = pd.read_csv(f, sep="\t", index_col=0)
    x.index = x.index.astype(int)  # programs x genes
    return x


def corr(a, b) -> tuple[pd.DataFrame, int]:
    """Pearson r between every row of a and of b over their shared genes."""
    g = a.columns.intersection(b.columns)
    A = a[g].to_numpy(float)
    B = b[g].to_numpy(float)
    A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    return pd.DataFrame(A @ B.T / len(g), index=a.index, columns=b.index), len(g)


def main() -> None:
    sav_s = load(SAVED_J, "gene_spectra_score")
    rep_s = load(REPO_J, "gene_spectra_score")
    sav_p, rep_p = load(SAVED_J, "spectra"), load(REPO_J, "spectra")
    R, ng = corr(sav_s, rep_s)
    Rp, ngp = corr(sav_p, rep_p)
    hv_i = set((SAVED_J / "joint.overdispersed_genes.txt").read_text().split())
    hv_r = set((REPO_J / "joint.overdispersed_genes.txt").read_text().split())
    print(
        f"overdispersed genes: saved {len(hv_i)}, repo {len(hv_r)}, "
        f"shared {len(hv_i & hv_r)}; genes correlated: scores {ng}, spectra {ngp}"
    )
    exact = sav_s.shape == rep_s.shape and np.allclose(
        sav_s.sort_index().to_numpy(), rep_s.sort_index().to_numpy(), atol=0, rtol=0
    )
    print("gene_spectra_score bit-identical:", exact)
    for kind, a, b in (("gene_spectra_score", sav_s, rep_s), ("spectra", sav_p, rep_p)):
        if a.shape == b.shape and a.index.equals(b.index):
            d = np.abs(a.to_numpy(float) - b[a.columns].to_numpy(float))
            scale = np.abs(a.to_numpy(float)).max()
            print(f"{kind}: max |saved - repo| {d.max():.3e} (max |value| {scale:.3e})")
    ri, ci = linear_sum_assignment(-R.to_numpy())
    rows = []
    for i, j in zip(ri, ci, strict=True):
        p, q = R.index[i], R.columns[j]
        others = R.loc[p].drop(q)
        rows.append(
            dict(
                **{SAVED_PROGRAM_COL: int(p)},
                repo_program=int(q),
                r_gene_score=float(R.loc[p, q]),
                r_spectra=float(Rp.loc[p, q]),
                best_other_r=float(others.max()),
                argmax_agrees=bool(R.loc[p].idxmax() == q),
            )
        )
    m = pd.DataFrame(rows).sort_values(SAVED_PROGRAM_COL)
    m["in_figure"] = m[SAVED_PROGRAM_COL].isin(FIG_PROGS)
    m.to_csv(PROGRAM_MAP, index=False)
    print(m[m.in_figure].round(4).to_string(index=False))
    print(
        f"programs matched at r >= 0.9: {(m.r_gene_score >= 0.9).sum()} / {len(m)}; "
        f">= 0.95: {(m.r_gene_score >= 0.95).sum()}; >= 0.99: "
        f"{(m.r_gene_score >= 0.99).sum()}; identity map: "
        f"{(m[SAVED_PROGRAM_COL] == m.repo_program).all()}"
    )


if __name__ == "__main__":
    main()
