#!/usr/bin/env python
"""Parity check: repository trial-blocked DE vs the saved ``de_all.csv.gz``.

Analysis design of the saved table: I. Traniello.

Compares, per cell type, the repository's fine-type table
``bimp_st.paths.CASTE_DE_FINE`` (``caste_de_blocked_all.py``, pyDESeq2 0.5.4)
with the saved table (frozen copy in ``bimp_st.paths.CASTE_DE_SAVED_INPUTS``,
pyDESeq2 0.5.2, same settings). The saved analysis fits the 15 fine types only,
so the whole brain and the coarse classes have no counterpart to compare. Only
types fitted on the same samples are compared: the repository keeps whole trial
pairs and tests a type only with all three, so a type missing from one trial
pair (Other glia) is not tested, while the saved fit keeps it; such types are listed and
skipped.

Tolerance, and why:
- genes fit (all-zero genes dropped): must be IDENTICAL; it checks the inputs.
- Wald statistic over the common genes: Pearson r >= 0.999, and at most 1% of
  genes with |delta Wald| > 0.01. The model and data are the same, but a
  different Cook's-outlier replacement between pyDESeq2 versions can move the
  Wald statistic of a few highly expressed genes while the rest agree, so the
  check counts genes that moved instead of requiring every gene to match.
- genes tested (padj not NA) may differ: pyDESeq2 0.5.4 and 0.5.2 choose the
  independent-filtering cutoff differently. This moves padj, so DEG counts may
  differ by a few genes; reported, not failed, when |delta| <= 2% of the saved count
  or <= 3 genes.

Also reports Pearson r of log2FC, and padj agreement over genes both tables
test: the fraction with the same call at FDR 0.05 and the median |delta padj|.

Writes ``bimp_st.paths.CASTE_DE_PARITY_SAVED`` (next to the fit)
and exits non-zero on a hard failure.

    PYTHONPATH=src:scripts .venv/bin/python scripts/caste_de/parity_saved_de.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bimp_st.caste_de import MIN_NUCLEI
from bimp_st.paths import (
    CASTE_DE_COUNTS,
    CASTE_DE_FINE,
    CASTE_DE_NUCLEI,
    CASTE_DE_PARITY_SAVED,
    CASTE_DE_SAVED_INPUTS,
)

REPO = Path(__file__).resolve().parents[2]
OURS = CASTE_DE_FINE
OUT = CASTE_DE_PARITY_SAVED
SAVED = CASTE_DE_SAVED_INPUTS / "de_all.csv.gz"
FDR = 0.05


def summ(d: pd.DataFrame) -> pd.DataFrame:
    sig = d.padj < FDR
    return (
        pd.DataFrame(
            {
                "fit": d.groupby("ct").size(),
                "tested": d[d.padj.notna()].groupby("ct").size(),
                "q": d[sig & (d.log2FoldChange > 0)].groupby("ct").size(),
                "w": d[sig & (d.log2FoldChange < 0)].groupby("ct").size(),
            }
        )
        .fillna(0)
        .astype(int)
    )


def main() -> int:
    ours = pd.read_csv(OURS)
    saved = pd.read_csv(SAVED)
    for d in (ours, saved):
        d["ct"] = d.cell_type.str.replace(r"\s+", " ", regex=True)
    # the saved fit drops only samples below MIN_NUCLEI, so it uses all six samples
    # exactly where every sample passes; ours must have kept all six there too
    cnt = pd.read_csv(CASTE_DE_COUNTS)
    cnt = cnt[cnt.level == "fine"].set_index("cell_type")
    nuc = pd.read_csv(CASTE_DE_NUCLEI, index_col=0).drop(columns="major_cell_type")
    differ = sorted(ct for ct in cnt.index if cnt.at[ct, "samples"] != 6)
    saved_all6 = sorted(ct for ct in nuc.index if (nuc.loc[ct] >= MIN_NUCLEI).all())
    assert sorted(set(cnt.index) - set(differ)) == saved_all6, (differ, saved_all6)
    print("skipped (samples differ from the saved fit):", differ)
    ours = ours[~ours.ct.isin(differ)]
    saved = saved[~saved.ct.isin(differ)]
    so, si = summ(ours), summ(saved)
    rows, hard = [], []
    for ct in sorted(set(so.index) | set(si.index)):
        if ct not in so.index or ct not in si.index:
            hard.append(f"{ct}: present in only one table")
            continue
        a = saved[saved.ct == ct].set_index("gene")
        b = ours[ours.ct == ct].set_index("gene")
        g = a.index.intersection(b.index)
        ok = a.loc[g, "stat"].notna() & b.loc[g, "stat"].notna()
        r_stat = np.corrcoef(a.loc[g, "stat"][ok], b.loc[g, "stat"][ok])[0, 1]
        okp = a.loc[g, "pvalue"].notna() & b.loc[g, "pvalue"].notna()
        r_p = np.corrcoef(a.loc[g, "pvalue"][okp], b.loc[g, "pvalue"][okp])[0, 1]
        sa = set(a.index[a.padj < FDR])
        sb = set(b.index[b.padj < FDR])
        row = dict(
            cell_type=ct,
            fit_ours=so.at[ct, "fit"],
            fit_saved=si.at[ct, "fit"],
            tested_ours=so.at[ct, "tested"],
            tested_saved=si.at[ct, "tested"],
            queen_up_ours=so.at[ct, "q"],
            queen_up_saved=si.at[ct, "q"],
            worker_up_ours=so.at[ct, "w"],
            worker_up_saved=si.at[ct, "w"],
            deg_overlap=len(sa & sb),
            r_wald=r_stat,
            r_pvalue=r_p,
        )
        okl = a.loc[g, "log2FoldChange"].notna() & b.loc[g, "log2FoldChange"].notna()
        row["r_log2fc"] = np.corrcoef(
            a.loc[g, "log2FoldChange"][okl], b.loc[g, "log2FoldChange"][okl]
        )[0, 1]
        bt = a.loc[g, "padj"].notna() & b.loc[g, "padj"].notna()
        pa, pb = a.loc[g, "padj"][bt], b.loc[g, "padj"][bt]
        row["n_both_tested"] = int(bt.sum())
        row["frac_same_call"] = float(((pa < FDR) == (pb < FDR)).mean())
        row["median_abs_dpadj"] = float((pa - pb).abs().median())
        row["total_ours"] = row["queen_up_ours"] + row["worker_up_ours"]
        row["total_saved"] = row["queen_up_saved"] + row["worker_up_saved"]
        dw = (a.loc[g, "stat"] - b.loc[g, "stat"]).abs()
        still = ok & (dw <= 0.01)
        row["n_wald_moved"] = int((dw > 0.01).sum())
        row["r_wald_unmoved"] = np.corrcoef(
            a.loc[g, "stat"][still], b.loc[g, "stat"][still]
        )[0, 1]
        rows.append(row)
        if row["fit_ours"] != row["fit_saved"]:
            hard.append(f"{ct}: genes fit {row['fit_ours']} vs {row['fit_saved']}")
        if r_stat < 0.999 or row["n_wald_moved"] > 0.01 * len(g):
            hard.append(f"{ct}: Wald r {r_stat:.5f}, {row['n_wald_moved']} moved")
        dt = abs(row["total_ours"] - row["total_saved"])
        if dt > max(3, 0.02 * row["total_saved"]):
            hard.append(f"{ct}: DEGs {row['total_ours']} vs {row['total_saved']}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(6).to_string(index=False))
    tot = out[["total_ours", "total_saved"]].sum()
    print(
        f"\nDEGs summed over types: ours {tot.total_ours}, "
        f"saved {tot.total_saved}"
    )
    for k in ("ours", "saved"):
        d = ours if k == "ours" else saved
        n = d[d.padj < FDR].gene.nunique()
        print(f"genes DE in >= 1 type ({k}): {n}")
    if hard:
        print("\nHARD FAILURES:\n  " + "\n  ".join(hard))
        return 1
    print("\nPARITY OK (fit counts identical, Wald r >= 0.999, <= 1% genes moved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
