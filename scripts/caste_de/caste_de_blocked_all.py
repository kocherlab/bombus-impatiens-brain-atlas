#!/usr/bin/env python
"""The snRNA caste DE: one recipe, every level, one job.

Fits the trial-blocked recipe (design: I. Traniello; ``bimp_st.caste_de``: ``~ trial +
caste``, samples with >= 10 nuclei kept as whole queen + worker trial pairs,
>= 2 pairs, all-zero genes dropped, Wald, BH) to

- the whole brain (``all``) and the six coarse classes, from the pseudobulk
  counts in ``paths.CASTE_DE_PSEUDOBULK_COARSE`` (``pseudobulk_deseq2.py``;
  each class's counts are exactly the sum of its fine types' counts below);
  coarse nuclei per sample are the per-sample sums of ``paths.CASTE_DE_NUCLEI``
  over ``major_cell_type``;
- the 15 central-brain fine types, from ``paths.CASTE_DE_PSEUDOBULK_FINE``
  (the fine types of the caste-DE analysis notebook).

Inputs are read, never written. Everything lands in ``paths.CASTE_DE_FIT_DIR``:

- ``de_coarse_blocked.csv``: whole brain + coarse classes (``cell_type``
  underscored, ``all`` = whole brain) plus ``tested``;
- ``de_fine_blocked.csv``: fine types (``cell_type`` with spaces);
- ``deg_counts.csv``: per level and type: samples, nuclei, genes fit/tested,
  queen-up, worker-up;
- ``focal_genes.csv``: Kr-h1, sNPF, Tk, Vgn and Vg-like in every type;
- ``per_trial_signs.csv``: every fine-type DEG's queen/worker log ratio in each
  retained trial pair, and whether all of them have the fitted sign;
- ``parity_previous.csv``: the fit against
  ``paths.CASTE_DE_FIT_DIR_TWO_PAIRS`` (Other glia fit on two trial pairs, one
  residual df). Every
  type whose retained samples did not change must be identical; exits non-zero
  if not. Parity with the saved analysis table is ``parity_saved_de.py``.

    PYTHONPATH=src .venv-deseq2/bin/python scripts/caste_de/caste_de_blocked_all.py
"""

from __future__ import annotations

import json
import sys
import warnings

import pandas as pd

from bimp_st import caste_de as cd
from bimp_st.paths import (
    CASTE_DE_FIT_DIR,
    CASTE_DE_FIT_DIR_TWO_PAIRS,
    CASTE_DE_NUCLEI,
    CASTE_DE_PSEUDOBULK_COARSE,
    CASTE_DE_PSEUDOBULK_FINE,
)

warnings.filterwarnings("ignore")

OUT = CASTE_DE_FIT_DIR
# the previous fit, where two trial pairs were enough (Other glia 2 vs 2)
PREVIOUS = CASTE_DE_FIT_DIR_TWO_PAIRS

COARSE = [
    "all",
    "Glia",
    "Hemocytes",
    "Kenyon_cells",
    "Other_neurons",
    "Photoreceptors",
    "Projection_neurons",
]
FOCAL = {g: s for s, g in cd.focal_gene_ids().items()}
# gene_name overrides in the tables, as in pseudobulk_deseq2.py, so downstream
# readers keep working
_ids = cd.focal_gene_ids()
NAMED = {_ids["Kr-h1"]: "Kr-h1", _ids["Vg-like"]: "vg", _ids["Tk"]: "Tk"}


def main() -> int:
    from pydeseq2 import __version__ as pdv

    OUT.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(CASTE_DE_PSEUDOBULK_COARSE / "gene_desc.csv", index_col=0)
    names = {g: NAMED.get(g, str(d)[:40]) for g, d in desc.iloc[:, 0].items()}
    nuc = pd.read_csv(CASTE_DE_NUCLEI, index_col=0)
    coarse_nuc = nuc.groupby("major_cell_type").sum()
    coarse_nuc.index = [cd.fn(c) for c in coarse_nuc.index]
    fine_nuc = nuc.drop(columns="major_cell_type")

    jobs = [
        ("coarse", ct, CASTE_DE_PSEUDOBULK_COARSE / f"counts_{ct}.csv") for ct in COARSE
    ] + [
        ("fine", ct, CASTE_DE_PSEUDOBULK_FINE / f"counts_{cd.fn(ct)}.csv")
        for ct in json.load(open(CASTE_DE_PSEUDOBULK_FINE / "groups.json"))
    ]
    res = {"coarse": [], "fine": []}
    counts_rows, signs = [], []
    for level, ct, path in jobs:
        c = pd.read_csv(path, index_col=0)
        if level == "fine":
            nuclei = fine_nuc.loc[ct]
        else:
            nuclei = None if ct == "all" else coarse_nuc.loc[ct]
        r, dds = cd.fit(c, nuclei, cd.DESIGN)
        row = dict(level=level, cell_type=ct, design=cd.DESIGN)
        if r is None:
            counts_rows.append(row | dict(status="skipped"))
            print(f"SKIP {level} {ct}", flush=True)
            continue
        q, w = cd.n_sig(r)
        kept = list(dds.obs_names)
        row |= dict(
            status="tested",
            samples=dds.n_obs,
            queen_samples=int((dds.obs.caste == "Queen").sum()),
            worker_samples=int((dds.obs.caste == "Worker").sum()),
            nuclei=None if nuclei is None else int(nuclei[kept].sum()),
            genes_fit=len(r),
            genes_tested=int(r.tested.sum()),
            queen_up=q,
            worker_up=w,
            min_padj=float(r.padj.min()),
            pydeseq2=pdv,
        )
        counts_rows.append(row)
        print(row, flush=True)
        r["cell_type"] = ct
        r["gene"] = r.index
        r["gene_name"] = [names.get(g, g) for g in r.index]
        res[level].append(r)
        if level == "fine":
            s = cd.per_trial_consistency(c, r, kept)
            s.insert(0, "cell_type", ct)
            signs.append(s)

    tables = {}
    for level, rs in res.items():
        t = pd.concat(rs)
        t.index.name = None
        t.to_csv(OUT / f"de_{level}_blocked.csv")
        tables[level] = t
    pd.DataFrame(counts_rows).to_csv(OUT / "deg_counts.csv", index=False)
    signs = pd.concat(signs, ignore_index=True)
    signs.to_csv(OUT / "per_trial_signs.csv", index=False)
    print(
        "per-trial signs:",
        int(signs.all_agree.sum()),
        "of",
        len(signs),
        "DEG rows agree in every trial pair",
        flush=True,
    )

    focal = []
    for level, t in tables.items():
        f = t[t.gene.isin(FOCAL)].copy()
        f["gene_name"] = f.gene.map(FOCAL)
        f.insert(0, "level", level)
        focal.append(f)
    focal = pd.concat(focal)[
        ["level", "gene", "gene_name", "cell_type", "baseMean", "log2FoldChange"]
        + ["lfcSE", "stat", "pvalue", "padj", "tested"]
    ]
    focal.to_csv(OUT / "focal_genes.csv", index=False)
    print(focal.to_string(), flush=True)
    return parity(tables, pd.DataFrame(counts_rows))


def parity(tables: dict, counts: pd.DataFrame) -> int:
    """Types whose retained samples are unchanged must reproduce the previous
    fit exactly; the others are listed with their changes."""
    prev_counts = pd.read_csv(PREVIOUS / "deg_counts.csv")
    key = ["level", "cell_type"]
    m = counts.merge(prev_counts, on=key, suffixes=("", "_prev"))
    rows = []
    for _, k in m.iterrows():
        ours = tables[k.level]
        prev = pd.read_csv(PREVIOUS / f"de_{k.level}_blocked.csv", index_col=0)
        a = ours[ours.cell_type == k.cell_type].set_index("gene")
        b = prev[prev.cell_type == k.cell_type].set_index("gene")
        g = a.join(b, rsuffix="_prev", how="outer")
        both = g.padj.notna() & g.padj_prev.notna()
        rows.append(
            dict(
                level=k.level,
                cell_type=k.cell_type,
                samples=k.samples,
                samples_prev=k.samples_prev,
                genes=len(a),
                genes_prev=len(b),
                tested=int(a.tested.sum()),
                tested_prev=int(b.tested.sum()),
                sig=int((a.padj < cd.FDR).sum()),
                sig_prev=int((b.padj < cd.FDR).sum()),
                max_abs_dstat=float((g.stat - g.stat_prev).abs().max()),
                max_abs_dpadj=float((g.padj[both] - g.padj_prev[both]).abs().max()),
            )
        )
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "parity_previous.csv", index=False)
    print(p.to_string(), flush=True)
    same = p[p.samples == p.samples_prev]
    ok = (
        (same.genes == same.genes_prev).all()
        and (same.tested == same.tested_prev).all()
        and (same.sig == same.sig_prev).all()
        and (same.max_abs_dstat == 0).all()
    )
    changed = p[p.samples != p.samples_prev]
    print("types with changed samples:", list(changed.cell_type), flush=True)
    print("PARITY of unchanged types:", "OK" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
