#!/usr/bin/env python
"""Caste DE refit with the fat-body-profile nuclei removed.

``snrna_controls.py`` flags queen central-brain nuclei that carry a fat-body
profile (queen, Vgn UMI >= 5 and Kunitz-inhibitor UMI >= 10, as in
``vgn_fat_body_origin.py``) and measures how much of each queen-up DEG's queen
counts they contribute.  This refits the pseudobulk DESeq2 for every class in
``bimp_st.paths.CASTE_DE_COARSE`` twice, on all nuclei (the reproduction check)
and with the flagged nuclei removed, under the ``~ trial + caste`` design.

Recipe: ``bimp_st.caste_de``. Raw ``counts`` summed per
(sample x ``major_cell_type``, plus ``all``) over the six
``BIMP_{Queen,Worker}_t{1,2,3}`` libraries; all-zero genes dropped; pyDESeq2
with Cook's refit; Wald Queen vs Worker (+ = queen-up), unshrunk LFC, BH within
class, FDR 0.05. Every coarse class keeps all six samples. The all-nuclei
``~trial + caste`` fit is asserted to reproduce the DEG count of every class in
``bimp_st.paths.CASTE_DE_COARSE``.

Recurring core: for each fit, a gene's ``n_coarse_queen_up`` is the number of
the seven coarse lists in which it is FDR < 0.05 and queen-up (the
``si_deg_recurrence.py`` definition restricted to the coarse lists); the core
is ``n_coarse_queen_up >= 3``.

Input: ``data/sn/BIMP_atlas_qw_CentralBrain.h5ad`` (layers['counts']);
``bimp_st.paths.CASTE_DE_COARSE`` (read only).

Output: results/caste_de/no_fatbody/
  pseudobulk/{full,nofat}/counts_<class>.csv, metadata.csv
  fatbody_nuclei.csv           the removed nuclei
  de_{full,nofat}_trial_blocked.csv   the two fits
  comparison.csv               per class x design: DEG counts before/after,
                               queen-up/worker-up, overlap, same-sign retention
  core.csv                     per design: core size before/after, retained
  core_genes.csv               per gene: n_coarse_queen_up before/after
  summary.md

Run (Slurm; one interpreter -- .venv-deseq2 has anndata, scipy and
pyDESeq2, pinned in config/envs/deseq2.txt)
    PYTHONPATH=src:scripts .venv-deseq2/bin/python -u \\
        scripts/caste_de/de_without_fat_body_nuclei.py --stage all
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from bimp_st.paths import (
    CASTE_DE_COARSE,
    CASTE_DE_DIR,  # noqa: E402
)

REPO = Path(__file__).resolve().parents[2]
ATLAS = REPO / "data/sn/BIMP_atlas_qw_CentralBrain.h5ad"
PUBLISHED = CASTE_DE_COARSE
OUT = CASTE_DE_DIR / "no_fatbody"
SAMPLES = [f"BIMP_{c}_t{t}" for c in ("Queen", "Worker") for t in (1, 2, 3)]
FDR = 0.05
VGN, KUNITZ = "BIMP_v4.3_005500", "BIMP_v4.3_000091"
HIGH, FAT = 5, 10
DESIGNS = {"trial_blocked": "~ trial + caste"}
VARIANTS = ("full", "nofat")


def stage_pseudobulk():
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(ATLAS)
    X = sp.csr_matrix(a.layers["counts"])
    samp = a.obs["sample"].astype(str).to_numpy()
    mct = a.obs["major_cell_type"].astype(str).to_numpy()
    queen = (a.obs["caste"] == "Queen").to_numpy()
    vgn = X[:, a.var_names.get_loc(VGN)].toarray().ravel()
    kun = X[:, a.var_names.get_loc(KUNITZ)].toarray().ravel()
    fat = queen & (vgn >= HIGH) & (kun >= FAT)
    assert fat.sum() == 13, fat.sum()
    OUT.mkdir(parents=True, exist_ok=True)
    a.obs.loc[fat, ["sample", "major_cell_type", "cell_type"]].assign(
        vgn=vgn[fat], kunitz=kun[fat]
    ).to_csv(OUT / "fatbody_nuclei.csv")
    meta = pd.DataFrame(
        {
            "caste": [s.split("_")[1] for s in SAMPLES],
            "trial": [s.rsplit("_", 1)[1] for s in SAMPLES],
        },
        index=pd.Index(SAMPLES, name="sample"),
    )
    for variant, keep in (("full", np.ones(len(samp), bool)), ("nofat", ~fat)):
        d = OUT / "pseudobulk" / variant
        d.mkdir(parents=True, exist_ok=True)
        meta.to_csv(d / "metadata.csv")
        for ct in ["all", *sorted(set(mct))]:
            rows = []
            for s in SAMPLES:
                m = keep & (samp == s) & ((ct == "all") | (mct == ct))
                rows.append(np.asarray(X[m].sum(axis=0)).ravel())
            mat = pd.DataFrame(
                np.vstack(rows).astype(int), index=SAMPLES, columns=a.var_names
            )
            mat.to_csv(d / f"counts_{ct.replace(' ', '_').replace('/', '_')}.csv")
        print(variant, "pseudobulk written, nuclei kept", int(keep.sum()), flush=True)


def stage_deseq():
    from bimp_st.caste_de import fit

    for variant in VARIANTS:
        d = OUT / "pseudobulk" / variant
        meta = pd.read_csv(d / "metadata.csv", index_col=0)
        for tag, design in DESIGNS.items():
            allres = []
            for f in sorted(glob.glob(str(d / "counts_*.csv"))):
                ct = os.path.basename(f)[len("counts_") : -len(".csv")]
                c = pd.read_csv(f, index_col=0).loc[meta.index]
                r, _ = fit(c, None, design)
                r["cell_type"], r["gene"] = ct, r.index
                allres.append(r)
                print(f"{variant} {tag} {ct}: {(r.padj < FDR).sum()} DEGs", flush=True)
            pd.concat(allres).to_csv(OUT / f"de_{variant}_{tag}.csv")


def _sig(df):
    s = df[df.padj < FDR]
    return {
        ct: (set(g.gene[g.log2FoldChange > 0]), set(g.gene[g.log2FoldChange < 0]))
        for ct, g in s.groupby("cell_type")
    }


def stage_compare():
    pub = pd.read_csv(PUBLISHED, index_col=0)
    classes = sorted(pub.cell_type.unique())
    rows, core_rows, gene_tabs = [], [], []
    for tag in DESIGNS:
        full = pd.read_csv(OUT / f"de_full_{tag}.csv", index_col=0)
        nofat = pd.read_csv(OUT / f"de_nofat_{tag}.csv", index_col=0)
        for ct in classes:  # reproduction of CASTE_DE_COARSE
            a = int((pub[pub.cell_type == ct].padj < FDR).sum())
            b = int((full[full.cell_type == ct].padj < FDR).sum())
            assert a == b, f"{ct}: round9 {a} vs refit {b}"
        sf, sn = _sig(full), _sig(nofat)
        for ct in classes:
            fq, fw = sf.get(ct, (set(), set()))
            nq, nw = sn.get(ct, (set(), set()))
            before, after = fq | fw, nq | nw
            same = len(fq & nq) + len(fw & nw)
            rows.append(
                dict(
                    design=tag,
                    cell_type=ct,
                    deg_before=len(before),
                    deg_after=len(after),
                    queen_up_before=len(fq),
                    queen_up_after=len(nq),
                    worker_up_before=len(fw),
                    worker_up_after=len(nw),
                    overlap=len(before & after),
                    same_sign_retained=same,
                    frac_retained=same / len(before) if before else np.nan,
                    lost=len(before - after),
                    gained=len(after - before),
                )
            )
        cnt = {}
        for v, s in (("before", sf), ("after", sn)):
            c = pd.Series(0, index=sorted(set(full.gene) | set(nofat.gene)))
            for ct in classes:
                q = list(s.get(ct, (set(), set()))[0])
                c.loc[q] += 1
            cnt[v] = c
        g = pd.DataFrame(
            {
                "n_coarse_queen_up_before": cnt["before"],
                "n_coarse_queen_up_after": cnt["after"],
            }
        )
        g["design"] = tag
        gene_tabs.append(
            g[(g.n_coarse_queen_up_before >= 3) | (g.n_coarse_queen_up_after >= 3)]
        )
        cb = set(g.index[g.n_coarse_queen_up_before >= 3])
        ca = set(g.index[g.n_coarse_queen_up_after >= 3])
        core_rows.append(
            dict(
                design=tag,
                core_before=len(cb),
                core_after=len(ca),
                core_retained=len(cb & ca),
                core_frac_retained=len(cb & ca) / len(cb) if cb else np.nan,
                core_new=len(ca - cb),
            )
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison.csv", index=False)
    core = pd.DataFrame(core_rows)
    core.to_csv(OUT / "core.csv", index=False)
    pd.concat(gene_tabs).to_csv(OUT / "core_genes.csv", index_label="gene")

    L = [
        "# Caste DE without the 13 fat-body nuclei",
        "",
        "Refit of the paper's pseudobulk DESeq2 (all-nuclei ~trial + caste "
        "refit reproduces every round9 class DEG count). Numbers from "
        "`comparison.csv` and `core.csv`.",
        "",
    ]
    for tag, gc in comp.groupby("design", sort=False):
        L.append(f"## {tag} ({DESIGNS[tag]})")
        for r in gc.itertuples():
            L.append(
                f"- {r.cell_type}: {r.deg_before} -> {r.deg_after} DEGs "
                f"(queen-up {r.queen_up_before} -> {r.queen_up_after}, "
                f"worker-up {r.worker_up_before} -> {r.worker_up_after}); "
                f"{r.same_sign_retained} of {r.deg_before} retained with "
                f"the same sign ({r.frac_retained:.1%})."
            )
        c = core[core.design == tag].iloc[0]
        L.append(
            f"- Recurring core (n_coarse_queen_up >= 3): {c.core_before} -> "
            f"{c.core_after}, {c.core_retained} retained "
            f"({c.core_frac_retained:.1%}), {c.core_new} new."
        )
        L.append(
            f"- Classes with >= 1 DEG after removal: "
            f"{int((gc.deg_after > 0).sum())} of {len(gc)}."
        )
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", default="all", choices=["pseudobulk", "deseq", "compare", "all"]
    )
    st = ap.parse_args().stage
    if st in ("pseudobulk", "all"):
        stage_pseudobulk()
    if st in ("deseq", "all"):
        stage_deseq()
    if st in ("compare", "all"):
        stage_compare()


if __name__ == "__main__":
    main()
