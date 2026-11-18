#!/usr/bin/env python
"""``fig:caste_de`` panel d inputs from the in-repo run of the cNMF / GSEA pipeline.

Same files and columns as ``bimp_st.paths.CASTE_DE_SAVED_INPUTS`` (written by
``caste_de_freeze_saved_outputs.py`` from the SAVED run), built only from this
pipeline's outputs under ``results/caste_programs/pipeline/``:

  cnmf_joint/joint/            the repo factorization (cnmf_prepare.py, run_joint.sh)
  repo/joint_k50_caste.csv     cell 25 on it (cnmf_caste.py)
  repo/joint_k50_figure_programs_go_bp.csv.gz   cell 27, 1,000 permutations
                               (program_go.py): the panel d labels
  repo/cnmf_k_selection.csv    cell 23 (cnmf_caste.py)
  program_map.csv              saved program numbers -> the repo's (program_map.py)

Written to ``pipeline/fig3_inputs/`` (with ``--cell109``, to
``pipeline/fig3_inputs_cell109/``, labels from original notebook cell 109's
500-permutation table for programs 18 and 12, ``program_go.py --cell109``: the
labels of the original notebook's figure; comparison only, ``fig:caste_de``
never reads it):
  cnmf_k50_fig_usage.csv.gz    per-nucleus usage (%), normalized over all 50
                               programs, of the five figure programs + cell type,
                               caste, trial
  cnmf_k50_fig_gene_scores.csv.gz  gene x program gene-spectra scores (5 programs)
  joint_k50_caste.csv          per (cell type, program) trial effects + usage
  program_labels.csv           panel d row labels: top BP term at FDR < 0.05
                               (clean notebook cell 30, ``make_d.row_label``;
                               original cell 111), else the notebook's manual label
                               (dagger, cell 30 DESCR)
  cnmf_k_selection.csv         k-selection table
  PROVENANCE.md
Programs keep the SAVED run's numbers (columns p18, p20, ...; ``program``) so
the figure code is unchanged; ``repo_program`` records the repo run's own
number. ``row_label`` and DESCR come from cell 30 of the clean notebook
01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb (md5
184efd9af8192fd09f7329f1eeaa9f17).

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/fig3_inputs.py [--cell109]
"""

import argparse
import hashlib
import re

import anndata as ad
import numpy as np
import pandas as pd
from cnmf_gsea_paths import (
    CNMF_NAME,
    FIG3_INPUTS,
    FIG_PROGS,
    PROGRAM_MAP,
    REPO_CNMF,
    RUN,
    SAVED_PROGRAM_COL,
)

J = REPO_CNMF / CNMF_NAME
REP = RUN / "repo"
OUT = FIG3_INPUTS
# clean notebook cell 30: manual labels for programs with no BP term at FDR < 0.05
DESCR = {
    18: "Transporters, oxidoreductases, esterases†",
    7: "ECM, Notch, Rho signaling†",
}


def md5(p) -> str:
    """md5 of a file, read in 4 MB blocks."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--cell109",
        action="store_true",
        help="labels from the original-cell-109 (500-permutation) table, into "
        "fig3_inputs_cell109/ (comparison only)",
    )
    args = ap.parse_args()
    out = OUT.with_name(OUT.name + "_cell109") if args.cell109 else OUT
    go_name = "joint_k50_figure_programs_go_bp" + ("_cell109" if args.cell109 else "")
    out.mkdir(parents=True, exist_ok=True)
    pm = pd.read_csv(PROGRAM_MAP)
    to_repo = pm.set_index(SAVED_PROGRAM_COL).repo_program.to_dict()
    to_saved = {v: k for k, v in to_repo.items()}
    srcs = []

    usage_f = J / "joint.usages.k_50.dt_2_0.consensus.txt"
    u = pd.read_csv(usage_f, sep="\t", index_col=0)
    u.columns = u.columns.astype(int)
    u = u.div(u.sum(axis=1), axis=0) * 100
    counts = REPO_CNMF / "joint.counts.h5ad"
    obs = ad.read_h5ad(counts, backed="r").obs[["cell_type", "caste", "trial"]]
    obs = obs.astype(str).reindex(u.index)
    assert obs.notna().all().all()
    obs["cell_type"] = obs.cell_type.map(lambda s: re.sub(r"\s+", " ", s))
    fu = u[[to_repo[p] for p in FIG_PROGS]].copy()
    fu.columns = [f"p{p}" for p in FIG_PROGS]
    fu = obs.join(fu)
    fu.index.name = "nucleus"
    fu.to_csv(out / "cnmf_k50_fig_usage.csv.gz")
    srcs += [(usage_f, md5(usage_f)), (counts, "obs only")]

    score_f = J / "joint.gene_spectra_score.k_50.dt_2_0.txt"
    s = pd.read_csv(score_f, sep="\t", index_col=0)
    s.index = s.index.astype(int)
    sc = s.loc[[to_repo[p] for p in FIG_PROGS]].T
    sc.columns = [f"p{p}" for p in FIG_PROGS]
    sc.index.name = "gene"
    sc.to_csv(out / "cnmf_k50_fig_gene_scores.csv.gz")
    srcs.append((score_f, md5(score_f)))

    jc_f = REP / "joint_k50_caste.csv"
    jc = pd.read_csv(jc_f)
    jc.insert(1, "repo_program", jc.program)
    jc["program"] = jc.repo_program.map(to_saved)
    jc.to_csv(out / "joint_k50_caste.csv", index=False)
    srcs.append((jc_f, md5(jc_f)))

    ks_f = REP / "cnmf_k_selection.csv"
    pd.read_csv(ks_f).to_csv(out / "cnmf_k_selection.csv", index=False)
    srcs.append((ks_f, md5(ks_f)))

    # make_d.row_label (clean notebook cell 30) on cell 27's BP table
    # (original cell 109's with --cell109)
    go_f = REP / f"{go_name}.csv.gz"
    g = pd.read_csv(go_f)
    lab = []
    for p in FIG_PROGS:
        t = g[g.program == to_repo[p]].sort_values(
            ["FDR q-val", "NES", "name"], ascending=[True, False, True]
        )
        t = t[t["FDR q-val"] < 0.05]
        if len(t):
            r = t.iloc[0]
            lab.append(
                dict(
                    program=p,
                    repo_program=to_repo[p],
                    label=r["name"][0].upper() + r["name"][1:],
                    go_id=r.go_id,
                    fdr=float(r["FDR q-val"]),
                    nes=float(r.NES),
                    manual=False,
                )
            )
        else:
            lab.append(
                dict(
                    program=p,
                    repo_program=to_repo[p],
                    label=DESCR[p],
                    go_id="",
                    fdr=np.nan,
                    nes=np.nan,
                    manual=True,
                )
            )
    lab = pd.DataFrame(lab)
    lab.to_csv(out / "program_labels.csv", index=False)
    srcs.append((go_f, md5(go_f)))
    print(lab.to_string())

    with open(out / "PROVENANCE.md", "w") as f:
        f.write(
            "# fig:caste_de panel d inputs from the in-repo pipeline\n\n"
            "Written by `scripts/caste_de/cnmf_gsea/fig3_inputs.py` from the repo's "
            "own run of the joint cNMF / GSEA code "
            "(`scripts/caste_de/cnmf_gsea/`, order in `scripts/README.md`). "
            "Programs keep the saved run's numbers; `program_map.csv` gives the "
            "repo run's "
            "number and the spectra correlation. Parity with the saved outputs: "
            "`results/caste_programs/pipeline/parity/parity_summary.txt`.\n\n"
            "| source | md5 |\n|---|---|\n"
        )
        for p, h in srcs:
            f.write(f"| `{p}` | {h} |\n")


if __name__ == "__main__":
    main()
