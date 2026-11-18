#!/usr/bin/env python
"""Freeze the saved cNMF / GSEA outputs of the caste-DE analysis, with provenance.

Analysis design: I. Traniello. ``fig:caste_de`` follows the analysis in
``<snrna_source>/src/scripts/03_analysis/
01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb``. The DE is recomputed in
the repo (``caste_de_blocked_all.py``); with ``fig_caste_de.py --saved-inputs``
the joint cNMF (k = 50) and the GSEA program labels are READ from the saved
outputs, never re-factorized. This script copies the small files and writes
compact derived tables so the figure script never opens an h5ad or the
snRNA-seq analysis tree:

``bimp_st.paths.CASTE_DE_SAVED_INPUTS``
    de_all.csv.gz              the saved DE table (parity reference only)
    joint_k50_caste.csv        per (cell type, program) trial effects + usage
    joint_k40_caste.csv        the same at k = 40 (Methods robustness sentence)
    cnmf_k_selection.csv       stability and prediction error for k = 30..70
    joint_k50_go_labels.csv    GSEA on program gene scores, top 5 terms each
    gsea_all.csv.gz            per-cell-type GSEA (optional supplementary plot)
    BIMP_lookup.csv            gene name / description / biotype
    cnmf_k50_fig_usage.csv.gz  per-nucleus usage (%) of the five figure programs
                               (normalized over all 50) + cell type, caste, trial
    cnmf_k50_fig_gene_scores.csv.gz  gene x program cNMF gene scores (5 programs)
    program_labels.csv         panel d row labels (GO BP term or manual, dagger)
    PROVENANCE.md              source path and md5 of every input

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/caste_de/caste_de_freeze_saved_outputs.py
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from bimp_st.paths import CASTE_DE_SAVED_INPUTS  # noqa: E402
from bimp_st.site import get as site_path

REPO = Path(__file__).resolve().parents[2]
OUT = CASTE_DE_SAVED_INPUTS
BASE = site_path("snrna_source")
CA = BASE / "out/clustering_and_annotation"
JOINT = CA / "cnmf_joint/joint"
USAGE = JOINT / "joint.usages.k_50.dt_2_0.consensus.txt"
SCORE = JOINT / "joint.gene_spectra_score.k_50.dt_2_0.txt"
COUNTS = CA / "cnmf_joint/joint.counts.h5ad"  # obs only
KSEL = JOINT / "joint.k_selection_stats.df.npz"
OBO = site_path("snrna_shared_scripts") / "go-basic.obo"
NOTEBOOK = (
    BASE / "src/scripts/03_analysis/01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb"
)
COPY = {
    "de_all.csv.gz": CA / "de_all.csv.gz",
    "joint_k50_caste.csv": CA / "joint_k50_caste.csv",
    "joint_k40_caste.csv": CA / "joint_k40_caste.csv",
    "joint_k50_go_labels.csv": CA / "joint_k50_go_labels.csv",
    "gsea_all.csv.gz": CA / "gsea_all.csv.gz",
    "BIMP_lookup.csv": BASE / "src/reference_genome/tables/BIMP_lookup.csv",
}
# Notebook cell 30: PROGS and the manual labels (dagger) for programs whose
# gene-score GSEA has no biological-process term at FDR < 0.05.
FIG_PROGS = [18, 20, 6, 7, 12]
DESCR = {
    18: "Transporters, oxidoreductases, esterases†",
    7: "ECM, Notch, Rho signaling†",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_obo(path: Path):
    """name -> (primary id, namespace); also the data-version header."""
    meta, version, cur = {}, None, {}
    for line in open(path):
        line = line.strip()
        if line.startswith("data-version:"):
            version = line.split(":", 1)[1].strip()
        if line.startswith("["):
            if cur.get("name") and cur.get("id"):
                meta.setdefault(cur["name"], (cur["id"], cur.get("ns")))
            cur = {}
        elif line.startswith("id: "):
            cur["id"] = line[4:]
        elif line.startswith("name: "):
            cur["name"] = line[6:]
        elif line.startswith("namespace: "):
            cur["ns"] = line[11:]
    if cur.get("name") and cur.get("id"):
        meta.setdefault(cur["name"], (cur["id"], cur.get("ns")))
    return meta, version


def main() -> None:
    import anndata as ad

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dst, src in COPY.items():
        shutil.copy2(src, OUT / dst)
        rows.append((dst, src, md5(src)))

    # k selection (stability = silhouette, prediction error), notebook cell 23
    z = np.load(KSEL, allow_pickle=True)
    pd.DataFrame(z["data"], columns=z["columns"]).to_csv(
        OUT / "cnmf_k_selection.csv", index=False
    )
    rows.append(("cnmf_k_selection.csv", KSEL, md5(KSEL)))

    # per-nucleus usage, normalized over all 50 programs (notebook make_d)
    u = pd.read_csv(USAGE, sep="\t", index_col=0)
    u.columns = u.columns.astype(int)
    u = u.div(u.sum(axis=1), axis=0) * 100
    obs = ad.read_h5ad(COUNTS, backed="r").obs[["cell_type", "caste", "trial"]]
    obs = obs.astype(str).reindex(u.index)
    assert obs.notna().all().all(), "usage nuclei missing from the counts obs"
    obs["cell_type"] = obs.cell_type.map(lambda s: re.sub(r"\s+", " ", s))
    fu = u[FIG_PROGS].copy()
    fu.columns = [f"p{p}" for p in FIG_PROGS]
    fu = obs.join(fu)
    fu.index.name = "nucleus"
    fu.to_csv(OUT / "cnmf_k50_fig_usage.csv.gz")
    rows += [
        ("cnmf_k50_fig_usage.csv.gz (usage)", USAGE, md5(USAGE)),
        (
            "cnmf_k50_fig_usage.csv.gz (obs)",
            COUNTS,
            "obs only; md5 not computed (1 GB)",
        ),
    ]

    s = pd.read_csv(SCORE, sep="\t", index_col=0)
    s.index = s.index.astype(int)
    s = s.T if set(FIG_PROGS) <= set(s.index) else s
    sc = s[FIG_PROGS] if set(FIG_PROGS) <= set(s.columns) else s.loc[FIG_PROGS].T
    sc.columns = [f"p{p}" for p in FIG_PROGS]
    sc.index.name = "gene"
    sc.to_csv(OUT / "cnmf_k50_fig_gene_scores.csv.gz")
    rows.append(("cnmf_k50_fig_gene_scores.csv.gz", SCORE, md5(SCORE)))

    # panel d labels: top biological-process term at FDR < 0.05 (sorted by FDR,
    # then NES descending, then name), else the manual label (notebook row_label)
    meta, version = parse_obo(OBO)
    g = pd.read_csv(OUT / "joint_k50_go_labels.csv")
    g["go_id"] = g.name.map(lambda n: meta.get(n, (None, None))[0])
    g["ns"] = g.name.map(lambda n: meta.get(n, (None, None))[1])
    lab = []
    for p in FIG_PROGS:
        t = g[(g.program == p) & (g.ns == "biological_process")]
        t = t[t["FDR q-val"] < 0.05].sort_values(
            ["FDR q-val", "NES", "name"], ascending=[True, False, True]
        )
        if len(t):
            r = t.iloc[0]
            lab.append(
                dict(
                    program=p,
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
                    label=DESCR[p],
                    go_id="",
                    fdr=np.nan,
                    nes=np.nan,
                    manual=True,
                )  # fmt: skip
            )
    pd.DataFrame(lab).to_csv(OUT / "program_labels.csv", index=False)
    rows.append(("program_labels.csv (GO names/namespaces)", OBO, md5(OBO)))
    print(pd.DataFrame(lab).to_string())

    with open(OUT / "PROVENANCE.md", "w") as f:
        f.write(
            "# Inputs from the saved caste-DE analysis (frozen for fig:caste_de)\n\n"
            "Written by `scripts/caste_de/caste_de_freeze_saved_outputs.py`. "
            "Analysis notebook: "
            f"`{NOTEBOOK}` (md5 {md5(NOTEBOOK)}).\n\n"
            "The joint cNMF (all central-brain nuclei except photoreceptors, 3,000 "
            "overdispersed genes, k = 50, density threshold 2.0, cnmf 1.7.1) and the "
            "GSEA program labels (GSEApy 1.2.1, eggNOG-mapper 2.1.12 GO sets, "
            f"GO release {version}) are the saved analysis's; nothing here was "
            "re-factorized or re-run. The program labels come from "
            "`joint_k50_go_labels.csv` (Nov 4), which matches the circulated "
            "figure (long-term synaptic potentiation FDR 0.016); a later "
            "six-program rerun in "
            "`out/fig4_supp/prog_go_six.csv.gz` gives FDR 0.021 for that term "
            "(GSEA permutation noise).\n\n"
            "| file | source | md5 of source |\n|---|---|---|\n"
        )
        for dst, src, h in rows:
            f.write(f"| `{dst}` | `{src}` | {h} |\n")
    print((OUT / "PROVENANCE.md").read_text())


if __name__ == "__main__":
    main()
