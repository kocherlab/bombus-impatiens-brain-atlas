#!/usr/bin/env python
"""Pseudobulk counts of every FINE cell type of the snRNA central brain.

Sums the ambient-corrected (CellBender) integer counts (``layers['counts']``)
of ``data/sn/BIMP_atlas_qw_CentralBrain.h5ad`` per sample x ``cell_type`` (15
fine types: glia subtypes, Kenyon subtypes, projection neuron subtypes, Other
neurons, Hemocytes, Photoreceptors) over the six libraries
``BIMP_{Queen,Worker}_t{1,2,3}``, and counts the nuclei per sample x type.
Writes ``counts_<fn(type)>.csv``, ``groups.json``, ``metadata.csv`` and
``gene_desc.csv`` to ``bimp_st.paths.CASTE_DE_PSEUDOBULK_FINE`` and
``nuclei_per_sample.csv`` to ``bimp_st.paths.CASTE_DE_NUCLEI``. The caste DE
fit on them is ``caste_de_blocked_all.py`` (recipe ``bimp_st.caste_de``).

The 13 optic-lobe types (OL_1..OL_13) and the optic-lobe Kenyon entry exist in
one optic-lobe library per caste, so no replicated test is possible for them;
they are not summed.

    PYTHONPATH=src:scripts .venv/bin/python scripts/caste_de/fine_type_deseq2.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bimp_st.caste_de import fn
from bimp_st.paths import CASTE_DE_NUCLEI, CASTE_DE_PSEUDOBULK_FINE
from bimp_st.site import get as site_path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
# data/ is gitignored and absent from worktrees; fall back to the main checkout
DATA_REPO = REPO if (REPO / "data/sn").exists() else site_path("data_repo")
ATLAS = DATA_REPO / "data/sn/BIMP_atlas_qw_CentralBrain.h5ad"
OUT = CASTE_DE_NUCLEI.parent
PB = CASTE_DE_PSEUDOBULK_FINE
SAMPLES = [f"BIMP_{c}_t{i}" for c in ("Queen", "Worker") for i in (1, 2, 3)]
CASTES = ["Queen"] * 3 + ["Worker"] * 3


def stage_pseudobulk() -> None:
    import scanpy as sc
    import scipy.sparse as sp

    PB.mkdir(parents=True, exist_ok=True)
    a = sc.read_h5ad(ATLAS)
    if "counts" not in a.layers:
        # X is normalised; DESeq2 needs the integer counts
        raise SystemExit(f"ERROR: {ATLAS} has no 'counts' layer")
    X = a.layers["counts"]
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    samp = a.obs["sample"].astype(str).values
    ct = a.obs["cell_type"].astype(str).values
    types = sorted(set(ct))
    major = (
        a.obs.groupby("cell_type", observed=True)["major_cell_type"].first().astype(str)
    )
    n = pd.DataFrame(
        {t: [int(((samp == s) & (ct == t)).sum()) for s in SAMPLES] for t in types},
        index=SAMPLES,
    ).T
    n["major_cell_type"] = major.reindex(n.index)
    n.to_csv(CASTE_DE_NUCLEI)
    print(n)
    pd.DataFrame({"sample": SAMPLES, "caste": CASTES}).set_index("sample").to_csv(
        PB / "metadata.csv"
    )
    desc = a.var["description"] if "description" in a.var else a.var.iloc[:, 0]
    desc.astype(str).to_csv(PB / "gene_desc.csv")
    genes = a.var_names.to_numpy()
    for t in types:
        rows = [
            np.asarray(X[(samp == s) & (ct == t)].sum(axis=0)).ravel() for s in SAMPLES
        ]
        pd.DataFrame(np.vstack(rows).astype(int), index=SAMPLES, columns=genes).to_csv(
            PB / f"counts_{fn(t)}.csv"
        )
    json.dump(types, open(PB / "groups.json", "w"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    stage_pseudobulk()
