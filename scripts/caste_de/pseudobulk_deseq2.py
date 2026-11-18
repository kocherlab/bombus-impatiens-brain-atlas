"""Pseudobulk counts of the whole brain and the coarse classes (snRNA central brain).

Sums the ambient-corrected (CellBender) integer counts of
``data/sn/BIMP_atlas_qw_CentralBrain.h5ad`` (``layers['counts']``) per sample x
``major_cell_type``, plus a pseudo-cell-type ``all`` pooling every nucleus,
over the six libraries ``BIMP_{Queen,Worker}_t{1,2,3}``. Writes
``counts_<type>.csv`` (samples x genes), ``metadata.csv`` and ``gene_desc.csv``
to ``bimp_st.paths.CASTE_DE_PSEUDOBULK_COARSE``. The caste DE fit on them is
``caste_de_blocked_all.py`` (recipe ``bimp_st.caste_de``, ``~ trial + caste``).

    PYTHONPATH=src .venv/bin/python scripts/caste_de/pseudobulk_deseq2.py
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bimp_st.caste_de import fn
from bimp_st.paths import CASTE_DE_PSEUDOBULK_COARSE, SN_CB

warnings.filterwarnings("ignore")

# ── Inputs / outputs ─────────────────────────────────────────────────────────
ATLAS = SN_CB
WORK_DIR = CASTE_DE_PSEUDOBULK_COARSE

# The six central-brain libraries: trial t1/t2/t3 = three separate bees/caste.
SAMPLES = [
    "BIMP_Queen_t1",
    "BIMP_Queen_t2",
    "BIMP_Queen_t3",
    "BIMP_Worker_t1",
    "BIMP_Worker_t2",
    "BIMP_Worker_t3",
]
CASTES = ["Queen"] * 3 + ["Worker"] * 3


def make_pseudobulk(atlas: Path, work_dir: Path) -> None:
    """Sum counts per (sample x cell type) and write one CSV per cell type."""
    import scanpy as sc
    import scipy.sparse as sp

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {atlas}...", flush=True)
    a = sc.read_h5ad(atlas)

    if "counts" not in a.layers:
        # X is normalised; DESeq2 needs the integer counts
        raise SystemExit(f"ERROR: {atlas} has no 'counts' layer")
    X = a.layers["counts"]
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    genes = a.var_names.to_numpy()
    samp = a.obs["sample"].astype(str).values
    mct = a.obs["major_cell_type"].astype(str).values

    missing = sorted(set(SAMPLES) - set(samp))
    if missing:
        raise SystemExit(f"ERROR: atlas is missing expected samples: {missing}")

    meta = pd.DataFrame({"sample": SAMPLES, "caste": CASTES}).set_index("sample")
    meta.to_csv(work_dir / "metadata.csv")

    def desc(g):
        for c in ("description", "gene_name", "Name"):
            if c in a.var.columns:
                return str(a.var.loc[g, c])
        return ""

    pd.Series({g: desc(g) for g in genes}).to_csv(work_dir / "gene_desc.csv")

    for ct in ["all"] + sorted(set(mct)):
        rows = []
        for s in SAMPLES:
            m = (samp == s) if ct == "all" else ((samp == s) & (mct == ct))
            rows.append(np.asarray(X[m].sum(axis=0)).ravel())
        mat = pd.DataFrame(
            np.vstack(rows).astype(int), index=SAMPLES, columns=genes
        )  # samples x genes
        mat.to_csv(work_dir / f"counts_{fn(ct)}.csv")
        print(f"  {ct}: {mat.shape} written", flush=True)
    print("Pseudobulk DONE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--atlas", type=Path, default=ATLAS)
    p.add_argument("--work-dir", type=Path, default=WORK_DIR)
    args = p.parse_args()
    make_pseudobulk(args.atlas, args.work_dir)


if __name__ == "__main__":
    main()
