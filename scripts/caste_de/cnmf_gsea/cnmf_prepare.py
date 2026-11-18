#!/usr/bin/env python
"""Step 1: joint cNMF prepare (clean notebook cell 21).

Source: 01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb cell 21
(md5 184efd9af8192fd09f7329f1eeaa9f17), function ``prepare_joint_cnmf``.

All central-brain nuclei except photoreceptors; genes detected in >= 10
nuclei; ``cnmf prepare`` with k = 30, 40, 50, 60, 70, 10 iterations, 3,000
overdispersed genes, seed 14. Writes ``pipeline/cnmf_joint/joint.counts.h5ad``,
the cNMF tree ``pipeline/cnmf_joint/joint/`` and ``run_joint.sh``, the
factorize / combine / k_selection_plot script that ``run_pipeline.sh
factorize`` then runs as one Slurm job (16 workers). The script's text is the
notebook's; ``LD_LIBRARY_PATH`` points at this env's lib/ instead of the
notebook's conda env.

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/cnmf_prepare.py
"""

import os
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import scanpy as sc
from cnmf import cNMF
from cnmf_gsea_common import CT, load_atlas, setup
from cnmf_gsea_paths import CNMF_NAME, REPO_CNMF

from bimp_st.paths import SN_CB

K_RANGE = np.array([30, 40, 50, 60, 70])
N_ITER = 10
NHVG = 3000
N_WORKERS = 16
SEED = 14


def counts_without_photoreceptors(adata) -> sc.AnnData:
    """Raw counts of every nucleus except photoreceptors."""
    ct = adata.obs[CT].astype(str).str.replace("\n", " ", regex=False)
    keep = (ct != "Photoreceptors").values
    if "counts" in adata.layers:
        return sc.AnnData(
            adata.layers["counts"][keep],
            obs=adata.obs[keep].copy(),
            var=adata.var.copy(),
        )
    if adata.raw is not None:
        return adata.raw.to_adata()[keep].copy()
    raise ValueError("no raw counts found")


def run_script(out_j: Path) -> str:
    """Cell 21's shell script: factorize with N_WORKERS workers, combine, plot."""
    cnmf_bin = shutil.which("cnmf") or os.path.join(
        os.path.dirname(sys.executable), "cnmf"
    )
    env_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    return textwrap.dedent(f"""\
        #!/bin/bash
        export LD_LIBRARY_PATH={env_lib}:$LD_LIBRARY_PATH
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        CNMF={cnmf_bin}; OUT={out_j}; N={N_WORKERS}; name={CNMF_NAME}
        seq 0 $((N-1)) | xargs -P $N -I{{}} $CNMF factorize --output-dir $OUT \
--name $name --worker-index {{}} --total-workers $N
        $CNMF combine --output-dir $OUT --name $name
        $CNMF k_selection_plot --output-dir $OUT --name $name
        echo ALL DONE
        """)


def prepare_joint_cnmf(adata, out_j: Path) -> Path:
    """Write the counts matrix, run ``cnmf prepare`` and write run_joint.sh."""
    out_j.mkdir(parents=True, exist_ok=True)
    a = counts_without_photoreceptors(adata)
    sc.pp.filter_genes(a, min_cells=10)
    fn = out_j / "joint.counts.h5ad"
    a.write_h5ad(fn)
    print(a.n_obs, "cells,", a.n_vars, "genes")
    cNMF(output_dir=str(out_j), name=CNMF_NAME).prepare(
        counts_fn=str(fn),
        components=K_RANGE,
        n_iter=N_ITER,
        seed=SEED,
        num_highvar_genes=NHVG,
    )
    sh = out_j / "run_joint.sh"
    sh.write_text(run_script(out_j))
    sh.chmod(0o755)
    print("factorize with: bash", sh)
    return sh


def main() -> None:
    setup()
    adata, _, _ = load_atlas(SN_CB)
    prepare_joint_cnmf(adata, REPO_CNMF)


if __name__ == "__main__":
    main()
