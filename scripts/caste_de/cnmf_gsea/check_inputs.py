#!/usr/bin/env python
"""Check the repo snRNA atlas against the one the saved analysis read.

Compares a content fingerprint of ``bimp_st.paths.SN_CB`` (counts layer, X,
obs_names, var_names, whitespace-normalized cell_type, caste, trial) with the
fingerprint of the source ``BIMP_atlas_adata_qw_CentralBrain.h5ad`` stored by
``freeze_inputs.py`` as ``atlas_fingerprint.json`` in ``cnmf_gsea_paths.SAVED``.
Writes ``inputs/input_check.csv`` (the saved side in column ``saved``) and exits
non-zero on any difference. Two projection-neuron labels in the source carry a
line break (``Anterodorsal\\n projection neurons``) where the repo copy has a
space; the notebook normalizes whitespace before every use, so cell_type is
compared normalized. (The two files differ by 128 bytes of HDF5 container
metadata.)

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/check_inputs.py
"""

import hashlib
import json
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cnmf_gsea_paths import INPUTS, SAVED

from bimp_st.paths import SN_CB

INP = INPUTS


def array_md5(*arrays) -> str:
    """md5 over the raw bytes of one or more arrays."""
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def fingerprint(adata) -> dict:
    """Content hash of an snRNA atlas: counts, X, names, labels."""

    def mat(m):
        """md5 of a matrix in canonical CSR form."""
        m = sp.csr_matrix(m)
        m.sort_indices()
        return array_md5(m.data, m.indices, m.indptr)

    norm = [re.sub(r"\s+", " ", s) for s in adata.obs.cell_type.astype(str)]
    return {
        "shape": list(adata.shape),
        "counts": mat(adata.layers["counts"]),
        "X": mat(adata.X),
        "obs_names": array_md5(np.array(adata.obs_names, dtype="U")),
        "var_names": array_md5(np.array(adata.var_names, dtype="U")),
        "cell_type_whitespace_normalized": array_md5(np.array(norm, dtype="U")),
        "caste": array_md5(np.array(adata.obs.caste.astype(str), dtype="U")),
        "trial": array_md5(np.array(adata.obs.trial.astype(str), dtype="U")),
    }


def main() -> None:
    saved = json.loads((SAVED / "atlas_fingerprint.json").read_text())
    ours = fingerprint(ad.read_h5ad(SN_CB))
    tab = pd.DataFrame(
        [(k, str(ours[k]), str(saved[k]), ours[k] == saved[k]) for k in saved],
        columns=["field", "repo", "saved", "identical"],
    )
    tab.to_csv(INP / "input_check.csv", index=False)
    print(tab.to_string(index=False))
    if not tab.identical.all():
        sys.exit("repo atlas content differs from the saved analysis's: stop")


if __name__ == "__main__":
    main()
