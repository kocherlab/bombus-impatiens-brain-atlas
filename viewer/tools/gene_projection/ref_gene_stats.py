"""Per-cell-type detection fraction of every snRNA reference gene.

Step 1 of the atlas gene projection (see README.md). For each of the 29
`cell_type` groups of paths.SN_COMBINED, the fraction of nuclei with a nonzero
raw count, plus the whole-reference fraction and the maximum over types
(`max_type_frac`, the column the gene rule thresholds).

    PYTHONPATH=src:scripts .venv/bin/python \
        viewer/tools/gene_projection/ref_gene_stats.py [OUT_CSV]
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from bimp_st.paths import RESULTS, SN_COMBINED

OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else RESULTS / "gene_projection" / "ref_gene_stats.csv"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

a = ad.read_h5ad(SN_COMBINED)
C = a.layers["counts"].tocsr()
ct = a.obs["cell_type"].astype(str).values
types = sorted(set(ct))
B = (C > 0).astype(np.float32)
df = pd.DataFrame(
    {"frac_" + t: np.asarray(B[ct == t].mean(axis=0)).ravel() for t in types},
    index=a.var_names,
)
type_cols = list(df.columns)
df["frac_all"] = np.asarray(B.mean(axis=0)).ravel()
df["n_cells"] = np.asarray((C > 0).sum(axis=0)).ravel()
df["highly_variable"] = a.var["highly_variable"].values
df["max_type_frac"] = df[type_cols].max(axis=1)
df.to_csv(OUT)
print(pd.Series(ct).value_counts().to_string())
print("written", OUT, df.shape)
