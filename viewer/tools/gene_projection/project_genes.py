"""Re-project a WIDER reference gene set through the canonical Tangram mapping.

The canonical mapping (paths.CANONICAL_TANGRAM, g2_R2_s42) ran in clusters mode, so
tangram2.project_genes computed, per spatial cell i and gene g,

    imputed[i, g] = sum_t M[t, i] * C[t, g]

with C[t, g] = SUM over nuclei of type t of normalize_total(counts)[., g]
(adata_to_cluster_expression(scale=True) sums, it does not average). The mapping
matrix M was never written; the tangram object stores only its row-normalised
transpose, obsm['tangram_ct_pred'] = M.T / rowsum. The per-cell scale lost there
is recovered from the canonical imputed object (which IS M.T @ C on its genes)
by least squares, and the reconstruction is required to reproduce every one of
those columns before any new gene is projected.

Step 3 of the atlas gene projection (see README.md). Writes a NEW file
(default ATLAS_V2_IMPUTED below) and refuses to overwrite; never touches the
existing imputed objects, which the analyses read. The per-cell
scale is kept as obs['tangram_map_scale'], the provenance as uns['provenance'].

    PYTHONPATH=src:scripts .venv/bin/python \
        viewer/tools/gene_projection/project_genes.py \
        --genes viewer/tools/gene_projection/genes_atlas_v2.txt \
        [--out PATH] [--check-only]
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

from bimp_st.paths import CANONICAL_TANGRAM, SN_COMBINED

REF = SN_COMBINED
CANON_IMPUTED = CANONICAL_TANGRAM.with_name(
    CANONICAL_TANGRAM.name.replace("_tangram.h5ad", "_imputed.h5ad")
)
#: The widened projection the web atlas reads (2026-10-16). Named
#: after the mapping it came from, per the tag-records-provenance convention.
ATLAS_V2_IMPUTED = CANONICAL_TANGRAM.with_name(
    CANONICAL_TANGRAM.name.replace("_tangram.h5ad", "_imputed_atlas_v2.h5ad")
)
CLUSTER_LABEL = "cell_type"
BLOCK = 100_000

ap = argparse.ArgumentParser()
ap.add_argument(
    "--genes", type=Path, required=True, help="one reference var name per line"
)
ap.add_argument(
    "--rule",
    default=(
        "reference genes detected (counts>0) in >=1% of nuclei of at least "
        "one of the 29 "
        "cell_type groups, plus all BIMPv4.3_neuropeptides_MALDI.tsv "
        "precursors and Xenium "
        "panel genes present in the reference (select_genes.py)"
    ),
    help="gene-selection rule, recorded in uns",
)
ap.add_argument("--out", type=Path, default=ATLAS_V2_IMPUTED)
ap.add_argument("--check-only", action="store_true")
args = ap.parse_args()
assert args.check_only or not args.out.exists(), f"refusing to overwrite {args.out}"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def _str(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def _col(f, path):
    """obs column as str array; handles categorical groups and plain datasets."""
    return np.asarray(ad.io.read_elem(f[path])).astype(str)


# ---- mapping posterior (row-normalised M.T), named columns --------------------
log("canonical tangram:", CANONICAL_TANGRAM)
with h5py.File(CANONICAL_TANGRAM, "r") as f:
    g = f["obsm/tangram_ct_pred"]
    cols = list(_str(g.attrs["column-order"]))
    P = np.stack([g[c][:] for c in cols], axis=1).astype(np.float64)
    tg_obs = _str(f["obs"][f["obs"].attrs["_index"]][:])
    tg_cell = _col(f, "obs/cell_id") if "cell_id" in f["obs"] else None
log("posterior", P.shape, "row sums", P.sum(1).min(), P.sum(1).max())

# ---- canonical imputed object (M.T @ C on its own gene set) -------------------
log("canonical imputed:", CANON_IMPUTED)
fi = h5py.File(CANON_IMPUTED, "r")
im_obs = _str(fi["obs"][fi["obs"].attrs["_index"]][:])
im_var = list(_str(fi["var"][fi["var"].attrs["_index"]][:]))
im_cell = _col(fi, "obs/cell_id") if "cell_id" in fi["obs"] else None
assert len(im_obs) == len(tg_obs) == P.shape[0], (len(im_obs), len(tg_obs), P.shape)
assert (im_obs == tg_obs).all(), "obs order differs between tangram and imputed objects"
if tg_cell is not None and im_cell is not None:
    assert (tg_cell == im_cell).all(), "cell_id order differs"
n = P.shape[0]
log("n cells", n, "imputed genes", len(im_var))

# ---- reference cluster expression, exactly as tangram_mapping.train ----------
log("reference:", REF)
ref = ad.read_h5ad(REF)
ref.X = ref.layers["counts"].copy()
del ref.layers["counts"]
sc.pp.normalize_total(ref)  # over ALL genes, before any subsetting
lower = [v.lower() for v in ref.var_names]
assert len(set(lower)) == len(lower)
ct = ref.obs[CLUSTER_LABEL].astype(str).values
X = ref.X.tocsr()
C = np.zeros((len(cols), X.shape[1]))
for k, t in enumerate(cols):
    m = ct == t
    assert m.any(), f"type {t} absent from reference"
    C[k] = np.asarray(X[m].sum(axis=0), dtype=np.float32).ravel()
lpos = {g: i for i, g in enumerate(lower)}
log("cluster sums", C.shape)

# ---- recover per-cell scale, then require exact reproduction ------------------
Cold = C[:, [lpos[g] for g in im_var]]
scale = np.empty(n)
max_rel = 0.0
worst = []
for r in range(0, n, BLOCK):
    Y = fi["X"][r : r + BLOCK].astype(np.float64)
    Q = P[r : r + BLOCK] @ Cold
    s = (Y * Q).sum(1) / np.maximum((Q * Q).sum(1), 1e-300)
    scale[r : r + BLOCK] = s
    R = Y - s[:, None] * Q
    rel = np.abs(R).max(1) / np.maximum(np.abs(Y).max(1), 1e-300)
    max_rel = max(max_rel, float(rel.max()))
    worst.append(float(np.percentile(rel, 99.9)))
fi.close()
log(f"scale: min {scale.min():.4g} median {np.median(scale):.4g} max {scale.max():.4g}")
log(
    f"reproduction of {len(im_var)} canonical imputed genes: max rel err {max_rel:.3g} "
    f"(99.9th pct per block max {max(worst):.3g})"
)
if max_rel > 1e-4:
    sys.exit(
        "ABORT: reconstruction does not reproduce the canonical imputed "
        f"object (max rel {max_rel})"
    )
if args.check_only:
    sys.exit(0)

# ---- project the widened gene list ------------------------------------------
genes = [g.strip().lower() for g in args.genes.read_text().split() if g.strip()]
missing = [g for g in genes if g not in lpos]
assert not missing, f"{len(missing)} genes not in reference: {missing[:5]}"
genes = list(dict.fromkeys(genes))
nz = X.getnnz(axis=0) > 0
genes = [g for g in genes if nz[lpos[g]]]  # project_genes drops all-zero genes
Cnew = C[:, [lpos[g] for g in genes]]
log("projecting", len(genes), "genes")

with h5py.File(CANON_IMPUTED, "r") as f0:
    obs = ad.io.read_elem(f0["obs"])
obs["tangram_map_scale"] = scale
training = set(
    json.loads(
        CANON_IMPUTED.with_name(
            CANON_IMPUTED.name.replace("_imputed.h5ad", "_gene_split.json")
        ).read_text()
    )["training_genes"]
)
var = pd.DataFrame(index=pd.Index(genes))
var["is_training"] = [g in {t.lower() for t in training} for g in genes]
var["in_canonical_imputed"] = var.index.isin(im_var)

Xout = np.empty((n, len(genes)), dtype=np.float32)
for r in range(0, n, BLOCK):
    Xout[r : r + BLOCK] = (scale[r : r + BLOCK, None] * P[r : r + BLOCK]) @ Cnew
log("projected")

out = ad.AnnData(X=Xout, obs=obs, var=var)
out.uns["provenance"] = json.dumps(
    {
        "mapping": str(CANONICAL_TANGRAM),
        "scale_recovered_from": str(CANON_IMPUTED),
        "reference": str(REF),
        "cluster_label": CLUSTER_LABEL,
        "projection": "imputed = (scale * tangram_ct_pred) @ "
        "cluster_sum(normalize_total(counts))",
        "reproduction_max_rel_err_on_canonical_genes": max_rel,
        "gene_rule": args.rule,
        "n_genes": len(genes),
        "date": str(date.today()),
        "script": "viewer/tools/gene_projection/project_genes.py",
    }
)
out.write_h5ad(args.out)
log("wrote", args.out, out.shape)
