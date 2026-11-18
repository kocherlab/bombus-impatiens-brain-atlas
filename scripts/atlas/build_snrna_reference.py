"""Build the combined CB+OL snRNA reference for Tangram.

Concatenates I. Traniello's Central Brain + Optic Lobe annotated atlases on their
shared gene set (join="inner"), keeping cell_type per-region and region-namespacing
the Leiden labels. Self-contained (no bimp_st import) so it runs under a partial venv.

Run: .venv/bin/python scripts/atlas/build_snrna_reference.py
"""

from pathlib import Path

import anndata as ad

SN = Path("data/sn")
CB = SN / "BIMP_atlas_qw_CentralBrain.h5ad"
OL = SN / "BIMP_atlas_qw_OpticLobe.h5ad"
OUT = SN / "BIMP_atlas_qw_combined.h5ad"


def _norm_ct(name: str) -> str:
    """Strip embedded newlines / collapse whitespace in a cell_type label."""
    return " ".join(str(name).replace("\n", " ").split())


def load(path: Path, region: str, leiden_col: str, prefix: str):
    a = ad.read_h5ad(path)
    a.obs["region"] = region
    a.obs["cell_type"] = [_norm_ct(x) for x in a.obs["cell_type"].astype(str)]
    a.obs["leiden"] = [f"{prefix}_{v}" for v in a.obs[leiden_col].astype(str)]
    return a


cb = load(CB, "CentralBrain", "leiden_res0_60", "CB")
ol = load(OL, "OpticLobe", "leiden_res0_40", "OL")

comb = ad.concat([cb, ol], join="inner", label=None, index_unique=None)
comb.obs["region"] = comb.obs["region"].astype("category")
comb.obs["cell_type"] = comb.obs["cell_type"].astype("category")
comb.obs["leiden"] = comb.obs["leiden"].astype("category")

# sanity
assert "counts" in comb.layers, "raw counts layer lost in concat"
assert comb.obs["leiden"].astype(str).str.startswith(("CB_", "OL_")).all()
assert all("\n" not in c for c in comb.obs["cell_type"].cat.categories)

comb.write_h5ad(OUT)
print(
    f"wrote {OUT}\n  {comb.n_obs:,} cells x {comb.n_vars:,} genes\n"
    f"  regions: {dict(comb.obs['region'].value_counts())}\n"
    f"  cell_type categories ({comb.obs['cell_type'].nunique()}): "
    f"{sorted(comb.obs['cell_type'].cat.categories.tolist())}"
)
