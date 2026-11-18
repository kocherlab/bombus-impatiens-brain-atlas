"""Build a combined CB+OL reference with the optic lobe's Kenyon cells removed.

Why this exists
---------------
In the canonical combined reference the optic-lobe atlas contributes exactly one
Kenyon-cell entry: a generic ``Kenyon cells`` group of 662 nuclei out of its
20,233. Kenyon cells are a mushroom-body population; 662 of them in an
optic-lobe dissection are central-brain carryover, and the central-brain atlas
already types Kenyon cells properly (Class II / Large-type / Middle-type /
Small-type / Other).

That entry can behave as a sink under Tangram: a small reference entry may
take far more spatial cells than the larger Kenyon entries of the reference.
This reference tests the mapping without it.

Dropping it costs no optic-lobe type: every OL_1..OL_13 cluster, and the glia,
hemocytes and photoreceptors, are untouched. This is the narrow alternative to
mapping against the central brain alone, which would leave the optic lobe with
no types of its own to map to.

Method
------
Filter the built combined object rather than re-concatenating, so the result is
provably the canonical reference minus those nuclei -- same genes, same layers,
same obs columns, same everything else.

Run: PYTHONPATH=src .venv/bin/python \
        scripts/atlas/build_snrna_reference_no_optic_kc.py
"""

from pathlib import Path

import anndata as ad

SN = Path("data/sn")
SRC = SN / "BIMP_atlas_qw_combined.h5ad"
OUT = SN / "BIMP_atlas_qw_combined_noOLKC.h5ad"

DROP_REGION = "OpticLobe"
DROP_TYPE = "Kenyon cells"

a = ad.read_h5ad(SRC)
region = a.obs["region"].astype(str)
ctype = a.obs["cell_type"].astype(str)

drop = (region == DROP_REGION) & (ctype == DROP_TYPE)
n_drop = int(drop.sum())
print(f"source {SRC}: {a.n_obs:,} nuclei x {a.n_vars:,} genes")
print(f"dropping region={DROP_REGION!r} cell_type={DROP_TYPE!r}: {n_drop:,} nuclei")

if n_drop == 0:
    raise SystemExit("nothing matched -- refusing to write an identical copy")

# The generic entry must exist ONLY in the optic lobe. If the central brain
# also carries it, dropping the optic-lobe half would silently keep a
# half-sized sink and the comparison would be meaningless.
cb_generic = int(((region != DROP_REGION) & (ctype == DROP_TYPE)).sum())
if cb_generic:
    raise SystemExit(
        f"{cb_generic} {DROP_TYPE!r} nuclei outside {DROP_REGION} -- this "
        "script assumes the generic entry is optic-lobe-only"
    )

out = a[~drop].copy()
out.obs["cell_type"] = out.obs["cell_type"].astype(str).astype("category")
out.obs["region"] = out.obs["region"].astype(str).astype("category")
out.obs["leiden"] = out.obs["leiden"].astype(str).astype("category")

assert out.n_obs == a.n_obs - n_drop
assert out.n_vars == a.n_vars
assert DROP_TYPE not in set(out.obs["cell_type"].cat.categories)
assert "counts" in out.layers, "raw counts layer lost"
# every optic-lobe cluster survives
ol_before = {c for c in a.obs["cell_type"].astype(str)[region == DROP_REGION]}
ol_after = {
    c
    for c in out.obs["cell_type"].astype(str)[
        out.obs["region"].astype(str) == DROP_REGION
    ]
}
assert ol_before - ol_after == {DROP_TYPE}, ol_before - ol_after

out.write_h5ad(OUT)
print(f"\nwrote {OUT}")
print(f"  {out.n_obs:,} nuclei x {out.n_vars:,} genes")
print(
    f"  cell_type categories: {a.obs['cell_type'].nunique()} -> "
    f"{out.obs['cell_type'].nunique()}"
)
print(f"  regions: {dict(out.obs['region'].value_counts())}")
