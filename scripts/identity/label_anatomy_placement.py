"""Where the canonical Kenyon and Optic lobe labels lie relative to anatomy.

Placement statistics of the canonical labelling, all computed with one
labelling (the canonical cluster vote, plus the per-cell Tangram call for
comparison) and one flag set (nine-region flags):

0. Kenyon cells against the Kenyon compartments (Small-type, Large-type and
   Class II together): precision (labelled cells inside a compartment),
   recall (compartment cells carrying the label) and F1, over all cells and,
   as in results/reframe/celltype_anatomy/, over the cells that lie in exactly
   one traced region.

1. Of the cells whose cluster vote is *Optic lobe cells*, the share that sits
   inside the outlined lamina, medulla or lobula, inside some other traced
   region, or in no traced region at all (nine-region flags, joined on
   slice_name + cell_id; untraced counts as outside).
2. The number of panel genes the canonical Tangram cell type mapping was
   trained on (``uns['training_genes']``) against the genes shared with the
   reference (``uns['overlap_genes']``).

Inputs: ``paths.CANONICAL_TANGRAM`` (obs slice_name / cell_id / leiden, uns),
results/round6/vote_stats/per_cell_labels_canonical.parquet (the canonical vote,
row-aligned with the canonical object), results/nine_region/region_flags.parquet
via ``nine_region_common.flags_table``.

Output: results/handoff/label_placement/summary.csv (name, value)

Run
    PYTHONPATH=src:scripts .venv/bin/python scripts/identity/label_anatomy_placement.py
"""

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from bimp_st.cluster_labels import get_pooled
from bimp_st.nine_region_common import (
    STATISTICAL_REGIONS,
    flags_table,
    kc_mask,
    optic_mask,
    region_mask,
    region_of,
    traced_mask,
)
from bimp_st.paths import CANONICAL_TANGRAM, RESULTS

OUT = RESULTS / "handoff" / "label_placement"
VOTE = RESULTS / "round6" / "vote_stats" / "per_cell_labels_canonical.parquet"
OPTIC = ["lamina", "medulla", "lobula"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(CANONICAL_TANGRAM, "r") as h:
        obs = read_elem(h["obs"])[
            ["slice_name", "cell_id", "leiden", "tangram_cell_type"]
        ]
        n_train = len(read_elem(h["uns"]["training_genes"]))
        n_overlap = len(read_elem(h["uns"]["overlap_genes"]))
    v = pd.read_parquet(VOTE)
    assert len(v) == len(obs)
    assert (
        v.slice_name.astype(str).to_numpy() == obs.slice_name.astype(str).to_numpy()
    ).all()
    assert (v.leiden.astype(str).to_numpy() == obs.leiden.astype(str).to_numpy()).all()

    t = flags_table()
    t = (
        t.assign(_k=t.slice_name.astype(str) + "||" + t.cell_id.astype(str))
        .drop_duplicates("_k")
        .set_index("_k")
    )
    idx = t.index.get_indexer(
        obs.slice_name.astype(str) + "||" + obs.cell_id.astype(str)
    )
    assert (idx >= 0).all(), int((idx < 0).sum())
    regions = [
        r for r in STATISTICAL_REGIONS + ["central_complex"] if f"in_{r}" in t.columns
    ]
    F = pd.DataFrame(
        {
            f"in_{r}": pd.array(t[f"in_{r}"].to_numpy()[idx], dtype="boolean")
            for r in regions
        }
    )
    # The optic and Kenyon flags are traced on every section (traced_mask raises
    # otherwise). Only the central complex has nulls (untraced on queen 0040240):
    # "in no traced region" counts a cell as inside only where it is known to be.
    optic = traced_mask(optic_mask(F))
    anyreg = region_mask(F, regions).to_numpy(dtype=bool, na_value=False)
    ol = (v.vote.astype(str) == "Optic lobe").to_numpy()
    calyx = traced_mask(kc_mask(F))
    one = np.isin(region_of(F, STATISTICAL_REGIONS), STATISTICAL_REGIONS)
    percell_kc = np.array(
        [get_pooled(x) == "Kenyon cells" for x in obs.tangram_cell_type.astype(str)]
    )
    rows = []
    for lab, kc in (
        ("vote", (v.vote.astype(str) == "Kenyon cells").to_numpy()),
        ("percell", percell_kc),
    ):
        for scope, m in (("all", np.ones(len(kc), bool)), ("one_region", one)):
            p = calyx[m & kc].mean()
            r = kc[m & calyx].mean()
            rows += [
                (f"kc_{lab}_{scope}_precision_pct", 100 * p),
                (f"kc_{lab}_{scope}_recall_pct", 100 * r),
                (f"kc_{lab}_{scope}_f1", 2 * p * r / (p + r)),
            ]
    rows += [
        ("optic_vote_cells", int(ol.sum())),
        ("optic_vote_pct_of_atlas", 100 * ol.mean()),
        (
            "optic_vote_pct_in_lamina_medulla_lobula",
            100 * (ol & optic).sum() / ol.sum(),
        ),
        (
            "optic_vote_pct_in_other_region_only",
            100 * (ol & anyreg & ~optic).sum() / ol.sum(),
        ),
        ("optic_vote_pct_in_no_region", 100 * (ol & ~anyreg).sum() / ol.sum()),
        (
            "optic_vote_pct_outside_lamina_medulla_lobula",
            100 * (ol & ~optic).sum() / ol.sum(),
        ),
        ("tangram_training_genes", n_train),
        ("tangram_overlap_genes", n_overlap),
    ]
    df = pd.DataFrame(rows, columns=["name", "value"])
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
