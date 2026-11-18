#!/usr/bin/env python
"""Supplementary table: queen vs worker effect sizes by region (spatial arm).

Caste DE by region exists only in the Xenium spatial data: the snRNA libraries carry
no region annotation (and the optic lobe has no replicated caste contrast), so the
n=3 DESeq2 cannot be split by region. This recomputes, with the functions that
draw the regional figure (``scripts/caste_de/si_regional_degs.py``), the
per-region caste test on the 100 panel genes:

* per-section pseudobulk = mean of the stored log-normalised expression over the
  region's cells in that section (sections with >= 5 cells in the region);
* a region needs >= 3 sections per caste;
* log2FC = log2((worker mean + 0.01) / (queen mean + 0.01)), so positive = worker-up
  (the opposite sign convention to the snRNA DESeq2, which is queen/worker);
* slice consistency = rank-biserial r of the queen and worker section means,
  1 - 2 U_queen / (n_queen n_worker) from the Mann-Whitney U statistic: +1 when every
  worker section is above every queen section, -1 for the reverse;
* one brain per caste: sections are pseudoreplicates of one animal, so no P value is
  computed (effect sizes only).

Tables cover ``CASTE_TEST_REGIONS`` (lamina, medulla, lobula, KC-IC, KC-NC, KC-OC).
The antennal lobe and central complex are listed in ``region_availability.csv`` only.

The result is checked against the stored
``results/regional_degs/caste_effects/degs_<region>.csv``
(``si_regional_degs.py``) and written, with the same columns plus ``region`` / ``n``
/ ``test``, to ``results/round4/regional_degs/caste_effects/``.
Nothing in ``results/regional_degs`` is written.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/caste_de/regional_deg_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / "src"),
    str(REPO / "scripts"),
]
import caste_de.si_regional_degs as R  # noqa: E402

from bimp_st.nine_region_common import BASE_REGIONS, attach_region_flags  # noqa: E402

OUT = REPO / "results/round4/regional_degs/caste_effects"
STORED = R.EFFECTS_DIR
TEST = (
    "effect sizes on per-section pseudobulk mean log-normalised expression: "
    "log2FC = log2((worker+0.01)/(queen+0.01)); slice consistency = rank-biserial r "
    "of the section means; no P value (one brain per caste)"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adatas = {}
    for brain, path in R.BRAINS.items():
        print(f"loading {brain}: {path}", flush=True)
        a = ad.read_h5ad(path)
        attach_region_flags(a, regions=BASE_REGIONS)
        adatas[brain] = a

    degs = R.regional_degs(adatas)

    avail = []
    for region in list(R.COMMON_REGIONS) + ["central_complex"]:
        rec = dict(region=region, caste_test_allowed=region in R.CASTE_TEST_REGIONS)
        for brain, a in adatas.items():
            col = f"in_{region}"
            if col not in a.obs.columns:
                rec[f"cells_{brain}"] = np.nan
                rec[f"sections_{brain}"] = np.nan
                continue
            m = a.obs[col].to_numpy(dtype=bool, na_value=False)
            vc = a.obs.loc[m, "slice_name"].astype(str).value_counts()
            rec[f"cells_{brain}"] = int(m.sum())
            rec[f"sections_{brain}"] = int((vc >= R.MIN_CELLS_PER_SLICE).sum())
        avail.append(rec)
    pd.DataFrame(avail).to_csv(OUT / "region_availability.csv", index=False)

    rows, checks = [], []
    for region, df in degs.items():
        if not len(df):
            continue
        d = df.copy()
        d.insert(1, "region", region)
        d["test"] = TEST
        d["n_cells_queen"] = int(
            adatas["queen"]
            .obs[f"in_{region}"]
            .to_numpy(dtype=bool, na_value=False)
            .sum()
        )
        d["n_cells_worker"] = int(
            adatas["worker"]
            .obs[f"in_{region}"]
            .to_numpy(dtype=bool, na_value=False)
            .sum()
        )
        d["pseudoreplicated"] = True
        rows.append(d)
        f = STORED / f"degs_{region}.csv"
        if f.exists():
            s = pd.read_csv(f).set_index("gene")
            m = d.set_index("gene").join(
                s[["log2fc", "slice_consistency"]], rsuffix="_stored", how="inner"
            )
            checks.append(
                dict(
                    region=region,
                    n_genes=len(m),
                    n_stored=len(s),
                    max_abs_dlog2fc=float((m.log2fc - m.log2fc_stored).abs().max()),
                    max_abs_dconsistency=float(
                        (m.slice_consistency - m.slice_consistency_stored).abs().max()
                    ),
                )
            )
    tab = pd.concat(rows, ignore_index=True).rename(
        columns={
            "log2fc": "log2FC_worker_over_queen",
            "n_slices_queen": "n_sections_queen",
            "n_slices_worker": "n_sections_worker",
        }
    )
    cols = [
        "gene",
        "gene_short",
        "region",
        "log2FC_worker_over_queen",
        "slice_consistency",
        "mean_queen",
        "mean_worker",
        "n_sections_queen",
        "n_sections_worker",
        "n_cells_queen",
        "n_cells_worker",
        "test",
        "pseudoreplicated",
    ]
    tab[cols].to_csv(OUT / "supp_table_caste_effects_by_region.csv", index=False)
    pd.DataFrame(checks).to_csv(
        OUT / "reproduction_check_vs_stored_effects.csv", index=False
    )
    print(pd.DataFrame(avail).to_string(index=False))
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == "__main__":
    main()
