"""Persist the restricted-projection numbers the SI null panel needs.

Reads the canonical Tangram run (``bimp_st.paths.CANONICAL_TANGRAM``), whose
posterior columns are named by cell-type identity.  This script's lookups are
all by column name (``MORPH`` list, ``"Kenyon" in c``), so it needs columns
named by identity, not by position.

Writes:

  kc_prob_mass_by_region.csv     share of each cell's Tangram probability mass
                                 sitting on Kenyon-cell reference columns, per
                                 region.
  morph_comp_by_region.csv       composition over the 4 named morphological KC
                                 types after restricting and renormalising, for
                                 all regions including the negative controls
                                 (optic tissue, unannotated tissue).  Neither a
                                 clean failure nor a clean pass -- see
                                 morph_expectation_check.csv.
  morph_meanprob_by_region.csv   same on mean probability rather than argmax,
                                 so the picture cannot be blamed on argmax.
  morph_share_of_kc_mass.csv     mechanism diagnostic: what fraction of each
                                 region's total Kenyon-cell probability mass
                                 (kc_cols) is carried by the 4 MORPH columns
                                 vs. the 2 excluded generic columns ("Kenyon
                                 cells", "Other Kenyon cells").  Inside the KC
                                 compartments the 4 MORPH columns hold a
                                 minority of the Kenyon mass and the generic
                                 "Kenyon cells" column (mostly calyx) holds the
                                 rest, so the restricted composition is built
                                 from a minority of the Kenyon-cell signal.
  morph_expectation_check.csv    the literature expectation check.
                                 argmax_pass/meanprob_pass only check which of
                                 the 3 KC compartments each type peaks in, so
                                 they cannot tell anatomical enrichment from
                                 noise that happens to rank the compartments
                                 correctly.  control_argmax_pass/
                                 control_meanprob_pass additionally require the
                                 expected compartment's value to exceed the
                                 best negative control (see the comment in the
                                 code for which controls) -- the same standard
                                 panel A uses.

Run:
  PYTHONPATH=src:scripts .venv/bin/python \\
      scripts/kenyon/restricted_projection_numbers.py
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

OUT = REPO / "results" / "fig5_kc"
OUT.mkdir(parents=True, exist_ok=True)

MORPH = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
]
SHORT = {
    "Large-type Kenyon cells": "Large",
    "Middle-type Kenyon cells": "Middle",
    "Small-type Kenyon cells": "Small",
    "Class II Kenyon cells": "Class II",
}
ORDER = ["kc_ic", "kc_nc", "kc_oc", "medulla", "lamina", "lobula", "none"]

print("loading ...", flush=True)
a = ad.read_h5ad(CANONICAL_TANGRAM, backed="r")
obs = a.obs.reset_index(drop=True)
P = a.obsm["tangram_ct_pred"].reset_index(drop=True)
del a

flags = ["in_kc_ic", "in_kc_nc", "in_kc_oc", "in_medulla", "in_lamina", "in_lobula"]
F = np.column_stack([obs[f].values.astype(bool) for f in flags])
nflag = F.sum(1)
reg = np.full(len(obs), "none", dtype=object)
for i, f in enumerate(flags):
    reg[(nflag == 1) & F[:, i]] = f[3:]
reg[nflag > 1] = "multi"
obs["reg"] = reg

kc_cols = [c for c in P.columns if "Kenyon" in c]
generic_cols = [c for c in kc_cols if c not in MORPH]
tot = P.sum(1)
mass = P[kc_cols].sum(1) / tot
mm = (
    pd.DataFrame({"reg": obs["reg"].values, "mass": mass.values})
    .groupby("reg")["mass"]
    .describe()
    .reindex(ORDER)[["count", "mean", "25%", "50%", "75%"]]
)
mm.to_csv(OUT / "kc_prob_mass_by_region.csv")
print("\nKC probability mass by region:\n", mm.round(4).to_string(), flush=True)

# Mechanism diagnostic: how much of that KC mass is even eligible to enter the
# 4-type restricted-and-renormalised composition below?  (generic_cols =
# "Kenyon cells" + "Other Kenyon cells", excluded from MORPH by construction.)
morph_mass = P[MORPH].sum(1) / tot
generic_mass = P[generic_cols].sum(1) / tot
share = pd.DataFrame(
    {
        "reg": obs["reg"].values,
        "morph_mass": morph_mass.values,
        "generic_mass": generic_mass.values,
        "kc_mass": mass.values,
    }
)
share = (
    share.groupby("reg")[["morph_mass", "generic_mass", "kc_mass"]]
    .mean()
    .reindex(ORDER)
)
share["morph_share_of_kc"] = share["morph_mass"] / share["kc_mass"]
share["generic_share_of_kc"] = share["generic_mass"] / share["kc_mass"]
share.to_csv(OUT / "morph_share_of_kc_mass.csv")
print(
    "\nMORPH share of total Kenyon-cell mass by region:\n",
    share.round(4).to_string(),
    flush=True,
)

Pm = P[MORPH]
Pm_n = Pm.div(Pm.sum(1), axis=0)
lab = np.array([SHORT[c] for c in Pm_n.idxmax(1).values])
sh = [SHORT[c] for c in MORPH]

comp = (pd.crosstab(obs["reg"], lab, normalize="index") * 100).reindex(ORDER)[sh]
comp.to_csv(OUT / "morph_comp_by_region.csv")
print(
    "\n4-morph argmax composition by region (%):\n",
    comp.round(2).to_string(),
    flush=True,
)

mp = Pm_n.groupby(obs["reg"].values).mean().rename(columns=SHORT)
mp = mp.reindex(ORDER)[sh]
mp.to_csv(OUT / "morph_meanprob_by_region.csv")
print("\n4-morph mean probability by region:\n", mp.round(4).to_string(), flush=True)

# the literature expectation, machine-readable so the figure marks it rather
# than hardcoding a claim
exp = {"Large": "kc_nc", "Middle": "kc_ic", "Small": "kc_ic", "Class II": "kc_oc"}
rows = []
kc3 = ["kc_ic", "kc_nc", "kc_oc"]
# Adjudicating controls are restricted to medulla and unannotated tissue:
# lamina and especially lobula carry little KC mass
# (kc_prob_mass_by_region.csv), so their restricted-and-renormalised
# composition is close to renormalised noise -- a weak signal amplified by
# a near-empty denominator. medulla and unannotated tissue hold more KC
# mass and many more cells, and are the controls a percentage-point
# comparison can be made against.
neg = ["medulla", "none"]
for t in sh:
    own_am, own_mp = comp.loc[exp[t], t], mp.loc[exp[t], t]
    negmax_am, negmax_mp = comp.loc[neg, t].max(), mp.loc[neg, t].max()
    rows.append(
        dict(
            type=t,
            expected_peak=exp[t],
            argmax_peak=comp.loc[kc3, t].idxmax(),
            meanprob_peak=mp.loc[kc3, t].idxmax(),
            argmax_pass=comp.loc[kc3, t].idxmax() == exp[t],
            meanprob_pass=mp.loc[kc3, t].idxmax() == exp[t],
            # Control-aware: does the expected compartment also beat the best
            # negative control (not just the other 2 KC compartments)?  Same
            # standard panel A already applies.
            own_argmax=own_am,
            best_neg_argmax=negmax_am,
            best_neg_argmax_region=comp.loc[neg, t].idxmax(),
            argmax_gap_vs_controls=own_am - negmax_am,
            control_argmax_pass=own_am > negmax_am,
            own_meanprob=own_mp,
            best_neg_meanprob=negmax_mp,
            best_neg_meanprob_region=mp.loc[neg, t].idxmax(),
            meanprob_gap_vs_controls=own_mp - negmax_mp,
            control_meanprob_pass=own_mp > negmax_mp,
        )
    )
ex = pd.DataFrame(rows)
ex.to_csv(OUT / "morph_expectation_check.csv", index=False)
print(
    "\nliterature expectation check (3-compartment peak only):\n",
    ex[
        [
            "type",
            "expected_peak",
            "argmax_peak",
            "meanprob_peak",
            "argmax_pass",
            "meanprob_pass",
        ]
    ].to_string(index=False),
    flush=True,
)
print(
    f"\n  PASS on argmax: {int(ex.argmax_pass.sum())}/4 ; "
    f"on mean prob: {int(ex.meanprob_pass.sum())}/4",
    flush=True,
)
print(
    "\ncontrol-aware check (expected compartment must also beat the best "
    "negative control):\n",
    ex[
        [
            "type",
            "best_neg_argmax_region",
            "argmax_gap_vs_controls",
            "control_argmax_pass",
            "best_neg_meanprob_region",
            "meanprob_gap_vs_controls",
            "control_meanprob_pass",
        ]
    ]
    .round(4)
    .to_string(index=False),
    flush=True,
)
print(
    f"\n  CONTROL-AWARE PASS on argmax: {int(ex.control_argmax_pass.sum())}/4 ; "
    f"on mean prob: {int(ex.control_meanprob_pass.sum())}/4",
    flush=True,
)
