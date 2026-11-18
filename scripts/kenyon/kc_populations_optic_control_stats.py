"""Matched effect-size comparison: the restricted five-population KC compartment
x call table (`fig:kc_subtypes` C) vs the
optic-neuropil negative control's, plus a within-KC permutation floor.

Reads:
  results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad   (`fig:kc_subtypes` C)
  results/reframe/kc5pop_negative_control/optic_negcontrol_calls.parquet
Writes:
  results/reframe/kc5pop_negative_control/negcontrol_comparison.txt
  results/reframe/kc5pop_negative_control/{kc,optic}_compartment_x_call.csv
"""

import faulthandler
from pathlib import Path

faulthandler.enable()

import sys

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

KC_H5 = REPO / "results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad"
OUTDIR = REPO / "results/reframe/kc5pop_negative_control"
OL_PQ = OUTDIR / "optic_negcontrol_calls.parquet"

CLUSTERS = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
]
SHORT = {
    "Large-type Kenyon cells": "Large",
    "Middle-type Kenyon cells": "Middle",
    "Small-type Kenyon cells": "Small",
    "Class II Kenyon cells": "ClassII",
    "Other Kenyon cells": "Other",
}
SHORTS = [SHORT[c] for c in CLUSTERS]


def read_obs_col(f, name):
    d = f["obs"][name]
    if isinstance(d, h5py.Group):
        codes = d["codes"][:]
        cats = [x.decode() if isinstance(x, bytes) else x for x in d["categories"][:]]
        return np.array([cats[i] if i >= 0 else "nan" for i in codes])
    a = d[:]
    if a.dtype.kind == "S" or a.dtype.kind == "O":
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def cramers_v(tab):
    """Cramer's V on a raw contingency table (counts). Sample-size normalised
    effect size, directly comparable between tables of the same shape."""
    tab = np.asarray(tab, dtype=float)
    n = tab.sum()
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0))
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def summarise(name, comp, call, order_rows, lines):
    tab = pd.crosstab(pd.Series(comp, name="compartment"), pd.Series(call, name="call"))
    tab = tab.reindex(index=order_rows, columns=CLUSTERS).fillna(0).astype(int)
    tab.columns = SHORTS
    frac = tab.div(tab.sum(1), axis=0)
    v = cramers_v(tab.values)
    dom = frac.idxmax(axis=1)
    domshare = frac.max(axis=1)
    lines.append(f"\n=== {name}  (n={tab.values.sum():,}) ===")
    lines.append("row % (each row sums to 100):")
    lines.append((100 * frac).round(1).to_string())
    lines.append("row n: " + ", ".join(f"{r} {tab.loc[r].sum():,}" for r in tab.index))
    lines.append(
        "dominant call per compartment: "
        + ", ".join(f"{r} -> {dom[r]} ({100 * domshare[r]:.1f}%)" for r in tab.index)
    )
    lines.append(
        f"distinct dominant calls across the 3 compartments: "
        f"{dom.nunique()} / 3  (3 = one-to-one permutation matrix)"
    )
    lines.append(f"Cramer's V (compartment x call) = {v:.4f}")
    return tab, v, dom


lines = [
    "Negative control for the doubly-restricted 5-KC-population projection "
    "(fig:kc_subtypes panel C).",
    "Control run: scripts/kenyon/project_kc_populations_optic_control.py "
    "(query = optic neuropils; reference, marker recipe, d_source, "
    "epochs, lr and seed identical to the panel C run).",
    "Run twice, on GPU and on CPU, to check that the device does not change "
    "the calls.",
]

# ---- restricted five-population run (`fig:kc_subtypes` C) -----------------
with h5py.File(KC_H5, "r") as f:
    kc_call = read_obs_col(f, "kc5pop_call")
    ic = read_obs_col(f, "in_kc_ic").astype(bool)
    nc = read_obs_col(f, "in_kc_nc").astype(bool)
    kc_brain = read_obs_col(f, "brain")
# identical cascade to kc_subtypes_data.load_kc5pop()
kc_comp = np.where(ic, "KC-IC", np.where(nc, "KC-NC", "KC-OC"))
kc_rows = ["KC-IC", "KC-NC", "KC-OC"]
kc_tab, kc_v, kc_dom = summarise(
    "Kenyon compartments (real query)", kc_comp, kc_call, kc_rows, lines
)

# ---- Negative control -----------------------------------------------------
ol = pd.read_parquet(OL_PQ)
ol_rows = ["Lamina", "Lobula", "Medulla"]
ol_tab, ol_v, ol_dom = summarise(
    "NEGATIVE CONTROL: optic neuropils (no Kenyon cells present)",
    ol["compartment"].values,
    ol["call"].values,
    ol_rows,
    lines,
)

# ---- Permutation floor ----------------------------------------------------
rng = np.random.default_rng(0)
floor = []
for _ in range(200):
    floor.append(
        cramers_v(
            pd.crosstab(pd.Series(rng.permutation(kc_comp)), pd.Series(kc_call))
            .reindex(index=kc_rows, columns=CLUSTERS)
            .fillna(0)
            .values
        )
    )
floor = np.array(floor)
lines.append(
    "\n=== permutation floor (compartment labels shuffled within the "
    "KC run, 200 draws) ==="
)
lines.append(
    f"Cramer's V: mean {floor.mean():.5f}  max {floor.max():.5f}  (chance level)"
)

# ---- Caste replication of the control -------------------------------------
lines.append("\n=== negative control by caste ===")
for b in ("queen", "worker"):
    m = ol["brain"].values == b
    t = (
        pd.crosstab(
            pd.Series(ol["compartment"].values[m]), pd.Series(ol["call"].values[m])
        )
        .reindex(index=ol_rows, columns=CLUSTERS)
        .fillna(0)
        .astype(int)
    )
    t.columns = SHORTS
    fr = t.div(t.sum(1), axis=0)
    lines.append(
        f"{b}: V={cramers_v(t.values):.4f}; dominant "
        + ", ".join(
            f"{r}->{fr.loc[r].idxmax()} "
            f"({100 * fr.loc[r].max():.1f}%, n={t.loc[r].sum():,})"
            for r in ol_rows
        )
    )

# ---- Confidence comparison ------------------------------------------------
lines.append("\n=== confidence (median max renormalised probability) ===")
lines.append(
    f"optic negative control: median {ol['maxprob_renorm'].median():.4f}"
    f"  frac<0.5 {(ol['maxprob_renorm'] < 0.5).mean():.4f}"
)

# ---- Baseline heterogeneity (unrestricted 29-category corrected labels) ---
# The question is whether DOUBLE RESTRICTION reveals compartment structure the
# unrestricted whole-atlas run cannot resolve.  Raw V across a 3x29 and a 3x5
# table is not commensurate, so the comparable quantity is the WITHIN-TISSUE
# amplification: V_restricted / V_unrestricted, for each tissue.
SP = CANONICAL_TANGRAM
with h5py.File(SP, "r") as f:
    ct_all = read_obs_col(f, "tangram_cell_type")
    _la = f["obs"]["in_lamina"][:].astype(bool)
    _lo = f["obs"]["in_lobula"][:].astype(bool)
    _me = f["obs"]["in_medulla"][:].astype(bool)
    _ic = f["obs"]["in_kc_ic"][:].astype(bool)
    _nc = f["obs"]["in_kc_nc"][:].astype(bool)
    _oc = f["obs"]["in_kc_oc"][:].astype(bool)
_kcm, _olm = _ic | _nc | _oc, _la | _lo | _me

# ---- Premise of the control: the optic neuropils contain no Kenyon cells --
_kc_like = [c for c in set(ct_all) if "enyon" in c]
lines.append(
    "\n=== premise check: fraction called ANY Kenyon type by the "
    "UNRESTRICTED 29-category whole-atlas run ==="
)
for nm, m in (
    ("Kenyon compartments (KC query)", _kcm),
    ("optic neuropils (control query)", _olm),
):
    s = pd.Series(ct_all[m]).value_counts(normalize=True)
    lines.append(
        f"  {nm}: n={m.sum():,}  any-Kenyon "
        f"{100 * sum(s.get(c, 0.0) for c in _kc_like):.2f}%  "
        f"(modal {s.index[0]} {100 * s.iloc[0]:.1f}%)"
    )
for nm, m in (("Lamina", _la), ("Lobula", _lo), ("Medulla", _me)):
    s = pd.Series(ct_all[m]).value_counts(normalize=True)
    lines.append(
        f"    {nm} (n={m.sum():,}): any-Kenyon "
        f"{100 * sum(s.get(c, 0.0) for c in _kc_like):.2f}%"
    )

base = {}
lines.append(
    "\n=== baseline: same compartments under the UNRESTRICTED "
    "29-category whole-atlas run ==="
)
for nm, m, cp in (
    ("KC", _kcm, np.where(_ic, "KC-IC", np.where(_nc, "KC-NC", "KC-OC"))[_kcm]),
    ("optic", _olm, np.where(_la, "Lamina", np.where(_lo, "Lobula", "Medulla"))[_olm]),
):
    t = pd.crosstab(pd.Series(cp), pd.Series(ct_all[m]))
    base[nm] = cramers_v(t.values)
    fr = t.div(t.sum(1), axis=0)
    lines.append(
        f"  {nm}: V={base[nm]:.4f} (table {t.shape}); distinct modal "
        f"calls {fr.idxmax(axis=1).nunique()}/3; "
        + ", ".join(
            f"{r}->{fr.loc[r].idxmax()} {100 * fr.loc[r].max():.1f}%" for r in t.index
        )
    )

# ---- Verdict --------------------------------------------------------------
lines.append("\n=== headline ===")
lines.append(
    f"Cramer's V: KC = {kc_v:.4f}; optic control = {ol_v:.4f}; "
    f"chance (label shuffle) = {floor.mean():.5f}"
)
lines.append(
    f"one-to-one dominant calls: KC {kc_dom.nunique()}/3, optic {ol_dom.nunique()}/3"
)
lines.append("\nAMPLIFICATION by double restriction (the comparable quantity):")
lines.append(f"  KC    : {base['KC']:.4f} -> {kc_v:.4f}  = {kc_v / base['KC']:.2f}x")
lines.append(
    f"  optic : {base['optic']:.4f} -> {ol_v:.4f}  = {ol_v / base['optic']:.2f}x"
)
lines.append(
    f"  optic's ARITHMETIC CEILING is {1 / base['optic']:.2f}x "
    f"(V <= 1), so the control cannot reach KC's "
    f"{kc_v / base['KC']:.2f}x whatever it returns."
)
lines.append(
    f"\n(raw, NON-commensurate cross-tissue ratio ol_V/kc_V = "
    f"{ol_v / kc_v:.3f} -- do not quote this alone: the optic "
    f"subcompartments start out far more distinct than the Kenyon "
    f"compartments do, see baseline above)"
)

txt = "\n".join(lines)
print(txt)
(OUTDIR / "negcontrol_comparison.txt").write_text(txt + "\n")
kc_tab.to_csv(OUTDIR / "kc_compartment_x_call.csv")
ol_tab.to_csv(OUTDIR / "optic_compartment_x_call.csv")
print(f"\nwrote {OUTDIR}/negcontrol_comparison.txt")
