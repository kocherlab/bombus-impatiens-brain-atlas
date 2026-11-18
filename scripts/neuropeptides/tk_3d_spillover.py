"""Does the Tk optic-lobe spillover hold up as a 3D phenomenon?

fig_neuropeptide_spatial.py's specificity number -- Tk's KC-vs-rest fold
enrichment per caste -- is pooled across every section of each brain. This
script asks two further questions using the 3D reconstruction, importing the
existing pieces rather than reimplementing them:

  A. Z-stratified robustness. Is the pooled fold consistent across the
     reconstructed volume, or driven by a few sections? Recomputes the fold
     per section with the same _kc_enrichment call imported from
     fig_neuropeptide_spatial.py, and looks at it along the A-P axis.

  B. 3D distance gradient. Within each caste, are Tk-expressing non-KC cells
     closer in 3D (across sections) to the Kenyon-cell compartments than
     non-expressing non-KC cells? Reported as medians and the rank-biserial
     effect size, with no P value: cells of one brain per caste are not
     independent replicates.

Coordinates match region_prediction_3d.py, from which Z_SCALE is
imported: (x, y) is obsm['spatial_aligned'] (Xenium camera pixels, identical
to obsm['spatial']), z = z_level (section index x 10 um nominal thickness,
in microns) x Z_SCALE (pixels per um). One KD-tree per brain; brains
are never linked.

Units: obsm['spatial_aligned'] is a rigid transform of each crop's raw
section coordinates in camera pixels (aligned / Z_SCALE reproduces raw
per-crop micrometre pairwise distances exactly, every one of the 65 crops),
so dividing by Z_SCALE gives micrometres in-plane. The raw per-crop
coordinates themselves cannot be used here: they are not co-registered
across sections, and the distance is measured across sections.

Kenyon-cell mask: the nine-region flags (in_kc_{ic,nc,oc} from
bimp_st.nine_region_common.attach_region_flags, joined on (slice_name,
cell_id)), the same flags behind the pooled per-caste folds, through the
nullable kc_mask: "outside" is a traced cell inside no Kenyon compartment, and
untraced cells are in neither arm.

Correctness gate: the pooled fold is first recomputed with the unmodified
_kc_enrichment() and checked against the reported values (GATE_Q, GATE_W);
the script stops if they do not match.

Data provenance: SPATIAL_H5AD (combined_clustered.h5ad) carries a
pre-correction tangram_cell_type column and 562,028 duplicate obs_names. This
script never reads cell-type identity or region flags from it and never uses
`.loc`; it reads only expression (via _get_expr), slice_name / cell_id (for
the region join) and coordinates, positionally, as fig_neuropeptide_spatial.py does.

Writes bimp_st.paths.TK_3D_SPILLOVER only (BIMP_TK3D_OUT: a scratch dir).
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuropeptides.fig_neuropeptide_spatial import _kc_enrichment, resolve_gene  # noqa: E402
from regions.region_prediction_3d import Z_SCALE  # noqa: E402

from bimp_st.nine_region_common import (  # noqa: E402
    KC_REGIONS,
    attach_region_flags,
    region_mask,
)
from bimp_st.paths import TK_3D_SPILLOVER  # noqa: E402
from bimp_st.si_common import SPATIAL_H5AD  # noqa: E402
from bimp_st.si_common import get_expr as _get_expr

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("BIMP_TK3D_OUT") or TK_3D_SPILLOVER)
OUT.mkdir(parents=True, exist_ok=True)

GATE_TOL = 0.15  # the gate values are rounded to 1 decimal
GATE_Q, GATE_W = 2.9, 6.7
MIN_CELLS_PER_ARM = 20  # per section, per KC/non-KC arm, for a stable fold


class _Shim:
    """attach_region_flags writes onto .obs; keep it off the loaded object."""

    def __init__(self, obs):
        self.obs = obs


def nine_region_mask(obs, regions):
    """Nullable union of the nine-region in_<region> flags (region_mask), joined
    on (slice_name, cell_id) -- never positionally, never the columns stored in
    obs. NA = the cell's section has no tracing for these regions."""
    d = obs[["slice_name", "cell_id"]].copy()
    attach_region_flags(_Shim(d), regions=list(regions))
    return region_mask(d, regions)


def nine_region_flags(obs, regions):
    """Cells inside any of *regions* (an untraced cell is not inside)."""
    return nine_region_mask(obs, regions).to_numpy(dtype=bool, na_value=False)


def nine_region_kc_flags(obs):
    """(inside a Kenyon compartment, traced for the Kenyon compartments)."""
    m = nine_region_mask(obs, KC_REGIONS)
    kc = m.to_numpy(dtype=bool, na_value=False)
    traced = ~np.asarray(m.isna())
    print(
        f"  nine-region Kenyon-compartment cells: {int(kc.sum()):,}; "
        f"untraced cells: {int((~traced).sum()):,}",
        flush=True,
    )
    if not kc.any():
        raise SystemExit("ERROR: no nine-region Kenyon-compartment cells")
    return kc, traced


def load():
    print(f"loading {SPATIAL_H5AD} ...", flush=True)
    adata = sc.read_h5ad(REPO / SPATIAL_H5AD)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars} genes", flush=True)

    queen_mask = adata.obs["brain"].values == "queen"
    brain = adata.obs["brain"].astype(str).values
    slice_name = adata.obs["slice_name"].astype(str).values
    z_level = adata.obs["z_level"].to_numpy(dtype=float)
    sp = adata.obsm["spatial_aligned"]

    var = resolve_gene("Tk", adata.var_names)
    if var is None:
        raise SystemExit("ERROR: Tk not found in data")
    expr_tk = _get_expr(adata, var)

    kc_flags, traced = nine_region_kc_flags(adata.obs)

    return dict(
        expr_tk=expr_tk,
        queen_mask=queen_mask,
        kc_flags=kc_flags,
        traced=traced,
        out_flags=traced & ~kc_flags,
        brain=brain,
        slice_name=slice_name,
        z_level=z_level,
        sp=sp,
    )


def correctness_gate(d):
    q, w = d["queen_mask"] & d["traced"], ~d["queen_mask"] & d["traced"]
    fold_q = _kc_enrichment(d["expr_tk"], q, d["kc_flags"])
    fold_w = _kc_enrichment(d["expr_tk"], w, d["kc_flags"])
    print(
        f"gate: pooled fold queen={fold_q:.3f} worker={fold_w:.3f} "
        f"(paper reports ~{GATE_Q}x / ~{GATE_W}x)",
        flush=True,
    )
    if abs(fold_q - GATE_Q) > GATE_TOL or abs(fold_w - GATE_W) > GATE_TOL:
        raise SystemExit(
            "GATE FAILED: recomputed fold does not match the registered "
            "manuscript numbers -- refusing to build new claims on this"
        )
    return fold_q, fold_w


# ── A. Z-stratified robustness ──────────────────────────────────────────────


def per_section_fold(d):
    rows = []
    for brain_label, bmask in [
        ("queen", d["queen_mask"]),
        ("worker", ~d["queen_mask"]),
    ]:
        sections = sorted(
            set(d["slice_name"][bmask]),
            key=lambda s: d["z_level"][d["slice_name"] == s][0],
        )
        for s in sections:
            smask = bmask & d["traced"] & (d["slice_name"] == s)
            n_kc = int((smask & d["kc_flags"]).sum())
            n_out = int((smask & d["out_flags"]).sum())
            if n_kc < MIN_CELLS_PER_ARM or n_out < MIN_CELLS_PER_ARM:
                continue
            fold = _kc_enrichment(d["expr_tk"], smask, d["kc_flags"])
            rows.append(
                dict(
                    brain=brain_label,
                    slice_name=s,
                    z_level=float(d["z_level"][smask][0]),
                    n_kc=n_kc,
                    n_out=n_out,
                    fold=fold,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_section_fold.csv", index=False)
    print(
        f"\nper-section fold: {len(df)} sections retained "
        f"(>= {MIN_CELLS_PER_ARM} cells/arm)",
        flush=True,
    )
    for b in ("queen", "worker"):
        sub = df[df.brain == b]["fold"]
        print(
            f"  {b}: median {sub.median():.2f}x, "
            f"IQR [{sub.quantile(0.25):.2f}, {sub.quantile(0.75):.2f}], "
            f"range [{sub.min():.2f}, {sub.max():.2f}], n={len(sub)}",
            flush=True,
        )
    return df


# ── B. True 3D distance-to-KC-boundary gradient ─────────────────────────────


def distance_gradient(d):
    from scipy.spatial import cKDTree

    z_px = d["z_level"] * Z_SCALE
    xyz = np.column_stack([d["sp"], z_px])

    rows = []
    dist_all = np.full(len(d["expr_tk"]), np.nan)
    for brain_label, bmask in [
        ("queen", d["queen_mask"]),
        ("worker", ~d["queen_mask"]),
    ]:
        kc_idx = np.flatnonzero(bmask & d["kc_flags"])
        non_idx = np.flatnonzero(bmask & d["out_flags"])
        if len(kc_idx) == 0 or len(non_idx) == 0:
            continue
        tree = cKDTree(xyz[kc_idx])
        dist_px, _ = tree.query(xyz[non_idx], k=1)
        dist_um = dist_px / Z_SCALE
        dist_all[non_idx] = dist_um

        expr = d["expr_tk"][non_idx]
        pos = expr > 0
        d_pos, d_neg = dist_um[pos], dist_um[~pos]
        if pos.sum() >= 10 and (~pos).sum() >= 10:
            # Rank-biserial effect size from the Mann-Whitney U of the Tk+ arm
            # (positive = Tk+ cells closer); no P value, cells are not replicates.
            n1, n2 = len(d_pos), len(d_neg)
            ranks = rankdata(np.concatenate([d_pos, d_neg]))
            u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
            r = 1 - (2 * u) / (n1 * n2)
        else:
            r = np.nan
        rows.append(
            dict(
                brain=brain_label,
                n_tk_pos=int(pos.sum()),
                n_tk_neg=int((~pos).sum()),
                median_dist_um_tk_pos=float(np.median(d_pos)) if pos.sum() else np.nan,
                median_dist_um_tk_neg=float(np.median(d_neg))
                if (~pos).sum()
                else np.nan,
                rank_biserial_r=float(r),
            )
        )
        print(
            f"\n{brain_label}, non-KC cells: Tk+ median distance to nearest "
            f"KC cell = {np.median(d_pos):.1f} um (n={pos.sum():,}), "
            f"Tk- = {np.median(d_neg):.1f} um (n={(~pos).sum():,}); "
            f"rank-biserial r={r:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "distance_gradient_summary.csv", index=False)

    out = pd.DataFrame(
        {
            "brain": d["brain"],
            "slice_name": d["slice_name"],
            "is_kc": d["kc_flags"],
            "traced": d["traced"],
            "expr_tk": d["expr_tk"],
            "dist_to_kc_um": dist_all,
        }
    )
    out.to_parquet(OUT / "per_cell_distance_to_kc.parquet")
    return df


def main():
    d = load()
    correctness_gate(d)
    per_section_fold(d)
    distance_gradient(d)
    print(f"\nDone. Outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
