#!/usr/bin/env python3
"""SI figure: sensitivity of the caste-composition result to the cell-type
labelling scheme (combined-reference vs region-stitched Tangram labels).

Two defensible Tangram labellings coexist in this pipeline:

  * combined -- ``tangram_cell_type`` of the canonical run
    (``bimp_st.paths.CANONICAL_TANGRAM``), one mapping against the joined
    central-brain + optic-lobe reference. Canonical: it recovers the
    Kenyon-cell <-> calyx correspondence but leaks optic types into the
    central brain at the per-cell level.
  * stitched -- separate central-brain and optic-lobe mappings, stitched
    together. No optic leakage, but it breaks the KC <-> calyx block. Composed
    here from the canonical central and optic runs (see below); the
    ``tangram_cell_type_stitched`` column stored in the combined objects
    is not composed from them and is not read.

The canonical labelling is the combined labels, passed through the
expression-only hierarchical cluster vote (``hierarchical_vote``,
anatomy-free), with a scCODA compositional contrast on the vote's six labels
(``Mixed`` is drawn as
*Other*; the vote's pooled ``Glia`` is the unresolved-glia fallback and is
drawn as *Glia (unresolved)*). This
figure discloses how much of that result depends on the choice, by
re-running the *identical* pipeline end to end under the stitched labels.

    PYTHONPATH=src .venv/bin/python scripts/identity/si_label_sensitivity.py

Runtime ~6 min (two scCODA NUTS chains); intermediate tables are cached under
``results/figures/si/_cache_label_sensitivity_v2/``.

Two files, one registered figure
--------------------------------
On a single canvas at ``width=\\linewidth``, panel A's 29 spelled-out tick labels
print at ~2 pt; a 29 x 29 matrix with full cell-type names needs the full width.
So the figure is split here, in the renderer, rather than by cropping a PNG (as
``scripts/paper/split_tall_si_figures.py`` does). Each half has a ``results/`` origin
and is checksummed by ``verify_manuscript_figures.py``:

  * ``fig_s_label_sensitivity_a`` -- the 29 x 29 confusion matrix. Not
    registered (its agreement numbers are panel A of _b); rendered only when
    ``BIMP_LABEL_SENS_A=1``.
  * ``figS_label_sensitivity`` -- the registered figure: panels A (agreement),
    B (caste effect), C (relabeled cells) stacked.

Font sizes are asked for in printed points via ``_fs()`` and converted to canvas
points, so the numbers in this file are what a reader measures on the page. No
panel titles: captions own titles (house rule), which also saves height in B/C/D.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.text import Text

from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM
from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import (
    LINEWIDTH_IN,
    QUEEN_COLOR,
    WORKER_COLOR,
    add_panel_headers,
    set_style,
)
from bimp_st.si_common import (
    save as _save,
)

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.environ.get("BIMP_LABEL_SENS_OUT", "results/figures/si"))
# Caches are named for the vote rule they hold. The older ``_cache/`` tables
# predate the deterministic hierarchical_vote (Leiden cluster 19 voted Glia
# there and not here) and are kept only as a record.
CACHE_DIR = OUTPUT_DIR / "_cache_label_sensitivity_v2"

## Both schemes use the column-corrected Tangram runs. The canonical side is the
## corrected combined run below.
##
## The stitched labelling is not a Tangram run of its own:
## build_region_stitched_labels.py composes it from the separate central-brain and
## optic-lobe runs. The `tangram_cell_type_stitched` column stored in the
## combined objects does not match a recomposition from the canonical
## central and optic runs (bimp_st.paths.POISONED_COLUMNS), so it is not
## read. The stitched column is composed here from the
## canonical central and optic runs, whose cell-type columns are keyed by
## name (both checked in their *_column_identity.csv).
TANGRAM_H5AD = str(CANONICAL_TANGRAM)
CENTRAL_H5AD = "results/tangram/combined_clustered_central_colfix_tangram.h5ad"
OPTIC_H5AD = "results/tangram/combined_clustered_optic_colfix_tangram.h5ad"
COMBINED_COL = "tangram_cell_type"
STITCHED_COL = "tangram_cell_type_stitched"
OPTIC_FLAGS = ["in_lamina", "in_medulla", "in_lobula"]
SEED = 42

_OL_VARIANTS = {
    "Optic lobe (lamina)": "Optic lobe",
    "Optic lobe (medulla)": "Optic lobe",
    "Optic lobe (lobula)": "Optic lobe",
}

# White -> deep teal; colourblind-safe sequential, prints legibly in grey.
CONF_CMAP = LinearSegmentedColormap.from_list(
    "conf", ["#FFFFFF", "#D6EDEA", "#8FCFC6", "#3F9E93", "#14625C", "#062F2C"], N=256
)

NEUTRAL = "#4C4C4C"
ACCENT = "#C0392B"

# ── Printed-size machinery ───────────────────────────────────────────────────
# Canvas width of each half, and the fraction of \linewidth it is placed at.
FIG_A_W, FIG_A_FRAC = 10.0, 1.0  # panel A alone
FIG_B_W, FIG_B_FRAC = 9.0, 1.0  # panels B, C, D


def _fs(printed_pt, fig_w, width_frac=1.0):
    """Canvas fontsize that prints at ``printed_pt`` on the page.

    A matplotlib fontsize is in points of the figure canvas, not of the page:
    printed = fontsize * (printed_width / fig_width). Same identity as
    ``_si_common.panel_label_size``, generalised off panel letters -- every size
    in this figure is specified as the size a reader measures with a ruler.
    """
    return printed_pt * fig_w / (width_frac * LINEWIDTH_IN)


def _printed(fontsize, fig_w, width_frac=1.0):
    """Inverse of ``_fs``: what a canvas fontsize actually prints at."""
    return fontsize * (width_frac * LINEWIDTH_IN) / fig_w


def _fit_xlim(ax, texts, pad=0.02):
    """Grow the x limit until none of ``texts`` runs past the right spine.

    The inline value annotations are placed in data coordinates with paddings
    that were tuned to one particular canvas width. A string occupies a fixed
    number of points but a variable number of data units, so the moment the
    axes gets narrower the annotations run off the end. Measure instead.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    lo, hi = ax.get_xlim()
    need = hi
    for t in texts:
        bb = t.get_window_extent(r)
        need = max(need, inv.transform((bb.x1, bb.y0))[0])
    ax.set_xlim(lo, need + (need - lo) * pad)


def _display_name(name: str) -> str:
    """Figure label for a fine (29-type) cell-type value.

    Routes through ``get_canonical_display`` (single source of truth, see
    ``bimp_st.cluster_labels``) for the 16 named central-brain types -- this
    is what spells the raw "Kenyon cells" residual out as "Kenyon cells
    (unclassified)" so it reads the same way here as in every other
    figure. The 13 OL_* optic-lobe subclusters have no canonical full
    name yet (unannotated placeholders pending annotation; the canonical
    vocabulary only defines a single pooled "Optic lobe"), so they pass
    through unchanged rather than being collapsed onto one label.
    """
    from bimp_st.cluster_labels import CANONICAL_ORDER, get_canonical_display

    return get_canonical_display(name) if name in CANONICAL_ORDER else name


# ── Data ─────────────────────────────────────────────────────────────────────


class _ObsShim:
    """hierarchical_vote() only touches .obs."""

    def __init__(self, obs):
        self.obs, self.n_obs = obs, len(obs)


def _compose_stitched(obs):
    """Region-stitched labels, composed from the canonical central + optic runs.

    ``stitched = optic-run label where the cell is optic-flagged, else
    central-run label`` -- the same rule as build_region_stitched_labels.py.
    Joined on
    slice_name||cell_id, never on row order.
    """

    def _side(path):
        o = read_obs(path)
        k = pd.Index(
            [
                f"{s}||{c}"
                for s, c in zip(
                    o["slice_name"].astype(str).values,
                    o["cell_id"].astype(str).values,
                    strict=True,
                )
            ]
        )
        return pd.Series(o[COMBINED_COL].astype(str).values, index=k)

    key = pd.Index(
        [
            f"{s}||{c}"
            for s, c in zip(
                obs["slice_name"].astype(str).values,
                obs["cell_id"].astype(str).values,
                strict=True,
            )
        ]
    )
    cen, opt = _side(CENTRAL_H5AD), _side(OPTIC_H5AD)
    if not (key.is_unique and cen.index.is_unique and opt.index.is_unique):
        raise SystemExit("non-unique slice_name||cell_id keys; cannot stitch")
    flag = obs[OPTIC_FLAGS].astype(bool).to_numpy().any(1)
    out = np.where(flag, opt.reindex(key).values, cen.reindex(key).values)
    n_na = pd.isna(out).sum()
    if n_na:
        raise SystemExit(f"{n_na} cells matched neither region run")
    print(
        f"    Composed stitched labels from the central+optic "
        f"({flag.sum():,} optic-flagged of {len(flag):,})",
        flush=True,
    )
    return np.array([str(x) for x in out])


def load_labels(cache=CACHE_DIR / "label_sensitivity_obs.parquet"):
    """Per-cell obs with both label schemes and both cluster votes."""
    if cache.exists():
        print(f"  Loading cached labels: {cache}", flush=True)
        return pd.read_parquet(cache)

    import anndata as ad

    from bimp_st.cluster_labels import hierarchical_vote

    print(f"  Loading obs from {TANGRAM_H5AD}...", flush=True)
    a = ad.read_h5ad(TANGRAM_H5AD, backed="r")
    cols = [
        "leiden",
        "brain",
        "slice_name",
        "cell_id",
        COMBINED_COL,
        "in_kc_ic",
        "in_kc_nc",
        "in_kc_oc",
        "in_lamina",
        "in_medulla",
        "in_lobula",
    ]
    obs = a.obs[cols].copy()
    for c in ("leiden", "brain", "slice_name", COMBINED_COL):
        obs[c] = obs[c].astype(str)
    obs[STITCHED_COL] = _compose_stitched(obs)

    shim = _ObsShim(obs)
    hierarchical_vote(
        shim, cluster_col="leiden", label_col=COMBINED_COL, dst="vote_combined"
    )
    hierarchical_vote(
        shim, cluster_col="leiden", label_col=STITCHED_COL, dst="vote_stitched"
    )
    for c in ("vote_combined", "vote_stitched"):
        obs[c] = obs[c].astype(str).replace(_OL_VARIANTS)

    obs = obs.reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(cache)
    print(f"    Cached {cache}", flush=True)
    return obs


def agreement_stats(obs):
    """Per-cell agreement between the two schemes at each level of pooling."""
    from bimp_st.cluster_labels import get_pooled

    comb = obs[COMBINED_COL].to_numpy()
    stit = obs[STITCHED_COL].to_numpy()
    agree = comb == stit
    optic = obs[["in_lamina", "in_medulla", "in_lobula"]].to_numpy().any(axis=1)

    pooled_c = obs[COMBINED_COL].map(get_pooled).to_numpy()
    pooled_s = obs[STITCHED_COL].map(get_pooled).to_numpy()

    voted_agree = obs["vote_combined"].to_numpy() == obs["vote_stitched"].to_numpy()

    # vocabulary shared by both schemes within each anatomical compartment
    shared = np.zeros(len(obs), bool)
    for m in (optic, ~optic):
        vocab = set(np.unique(comb[m])) & set(np.unique(stit[m]))
        shared[m] = np.isin(comb[m], list(vocab)) & np.isin(stit[m], list(vocab))

    return [
        ("Fine label (29 types), all cells", agree.mean(), len(agree)),
        ("Fine label, central brain", agree[~optic].mean(), int((~optic).sum())),
        ("Fine label, optic lobe", agree[optic].mean(), int(optic.sum())),
        ("Fine label, shared vocabulary only", agree[shared].mean(), int(shared.sum())),
        ("Pooled to 7 display groups", (pooled_c == pooled_s).mean(), len(agree)),
        (
            "Cluster vote\n(labeling used by the manuscript)",
            voted_agree.mean(),
            len(agree),
        ),
    ]


def run_sccoda(obs, vote_col, scheme, cache_dir=CACHE_DIR):
    """scCODA compositional contrast, identical recipe to caste_comparison_sccoda.py."""
    cache = cache_dir / f"sccoda_{scheme}.csv"
    if cache.exists():
        print(f"  Loading cached scCODA ({scheme}): {cache}", flush=True)
        return pd.read_csv(cache)

    import scanpy._utils as _su

    for _old, _new in (
        ("check_use_raw", "_check_use_raw"),
        ("sanitize_anndata", "_sanitize_anndata"),
    ):
        if not hasattr(_su, _old) and hasattr(_su, _new):
            setattr(_su, _old, getattr(_su, _new))

    import anndata as ad
    from pertpy.tools import Sccoda

    from bimp_st.cluster_labels import POOLED_ORDER

    print(f"  Running scCODA ({scheme})...", flush=True)
    ct = obs[vote_col].astype(str)
    slice_ct = pd.crosstab(obs["slice_name"], ct)
    order = [g for g in POOLED_ORDER if g in set(slice_ct.columns)]
    order += [c for c in slice_ct.columns if c not in order]
    slice_ct = slice_ct[order]

    sb = obs.groupby("slice_name", observed=True)["brain"].first()
    qs, ws = sb[sb == "queen"].index, sb[sb == "worker"].index
    props = slice_ct.div(slice_ct.sum(axis=1), axis=0)

    cad = ad.AnnData(
        obs=pd.DataFrame(
            {
                "slice_name": obs["slice_name"].to_numpy(),
                "brain": obs["brain"].to_numpy(),
                "cell_type_pooled": ct.to_numpy(),
            }
        )
    )
    cad.obs_names_make_unique()

    sc_ = Sccoda()
    md = sc_.load(
        cad,
        type="cell_level",
        generate_sample_level=True,
        cell_type_identifier="cell_type_pooled",
        sample_identifier="slice_name",
        covariate_obs=["brain"],
    )
    ref_ct = str(slice_ct.sum(axis=0).idxmax())
    sc_.prepare(md, modality_key="coda", formula="brain", reference_cell_type=ref_ct)
    md["coda"].obs.index.name = None
    sc_.run_nuts(
        md, modality_key="coda", num_warmup=500, num_samples=1000, rng_key=SEED
    )
    sc_.credible_effects(md, modality_key="coda", est_fdr=0.05)

    coda = md["coda"]
    eff = coda.varm[[k for k in coda.varm if "effect_df" in k][0]]

    rows = []
    for c in order:
        q, w = float(props.loc[qs, c].mean()), float(props.loc[ws, c].mean())
        if c in eff.index:
            fp = float(eff.loc[c, "Final Parameter"])
            l2 = (
                float(eff.loc[c, "log2-fold change"])
                if "log2-fold change" in eff.columns
                else fp
            )
            ip = (
                float(eff.loc[c, "Inclusion probability"])
                if "Inclusion probability" in eff.columns
                else np.nan
            )
        else:
            fp, l2, ip = 0.0, float(np.log2(w / q)) if q > 0 else 0.0, np.nan
        rows.append(
            {
                "scheme": scheme,
                "cell_type": c,
                "is_reference": c == ref_ct,
                "credible": fp != 0,
                "final_parameter": fp,
                "log2fc_worker_vs_queen": l2,
                "inclusion_prob": ip,
                "prop_queen": q,
                "prop_worker": w,
                "n_queen_slices": len(qs),
                "n_worker_slices": len(ws),
                "n_cells": int(slice_ct[c].sum()),
            }
        )

    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ── Panels ───────────────────────────────────────────────────────────────────


def _natural(s):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def panel_confusion(ax, cax, obs, tick_fs, label_fs, cbtick_fs, cblab_fs):
    """Row-normalised 29 x 29 confusion, combined (rows) vs stitched (cols)."""
    conf = pd.crosstab(obs[COMBINED_COL], obs[STITCHED_COL])
    order = sorted(set(conf.index) | set(conf.columns), key=_natural)
    conf = conf.reindex(index=order, columns=order, fill_value=0)
    frac = conf.div(conf.sum(axis=1).replace(0, 1), axis=0) * 100

    im = ax.imshow(
        frac.to_numpy(),
        cmap=CONF_CMAP,
        vmin=0,
        vmax=100,
        aspect="equal",
        interpolation="nearest",
    )
    n = len(order)
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color=ACCENT, lw=1.0, ls="--", alpha=0.9)

    full = [_display_name(o) for o in order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(full, rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(full, fontsize=tick_fs)
    ax.set_xlabel("Stitched label (region-wise Tangram)", fontsize=label_fs, labelpad=3)
    ax.set_ylabel("Atlas label (combined reference)", fontsize=label_fs, labelpad=3)
    ax.tick_params(length=1.5, pad=1.5)

    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("right")
    cb.set_label("% of row", fontsize=cblab_fs, labelpad=5)
    cb.ax.tick_params(labelsize=cbtick_fs, length=2, pad=2)
    cb.outline.set_linewidth(0.5)
    return ax, cax


def panel_agreement(ax, stats, tick_fs, ann_fs, label_fs):
    labels = [s[0] for s in stats]
    vals = np.array([s[1] * 100 for s in stats])
    ns = [s[2] for s in stats]
    y = np.arange(len(labels))[::-1]

    colors = [ACCENT if i == len(labels) - 1 else NEUTRAL for i in range(len(labels))]
    ax.barh(y, vals, height=0.6, color=colors, zorder=2)
    texts = []
    for yi, v, nn in zip(y, vals, ns, strict=False):
        texts.append(
            ax.text(
                v + 1.6,
                yi,
                f"{v:.1f}%   ($n$ = {nn:,})",
                va="center",
                fontsize=ann_fs,
                zorder=3,
            )
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=tick_fs)
    ax.set_xlim(0, 100)
    ax.set_xlabel(
        "Cells receiving the same label under both schemes (%)", fontsize=label_fs
    )
    ax.tick_params(axis="x", labelsize=tick_fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return texts  # fitted by compose_b once the axes is at its final width


class _RefKey(HandlerLine2D):
    """Legend key for the reference category: the open circle with an R in it,
    drawn the way the panel draws it."""

    def __init__(self, fontsize):
        super().__init__(numpoints=1)
        self._fs = fontsize

    def create_artists(self, legend, orig, xd, yd, w, h, fontsize, trans):
        arts = super().create_artists(legend, orig, xd, yd, w, h, fontsize, trans)
        txt = Text(
            -xd + 0.5 * w,
            -yd + 0.5 * h,
            "R",
            ha="center",
            va="center",
            fontsize=self._fs,
            transform=trans,
        )
        return [*arts, txt]


def panel_composition(
    ax, sc_comb, sc_stit, tick_fs, ann_fs, label_fs, legend_fs, marker_s
):
    """The key panel: scCODA caste effect per pooled group, both schemes."""
    both = pd.concat([sc_comb, sc_stit])
    # All six vote labels, Mixed included (drawn as "Other"): it is a fitted
    # category in both schemes and the stitched reference.
    types = list(both["cell_type"].unique())
    # order by the canonical (combined) effect, largest worker-lean first
    key = sc_comb.set_index("cell_type")["log2fc_worker_vs_queen"]
    types = sorted(types, key=lambda t: -key.get(t, 0.0))
    y = np.arange(len(types))[::-1]

    ax.axvline(0, color="#666666", lw=0.8, zorder=1)
    drawn = []  # every marker centre, for the legend placement below
    styles = set()  # (marker, fill) pairs actually drawn; the legend keys these
    has_ref = False
    off = 0.19
    for df, dy, mk, _scheme in (
        (sc_comb, +off, "o", "combined"),
        (sc_stit, -off, "s", "stitched"),
    ):
        d = df.set_index("cell_type")
        for yi, t in zip(y, types, strict=False):
            if t not in d.index:
                continue
            r = d.loc[t]
            v = r["log2fc_worker_vs_queen"]
            col = WORKER_COLOR if v > 0 else QUEEN_COLOR
            ax.plot(
                [0, v],
                [yi + dy, yi + dy],
                color=col,
                lw=1.6,
                alpha=0.55,
                zorder=2,
                solid_capstyle="butt",
            )
            ax.scatter(
                [v],
                [yi + dy],
                s=marker_s,
                marker=mk,
                facecolor=col if r["credible"] else "white",
                edgecolor="black",
                linewidths=1.0,
                zorder=4,
            )
            drawn.append((v, yi + dy))
            drawn.append((v * 0.5, yi + dy))  # the stem, not just its head
            styles.add((mk, col if r["credible"] else "open"))
            if r["is_reference"]:
                has_ref = True
                ax.text(
                    v,
                    yi + dy,
                    "R",
                    ha="center",
                    va="center",
                    fontsize=ann_fs,
                    color="black",
                    zorder=5,
                )

    # flag the calls that change credibility between schemes
    cc = sc_comb.set_index("cell_type")["credible"]
    cs = sc_stit.set_index("cell_type")["credible"]
    changed_rows = []
    xlo, xhi = ax.get_xlim()
    xr = (xlo, xhi + (xhi - xlo) * 0.20)  # headroom for the "changes" flags
    for yi, t in zip(y, types, strict=False):
        a, b = bool(cc.get(t, False)), bool(cs.get(t, False))
        if a != b:
            ax.axhspan(yi - 0.45, yi + 0.45, color=ACCENT, alpha=0.10, zorder=0)
            ax.text(
                0.985,
                yi,
                "changes",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=ann_fs,
                color=ACCENT,
                fontweight="bold",
                zorder=5,
            )
            changed_rows.append(yi)
    ax.set_xlim(xr)

    from bimp_st.cluster_labels import get_pooled_display

    ax.set_yticks(y)
    from bimp_st.cluster_labels import DISPLAY_NAME_OVERRIDES

    # The vote's pooled "Glia" is the unresolved-glia fallback (clusters whose
    # fine glial call is undecided), not the union of the glial subtypes drawn
    # beside it; "Mixed" is the vote's undecided label, drawn as "Other".
    def _row_name(t):
        if t == "Mixed":
            return "Other"
        return DISPLAY_NAME_OVERRIDES.get(t, get_pooled_display(t))

    ax.set_yticklabels([_row_name(t) for t in types], fontsize=tick_fs)
    ax.set_ylim(-0.6, len(types) - 0.4)
    ax.set_xlabel(
        r"scCODA compositional effect, log$_2$FC (worker / queen)", fontsize=label_fs
    )
    ax.tick_params(axis="x", labelsize=tick_fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # The legend keys exactly what is drawn: marker shape = scheme, both drawn
    # open (outline only) because open = not credible; a filled marker is a
    # credible call, coloured by direction; the circled R is the reference.
    # No filled-grey "scheme" key -- nothing in the panel is drawn that way.
    ms = np.sqrt(marker_s) * 0.82

    def _key(marker, face, label):
        return Line2D(
            [],
            [],
            marker=marker,
            ls="none",
            markerfacecolor=face,
            markeredgecolor="black",
            markersize=ms,
            label=label,
        )

    # Two columns so the block stays above the first "changes" row: shapes
    # (scheme) and the reference key in the first, credible fills second.
    handles = [
        _key("o", "white", "Combined (atlas)"),
        _key("s", "white", "Stitched"),
    ]
    handler_map = {}
    if has_ref:
        ref = _key("o", "white", "Reference")
        handles.append(ref)
        handler_map[ref] = _RefKey(ann_fs)
    for col, lab in (
        (QUEEN_COLOR, "Queen-enriched"),
        (WORKER_COLOR, "Worker-enriched"),
    ):
        shapes = sorted(mk for mk, fill in styles if fill == col)
        if shapes:
            handles.append(_key(shapes[0], col, lab))
    # Inside the empty upper-left of this panel: its top rows are near-zero
    # effects, so everything left of zero is blank above the first stem.
    # A fixed position is used rather than scoring corners against the drawn
    # markers, which breaks whenever the rows or effect sizes change.
    ax.legend(
        handles=handles,
        handler_map=handler_map,
        title="Open marker = not credible",
        title_fontsize=legend_fs,
        alignment="left",
        fontsize=legend_fs,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.02),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
        labelspacing=0.3,
    )


def panel_instability(ax, obs, tick_fs, ann_fs, label_fs, top_n=16):
    """Which cell types are least reproducible between the two schemes."""
    d = pd.DataFrame(
        {
            "ct": obs[COMBINED_COL].to_numpy(),
            "agree": obs[COMBINED_COL].to_numpy() == obs[STITCHED_COL].to_numpy(),
        }
    )
    g = d.groupby("ct")["agree"].agg(["mean", "size"])
    g["disagree"] = (1 - g["mean"]) * 100
    g = g.sort_values("size", ascending=False).head(top_n)
    g = g.sort_values("disagree")

    y = np.arange(len(g))
    cols = [ACCENT if v >= 90 else NEUTRAL for v in g["disagree"]]
    ax.barh(y, g["disagree"].to_numpy(), height=0.68, color=cols, zorder=2)
    texts = []
    for yi, (v, nn) in enumerate(zip(g["disagree"], g["size"], strict=True)):
        texts.append(
            ax.text(
                v + 1.2,
                yi,
                f"{v:.0f}%  ($n$ = {nn:,})",
                va="center",
                fontsize=ann_fs,
                zorder=3,
            )
        )

    ax.set_yticks(y)
    ax.set_yticklabels([_display_name(t) for t in g.index], fontsize=tick_fs)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Cells relabeled by the stitched scheme (%)", fontsize=label_fs)
    ax.tick_params(axis="x", labelsize=tick_fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return texts  # fitted by compose_b once the axes is at its final width


# ── Figure ───────────────────────────────────────────────────────────────────


def _text_in(artist, renderer, dpi, dim="width"):
    bb = artist.get_window_extent(renderer)
    return (bb.width if dim == "width" else bb.height) / dpi


def compose_a(obs, output_dir, fig_w=FIG_A_W, frac=FIG_A_FRAC):
    """Panel A alone: the 29 x 29 confusion matrix, at width=\\linewidth.

    Laid out in inches rather than by gridspec fractions, because the whole
    point is to spend exactly as much of the canvas on the tick labels as the
    text they carry needs and give everything else to the matrix. The label
    band is measured off a first draw (a 31-character name at this size is
    ~1.4 in, and guessing that from a character count is how figures end up
    with their longest label clipped).
    """
    tick_fs = _fs(6.0, fig_w, frac)
    label_fs = _fs(8.0, fig_w, frac)
    cbtick_fs = _fs(6.0, fig_w, frac)
    cblab_fs = _fs(7.0, fig_w, frac)
    letter_fs = _fs(8.0, fig_w, frac)

    fig = plt.figure(figsize=(fig_w, fig_w))  # provisional; resized below
    ax = fig.add_axes([0.28, 0.28, 0.62, 0.62])
    cax = fig.add_axes([0.93, 0.40, 0.012, 0.30])
    panel_confusion(ax, cax, obs, tick_fs, label_fs, cbtick_fs, cblab_fs)

    fig.canvas.draw()
    r, dpi = fig.canvas.get_renderer(), fig.dpi
    ylab_band = max(_text_in(t, r, dpi, "width") for t in ax.get_yticklabels())
    xlab_band = max(_text_in(t, r, dpi, "height") for t in ax.get_xticklabels())
    # axis titles: the y title is rotated, so its *width* is its thickness
    y_title = _text_in(ax.yaxis.label, r, dpi, "width")
    x_title = _text_in(ax.xaxis.label, r, dpi, "height")
    # colorbar furniture sits left (ticks) and right (label) of the bar itself
    cb_ticks = max(_text_in(t, r, dpi, "width") for t in cax.get_yticklabels())
    cb_label = _text_in(cax.yaxis.label, r, dpi, "width")

    pad = 0.05
    left = ylab_band + y_title + 3 * pad
    bottom = xlab_band + x_title + 3 * pad
    cb_w = 0.10
    right = cb_ticks + cb_w + cb_label + 4 * pad
    top = 1.6 * letter_fs / 72.0 + 2 * pad  # letter sits 0.375 em above axes

    side = fig_w - left - right  # matrix is square
    fig_h = side + bottom + top
    fig.set_size_inches(fig_w, fig_h)
    ax.set_position([left / fig_w, bottom / fig_h, side / fig_w, side / fig_h])
    cax.set_position(
        [
            (left + side + cb_ticks + 2 * pad) / fig_w,
            bottom / fig_h,
            cb_w / fig_w,
            side / fig_h,  # same height as the matrix it keys
        ]
    )

    add_panel_headers(
        fig,
        [(ax, "A", "Label confusion")],
        width_frac=PRINTED_WIDTH_FRAC["fig_s_label_sensitivity_a"],
    )

    print(
        f"  A: {fig_w:.2f} x {fig_h:.2f} in, aspect {fig_h / fig_w:.3f}; "
        f"tick labels print at {_printed(tick_fs, fig_w, frac):.2f} pt; "
        f"matrix {side:.2f} in ({side / 29 * 72:.1f} pt per cell)",
        flush=True,
    )
    _save(fig, output_dir, "fig_s_label_sensitivity_a")


def compose_b(obs, stats, sc_comb, sc_stit, output_dir, fig_w=FIG_B_W, frac=FIG_B_FRAC):
    """Panels B, C, D stacked, at width=\\linewidth.

    All three are horizontal bar/dot charts whose category names run to ~35
    characters and which annotate each row with a value and an n. Side by side
    in half-width columns the annotations do not fit -- the strings occupy a
    fixed number of points and a half-width axes has too few data units to
    spare -- so this is a single column. That costs height, which is paid for
    by dropping the per-panel titles into the caption (house rule anyway).
    """
    tick_fs = _fs(6.5, fig_w, frac)
    ann_fs = _fs(6.0, fig_w, frac)
    label_fs = _fs(7.5, fig_w, frac)
    legend_fs = _fs(6.5, fig_w, frac)
    letter_fs = _fs(8.0, fig_w, frac)
    marker_s = (_fs(7.5, fig_w, frac)) ** 2  # printed marker dia. 7.5 pt

    # Row heights in inches of canvas; B has 6 rows, C 7, D 16. The gap and the
    # bottom margin have to carry each panel's tick labels and its axis title,
    # which live outside the axes rectangle -- undersize them and the label of
    # one panel lands on the panel below it.
    pad = 0.05
    heights = {"B": 1.95, "C": 2.30, "D": 2.68}
    gap, bottom = 0.76, 0.44
    top = 1.6 * letter_fs / 72.0 + 2 * pad  # letter sits 0.375 em above axes
    fig_h = sum(heights.values()) + 2 * gap + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    ys, y = {}, fig_h - top
    for k in ("B", "C", "D"):
        y -= heights[k]
        ys[k] = y
        y -= gap

    left0, right = 2.60, 0.12  # left is refitted below
    axes = {}
    for k in ("B", "C", "D"):
        axes[k] = fig.add_axes(
            [
                left0 / fig_w,
                ys[k] / fig_h,
                (fig_w - left0 - right) / fig_w,
                heights[k] / fig_h,
            ]
        )

    t_b = panel_agreement(axes["B"], stats, tick_fs, ann_fs, label_fs)
    panel_composition(
        axes["C"], sc_comb, sc_stit, tick_fs, ann_fs, label_fs, legend_fs, marker_s
    )
    t_d = panel_instability(axes["D"], obs, tick_fs, ann_fs, label_fs)

    # One shared left margin, wide enough for the longest name in any panel, so
    # the three value axes line up.
    fig.canvas.draw()
    r, dpi = fig.canvas.get_renderer(), fig.dpi
    band = max(
        _text_in(t, r, dpi, "width")
        for ax in axes.values()
        for t in ax.get_yticklabels()
    )
    left = band + 3 * pad
    for k, ax in axes.items():
        ax.set_position(
            [
                left / fig_w,
                ys[k] / fig_h,
                (fig_w - left - right) / fig_w,
                heights[k] / fig_h,
            ]
        )

    # Only now, at the final axes width, is it meaningful to ask whether the
    # inline annotations fit.
    # Generous pad: the print-scale hook may still grow these strings at save
    # time, and an annotation must never end at the image edge.
    _fit_xlim(axes["B"], t_b, pad=0.06)
    _fit_xlim(axes["D"], t_d, pad=0.06)

    add_panel_headers(
        fig,
        [
            (axes["B"], "A", "Label agreement"),
            (axes["C"], "B", "Caste effect"),
            (axes["D"], "C", "Relabeled cells"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["figS_label_sensitivity"],
    )

    print(
        f"  B: {fig_w:.2f} x {fig_h:.2f} in, aspect {fig_h / fig_w:.3f}; "
        f"tick labels print at {_printed(tick_fs, fig_w, frac):.2f} pt, "
        f"annotations at {_printed(ann_fs, fig_w, frac):.2f} pt, "
        f"axis titles at {_printed(label_fs, fig_w, frac):.2f} pt",
        flush=True,
    )
    _save(fig, output_dir, "figS_label_sensitivity")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        import matplotlib as mpl

        set_style()
        mpl.rcParams["savefig.dpi"] = 300
        mpl.rcParams["figure.dpi"] = 300
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print("Label-scheme sensitivity", flush=True)
        obs = load_labels()

        stats = agreement_stats(obs)
        for name, frac, n in stats:
            print(
                f"  {name.replace(chr(10), ' ')}: {frac * 100:.2f}% (n={n:,})",
                flush=True,
            )

        vote = pd.crosstab(obs["vote_combined"], obs["vote_stitched"])
        per_cluster = obs.groupby("leiden")[["vote_combined", "vote_stitched"]].first()
        n_same = int(
            (per_cluster["vote_combined"] == per_cluster["vote_stitched"]).sum()
        )
        print(
            f"  Leiden clusters voted identically: {n_same}/{len(per_cluster)}",
            flush=True,
        )

        sc_comb = run_sccoda(obs, "vote_combined", "combined")
        sc_stit = run_sccoda(obs, "vote_stitched", "stitched")

        both = pd.concat([sc_comb, sc_stit])
        both.to_csv(OUTPUT_DIR / "fig_s_label_sensitivity_sccoda.csv", index=False)
        per_cluster.to_csv(OUTPUT_DIR / "fig_s_label_sensitivity_clustervote.csv")
        pd.DataFrame(
            [
                {"level": n.replace("\n", " "), "agreement_pct": f * 100, "n_cells": nn}
                for n, f, nn in stats
            ]
        ).to_csv(OUTPUT_DIR / "fig_s_label_sensitivity_agreement.csv", index=False)

        print("\n  scCODA credibility, combined vs stitched:", flush=True)
        piv = both.pivot_table(
            index="cell_type",
            columns="scheme",
            values=["credible", "log2fc_worker_vs_queen", "inclusion_prob"],
        )
        print(piv.to_string(), flush=True)

        if os.environ.get("BIMP_LABEL_SENS_A") == "1":
            compose_a(obs, OUTPUT_DIR)
        compose_b(obs, stats, sc_comb, sc_stit, OUTPUT_DIR)
        print("Done.", flush=True)
