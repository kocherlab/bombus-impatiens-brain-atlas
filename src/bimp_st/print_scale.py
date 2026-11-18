r"""Make every registered figure's text print at the same size.

A matplotlib ``fontsize`` is in points of the figure canvas, and LaTeX scales
the image to a fixed width, so

    printed_pt = authored_pt * (printed_in / saved_in)

where ``saved_in`` is the width of the written file. With ``savefig.bbox =
"tight"`` that is the tight bounding box, not ``fig.get_figwidth()`` (for
fig:atlas_overview, about 28 in vs a 16 in canvas). Across the registered set
``saved_in`` ranges from about 5 to 28 in, so the same literal prints up to ~4x
larger in one figure than another.

At save time the page scale is computed,

    s = (width_frac * LINEWIDTH_IN) / tight_w_in      # page inches per fig inch

and every text artist is rescaled by one factor, ``HOUSE_PT["tick"] / p``, where
``p`` is the median printed tick size. A single factor per figure keeps each
figure's internal size hierarchy. If growing the text creates new overlaps
(text-vs-text, legend-vs-legend, legend-vs-reference-line; see
``text_collisions``), the factor is reduced and the shortfall is reported.
Text-vs-patch and text-vs-3D-scatter overlaps are not checked automatically.

``install()`` patches ``Figure.savefig``; it is called from
``bimp_st.style.set_pub_style`` and on import of ``bimp_st.si_common``. It acts
only on files whose stem is in ``PRINTED_WIDTH_FRAC`` (the registered figures).
"""

from __future__ import annotations

import os
import pathlib

# Page geometry, from bioRxiv.cls: \RequirePackage[a4paper, total={170mm,247mm}]
# 170 mm = 482.2 pt; bimp_st.si_common.LINEWIDTH_IN carries 481.9 pt measured
# off a build, which agrees with it. A full-width figure* spans \textwidth.
LINEWIDTH_IN = 481.9 / 72.27
TEXTHEIGHT_IN = 247.0 / 25.4

# Printed point sizes for the standard roles. `tick` is the anchor: it is the
# most consistently authored role and is present in almost every figure.
# Targets follow common journal minima for figure text (body text is ~9-10 pt),
# not the set's own median (~4.5 pt, too small). Most figures grow 1.3-2x; a
# figure that cannot reach these sizes without overlaps needs a looser layout.
HOUSE_PT = {
    "tick": 6.0,
    "axis_label": 7.0,
    "title": 7.0,
    "legend": 6.0,
    "legend_title": 6.5,
    "panel_letter": 8.0,
    "panel_tag": 7.0,  # short bold tag beside the letter (si_common.add_panel_headers)
}

# Output-file stem -> the \includegraphics width fraction it is placed at.
# Generated from the .tex; verify_manuscript_figures.py checks it still agrees,
# so a placement change in the .tex cannot silently desynchronise the renderer.
PRINTED_WIDTH_FRAC = {
    "celltype_lp_summary": 0.49,
    "figS_panel_concordance": 1.00,
    "fig_atlas_overview": 1.00,
    "fig_2": 1.00,  # stem of a layout no registered figure uses
    "fig_caste_comparison": 1.00,
    "fig_3": 1.00,
    "fig_caste_de": 1.00,
    "fig_neuropeptide_spatial": 1.00,
    "figS_caste_de_volcanoes": 1.00,
    "fig_kc_subtypes": 1.00,
    "figS_atlas_quality": 1.00,
    "fig_cnmf_programs_remade": 1.00,
    "fig_celltype_3d_adjacency": 0.75,
    "figS_beta_diversity": 0.60,
    "fig_s_bento_pergene": 1.00,
    "figS_subcellular_regions": 1.00,
    "fig_s_caste_overview": 0.55,
    "figS_deg_concordance": 1.00,
    "figS_exploded_regions": 0.85,
    "figS_focal_vg_krh1": 1.00,
    "fig_s_hill_diversity": 1.00,
    "figS_kc_projection_null": 1.00,
    "fig_s_kc_restricted_confidence": 1.00,
    "fig_s_label_sensitivity_a": 1.00,
    "figS_label_sensitivity": 0.92,
    "fig_s_ltkc_kcnc_caste_control": 0.55,
    "figS_mosaicfield": 1.0,
    "figS_multimodal_tk": 1.00,
    "figS_neuropeptide_landscape": 0.82,
    "no_generic": 0.82,  # BIMP_NP_OUT preview stem of figS_neuropeptide_landscape
    "figS_percell_vs_vote": 1.00,
    "figS_probe_clusters": 1.00,
    "fig_s_subcellular_proximity": 1.00,
    "figS_tangram_confidence": 0.90,
    "fig_s_tangram_confidence_a": 1.00,
    "fig_s_tangram_confidence_b": 1.00,
    "figS_tangram_validation": 1.00,
    "fig_s_vg_localization": 1.00,
    "figS_tk_3d_spillover": 1.00,
    "kc_threshold": 1.00,
    "overlapping_umaps_by_modality": 1.00,
    "figS_ap_gradient": 0.80,
    "figS_sn_panel_umap": 1.00,
    "region_lp_summary": 0.49,
    "figS_region_prediction": 1.00,
    "si_regional_lollipop": 1.00,
    "si_regional_overview": 0.95,
    "figS_regional_volcanoes": 1.00,
    "figS_snuc_kc_subclusters": 1.00,
}

# fig_s_xenium_qc is cut into _a/_b by split_tall_si_figures.py after rendering;
# both halves are placed at \linewidth, so the uncut figure is scaled as one.
PRINTED_WIDTH_FRAC["fig_s_xenium_qc"] = 1.00
PRINTED_WIDTH_FRAC["figS_leiden_stability"] = 1.00  # full width
PRINTED_WIDTH_FRAC["fig_s_xenium_celltype_overlay"] = 1.00  # full-width SI figure
# trimmed to clear the text block
PRINTED_WIDTH_FRAC["fig_s_caste_de_glia_subtypes"] = 0.94
PRINTED_WIDTH_FRAC["fig_xenium_celltype_overlay"] = 1.0
PRINTED_WIDTH_FRAC["fig_xenium_celltype_overlay_v1"] = 1.0
PRINTED_WIDTH_FRAC["fig_xenium_celltype_overlay_v2"] = 1.0
PRINTED_WIDTH_FRAC["fig_s_xenium_channels"] = 1.0
PRINTED_WIDTH_FRAC["figS_xenium_kc_populations"] = 1.0
PRINTED_WIDTH_FRAC["figS_reference_composition"] = 1.00  # full-width SI
PRINTED_WIDTH_FRAC["fig_s_caste_de_minor_types"] = 1.0  # two volcanoes, full width
PRINTED_WIDTH_FRAC["cross_list_dotmatrix"] = 1.0  # fig_s_deg_recurrence, full width

_ENV_OFF = "BIMP_NO_PRINT_SCALE"
_installed = False


# ── measurement ─────────────────────────────────────────────────────────────


def _visible_texts(fig):
    from matplotlib.text import Text

    out = []
    for t in fig.findobj(Text):
        try:
            if not t.get_visible():
                continue
            s = t.get_text()
        except Exception:
            continue
        if not s or not str(s).strip():
            continue
        out.append(t)
    return out


def _is_panel_letter(t):
    """A bold single A-Z character: how every generator draws a panel letter.

    Checks weight and length rather than position, so it does not depend on
    where a script places its letter.
    """
    try:
        if t.get_gid() == "panel_header_letter":
            return True
        txt = str(t.get_text()).strip()
        if len(txt) != 1 or not ("A" <= txt <= "Z"):
            return False
        return "bold" in str(t.get_fontweight()).lower()
    except Exception:
        return False


def _roles(fig):
    """id(Text) -> role, from matplotlib's own artist structure."""
    role = {}
    for ax in fig.axes:
        role[id(ax.title)] = "title"
        role[id(ax.xaxis.label)] = "axis_label"
        role[id(ax.yaxis.label)] = "axis_label"
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            role[id(t)] = "tick"
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                role[id(t)] = "legend"
            if leg.get_title() is not None:
                role[id(leg.get_title())] = "legend_title"
    for leg in getattr(fig, "legends", None) or []:
        for t in leg.get_texts():
            role[id(t)] = "legend"
    for t in _visible_texts(fig):
        if id(t) not in role and _is_panel_letter(t):
            role[id(t)] = "panel_letter"
        elif id(t) not in role and t.get_gid() == "panel_header_tag":
            role[id(t)] = "panel_tag"
    return role


def saved_width_in(fig, renderer=None, bbox_inches=None, pad_inches=None):
    """Width in inches of the file savefig will actually write.

    Under bbox_inches="tight" that is the tight bounding box plus pad_inches on
    each side; with an explicit Bbox it is that box; otherwise it is the canvas.
    Getting this wrong scales the text by the ratio between the two, which is
    large for the widest figures (fig:atlas_overview).
    """
    import matplotlib as mpl

    if bbox_inches is None:
        bbox_inches = mpl.rcParams.get("savefig.bbox", None)
    if pad_inches is None:
        pad_inches = mpl.rcParams.get("savefig.pad_inches", 0.0)
    try:
        pad = float(pad_inches)
    except Exception:
        pad = 0.0

    if bbox_inches == "tight":
        if renderer is None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(renderer)
        if bb is None or bb.width <= 0:
            return None
        return bb.width + 2 * pad
    if hasattr(bbox_inches, "width"):  # an explicit Bbox
        return float(bbox_inches.width)
    return float(fig.get_figwidth())


def page_scale(fig, width_frac, renderer=None, bbox_inches=None, pad_inches=None):
    """Page inches per figure inch, from the width actually written."""
    w = saved_width_in(fig, renderer, bbox_inches, pad_inches)
    if not w or w <= 0:
        return None
    return (width_frac * LINEWIDTH_IN) / w


# Fraction of the smaller text box that must be covered before a pair counts
# as colliding. Text bboxes include ascender/descender space, so adjacent tick
# labels often overlap by a pixel or two and still read clearly.
COLLISION_FRAC = 0.18


def _legends(fig):
    """Every distinct Legend artist in the figure (axes-level + figure-level)."""
    out, seen = [], set()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None and id(leg) not in seen:
            seen.add(id(leg))
            out.append(leg)
    for leg in getattr(fig, "legends", None) or []:
        if id(leg) not in seen:
            seen.add(id(leg))
            out.append(leg)
    return out


def _legend_own_text_ids(leg):
    ids = {id(t) for t in leg.get_texts()}
    title = leg.get_title()
    if title is not None:
        ids.add(id(title))
    return ids


def _reference_line_hits_bbox(line, bbox, n_sample=64):
    """True if a dashed/dotted line's plotted path crosses `bbox` (display px).

    Only non-solid styles are checked: these are usually reference levels
    (chance, thresholds) spanning the axes, whereas solid data traces passing
    near a legend are normal.
    """
    ls = line.get_linestyle()
    if ls in (None, "None", "none", "-", "solid", ""):
        return False
    try:
        import numpy as np

        verts = line.get_transform().transform(line.get_path().vertices)
    except Exception:
        return False
    if len(verts) < 2:
        return False
    idx = np.linspace(0, len(verts) - 1, n_sample)
    xs = np.interp(idx, np.arange(len(verts)), verts[:, 0])
    ys = np.interp(idx, np.arange(len(verts)), verts[:, 1])
    inside = (xs >= bbox.x0) & (xs <= bbox.x1) & (ys >= bbox.y0) & (ys <= bbox.y1)
    return bool(inside.any())


def text_collisions(fig, renderer, frac=COLLISION_FRAC):
    """Pairs of overlapping artists that would read as a layout defect.

    Checks text-vs-text, then legend-vs-legend, legend-vs-other-text, and
    dashed/dotted reference lines behind a legend. The legend passes only add
    hits, so they can only make the growth/backoff loop more conservative.

    Text-vs-patch and text-vs-3D-scatter overlaps are not checked: they cannot
    be told apart from normal bar charts or scatters automatically. Check those
    by eye.
    """
    boxes = []
    for t in _visible_texts(fig):
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t, bb))
    hits = []
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        ai = bi.width * bi.height
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox <= 0 or oy <= 0:
                continue
            aj = bj.width * bj.height
            smaller = ai if ai < aj else aj
            if smaller > 0 and (ox * oy) / smaller > frac:
                hits.append((str(ti.get_text())[:28], str(tj.get_text())[:28]))

    # -- legend vs. legend, legend vs. any text that isn't its own --
    legend_boxes = []
    for leg in _legends(fig):
        try:
            lb = leg.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if lb.width > 0 and lb.height > 0:
            legend_boxes.append((leg, lb, _legend_own_text_ids(leg)))

    for i in range(len(legend_boxes)):
        _, bi, _ = legend_boxes[i]
        ai = bi.width * bi.height
        for j in range(i + 1, len(legend_boxes)):
            _, bj, _ = legend_boxes[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox <= 0 or oy <= 0:
                continue
            aj = bj.width * bj.height
            smaller = ai if ai < aj else aj
            if smaller > 0 and (ox * oy) / smaller > frac:
                hits.append(("<legend>", "<legend>"))

    for _leg, lb, owned in legend_boxes:
        ai = lb.width * lb.height
        for t, tb in boxes:
            if id(t) in owned:
                continue
            ox = min(lb.x1, tb.x1) - max(lb.x0, tb.x0)
            oy = min(lb.y1, tb.y1) - max(lb.y0, tb.y0)
            if ox <= 0 or oy <= 0:
                continue
            at = tb.width * tb.height
            smaller = ai if ai < at else at
            if smaller > 0 and (ox * oy) / smaller > frac:
                hits.append(("<legend>", str(t.get_text())[:28]))

    # -- dashed/dotted reference lines running behind a legend --
    for ax in fig.axes:
        for line in ax.get_lines():
            try:
                if not line.get_visible():
                    continue
            except Exception:
                continue
            for _, lb, _ in legend_boxes:
                if _reference_line_hits_bbox(line, lb):
                    hits.append(("<legend>", "<reference line>"))

    return hits


# ── normalisation ───────────────────────────────────────────────────────────


def _median(v):
    v = sorted(v)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def target_factor(fig, width_frac, bbox_inches=None, pad_inches=None):
    """Factor to multiply every fontsize by so the figure prints at house size.

    Anchored on the median tick label. Figures without ticks fall back to axis
    labels, then to all text, so a figure made entirely of annotations still
    gets scaled rather than skipped.
    """
    s = page_scale(fig, width_frac, bbox_inches=bbox_inches, pad_inches=pad_inches)
    if not s:
        return None, None
    role = _roles(fig)
    sizes: dict[str, list[float]] = {}
    for t in _visible_texts(fig):
        sizes.setdefault(role.get(id(t), "other"), []).append(t.get_fontsize())

    for anchor in ("tick", "axis_label", "legend", "title"):
        cur = _median(sizes.get(anchor, []))
        if cur:
            printed_now = cur * s
            return HOUSE_PT[anchor] / printed_now, s
    cur = _median([x for v in sizes.values() for x in v])
    if not cur:
        return None, s
    return HOUSE_PT["tick"] / (cur * s), s


def normalize_for_print(
    fig,
    width_frac,
    label="",
    max_backoff=6,
    verbose=True,
    bbox_inches=None,
    pad_inches=None,
):
    """Rescale a figure's text to the house printed sizes, without new overlaps.

    Returns (applied_factor, page_scale, n_collisions_before, n_after).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    before = text_collisions(fig, renderer)

    factor, s = target_factor(fig, width_frac, bbox_inches, pad_inches)
    if not factor or not s:
        return None, s, len(before), len(before)

    texts = _visible_texts(fig)
    original = {id(t): t.get_fontsize() for t in texts}

    def apply(f):
        for t in texts:
            t.set_fontsize(original[id(t)] * f)
        fig.canvas.draw()
        return fig.canvas.get_renderer()

    # Shrinking cannot create an overlap, so it is applied directly. Growing
    # can, so step back toward 1.0 until it does not.
    applied = factor
    if factor > 1.0:
        lo, hi = 1.0, factor
        r = apply(hi)
        if os.environ.get("BIMP_PRINT_SCALE_DEBUG"):
            at_full = text_collisions(fig, r)
            print(
                "    [debug] at full x%.2f: %d collisions" % (factor, len(at_full)),
                flush=True,
            )
            for a, b in at_full[:10]:
                print("      [debug] %r / %r" % (a, b), flush=True)
        if len(text_collisions(fig, r)) > len(before):
            for _ in range(max_backoff):
                mid = 0.5 * (lo + hi)
                r = apply(mid)
                if len(text_collisions(fig, r)) > len(before):
                    hi = mid
                else:
                    lo = mid
            applied = lo
            apply(applied)
        else:
            applied = factor
    else:
        apply(applied)

    renderer = fig.canvas.get_renderer()
    after = text_collisions(fig, renderer)

    # Panel letters and tags are set to the exact house size regardless of the
    # tick-anchored factor, so letters match across figures. Reverted only if
    # this introduces a collision.
    role = _roles(fig)
    letters = [t for t in texts if role.get(id(t)) == "panel_letter"]
    tags = [t for t in texts if role.get(id(t)) == "panel_tag"]
    letters_forced = False
    tags_note = ""
    if letters:
        saved = {id(t): t.get_fontsize() for t in letters + tags}
        target_fig_pt = HOUSE_PT["panel_letter"] / s
        for t in letters:
            t.set_fontsize(target_fig_pt)
        for t in tags:
            t.set_fontsize(HOUSE_PT["panel_tag"] / s)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        forced_after = text_collisions(fig, renderer)
        if len(forced_after) > len(after):
            # Titles are the usual culprit (wider than letters), so fall back
            # to forcing the letters alone before giving up on both.
            for t in tags:
                t.set_fontsize(saved[id(t)])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            forced_after = text_collisions(fig, renderer)
            if tags and len(forced_after) <= len(after):
                after = forced_after
                letters_forced = True
                tags_note = "  (panel titles NOT forced: would collide)"
            else:
                for t in letters:
                    t.set_fontsize(saved[id(t)])
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
        else:
            after = forced_after
            letters_forced = True

    if verbose:
        short = " %s:" % label if label else ""
        note = ""
        if applied < factor * 0.995:
            note = "  (capped from %.2f -- layout cannot take the full size)" % factor
        letter_note = ""
        if letters:
            letter_note = (
                "  panel letters -> exact 8.0pt"
                if letters_forced
                else "  panel letters NOT forced (would collide)"
            ) + tags_note
        print(
            "    print-scale%s page %.3f, text x%.2f%s, collisions %d -> %d%s"
            % (short, s, applied, note, len(before), len(after), letter_note),
            flush=True,
        )
        if len(after) > len(before):
            for a, b in after[:6]:
                print("      COLLIDE %r / %r" % (a, b), flush=True)
    return applied, s, len(before), len(after)


# ── savefig hook ────────────────────────────────────────────────────────────


def width_frac_for(path):
    try:
        stem = pathlib.Path(str(path)).stem
    except Exception:
        return None
    return PRINTED_WIDTH_FRAC.get(stem)


def install():
    """Patch Figure.savefig so registered figures are normalised as written."""
    global _installed
    if _installed or os.environ.get(_ENV_OFF):
        return
    from matplotlib.figure import Figure

    orig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        frac = width_frac_for(fname) if isinstance(fname, (str, os.PathLike)) else None
        if frac is not None and not getattr(self, "_bimp_print_scaled", False):
            try:
                normalize_for_print(
                    self,
                    frac,
                    label=pathlib.Path(str(fname)).stem,
                    bbox_inches=kwargs.get("bbox_inches"),
                    pad_inches=kwargs.get("pad_inches"),
                )
                # A figure saved twice (PNG then PDF) must not be scaled twice.
                self._bimp_print_scaled = True
            except Exception as exc:  # never break a render over styling
                print("    print-scale skipped (%s)" % exc, flush=True)
        return orig(self, fname, *args, **kwargs)

    Figure.savefig = savefig
    _installed = True


__all__ = [
    "HOUSE_PT",
    "LINEWIDTH_IN",
    "PRINTED_WIDTH_FRAC",
    "TEXTHEIGHT_IN",
    "install",
    "normalize_for_print",
    "page_scale",
    "saved_width_in",
    "target_factor",
    "COLLISION_FRAC",
    "text_collisions",
    "width_frac_for",
]
