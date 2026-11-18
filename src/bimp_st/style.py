"""The shared figure style: one place for every figure's look.

Two calls cover a figure:

``set_pub_style()``
    Applies the house rcParams (font, sizes, line widths, spines, save
    settings) and installs the print-scale hook
    (:mod:`bimp_st.print_scale`), which rescales a registered figure's text at save
    time so it prints at the house sizes in ``HOUSE_PT`` (tick 6 pt, axis label
    and title 7 pt, panel letter 8 pt, panel title 7 pt) whatever the canvas
    width. Main and SI figures call it once at the top of the script.

``add_panel_headers(fig, [(axes, "A", "Title"), ...])``
    The house panel header: a BOLD UPPERCASE letter at the panel's top-left
    with a BOLD BLACK short title on the same baseline (Figs 1-6). The letter
    and title carry gids that the print-scale hook sets to exactly 8 pt and
    7 pt printed. ``panel_header`` is the one-panel form.

Colours shared across figures (queen gold, worker blue) live here too;
``bimp_st.si_common`` re-exports everything for older imports.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from bimp_st.print_scale import HOUSE_PT, LINEWIDTH_IN

# The font every figure renders in. Arial / Helvetica are not installed on the
# cluster, so the old ["Arial", "Helvetica Neue", "DejaVu Sans"] stack already
# resolved to DejaVu Sans; naming it alone stops a figure that lists another
# installed face (Liberation Sans) from rendering in a different font.
FONT_FAMILY = "DejaVu Sans"
DEFAULT_FONT_STACK = [FONT_FAMILY]

# Caste colours: queen gold, worker blue.
QUEEN_COLOR = "#DAA520"
WORKER_COLOR = "#4A90D9"
CASTE_COLORS = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}

# Printed size of the panel letter (pt); the hook enforces HOUSE_PT anyway.
PANEL_LABEL_PRINTED_PT = HOUSE_PT["panel_letter"]


def set_pub_style(
    *,
    font_size: float = 9,
    title_size: float = 11,
    label_size: float = 10,
    tick_size: float = 8,
    legend_size: float = 8,
    figure_dpi: int = 200,
    savefig_dpi: int = 300,
    axes_linewidth: float = 0.6,
    pad_inches: float = 0.02,
    font_family: str = "sans-serif",
    font_stack: list[str] | None = None,
    print_scale: bool = True,
) -> None:
    """Apply the house rcParams block.

    All parameters are keyword-only. Sizes are canvas points: at save time the
    print-scale hook rescales a registered figure's text so its median tick
    label prints at ``HOUSE_PT["tick"]`` (one factor per figure, keeping the
    figure's own size hierarchy) and its panel letters and titles print at
    exactly ``HOUSE_PT["panel_letter"]`` / ``HOUSE_PT["panel_tag"]``.

    ``print_scale`` installs :func:`bimp_st.print_scale.install`. Set False, or
    export BIMP_NO_PRINT_SCALE, to render at the authored sizes.
    """
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": font_stack or DEFAULT_FONT_STACK,
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.titleweight": "bold",
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": axes_linewidth,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": pad_inches,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    if print_scale:
        from bimp_st.print_scale import install

        install()


# ── Panel headers ───────────────────────────────────────────────────────────


def panel_label_size(fig_width_in, width_frac=1.0, printed_pt=PANEL_LABEL_PRINTED_PT):
    """Font size that prints panel letters at `printed_pt` on the page.

    A matplotlib fontsize is in points of the figure canvas, so the same literal
    prints at different sizes depending on figure width and the fraction of
    \\linewidth it is placed at (fontsize=16 prints at 6.7 pt on a 16 in figure
    and 11.2 pt on a 9.5 in one).

    printed = fontsize * (printed_width_in / fig_width_in); this inverts it.
    Example: a 13.5 in figure at full \\linewidth needs 8 * 13.5 / 6.67 = 16.2
    to print at 8 pt.
    """
    return printed_pt * fig_width_in / (width_frac * LINEWIDTH_IN)


def add_panel_headers(
    fig,
    items,
    *,
    tag_size=None,
    dy_pts=None,
    width_frac=1.0,
    letter_size=None,
    letter_ha="left",
):
    """Bold panel letter at each panel's top-left plus a short bold title.

    items : list of (anchor, letter, title_or_None). The anchor is an Axes, a
    SubplotSpec, or an (x, y) point in figure fraction; the letter's left edge
    sits at the anchor's x and its baseline just above the anchor's y.

    Letter and title share one baseline; the title is anchored to the right
    edge of the letter's own text box, so the gap follows the letter's width.
    All offsets are in font-size units (not points) and the anchor is the axes
    in axes-fraction coordinates, so everything survives bbox_inches="tight" and
    the print_scale hook's rescaling (which forces the letter to 8 pt and the
    title to 7 pt printed; artists carry gids 'panel_header_letter'/'_tag').
    Letter size defaults to panel_label_size(fig width, width_frac); title to
    0.65 of that. dy_pts overrides the default 0.375-em lift in points.
    letter_ha="right" hangs the letter left of the anchor (its right edge on
    the anchor's x), for letters set in the margin beside a panel.
    A SubplotSpec is re-read from its gridspec at every draw (works with the
    hook and subplots_adjust; with constrained/tight layout pass real Axes).
    """
    from matplotlib.gridspec import SubplotSpec

    ls = letter_size or panel_label_size(fig.get_figwidth(), width_frac)
    ts = tag_size or 0.65 * ls
    out = []
    for target, letter, tag in items:
        if isinstance(target, SubplotSpec):
            kw = dict(xy=(0, 1), xycoords=_cell_bbox(fig, target))
        elif isinstance(target, tuple):
            # the Transform object, not the string "figure fraction": under
            # bbox_inches="tight" the string is not shifted with the crop
            kw = dict(xy=target, xycoords=fig.transFigure)
        else:
            kw = dict(xy=(0, 1), xycoords=target.transAxes)
        if dy_pts is None:
            off, oc = (0, 0.375), "offset fontsize"
        else:
            off, oc = (0, dy_pts), "offset points"
        a = fig.add_artist(_mk_ann(fig, letter, kw, off, oc, ls, ha=letter_ha))
        a.set_gid("panel_header_letter")
        t = None
        if tag:
            toff, toc = (0.5, 0), "offset fontsize"
            t = _mk_ann(
                fig, tag, dict(xy=(0, 0), xycoords=_baseline_right(a)), toff, toc, ts
            )
            fig.add_artist(t)
            t.set_gid("panel_header_tag")
        out.append((a, t))
    return out


def panel_header(fig, anchor, letter, title=None, **kw):
    """One panel's header; see :func:`add_panel_headers`."""
    return add_panel_headers(fig, [(anchor, letter, title)], **kw)[0]


def add_panel_labels(fig, axes_and_labels, offset_x=-0.01, offset_y=0.01):
    """Letter-only wrapper kept for existing callers; see add_panel_headers."""
    return add_panel_headers(
        fig, [(item, label, None) for item, label in axes_and_labels]
    )


def _cell_bbox(fig, spec):
    """Lazy display-space bbox of a SubplotSpec (re-read at every draw)."""
    from matplotlib.transforms import TransformedBbox

    def get(renderer):
        return TransformedBbox(spec.get_position(fig), fig.transFigure)

    return get


def _baseline_right(letter):
    """Lazy zero-size bbox at the letter's right edge, ON its baseline."""
    from matplotlib.transforms import Bbox

    def get(renderer):
        bb = letter.get_window_extent(renderer)
        fp = letter.get_fontproperties()
        d = max(
            renderer.get_text_width_height_descent(letter.get_text(), fp, False)[2],
            renderer.get_text_width_height_descent("lp", fp, False)[2],
        )
        return Bbox([[bb.x1, bb.y0 + d], [bb.x1, bb.y0 + d]])

    return get


def _mk_ann(fig, s, kw, off, oc, size, ha="left"):
    from matplotlib.text import Annotation

    return Annotation(
        s,
        xytext=off,
        textcoords=oc,
        fontsize=size,
        fontweight="bold",
        color="black",
        va="baseline",
        ha=ha,
        annotation_clip=False,
        **kw,
    )


__all__ = [
    "CASTE_COLORS",
    "DEFAULT_FONT_STACK",
    "FONT_FAMILY",
    "HOUSE_PT",
    "PANEL_LABEL_PRINTED_PT",
    "QUEEN_COLOR",
    "WORKER_COLOR",
    "add_panel_headers",
    "add_panel_labels",
    "panel_header",
    "panel_label_size",
    "set_pub_style",
]
