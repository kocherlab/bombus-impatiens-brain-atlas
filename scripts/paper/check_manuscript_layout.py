r"""Check that no manuscript figure or caption overruns the page.

Two independent checks, because neither alone catches everything:

  --predict   From the PNG pixel dimensions and the \includegraphics width
              fraction, compute how tall each figure will render. Runs without
              building the PDF and works on any git ref, so it can answer
              "is main broken?" without checking main out.

  --measure   Parse a built PDF and report the lowest image and text box on
              every page. Ground truth.

LaTeX does not warn here: SI floats are ``\begin{figure*}[!h]``, and ``!``
ignores \floatpagefraction, so an oversized float runs off the page with a
clean log.

Usage:
    .venv/bin/python scripts/paper/check_manuscript_layout.py --predict
    .venv/bin/python scripts/paper/check_manuscript_layout.py --predict --ref \
        origin/main
    .venv/bin/python scripts/paper/check_manuscript_layout.py --measure path/to.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

MANUSCRIPT = Path(
    os.environ.get(
        "BIMP_MANUSCRIPT",
        Path(__file__).resolve().parents[2]
        / "submodules/bombus_impatiens_atlas_manuscript",
    )
)
TEX_FILES = ("02_Article_Supplementary.tex", "01_Article_MainText.tex")

# bioRxiv.cls on A4, measured off a build rather than read off the class file.
LINEWIDTH_PT = 481.9  # \linewidth in the SI's \onecolumn mode
TEXTHEIGHT_PT = 700.0
# Bottom of the last body line. Most pages bottom out at 75.9, but a final line
# with deep descenders reaches ~74.9; real overruns are far larger (tens to
# hundreds of pt).
TEXT_BOTTOM_Y = 74.5
FOOTER_BAND = (44.0, 58.0)  # running footer + bioRxiv logo, below the block by design

# A figure taller than this leaves no room for its caption. Captions in this SI
# run 6-14 lines, so budget generously.
MAX_FRACTION = 0.72


def _png_size(blob: bytes) -> tuple[int, int] | None:
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", blob[16:24])
    return w, h


def _read(path: str, ref: str | None) -> bytes:
    if ref is None:
        target = MANUSCRIPT / path
        return target.read_bytes() if target.exists() else b""
    out = subprocess.run(
        ["git", "-C", str(MANUSCRIPT), "show", f"{ref}:{path}"],
        capture_output=True,
    )
    return out.stdout


def predict(ref: str | None) -> int:
    """Report figures whose rendered height leaves no room for a caption."""
    bad = 0
    for tex_name in TEX_FILES:
        tex = _read(tex_name, ref).decode("utf8", "replace")
        includes = re.findall(r"\\includegraphics\[([^\]]*)\]\{figures/([^}]+)\}", tex)
        if not includes:
            continue
        print(f"\n=== {tex_name} @ {ref or 'working tree'} ===")
        print(
            f"{'#':>3} {'figure':<34} {'frac':>5} {'pixels':>12} "
            f"{'h/w':>5} {'height':>7} {'%TH':>5}"
        )
        for i, (opts, name) in enumerate(includes, 1):
            m = re.search(r"width=([\d.]+)\\linewidth", opts)
            frac = float(m.group(1)) if m else 1.0
            m = re.search(r"height=([\d.]+)\\textheight", opts)
            cap = float(m.group(1)) if m else None

            size = _png_size(_read(f"figures/{name}.png", ref))
            if size is None:
                print(f"{i:>3} {name:<34} {frac:>5}   (not a PNG / missing)")
                continue
            w, h = size

            # trim/clip is not modelled: it has silently failed to take effect in
            # this document, so modelling it could turn an overflow into a pass.
            # Split over-tall artwork into separate files instead.
            rendered = frac * LINEWIDTH_PT * (h / w)
            capped = cap is not None and cap * TEXTHEIGHT_PT < rendered
            if capped:
                rendered = cap * TEXTHEIGHT_PT
            pct = rendered / TEXTHEIGHT_PT
            flag = ""
            if pct > MAX_FRACTION:
                flag, bad = "  <== NO ROOM FOR CAPTION", bad + 1
            elif capped:
                # Not overflowing, but only because a height= cap shrinks it --
                # which wastes page width. Reshape the figure instead.
                flag = "  (held down by a height= cap)"
            print(
                f"{i:>3} {name:<34} {frac:>5} {w:>6}x{h:<5} {h / w:>5.2f} "
                f"{rendered:>7.0f} {100 * pct:>5.0f}{flag}"
            )
    return bad


def measure(pdf: str) -> int:
    """Report anything that actually landed below the text block."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTImage, LTTextContainer

    def walk(obj):
        yield obj
        if hasattr(obj, "__iter__"):
            for child in obj:
                yield from walk(child)

    def is_footer(obj) -> bool:
        return obj.y0 >= FOOTER_BAND[0] and obj.y1 <= FOOTER_BAND[1] + 2

    bad = pno = 0
    for pno, page in enumerate(extract_pages(pdf), 1):
        low_text, snippet, low_img = 1e9, None, 1e9
        for obj in walk(page):
            if isinstance(obj, LTTextContainer):
                if not obj.get_text().strip() or is_footer(obj):
                    continue
                if obj.y0 < low_text:
                    low_text, snippet = obj.y0, obj.get_text().strip()
            elif isinstance(obj, LTImage) and not is_footer(obj):
                low_img = min(low_img, obj.y0)
        lowest = min(low_text, low_img)
        if lowest < TEXT_BOTTOM_Y:
            kind = "text" if low_text <= low_img else "image"
            print(
                f"p{pno:<3} {kind} reaches y={lowest:.1f}, "
                f"{TEXT_BOTTOM_Y - lowest:.1f}pt below the text block"
                f"  {(snippet or '')[:60]!r}"
            )
            bad += 1
    if not bad:
        print(f"{pno} pages: nothing below the text block (y={TEXT_BOTTOM_Y}).")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--measure", metavar="PDF")
    ap.add_argument("--ref", help="git ref to inspect instead of the worktree")
    args = ap.parse_args()
    if not args.predict and not args.measure:
        ap.error("pass --predict, --measure PDF, or both")

    bad = 0
    if args.predict:
        bad += predict(args.ref)
    if args.measure:
        bad += measure(args.measure)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
