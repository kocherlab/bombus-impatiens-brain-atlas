#!/usr/bin/env python3
r"""Copy current results/ outputs into the manuscript, by the registry's map.

Every shipped figure file (manuscript_figure_registry.FIGURES: PNG and, where it
ships, PDF), every generated Supplementary table (TABLES) and every
Supplementary Data file (SUPPLEMENTARY_DATA) has one results/ source; this copies
the ones whose bytes differ. Dry run by default: it lists what would change.

    PYTHONPATH=src:scripts .venv/bin/python scripts/paper/sync_manuscript_figures.py
    PYTHONPATH=src:scripts .venv/bin/python scripts/paper/sync_manuscript_figures.py \
        --apply
    ... --manuscript DIR   a manuscript checkout other than the submodule

The aspect guard refuses a figure whose PNG or PDF aspect ratio changed by more
than ASPECT_TOLERANCE: that is a layout change, not a re-render (e.g. a stale
results/ copy with a different panel), and should be a deliberate decision
(--allow-reshape SLUG) rather than a sync. The guard runs per figure before any
of its files is copied, so a blocked figure never ships a new PDF (which LaTeX
prints) beside its old PNG. Bytes are compared as the figure check compares
them (manuscript_text.md5: a PDF's draw date is ignored).

Afterwards run scripts/paper/verify_manuscript_figures.py (all MATCH) and rebuild the
PDF. The Xenium QC figure is a crop: run split_tall_si_figures.py after
si_xenium_run_qc.py, before syncing.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manuscript_text import md5  # noqa: E402

ASPECT_TOLERANCE = 0.15  # fractional change that trips the layout-change guard

_NUM = rb"\s*([-\d.]+)"
_MEDIABOX = re.compile(rb"/MediaBox\s*\[" + _NUM * 4)


def aspect(p: Path) -> float:
    """Height / width of a PNG, or of a PDF's first MediaBox."""
    if p.suffix == ".pdf":
        m = _MEDIABOX.search(p.read_bytes())
        if not m:
            raise ValueError(f"{p}: no /MediaBox")
        x0, y0, x1, y1 = (float(v) for v in m.groups())
        return (y1 - y0) / (x1 - x0)
    w, h = Image.open(p).size
    return h / w


def reshaped(src: Path, dst: Path) -> str | None:
    """Why replacing image ``dst`` by ``src`` is a layout change, or None."""
    if dst.suffix not in (".png", ".pdf") or not (src.exists() and dst.exists()):
        return None
    a_old, a_new = aspect(dst), aspect(src)
    if abs(a_new - a_old) / a_old > ASPECT_TOLERANCE:
        return f"{dst.name}: aspect {a_old:.2f} -> {a_new:.2f}"
    return None


def registered_pairs() -> list[tuple[str, str, str]]:
    """(slug, manuscript path, results path) for every registered file."""
    from manuscript_figure_registry import FIGURES, SUPPLEMENTARY_DATA, TABLES

    pairs = [(f.slug, m, r) for f in FIGURES for m, r in f.files()]
    pairs += [(t.slug, f"tables/{t.slug}.tex", t.results) for t in TABLES if t.results]
    pairs += [(Path(m).stem, m, r) for m, r in SUPPLEMENTARY_DATA.items()]
    return pairs


def sync(pairs, repo: Path, ms: Path, *, apply=False, allow=()) -> dict:
    """Copy each results/ file whose bytes differ, guarding whole figures."""
    allow = set(allow)
    # The guard looks at every file of a figure before any of them is copied.
    blocked: dict[str, list[str]] = {}
    for slug, m, r in pairs:
        if slug not in allow and md5(repo / r) != md5(ms / m):
            why = reshaped(repo / r, ms / m)
            if why:
                blocked.setdefault(slug, []).append(why)
    for slug, whys in blocked.items():
        print(
            f"BLOCKED  {slug}: {'; '.join(whys)}, a layout change. Look at both "
            f"renders, then pass --allow-reshape {slug} if intended."
        )

    n = {"copy": 0, "same": 0, "block": 0, "missing": 0}
    for slug, m, r in pairs:
        src, dst = repo / r, ms / m
        if not src.exists():
            print(f"NO-SRC   {m}  <- {r}")
            n["missing"] += 1
        elif md5(src) == md5(dst):
            n["same"] += 1
        elif slug in blocked:
            n["block"] += 1
        else:
            print(f"{'copy' if apply else 'would copy':<10} {m}  <- {r}")
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            n["copy"] += 1
    return n


def main(argv=None) -> int:
    from manuscript_figure_registry import MANUSCRIPT, REPO

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="copy (default: dry run)")
    ap.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    ap.add_argument(
        "--allow-reshape",
        nargs="+",
        default=[],
        metavar="SLUG",
        help="figures whose aspect ratio may change, named one by one, after "
        "looking at BOTH renders. There is deliberately no allow-everything form.",
    )
    args = ap.parse_args(argv)
    pairs = registered_pairs()
    n = sync(
        pairs,
        REPO,
        args.manuscript.resolve(),
        apply=args.apply,
        allow=args.allow_reshape,
    )
    print(
        f"\n{len(pairs)} registered files: {n['same']} current, "
        f"{n['copy']} {'copied' if args.apply else 'to copy'}, {n['block']} blocked, "
        f"{n['missing']} with no results/ source"
        + ("" if args.apply else "  (dry run; --apply to copy)")
    )
    return 1 if (n["block"] or n["missing"]) else 0


if __name__ == "__main__":
    sys.exit(main())
