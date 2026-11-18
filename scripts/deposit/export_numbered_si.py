#!/usr/bin/env python
"""Journal-ready, numbered copies of the figures, SI items and Supplementary Data.

The LaTeX sources name every file by content slug and let LaTeX number the items.
Journals want numbered files, so this reads the numbers LaTeX actually printed
(the ``\\newlabel`` entries of the ``.aux`` files of a LaTeX build), maps each
label to its files through the registry (FIGURES, TABLES, SUPPLEMENTARY_DATA_ITEMS)
and copies

    Figure_<N>.<ext>                         main-text figures
    Supplementary_Figure_S<N>[_<k>].<ext>    (k numbers a float's files, if several)
    Supplementary_Table_S<N>.tex
    Supplementary_Data_<N>_<slug>.csv

into ``export/`` (gitignored; emptied first, so a reorder leaves no stale file),
with ``INDEX.md`` / ``INDEX.csv`` listing printed
number, label, source file, exported file and generator. Nothing is
renumbered by hand: rebuild the PDF, then rerun this.

    PYTHONPATH=src:scripts .venv/bin/python scripts/deposit/export_numbered_si.py \\
        --build DIR_WITH_AUX [--manuscript DIR] [--out export] [--readme]

``--build`` is the directory holding the compiled ``*.aux`` (a local build, or
Overleaf's "Other logs and files" download); ``--manuscript`` the checkout whose
files are copied (default: the submodule, or ``--build`` when it holds the
sources). ``--readme`` rewrites the item table of that checkout's README.md
between its ``<!-- items -->`` markers.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from paper.manuscript_figure_registry import (  # noqa: E402
    FIGURES,
    MANUSCRIPT,
    SUPPLEMENTARY_DATA_ITEMS,
    TABLES,
)

NEWLABEL = re.compile(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}")


def aux_numbers(build: Path) -> dict[str, str]:
    """label -> printed number, from every .aux in a LaTeX build directory."""
    out: dict[str, str] = {}
    for aux in sorted(build.glob("*.aux")):
        for m in NEWLABEL.finditer(aux.read_text(encoding="utf8", errors="replace")):
            out[m.group(1)] = m.group(2)
    if not out:
        raise SystemExit(f"no \\newlabel entries in {build}/*.aux: build the PDF first")
    return out


def items() -> list[dict]:
    """Every registered item: kind, label, LaTeX-side files, generator."""
    rows = []
    seen: dict[str, dict] = {}
    for f in FIGURES:
        row = seen.get(f.label)
        if row is None:
            row = {
                "kind": "Figure" if f.kind == "main" else "Supplementary Figure",
                "label": f.label,
                "parts": [],
                "generator": [],
            }
            seen[f.label] = row
            rows.append(row)
        # One part per file stem (PDF preferred: it is what LaTeX prints).
        ext = "pdf" if "pdf" in f.formats else f.formats[0]
        row["parts"].append(f"{f.manuscript_stem}.{ext}")
        if f.generator not in row["generator"]:
            row["generator"].append(f.generator)
    for t in TABLES:
        rows.append(
            {
                "kind": "Supplementary Table",
                "label": t.label,
                "parts": [f"tables/{t.slug}.tex"],
                "generator": [t.generator or "(written in the manuscript)"],
            }
        )
    for d in SUPPLEMENTARY_DATA_ITEMS:
        rows.append(
            {
                "kind": "Supplementary Data",
                "label": d.label,
                "parts": [f"supplementary_data/{d.slug}.csv"],
                "generator": ["scripts/deposit/supplementary_data_csvs.py"],
                "slug": d.slug,
            }
        )
    return rows


def export_name(row: dict, num: str, k: int, src: str) -> str:
    ext = Path(src).suffix
    many = len(row["parts"]) > 1
    part = f"_{k}" if many else ""
    if row["kind"] == "Figure":
        return f"Figure_{num}{part}{ext}"
    if row["kind"] == "Supplementary Figure":
        return f"Supplementary_Figure_{num}{part}{ext}"
    if row["kind"] == "Supplementary Table":
        return f"Supplementary_Table_{num}{ext}"
    return f"Supplementary_Data_{num}_{row['slug']}{ext}"


def printed(row: dict, num: str) -> str:
    return {
        "Figure": f"Fig. {num}",
        "Supplementary Figure": f"Supplementary Fig. {num}",
        "Supplementary Table": f"Supplementary Table {num}",
        "Supplementary Data": f"Supplementary Data {num}",
    }[row["kind"]]


def readme_table(rows: list[dict], nums: dict[str, str]) -> str:
    lines = ["| Printed | File | Generator (bimp-st) |", "|---|---|---|"]
    for row in rows:
        files = "<br>".join(f"`{p}`" for p in row["parts"])
        gens = "<br>".join(f"`{g}`" for g in row["generator"])
        lines.append(f"| {printed(row, nums[row['label']])} | {files} | {gens} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", type=Path, required=True, help="dir with built .aux")
    ap.add_argument("--manuscript", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "export")
    ap.add_argument("--readme", action="store_true")
    args = ap.parse_args()
    build = args.build.resolve()
    ms = args.manuscript or (build if (build / "figures").is_dir() else MANUSCRIPT)
    nums = aux_numbers(build)
    rows = items()
    absent = [r["label"] for r in rows if r["label"] not in nums]
    if absent:
        raise SystemExit(f"labels missing from the .aux (stale build?): {absent}")

    out = args.out
    if out.exists():
        if out.resolve() in (ROOT, ms.resolve()) or not (out / "INDEX.csv").exists():
            raise SystemExit(f"{out} exists and is not an earlier export; not clearing")
        shutil.rmtree(out)  # a stale Figure_7.* must not survive a reorder
    out.mkdir(parents=True)
    index = []
    for row in rows:
        num = nums[row["label"]]
        for k, src in enumerate(row["parts"], 1):
            name = export_name(row, num, k, src)
            shutil.copy2(ms / src, out / name)
            index.append(
                {
                    "printed": printed(row, num),
                    "label": row["label"],
                    "manuscript_file": src,
                    "export_file": name,
                    "generator": "; ".join(row["generator"]),
                }
            )
    with open(out / "INDEX.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    md = ["# Numbered export", "", f"Numbers read from `{build}/*.aux`.", ""]
    md += ["| Printed | Label | Manuscript file | Exported as |", "|---|---|---|---|"]
    md += [
        f"| {r['printed']} | `{r['label']}` | `{r['manuscript_file']}` | "
        f"`{r['export_file']}` |"
        for r in index
    ]
    (out / "INDEX.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"wrote {len(index)} files + INDEX.md/INDEX.csv to {out}")

    if args.readme:
        readme = ms / "README.md"
        text = readme.read_text(encoding="utf8")
        pat = re.compile(r"(<!-- items -->\n).*?(<!-- /items -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"{readme}: no <!-- items --> ... <!-- /items --> block")
        readme.write_text(
            pat.sub(
                lambda m: m.group(1) + readme_table(rows, nums) + "\n" + m.group(2),
                text,
            ),
            encoding="utf8",
        )
        print(f"rewrote the item table in {readme}")


if __name__ == "__main__":
    main()
