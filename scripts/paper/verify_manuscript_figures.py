"""Check every shipped manuscript item against the output it is supposed to be.

Walks manuscript_figure_registry: each figure file (PNG and, where it ships, the
PDF that LaTeX actually prints), each generated Supplementary table and each
Supplementary Data CSV is byte-compared with its results/ source; the registry is
checked for coverage (every file under figures/, tables/ and supplementary_data/
is registered, every \\includegraphics / \\input resolves to a registered file,
every registered generator exists and is git-tracked); and the Supplementary Data
list is checked to be in order of first citation (its order is what numbers it).

    PYTHONPATH=src:scripts .venv/bin/python scripts/paper/verify_manuscript_figures.py
    ... --manuscript DIR      a manuscript checkout other than the submodule
    ... --registry-only       skip the byte comparisons (no results/ needed)

`git add submodules/<name>` from the repo root commits the submodule pointer,
not the submodule's files, so a figure copied in can stay uncommitted while the
parent looks clean; comparing bytes catches this, mtimes do not. DIFFER means the
shipped copy and its source disagree: decide which one is current rather than
assuming.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import manuscript_text  # noqa: E402
from manuscript_figure_registry import (  # noqa: E402
    ANALYSIS_PREDECESSOR,
    FIGURES,
    MANUSCRIPT,
    NOT_REPRODUCIBLE,
    PRE_STEPS,
    SUPPLEMENTARY_DATA,
    SUPPLEMENTARY_DATA_ITEMS,
    SUPPLEMENTARY_DATA_SCRIPT,
    TABLES,
    declared_inputs,
)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--manuscript", type=pathlib.Path, default=MANUSCRIPT)
ap.add_argument(
    "--registry-only",
    action="store_true",
    help="skip byte comparisons (they need results/, a gitignored symlink)",
)
args = ap.parse_args()
MS = args.manuscript.resolve()
REGISTRY_ONLY = args.registry_only
W = pathlib.Path(__file__).resolve().parents[2]

if not (MS / "00_Article_Merge.tex").is_file():
    # The manuscript is a separate repository. Without a checkout, only the
    # registry's own consistency (every generator named exists and is tracked)
    # can be checked.
    if not REGISTRY_ONLY:
        raise SystemExit(
            f"no manuscript checkout at {MS}: set BIMP_MANUSCRIPT (or --manuscript) "
            "to compare the shipped items, or run with --registry-only"
        )
    _gens = (
        {f.generator for f in FIGURES}
        | {t.generator for t in TABLES if t.generator}
        | set(ANALYSIS_PREDECESSOR.values())
        | {s for steps in PRE_STEPS.values() for s in steps}
        | {SUPPLEMENTARY_DATA_SCRIPT}
    )
    _tracked = set(
        subprocess.run(
            ["git", "-C", str(W), "ls-files"], stdout=subprocess.PIPE, text=True
        ).stdout.split("\n")
    )
    _bad = sorted(s for s in _gens if not (W / s).exists() or s not in _tracked)
    print(
        f"skipped: manuscript checks (no manuscript checkout at {MS}; set "
        f"BIMP_MANUSCRIPT). Registry: {len(_gens)} generators, "
        f"{len(_bad)} missing or untracked."
    )
    for s in _bad:
        print(f"  generator missing or not git-tracked: {s}")
    raise SystemExit(1 if _bad else 0)


def md5(p: pathlib.Path) -> str:
    """Short content hash (PDF draw date blanked); see manuscript_text.md5."""
    h = manuscript_text.md5(p)
    return h if h == "MISSING" else h[:8]


def tex_body(name: str) -> str:
    """A .tex file with comments and \\iffalse ... \\fi blocks removed."""
    return manuscript_text.printed((MS / name).read_text(encoding="utf8"))


def tex_document() -> str:
    """The compiled text in reading order: \\include'd files, \\input expanded."""

    def expand(name: str) -> str:
        return re.sub(
            r"\\input\{([^}]+)\}",
            lambda m: expand(m.group(1).removesuffix(".tex") + ".tex"),
            tex_body(name),
        )

    merge = tex_body("00_Article_Merge.tex")
    return "".join(
        expand(m.group(1) + ".tex") for m in re.finditer(r"\\include\{([^}]+)\}", merge)
    )


compare_figures = manuscript_text.compare_figures
fatal: list[str] = []

# ── Figures: shipped bytes vs results/ ───────────────────────────────────────
n_match = n_differ = 0
if REGISTRY_ONLY:
    print("--registry-only: skipping byte comparisons (they require results/).\n")
else:
    n_match, n_differ, _fig_fatal = compare_figures(FIGURES, MS, W, NOT_REPRODUCIBLE)
    fatal += _fig_fatal

registered = {m for f in FIGURES for m, _ in f.files()}
shipped = sorted(
    str(p.relative_to(MS)) for p in (MS / "figures").rglob("*") if p.is_file()
)
untracked = [n for n in shipped if n not in registered]
missing = sorted(n for n in registered if not (MS / n).exists())
print(
    f"\nFIGURES: {n_match} MATCH, {n_differ} DIFFER; {len(registered)} registered "
    f"files ({len(FIGURES)} figures), {len(shipped)} shipped, "
    f"{len(untracked)} unregistered, {len(missing)} registered but not shipped"
)
for n in untracked:
    print(f"  unregistered: {n}")
for n in missing:
    print(f"  not shipped: {n}")
fatal += [f"unregistered figure file {n}" for n in untracked]
fatal += [f"registered figure not shipped {n}" for n in missing]

# Every \includegraphics names a registered stem, and every registered stem is
# included (an unreferenced file under figures/ is clutter in the Overleaf tree).
doc = tex_document()
included = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", doc))
stems = {f.manuscript_stem for f in FIGURES}
for s in sorted(included - stems):
    fatal.append(f"\\includegraphics{{{s}}} is not a registered figure")
for s in sorted(stems - included):
    fatal.append(f"registered figure {s} is not \\includegraphics'd")
# Each registered label is the label of the float that includes the file.
for f in FIGURES:
    m = re.search(
        r"\\includegraphics(?:\[[^\]]*\])?\{"
        + re.escape(f.manuscript_stem)
        + r"\}(.*?)\\end\{figure\*?\}",
        doc,
        flags=re.S,
    )
    if m and f"\\label{{{f.label}}}" not in m.group(1):
        fatal.append(f"{f.manuscript_stem}: its float is not labelled {f.label}")

# ── Supplementary tables ─────────────────────────────────────────────────────
tab_inputs = set(
    re.findall(
        r"\\input\{tables/([^}]+)\}",
        "".join(tex_body(p.name) for p in MS.glob("0*.tex")),
    )
)
tab_shipped = sorted(p.stem for p in (MS / "tables").glob("*.tex"))
print(f"\nTABLES: {len(TABLES)} registered, {len(tab_shipped)} shipped")
for t in TABLES:
    path = MS / "tables" / f"{t.slug}.tex"
    if not path.exists():
        fatal.append(f"registered table not shipped tables/{t.slug}.tex")
    if t.slug not in tab_inputs:
        fatal.append(f"tables/{t.slug}.tex is not \\input")
    if f"\\label{{{t.label}}}" not in path.read_text(encoding="utf8"):
        fatal.append(f"tables/{t.slug}.tex does not carry \\label{{{t.label}}}")
    if t.results and not REGISTRY_ONLY:
        a, b = md5(path), md5(W / t.results)
        print(f"{'MATCH ' if a == b else 'DIFFER'}  tables/{t.slug}.tex  {a}  {b}")
        if a != b or a == "MISSING":
            fatal.append(f"table DIFFER tables/{t.slug}.tex (shipped {a}, results {b})")
    elif not t.results:
        print(f"  hand   tables/{t.slug}.tex (written in the manuscript)")
for s in tab_shipped:
    if s not in {t.slug for t in TABLES}:
        fatal.append(f"unregistered table tables/{s}.tex")

# ── Supplementary Data files ─────────────────────────────────────────────────
# Deterministic text written by SUPPLEMENTARY_DATA_SCRIPT, so a DIFFER, a missing
# copy or a shipped file with no registry entry is a hard failure.
SDDIR = MS / "supplementary_data"
sd_differ = []
if not REGISTRY_ONLY:
    print()
    for m, s in SUPPLEMENTARY_DATA.items():
        a, b = md5(MS / m), md5(W / s)
        ok = a == b and a != "MISSING"
        if not ok:
            sd_differ.append(m)
        print(f"{'MATCH ' if ok else 'DIFFER'}  {m:<56} {a}  {b}")
sd_shipped = sorted(
    f"supplementary_data/{p.name}" for p in SDDIR.iterdir() if p.is_file()
)
sd_unregistered = [n for n in sd_shipped if n not in SUPPLEMENTARY_DATA]
print(
    f"\nSUPPLEMENTARY DATA: {len(SUPPLEMENTARY_DATA)} registered, "
    f"{len(sd_shipped)} shipped, {len(sd_differ)} DIFFER/MISSING, "
    f"{len(sd_unregistered)} shipped but unregistered"
)
fatal += [f"Supplementary Data DIFFER/MISSING {n}" for n in sd_differ]
fatal += [f"unregistered Supplementary Data file {n}" for n in sd_unregistered]

# The \suppdata list numbers the items, so its order must be the registry's and
# the order in which the paper first cites each label.
listed = re.findall(r"\\suppdata\{(sd:[^}]+)\}\{([^}]*)\}", doc)
first_cited: list[str] = []
for lab in re.findall(r"\\ref\{(sd:[^}]+)\}", doc):
    if lab not in first_cited:
        first_cited.append(lab)
want = [(d.label, f"{d.slug}.csv") for d in SUPPLEMENTARY_DATA_ITEMS]
got = [(lab, fn.replace("\\_", "_")) for lab, fn in listed]
order_ok = got == want and [lab for lab, _ in want] == first_cited
print(
    "  \\suppdata list = registry order = first-citation order: "
    + ("yes" if order_ok else "NO")
)
if not order_ok:
    print(f"    listed:      {[lab for lab, _ in got]}")
    print(f"    registry:    {[lab for lab, _ in want]}")
    print(f"    first cited: {first_cited}")
    fatal.append("Supplementary Data list order")
# No hand-typed numbers for SI items anywhere in the compiled text.
hand = re.findall(
    r"(?:Supplementary\s*(?:Fig(?:ure)?s?\.?|Tables?|Data|Videos?)|Table|Fig\.)"
    r"[~\s]*S?\d+\b",
    doc,
)
print(f"  hand-typed SI numbers in the text: {len(hand)}")
fatal += [f"hand-typed SI number: {h!r}" for h in hand]

for n, why in sorted(NOT_REPRODUCIBLE.items()):
    print(f"  NOT byte-reproducible: {n} -- {why}")

# ── Can each shipped item be recreated? ──────────────────────────────────────
# A generating script is named, exists and is tracked, and its declared inputs
# are on disk. Nothing is re-run, so this is not proof of byte-reproducibility.
generators = (
    {f.generator for f in FIGURES}
    | {t.generator for t in TABLES if t.generator}
    | set(ANALYSIS_PREDECESSOR.values())
    | {s for steps in PRE_STEPS.values() for s in steps}
    | {SUPPLEMENTARY_DATA_SCRIPT}
)
missing_script = sorted(s for s in generators if not (W / s).exists())
tracked = set(
    subprocess.run(
        ["git", "-C", str(W), "ls-files"], stdout=subprocess.PIPE, text=True
    ).stdout.split("\n")
)
untracked_script = sorted(
    s for s in generators if (W / s).exists() and s not in tracked
)
missing_inputs = [
    (s, v)
    for s in sorted({f.generator for f in FIGURES})
    for v in declared_inputs(s)
    if not pathlib.Path(v).exists()
]
print(
    f"\nRECREATE: {len(generators)} generators, {len(missing_script)} missing, "
    f"{len(untracked_script)} untracked, {len(missing_inputs)} missing declared inputs"
)
for s in missing_script:
    print(f"  generator does not exist: {s}")
for s in untracked_script:
    print(f"  generator not git-tracked (absent from a clean clone): {s}")
for s, v in missing_inputs:
    print(f"  declared input missing: {s} -> {v}")
fatal += [f"generator missing {s}" for s in missing_script]
fatal += [f"generator untracked {s}" for s in untracked_script]

# ── PRINTED_WIDTH_FRAC vs the .tex ───────────────────────────────────────────
# bimp_st.print_scale scales a shipped figure's text by the width fraction the
# paper places it at, keyed by the results/ stem the generator writes.
sys.path.insert(0, str(W / "src"))
from bimp_st.print_scale import PRINTED_WIDTH_FRAC  # noqa: E402

tex_frac = {}
for m in re.finditer(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", doc):
    w = re.search(r"width=([0-9.]*)\\linewidth", m.group(1))
    tex_frac[m.group(2)] = float(w.group(1)) if w and w.group(1) else 1.0
frac_problems = []
for f in FIGURES:
    stem = pathlib.PurePath(f.results).name
    if f.generator == "scripts/paper/split_tall_si_figures.py":
        stem = "fig_s_xenium_qc"  # a PIL crop; the scaled render is the tall source
    want_w, have = tex_frac.get(f.manuscript_stem), PRINTED_WIDTH_FRAC.get(stem)
    if want_w is None:
        continue
    if have is None:
        frac_problems.append(f"{stem}: not in PRINTED_WIDTH_FRAC (.tex: {want_w:g})")
    elif abs(have - want_w) > 1e-6:
        frac_problems.append(
            f"{stem}: table says {have:g}, .tex places it at {want_w:g}"
        )
print(
    f"\nWIDTH FRACTIONS: {len(PRINTED_WIDTH_FRAC)} entries, "
    f"{len(frac_problems)} disagree with the .tex"
)
for p in frac_problems:
    print("  " + p)
fatal += frac_problems

# ── Exit status ──────────────────────────────────────────────────────────────
# Every defect above is a hard failure, including a figure or table whose shipped
# bytes differ from results/ (only NOT_REPRODUCIBLE entries are excused).
if fatal:
    print("\nREGISTRY FAILURE:")
    for m in fatal:
        print(f"  {m}")
    raise SystemExit(1)
print("\nregistry OK: every shipped item is registered, cited and has a generator.")
