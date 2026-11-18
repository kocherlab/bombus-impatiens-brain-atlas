#!/usr/bin/env python
"""Flag result-style text in comments, docstrings and Markdown.

    python scripts/paper/leak_lint.py [--root DIR] [--allow FILE] [PATH ...]

Comments and docstrings say what code computes or tests. They do not quote
results (test statistics, P values, correlations, effect sizes, agreement
percentages) or describe a text that is not in the repository. The lint reads
the comments and docstrings of every .py, .smk, Snakefile and .sh file, the
comments of every .js and .mjs file and every line of every .md file, and
reports each line matching one of PATTERNS. Without PATH arguments it reads
every file ``git ls-files`` lists under DIR (every file under DIR outside a git
checkout).

The allow file (default ``scripts/paper/leak_lint_allow.txt``, if present) is
tab-separated; ``#`` starts a comment line. Two kinds of row:

    allow  <pattern id>  <path glob>  <regex the line must match>  <reason>
    skip   <path glob>   <reason>

An ``allow`` row names one pattern id (there is no wildcard); a ``skip`` row
leaves files out of the scan. Every row needs a reason. Exit status 1 on any
hit no row excuses.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

_STAT = (
    r"rho|AUROC|AUC|Spearman|Pearson|padj|FDR|TVD|Cram[eé]r|log2FC|LFC|kappa|"
    r"ARI|NMI|accuracy|recall|precision|correlation|median|pseudo-?F|R\^?2"
)
# a number after a comparison is a threshold or tolerance, not a result
_THRESHOLD = (
    r"(?<![<>≤≥\d.])(?<![<>≤≥] )(?<![<>]=)(?<![<>]= )"
    r"(?!0\.0?5\b|0\.0?1\b|0\.10?\b|0\.001\b)"
)
PATTERNS: dict[str, str] = {
    "p-value": r"\bP ?[=<≤]",
    "rho": r"(?i)(?:\brho|ρ) ?[=≈~]",
    "r-squared": r"R\^2|R²|\bR2 ?(?:=|of\b|\d)",
    "median": r"(?i)\bmedians? (?:\w+ ){0,2}-?\d*\.\d",
    "pseudo-f": r"(?i)\bpseudo-?F\b",
    "fold": (
        r"(?i)\d(?:\.\d+)? ?-?fold\b|\bfold\b[^\n]{0,25}?\d\.\d|"
        r"\b\d+\.\d+ ?[x×](?!\w)"
    ),
    "stat-value": rf"(?i)\b(?:{_STAT})\b[^\n]{{0,25}}?{_THRESHOLD}\d+\.\d{{2,}}",
    "pct-decimal": r"\b\d+\.\d+ ?%",
    "pct-score": (
        r"(?i)\d+(?:\.\d+)? ?%[^\n]{0,40}\b(?:agree|accura|recall|precision|"
        r"concordan|purit)|\b(?:agree|accura|recall|precision|concordan|purit)"
        r"\w*[^\n]{0,40}?\d+(?:\.\d+)? ?%"
    ),
    "pp": r"\b\d+(?:\.\d+)? ?pp\b",
    "text-quotes": r"(?i)\btext quotes\b",
    "the-paper": r"(?i)\bthe paper\b",
    "manuscript": r"(?i)\bmanuscript\b",
    "shipped": r"(?i)\bshipped\b",
    "printed-fig": r"(?i)\bprinted Fig",
    "fig-number": r"\bFig(?:ure)?\.? ?S?\d|\b(?:fig|figure)(?:\.|\. | )S?\d",
    "main-text": r"(?i)\bmain text\b",
    "supp-methods": r"(?i)\bSupplementary Methods\b",
    "amendment": r"(?i)\bamendment",
    "pre-registration": r"(?i)\bpre-?regist",
}
COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}
COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}
DEFAULT_ALLOW = "scripts/paper/leak_lint_allow.txt"
# Snakemake directives whose value may be a string on the line after the colon
SMK_DIRECTIVES = {
    "shell",
    "message",
    "input",
    "output",
    "params",
    "log",
    "benchmark",
    "script",
    "notebook",
    "wrapper",
    "conda",
    "container",
    "resources",
    "threads",
    "wildcard_constraints",
}


def _kind(path: str) -> str | None:
    name = path.rsplit("/", 1)[-1]
    if path.endswith((".py", ".smk")) or name == "Snakefile":
        return "py"
    if path.endswith((".js", ".mjs")):
        return "js"
    for ext in (".sh", ".md"):
        if path.endswith(ext):
            return ext[1:]
    return None


def _py_text(src: str) -> list[tuple[int, str]]:
    """(line, text) of comments and of string statements (docstrings)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return list(enumerate(src.splitlines(), 1))
    out: list[tuple[int, str]] = []
    for t in toks:
        if t.type == tokenize.COMMENT and not (
            t.start[0] == 1 and t.string[:2] == "#!"
        ):
            out.append((t.start[0], t.string))
    sig = [t for t in toks if t.type not in (tokenize.NL, tokenize.COMMENT)]
    breaks = (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    for i, t in enumerate(sig):
        if t.type != tokenize.STRING:
            continue
        before = sig[i - 1].type if i else tokenize.NEWLINE
        after = sig[i + 1].type if i + 1 < len(sig) else tokenize.ENDMARKER
        if before not in breaks or after not in (tokenize.NEWLINE, tokenize.ENDMARKER):
            continue
        # a Snakemake directive's value on its own line ("shell:" NEWLINE INDENT
        # STRING) is code, not a docstring
        j = i - 1
        while j >= 0 and sig[j].type in breaks:
            j -= 1
        if j >= 0 and sig[j].string == ":":
            k = j
            while k > 0 and sig[k - 1].type not in breaks:
                k -= 1
            if sig[k].string in SMK_DIRECTIVES:
                continue
        for n, ln in enumerate(t.string.splitlines()):
            out.append((t.start[0] + n, ln))
    return out


def _sh_text(src: str) -> list[tuple[int, str]]:
    out = []
    for i, ln in enumerate(src.splitlines(), 1):
        if i == 1 and ln.startswith("#!"):
            continue
        m = re.search(r"(?:^|\s)(#.*)$", ln)
        if m:
            out.append((i, m.group(1)))
    return out


def _js_text(src: str) -> list[tuple[int, str]]:
    """// and /* */ comments, skipping string and template literals."""
    out: list[tuple[int, str]] = []
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
        elif c in "'\"`":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n" and c != "`":
                    break
                j += 1
            line += src.count("\n", i, j + 1)
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append((line, src[i:j]))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k, ln in enumerate(src[i:j].split("\n")):
                out.append((line + k, ln))
            line += src.count("\n", i, j)
            i = j
        else:
            i += 1
    return out


def text_of(path: str, src: str) -> list[tuple[int, str]]:
    """The (line, text) pairs of ``src`` that the lint reads."""
    kind = _kind(path)
    if kind == "py":
        return _py_text(src)
    if kind == "sh":
        return _sh_text(src)
    if kind == "js":
        return _js_text(src)
    if kind == "md":
        return list(enumerate(src.splitlines(), 1))
    return []


def load_allow(path: Path | None) -> tuple[list, list]:
    """(allow rules, skip globs) from an allow file; ([], []) without one."""
    allow, skip = [], []
    if path is None or not Path(path).exists():
        return allow, skip
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if f[0] == "allow" and len(f) == 5 and f[1] in PATTERNS and f[4].strip():
            allow.append((f[1], f[2], re.compile(f[3]), f[4]))
        elif f[0] == "skip" and len(f) == 3 and f[2].strip():
            skip.append((f[1], f[2]))
        else:
            raise SystemExit(f"{path}: malformed row: {ln!r}")
    return allow, skip


def scan(path: str, src: str, allow=()) -> tuple[list, list]:
    """Hits (pattern, path, line, text) in ``src``: (unexcused, excused)."""
    hits, excused = [], []
    for i, ln in text_of(path, src):
        for k, rx in COMPILED.items():
            if not rx.search(ln):
                continue
            row = (k, path, i, ln.strip()[:200])
            ok = any(
                pk == k and fnmatch.fnmatchcase(path, pg) and lr.search(ln)
                for pk, pg, lr, _ in allow
            )
            (excused if ok else hits).append(row)
    return hits, excused


def scan_files(files: dict[str, str], allow=(), skip=()) -> tuple[list, list]:
    """Scan path -> text, leaving out paths a skip glob matches."""
    hits, excused = [], []
    for p in sorted(files):
        if any(fnmatch.fnmatchcase(p, g) for g, _ in skip):
            continue
        h, e = scan(p, files[p], allow)
        hits += h
        excused += e
    return hits, excused


def candidate_paths(root: Path) -> list[str]:
    r = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True
    )
    if r.returncode == 0:
        paths = [p for p in r.stdout.split("\0") if p]
    else:
        paths = [
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(root).parts
        ]
    return sorted(p for p in paths if _kind(p) and (root / p).is_file())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--allow", type=Path, default=None)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    allow_file = a.allow or root / DEFAULT_ALLOW
    allow, skip = load_allow(allow_file)
    paths = a.paths or candidate_paths(root)
    files = {p: (root / p).read_text(errors="replace") for p in paths if _kind(p)}
    hits, excused = scan_files(files, allow, skip)
    for k, p, i, t in hits:
        print(f"{p}:{i}: [{k}] {t}")
    by_rule: dict[str, int] = {}
    for k, *_ in excused:
        by_rule[k] = by_rule.get(k, 0) + 1
    print(
        f"leak lint: {len(files)} files, {len(hits)} hits, {len(excused)} excused "
        f"{by_rule or ''}"
    )
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
