"""Text and byte helpers shared by the manuscript checkers and the sync.

One implementation of each, so the numbers check, the figure check and the
sync agree on what counts as "in the paper" and as "the same file".
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_COMMENT = re.compile(r"(?<!\\)%.*")
# \if... opens a conditional (\iffalse, \ifx, \ifdefined, ...); \fi closes one.
_COND = re.compile(r"\\(if[A-Za-z@]*|fi)(?![A-Za-z@])")


def strip_comments(text: str) -> str:
    """Drop LaTeX comments (an unescaped % to the end of the line)."""
    return _COMMENT.sub("", text)


def strip_iffalse(text: str) -> str:
    r"""Remove every \iffalse ... \fi block, counting nested conditionals.

    A switched-off draft is not in the paper. TeX skips a false branch by
    matching \if.../\fi pairs, so a conditional nested inside the draft does not
    end it early; an indented \iffalse starts a block like any other.
    """
    out: list[str] = []
    pos = 0
    depth = 0  # > 0 while inside a skipped \iffalse block
    for m in _COND.finditer(text):
        tok = m.group(1)
        if depth == 0:
            if tok == "iffalse":
                out.append(text[pos : m.start()])
                depth = 1
        elif tok == "fi":
            depth -= 1
            if depth == 0:
                pos = m.end()
        else:
            depth += 1
    if depth == 0:
        out.append(text[pos:])
    return "".join(out)


def printed(text: str) -> str:
    """The text a reader sees: comments and \\iffalse blocks removed."""
    return strip_iffalse(strip_comments(text))


# ── Printed numerals ────────────────────────────────────────────────────────
# A registry row's `fmt` is a Python format spec ("{:.3f}", "{:,.0f}", "{:.1%}",
# "{:.1e}"). The check renders the recomputed value with it and requires the
# result among the numerals printed in the row's literal, so a number passes
# only if the text prints it at the text's own rounding.
_SCI = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|×)\s*10\s*\^\s*\{?\s*([-−]?\d+)\s*\}?"
)
_NUMERAL = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?")
# A minus sign rather than a hyphen: it follows one of these, or starts the text.
_SIGN_CONTEXT = set(" \t\n$({[=~:;<>/")


def normalise_numerals(text: str) -> str:
    """LaTeX number markup to plain text: {,} -> ',', \\% -> %, minus signs,
    a x 10^{k} -> aek, and -- (a range dash) -> a space."""
    t = text.replace("{,}", ",").replace("\\,", "").replace("\\%", "%")
    t = t.replace("−", "-").replace("$-$", "-").replace("\\textminus", "-")
    t = t.replace("---", " ").replace("--", " ")
    return _SCI.sub(lambda m: f"{m.group(1)}e{int(m.group(2).replace('−', '-'))}", t)


def numerals(text: str) -> list[str]:
    """The numerals a reader sees in ``text``, normalised as in
    :func:`normalise_numerals`. A '-' counts as a sign only where a minus can
    stand (after a space, $, (, = ...), not as a hyphen inside a word."""
    t = normalise_numerals(text)
    out = []
    for m in _NUMERAL.finditer(t):
        s, start = m.group(0), m.start()
        if s.startswith("-") and start > 0 and t[start - 1] not in _SIGN_CONTEXT:
            s = s[1:]
        if start > 0 and (t[start - 1].isalpha() or t[start - 1] == "_"):
            j = start
            while j > 0 and t[j - 1].isalpha():
                j -= 1
            if not (j > 0 and t[j - 1] == "\\"):
                continue  # part of an identifier (S1, round9), not \pm0.03
        out.append(s)
        if "e" in s:  # a value registered by its mantissa (a x 10^-b)
            out.append(s.split("e")[0])
    return out


def _clean(s: str) -> str:
    s = s.replace("%", "")
    if "e" in s:  # 4.4e-05 -> 4.4e-5
        mant, exp = s.split("e")
        s = f"{mant}e{int(exp)}"
    try:
        if float(s.replace(",", "")) == 0:
            s = s.lstrip("-")
    except ValueError:
        pass
    return s


def render(value: float, fmt: str) -> set[str]:
    """``value`` printed with ``fmt``, under round-half-even (Python's format)
    and round-half-up (how a person rounds a 5), normalised like numerals().

    A ``|sigN`` suffix first rounds to N significant figures, for a number the
    text gives as approximate ("roughly 340{,}000": ``{:,.0f}|sig2``).
    """
    from decimal import ROUND_HALF_UP, Decimal

    fmt, _, sig = fmt.partition("|sig")
    if sig:
        value = float(f"{float(value):.{int(sig)}g}")
    out = {_clean(fmt.format(value))}
    m = re.fullmatch(r"\{:(,?)\.(\d+)([f%])\}", fmt)
    if m:
        comma, nd, kind = m.group(1), int(m.group(2)), m.group(3)
        # 12 significant digits drop binary noise (0.6175 computed as
        # 0.61749999999999994) before a person's half-up rounding is applied
        d = Decimal(f"{float(value):.12g}") * (100 if kind == "%" else 1)
        q = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
        out.add(_clean(f"{q:{comma}.{nd}f}"))
    return out


def printed_match(value: float, fmt: str, literal: str) -> bool:
    """True if ``value`` rendered with ``fmt`` is a numeral of ``literal``."""
    return bool(render(value, fmt) & {_clean(n) for n in numerals(literal)})


# matplotlib stamps every PDF with the time it was drawn, so two renders of the
# same artwork never match byte for byte. Blank that field, and the trailing
# startxref offset that shifts with the date's length, before hashing.
_PDF_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")
_PDF_STARTXREF = re.compile(rb"startxref\s+\d+")


def md5(p: Path) -> str:
    """Content hash, with a PDF's draw date blanked; "MISSING" if absent."""
    p = Path(p)
    if not p.exists():
        return "MISSING"
    data = p.read_bytes()
    if p.suffix == ".pdf":
        data = _PDF_DATE.sub(b"/CreationDate ()", data)
        data = _PDF_STARTXREF.sub(b"startxref", data)
    return hashlib.md5(data).hexdigest()


def compare_figures(figures, ms: Path, repo: Path, not_reproducible):
    """Byte-compare each shipped figure file with its results/ source.

    ``figures`` are registry entries (``.files()`` yields (manuscript path,
    results path)). Returns (n_match, n_differ, fatal): every DIFFER or MISSING
    is fatal except a ``not_reproducible`` entry whose two files both exist.
    """
    n_match = n_differ = 0
    fatal = []
    for f in figures:
        for m, r in f.files():
            a, b = md5(Path(ms) / m)[:8], md5(Path(repo) / r)[:8]
            ok = a == b and a != "MISSING"
            n_match += ok
            n_differ += not ok
            excused = m in not_reproducible and "MISSING" not in (a, b)
            note = "   (see NOT_REPRODUCIBLE)" if excused and not ok else ""
            print(f"{'MATCH ' if ok else 'DIFFER'}  {m:<52} {a}  {b}  {r}{note}")
            if not ok and not excused:
                fatal.append(f"figure DIFFER {m} (shipped {a}, results {b})")
    return n_match, n_differ, fatal
