"""scripts/paper/leak_lint.py: planted result values and narration are caught,
code and string literals are not read, allow rows excuse only what they name."""

import importlib.util
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "scripts" / "paper"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ll():
    return _load("leak_lint_under_test", PAPER / "leak_lint.py")


def _ids(ll, path, src, allow=()):
    hits, _ = ll.scan(path, src, allow)
    return sorted({h[0] for h in hits})


PLANTED = {
    "p-value": "# the caste effect is significant (P = 0.003)",
    "rho": "# the two runs agree, rho = 0.71",
    "r-squared": "# subcluster explains it (R^2 0.89)",
    "median": "# median 0.79 across compartments",
    "pseudo-f": "# PERMANOVA pseudo-F 12.3",
    "fold": "# Tk is 6.6x higher in the queen",
    "stat-value": "# AUROC 0.92 for the calyx",
    "pct-decimal": "# 41.9% of cells change label",
    "pct-score": "# 92% agreement with the reference",
    "pp": "# rises by 12 pp",
    "text-quotes": "# the text quotes this value",
    "the-paper": "# as the paper reports",
    "manuscript": "# the manuscript states",
    "shipped": "# the shipped figure",
    "printed-fig": "# printed Fig. 5",
    "fig-number": "# see Fig 5C",
    "main-text": "# quoted in the main text",
    "supp-methods": "# see Supplementary Methods",
    "amendment": "# per the amendment",
    "pre-registration": "# the pre-registered test",
}


def test_every_pattern_has_a_planted_violation(ll):
    assert set(PLANTED) == set(ll.PATTERNS)


@pytest.mark.parametrize("pattern", sorted(PLANTED))
def test_planted_comment_is_caught(ll, pattern):
    assert pattern in _ids(ll, "scripts/x/planted.py", PLANTED[pattern] + "\nx = 1\n")


def test_docstring_markdown_shell_and_js_comments_are_read(ll):
    doc = '"""Compare runs.\n\nThe paper quotes rho = 0.71."""\nx = 1\n'
    assert "rho" in _ids(ll, "scripts/x/a.py", doc)
    assert "fig-number" in _ids(ll, "README.md", "Panel of Fig. 3\n")
    assert "shipped" in _ids(ll, "run.sh", "#!/bin/bash\necho hi  # shipped run\n")
    assert "fig-number" in _ids(ll, "viewer/a.js", "let a = 1; // as in Fig 6\n")
    assert "fig-number" in _ids(ll, "viewer/a.js", "/* Fig 6\n */\n")
    assert "shipped" in _ids(ll, "viewer/t.mjs", "const a = 1; // shipped once\n")


def test_code_and_string_literals_are_not_read(ll):
    assert _ids(ll, "scripts/x/a.py", 'label = "Fig 5C, rho = 0.71"\n') == []
    assert _ids(ll, "viewer/a.js", 'const t = "see Fig. 6 // shipped";\n') == []
    smk = (
        'rule a:\n    """Make a."""\n    shell:\n'
        '        """\n        echo P = 0.1\n        """\n'
    )
    assert _ids(ll, "pipelines/x/Snakefile", smk) == []


def test_parameters_after_a_comparison_are_not_results(ll):
    assert _ids(ll, "a.py", "# GO terms at FDR < 0.25\n") == []
    assert _ids(ll, "a.py", "# parity: Pearson r >= 0.999\n") == []
    assert _ids(ll, "a.py", "# results/fig5_kc/ tables\n") == []


def test_allow_rows_excuse_only_their_pattern_path_and_line(ll):
    allow = [("fold", "scripts/x/*", re.compile(r"1\.5x lead"), "vote rule")]
    assert _ids(ll, "scripts/x/a.py", "# a 1.5x lead\n", allow) == []
    assert _ids(ll, "scripts/y/a.py", "# a 1.5x lead\n", allow) == ["fold"]
    assert _ids(ll, "scripts/x/a.py", "# 6.6x higher\n", allow) == ["fold"]
    both = "# a 1.5x lead, see Fig 2\n"
    assert _ids(ll, "scripts/x/a.py", both, allow) == ["fig-number"]


def test_allow_file_rejects_rows_without_a_reason_or_a_pattern(ll, tmp_path):
    f = tmp_path / "allow.txt"
    f.write_text("allow\tfold\t*\t1\\.5x\t\n")
    with pytest.raises(SystemExit):
        ll.load_allow(f)
    f.write_text("allow\t*\t*\t.\twildcard\n")
    with pytest.raises(SystemExit):
        ll.load_allow(f)
    f.write_text("skip\tdocs/**\tnotes\nallow\tfold\t*\t1\\.5x\tvote rule\n")
    allow, skip = ll.load_allow(f)
    assert [a[0] for a in allow] == ["fold"] and skip == [("docs/**", "notes")]


def test_figure_map_descriptions_carry_no_leaks(ll):
    figmap = _load("make_figure_map_for_lint", PAPER / "make_figure_map.py")
    text = "\n".join(figmap.DESCRIPTIONS.values())
    assert _ids(ll, "README.md", text) == []
