"""Static lint for known data-provenance hazards. Reads text only; runs in seconds.

(``check_tangram_column_naming.py`` is the separate ~4-minute synthetic Tangram
run that tests the mapping code itself.)

Checks:

  1. Tracked code reading a superseded Tangram object by literal path. Those
     objects carry position-assigned cell-type names. Allowed only in
     ``paths.py`` (and in scripts that compare those objects with the
     canonical ones on purpose, listed in ``PROVENANCE_EXEMPT``).
  2. Label-based ``.loc[`` on ``.obs``: the atlas index has 562,028
     duplicates, so these lookups silently pull in unrelated rows.
  3. Every ``\\texttt{scripts/...}`` pointer in the manuscript resolves to a
     git-tracked file (``scripts/_*.py`` is gitignored).
  4. Manuscript pointers naming a superseded object.
  5. Reading ``tangram_cell_type`` off an object loaded from the spatial
     object (``paths.SPATIAL_COMBINED`` or a ``combined_clustered.h5ad``
     literal), whose copy of that column is not the canonical labelling.
     Followed by data flow: a name bound to a read of that path is tainted
     until its ``obs["tangram_cell_type"]`` is reassigned.
  6. Reads of the ``results/fig5_kc/`` tables built from position-named columns.

Run:
    python3 scripts/paper/check_data_provenance.py
    python3 scripts/paper/check_data_provenance.py --baseline \
        scripts/paper/provenance_baseline.txt

With ``--baseline``, each finding's key (``file:check``) is compared with the
reviewed list in that file: a finding not on the list fails, and so does a
listed key that no longer fires (remove it, so the list only shrinks). Without
it, the exit status is the number of findings. Checks 3 and 4 need a manuscript
checkout (``BIMP_MANUSCRIPT``, default the submodule); without one they are
skipped with a message and the other checks still run.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEX_DIR = Path(
    os.environ.get(
        "BIMP_MANUSCRIPT", REPO / "submodules" / "bombus_impatiens_atlas_manuscript"
    )
)

SUPERSEDED_BASENAMES = {
    "combined_clustered_tangram.h5ad",
    "combined_clustered_stitched_tangram.h5ad",
    "combined_clustered_central_tangram.h5ad",
    "combined_clustered_optic_tangram.h5ad",
}
# Scripts that compare superseded objects against canonical ones.
PROVENANCE_EXEMPT = re.compile(
    r"(^|/)(paths\.py|check_data_provenance\.py)$"
)

# results/fig5_kc/ holds two sets of tables side by side. The tables below were
# built from Tangram objects whose cell-type column names were assigned by
# position, so their Kenyon subtype labels are wrong. Their *_by_region
# siblings are the ones to read; fig:kc_subtypes panel C reads
# results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad.
READ_CALL = re.compile(
    r"\b(read_csv|read_parquet|read_table|np\.load|loadtxt|genfromtxt)\s*\("
    r"|\bopen\s*\([^)]*[\"']r"
)

PRECORRECTION_FIG5KC = {
    "morph_comp_caste.csv",
    "morph_comp_per_slice.csv",
    "restr_comp.csv",
    "restr_meanprob.csv",
    "full_comp.csv",
    "kc_compartment_diversity.csv",
    "kc_compartment_diversity_per_slice.csv",
}

findings = []  # (key, message); key = "file:check", stable across line moves


def report(rel, check, message):
    findings.append(("%s:%s" % (rel, check), message))


def tracked_files():
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    ).stdout.split("\n")
    return [REPO / p for p in out if p.strip()]


def is_tracked(rel):
    r = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return r.returncode == 0


_PY38 = sys.version_info >= (3, 8)


def _is_str_const(node):
    """True for a string literal. py3.8+ parses to Constant, py3.6/3.7 to Str."""
    import ast

    if _PY38:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)
    return isinstance(node, ast.Str)


def _str_val(node):
    return node.value if _PY38 else node.s


def _code_strings(path):
    """String literals that are real code, not docstrings or comments."""
    import ast

    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (SyntaxError, OSError, ValueError):
        return None  # fall back to the line scan
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr):
                v = body[0].value
                if _is_str_const(v):
                    docstrings.add(id(v))
    out = []
    for node in ast.walk(tree):
        if _is_str_const(node) and id(node) not in docstrings:
            out.append((getattr(node, "lineno", 0), _str_val(node)))
    return out


def check_superseded_reads(files):
    for f in files:
        if f.suffix not in (".py", ".sh") or not f.exists():
            continue
        rel = f.relative_to(REPO).as_posix()
        if PROVENANCE_EXEMPT.search(rel) or rel.startswith("archive/"):
            continue

        hits = []
        if f.suffix == ".py":
            strings = _code_strings(f)
            if strings is not None:
                for lineno, s in strings:
                    for base in SUPERSEDED_BASENAMES:
                        if base in s:
                            hits.append((lineno, base))
                            break
        if f.suffix == ".sh" or (f.suffix == ".py" and _code_strings(f) is None):
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.split("\n"), 1):
                if line.strip().startswith("#"):
                    continue
                for base in SUPERSEDED_BASENAMES:
                    if base in line:
                        hits.append((i, base))
                        break

        for lineno, base in hits:
            report(
                rel,
                "superseded-read",
                f"{rel}:{lineno}: reads superseded object {base!r} -- "
                f"import bimp_st.paths.CANONICAL_TANGRAM instead",
            )


# Reviewed .obs.loc lookups that are safe, with the reason. Keyed by
# repo-relative path; every flagged line in the file is allowed, so keep entries
# to files whose only .obs.loc is the reviewed one.
OBS_LOC_ALLOW = {
    "scripts/caste_de/cnmf_gsea/cnmf_caste.py": (
        "the snRNA reference (paths.SN_CB / BIMP_atlas_qw_*.h5ad), whose 53,230 "
        "obs_names are unique, indexed by cNMF usage rows "
        "named after those nuclei; not the spatial atlas"
    ),
    "scripts/caste_de/cnmf_gsea/label_swap_null.py": (
        "same snRNA lookup as cnmf_caste.py: unique nucleus barcodes, not the "
        "spatial atlas"
    ),
}


def check_obs_loc(files):
    # Only the label-lookup shape is dangerous with a duplicated index:
    #   obs.loc[other.index]   obs.loc[names_list]   obs.loc[a.obs_names]
    # Boolean masks and positional work fine, so do not flag those.
    pat = re.compile(r"\bobs\.loc\[[^\]]*(?:\.index|\.obs_names)")
    for f in files:
        if f.suffix != ".py" or not f.exists():
            continue
        rel = f.relative_to(REPO).as_posix()
        if PROVENANCE_EXEMPT.search(rel) or rel.startswith("archive/"):
            continue
        if rel in OBS_LOC_ALLOW:
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.split("\n"), 1):
            if line.strip().startswith("#"):
                continue
            if pat.search(line):
                report(
                    rel,
                    "obs-loc",
                    f"{rel}:{i}: label lookup on .obs -- the atlas has 562,028 "
                    f"duplicate obs_names, so this can silently expand. Index "
                    f"positionally, or load via paths.load_tangram("
                    f"allow_duplicate_obs_names=False)",
                )


SPATIAL_OBJ = "combined_clustered.h5ad"
SPATIAL_CONST = "SPATIAL_COMBINED"
LABEL = "tangram_cell_type"
_STMT_TYPES = ("Assign", "AnnAssign", "AugAssign", "Expr", "Return")


def _mentions_spatial(node, path_names):
    """True if the expression names the spatial object: the paths constant, a
    string literal with its basename, or a variable holding such a path."""
    import ast

    for n in ast.walk(node):
        if isinstance(n, ast.Name) and (n.id == SPATIAL_CONST or n.id in path_names):
            return True
        if isinstance(n, ast.Attribute) and n.attr == SPATIAL_CONST:
            return True
        if _is_str_const(n) and SPATIAL_OBJ in _str_val(n):
            return True
    return False


def _label_access(node):
    """(name, is_store) if ``node`` is name.obs["tangram_cell_type"] or
    name.obs.tangram_cell_type, else None."""
    import ast

    obs = None
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if not _PY38 and isinstance(sl, ast.Index):
            sl = sl.value
        if _is_str_const(sl) and _str_val(sl) == LABEL:
            obs = node.value
    elif isinstance(node, ast.Attribute) and node.attr == LABEL:
        obs = node.value
    if (
        isinstance(obs, ast.Attribute)
        and obs.attr == "obs"
        and isinstance(obs.value, ast.Name)
    ):
        return obs.value.id, isinstance(node.ctx, ast.Store)
    return None


def _passes_label(node, tainted, path_names):
    """The column named in a call on the spatial object: a helper given a
    tainted object and "tangram_cell_type" (add_canonical(a, src=...),
    label_col=...), or read_obs(<spatial path>)["tangram_cell_type"]."""
    import ast

    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
        sl = node.slice
        if not _PY38 and isinstance(sl, ast.Index):
            sl = sl.value
        if _is_str_const(sl) and _str_val(sl) == LABEL:
            return _mentions_spatial(node.value, path_names)
    if not isinstance(node, ast.Call):
        return False
    fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    if fname == "keyed_obs_column":
        return False  # reads the target's keys only; the labels come from its path
    args = list(node.args) + [k.value for k in node.keywords]
    has_obj = any(isinstance(a, ast.Name) and a.id in tainted for a in args)
    names_label = any(
        _is_str_const(n) and _str_val(n) == LABEL for a in args for n in ast.walk(a)
    )
    return has_obj and names_label


def _bound_names(target):
    import ast

    if isinstance(target, ast.Name):
        return [target]
    return [e for e in getattr(target, "elts", []) if isinstance(e, ast.Name)]


def spatial_label_reads(source):
    """Line numbers where tangram_cell_type is read off an object loaded from
    the spatial object. Statements are taken in source order; a name is tainted
    when bound to a call that reads the spatial path, and clean again when
    rebound to anything else or when its label column is reassigned."""
    import ast

    tree = ast.parse(source)
    path_names, tainted, hits = set(), set(), []
    stmts = sorted(
        (n for n in ast.walk(tree) if type(n).__name__ in _STMT_TYPES),
        key=lambda n: (n.lineno, n.col_offset),
    )
    for st in stmts:
        value = getattr(st, "value", None)
        if value is not None:
            for n in ast.walk(value):
                acc = _label_access(n)
                if acc and not acc[1] and acc[0] in tainted:
                    hits.append(n.lineno)
                if _passes_label(n, tainted, path_names):
                    hits.append(n.lineno)
        if isinstance(st, ast.Assign):
            targets = st.targets
        elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
            targets = [st.target]
        else:
            targets = []
        for t in targets:
            acc = _label_access(t)
            if acc and acc[1]:
                tainted.discard(acc[0])  # the stale column was overwritten
            for nm in _bound_names(t):
                reads = value is not None and _mentions_spatial(value, path_names)
                is_call = value is not None and any(
                    isinstance(c, ast.Call) for c in ast.walk(value)
                )
                tainted.discard(nm.id)
                path_names.discard(nm.id)
                if reads and is_call:
                    tainted.add(nm.id)
                elif reads:
                    path_names.add(nm.id)
    return sorted(set(hits))


def check_spatial_label_reads(files):
    """Flag reading tangram_cell_type off the spatial object.

    pipelines/combined_atlas/data/combined_clustered.h5ad carries its own
    tangram_cell_type, which is not the canonical labelling, so labels read from
    it are wrong.
    Loading the canonical object elsewhere in the same file is no excuse: only
    the object the labels are read from counts.
    """
    for f in files:
        if f.suffix != ".py" or not f.exists():
            continue
        rel = f.relative_to(REPO).as_posix()
        if PROVENANCE_EXEMPT.search(rel) or rel.startswith("archive/"):
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        if SPATIAL_OBJ not in text and SPATIAL_CONST not in text:
            continue
        try:
            lines = spatial_label_reads(text)
        except SyntaxError:
            pat = re.compile(r"[\"']tangram_cell_type[\"']|\.tangram_cell_type\b")
            lines = [
                i
                for i, line in enumerate(text.split("\n"), 1)
                if not line.strip().startswith("#") and pat.search(line)
            ][:1]
        if lines:
            report(
                rel,
                "spatial-label",
                f"{rel}:{lines[0]}: reads tangram_cell_type off the spatial object "
                f"({SPATIAL_OBJ} / paths.{SPATIAL_CONST}), whose copy of that "
                f"column is a mapping the paper does not use for cell types. For "
                f"cell-type labels read paths.CANONICAL_TANGRAM "
                f"(paths.keyed_obs_column)",
            )


def check_precorrection_fig5kc(files):
    """Flag reads of the fig5_kc tables built from position-named columns.

    Some projection-test scripts compare vintages on purpose, but a reported
    number must never come from one of these tables.
    """
    for f in files:
        if f.suffix not in (".py", ".sh") or not f.exists():
            continue
        rel = f.relative_to(REPO).as_posix()
        if PROVENANCE_EXEMPT.search(rel) or rel.startswith("archive/"):
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.split("\n"), 1):
            if line.strip().startswith("#"):
                continue
            if not READ_CALL.search(line):
                continue  # a to_csv writer, or a name in a copy list
            for base in PRECORRECTION_FIG5KC:
                if base in line:
                    report(
                        rel,
                        "fig5kc-precorrection",
                        f"{rel}:{i}: reads {base!r} from results/fig5_kc/, "
                        f"whose copy on disk was built from position-named "
                        f"cell-type columns. Regenerate it, or use the "
                        f"*_by_region tables, "
                        f"or results/tangram_ltKC/kc5pop_spatial_allKC_"
                        f"dsource.h5ad for the restricted reclassification",
                    )
                    break


def check_manuscript_pointers():
    if not (TEX_DIR / "00_Article_Merge.tex").is_file():
        print(
            f"check_data_provenance: skipped the manuscript-pointer check (no "
            f"manuscript checkout at {TEX_DIR}; set BIMP_MANUSCRIPT to run it)"
        )
        return
    pat = re.compile(r"\\texttt\{((?:scripts|results|notebooks|src)[^}]*)\}")
    for tex in sorted(TEX_DIR.glob("0*.tex")):
        text = tex.read_text(errors="replace")
        for line_no, line in enumerate(text.split("\n"), 1):
            if line.lstrip().startswith("%"):
                continue
            for m in pat.finditer(line):
                raw = m.group(1).replace("\\_", "_").replace("\\", "")
                if any(ch in raw for ch in "{}*$"):
                    continue  # brace-expansion globs like permanova_{a,b}
                target = REPO / raw
                if raw.endswith(".py") or raw.endswith(".sh"):
                    if not target.exists():
                        report(
                            tex.name,
                            "cites-missing:" + raw,
                            f"{tex.name}:{line_no}: cites {raw} which does not exist",
                        )
                    elif not is_tracked(raw):
                        report(
                            tex.name,
                            "cites-untracked:" + raw,
                            f"{tex.name}:{line_no}: cites {raw} which is NOT "
                            f"git-tracked -- a clean clone will not have it",
                        )
                if Path(raw).name in SUPERSEDED_BASENAMES:
                    report(
                        tex.name,
                        "cites-superseded:" + raw,
                        f"{tex.name}:{line_no}: cites superseded object {raw}",
                    )


SELF_TEST_CASES = [
    # (snippet, should_be_flagged)
    ('sub_lab = adata.obs.loc[sub.index, "tangram_cell_type"]', True),
    ("x = a.obs.loc[b.obs_names]", True),
    ("y = adata.obs.loc[mask]", False),  # boolean mask is safe
    ('z = adata.obs.loc[:, "brain"]', False),  # column select is safe
]


def self_test():
    """Prove the .obs.loc pattern still catches the bug it was written for."""
    pat = re.compile(r"\bobs\.loc\[[^\]]*(?:\.index|\.obs_names)")
    bad = 0
    for snippet, want in SELF_TEST_CASES:
        got = bool(pat.search(snippet))
        if got != want:
            bad += 1
            print(
                "  SELF-TEST FAIL: %r -> flagged=%s, expected %s" % (snippet, got, want)
            )
    if bad:
        return bad
    print(
        "self-test: %d/%d obs.loc cases correct"
        % (len(SELF_TEST_CASES), len(SELF_TEST_CASES))
    )
    return 0


def main():
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    if "--self-test" in sys.argv:
        return self_test()
    if self_test():
        print("linter self-test failed -- its own pattern is broken, fix that first")
        return 1
    files = tracked_files()
    check_superseded_reads(files)
    check_obs_loc(files)
    check_spatial_label_reads(files)
    check_precorrection_fig5kc(files)
    check_manuscript_pointers()

    if "--max-findings" in sys.argv:
        print(
            "--max-findings is retired: a count hides a new finding behind a "
            "fixed one. Use --baseline scripts/paper/provenance_baseline.txt."
        )
        return 2

    reviewed = {}
    if "--baseline" in sys.argv:
        bpath = Path(sys.argv[sys.argv.index("--baseline") + 1])
        reviewed = read_baseline_reasons(bpath)
    if not findings:
        print("check_data_provenance: clean")
    else:
        print("check_data_provenance: %d finding(s)\n" % len(findings))
        for key, msg in findings:
            print("  " + msg)
            if key in reviewed:
                why = reviewed[key] or "see the baseline file"
                print("    reviewed, accepted: " + why)
    if "--baseline" not in sys.argv:
        return len(findings)
    return compare_baseline([k for k, _ in findings], bpath)


def read_baseline_reasons(path):
    """key -> the reviewer's reason (the comment after the key, may be empty)."""
    out = {}
    for line in Path(path).read_text().split("\n"):
        key, _, why = line.partition("#")
        if key.strip():
            out[key.strip()] = why.strip()
    return out


def read_baseline(path):
    return list(read_baseline_reasons(path))


def compare_baseline(keys, path):
    """0 if the findings are exactly the reviewed baseline, else 1."""
    base = set(read_baseline(path))
    new = sorted(set(keys) - base)
    gone = sorted(base - set(keys))
    for k in new:
        print("NEW finding (not in %s): %s" % (path.name, k))
    for k in gone:
        print("FIXED (remove it from %s): %s" % (path.name, k))
    if new or gone:
        return 1
    print(
        "\nall %d findings are reviewed and accepted in %s: none of them feeds a "
        "shipped figure, table or quoted number" % (len(base), path.name)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
