"""Compare the GENE_SYMBOLS dicts in the web viewer and in bimp_st.genes.

viewer/export_data.py keeps its own copy of the focal symbols the registered
figures take from src/bimp_st/genes.py; if they disagree on a gene id, the
viewer and the figures label the same gene differently. Read-only; parses the
literals by AST rather than importing.
"""

import ast
import pathlib
import sys

VIEWER = "viewer/export_data.py"
PAPER = "src/bimp_st/genes.py"


def gene_symbols(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "GENE_SYMBOLS":
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        return None
    return None


def main() -> int:
    a = gene_symbols(VIEWER)
    b = gene_symbols(PAPER)
    print(f"{VIEWER} : {len(a) if a else 'not a literal'} entries")
    print(f"{PAPER}       : {len(b) if b else 'not a literal'} entries")
    if not (a and b):
        return 0

    shared = sorted(set(a) & set(b))
    print(f"\nshared gene ids: {len(shared)}")
    disagree = [(g, a[g], b[g]) for g in shared if a[g] != b[g]]
    if disagree:
        print("\n!! DISAGREEMENTS (viewer vs figures):")
        for g, x, y in disagree:
            print(f"  {g}   webviz={x!r}   figures={y!r}")
    else:
        print("  no disagreements on shared ids")
    rc = 1 if disagree else 0

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print(
        f"\nonly in webviz ({len(only_a)}): " + ", ".join(f"{g}={a[g]}" for g in only_a)
    )
    print(
        f"only in bimp_st.genes ({len(only_b)}): "
        + ", ".join(f"{g}={b[g]}" for g in only_b[:12])
        + (" ..." if len(only_b) > 12 else "")
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
