#!/usr/bin/env python
"""Compare the regenerated snRNA Supplementary Data tables with the saved ones.

Table design: I. Traniello. The saved tables
(``Supp_celltype_GO_terms_full_unfiltered.csv``,
``Supp_program_GO_terms.csv``, ``Supp_pseudobulk_DEGs_annotated.csv``,
``cellranger_cellbender_output.csv``, ``EGAPx_sample_tables.csv``) are read from
``--saved-dir`` (md5 printed per file; the three ``Supp_`` files match the
``out/fig4_supp/`` copies byte for byte). Each table is rebuilt with the
generator's ``build()``:
``source="repo"`` (what ships) and, where the generator takes it,
``source="saved"`` (the saved inputs), so a difference can be put down to the
port or to the input.

Per table: column names and order, row keys (shared / only saved / only ours),
row order, and per column the max abs / rel difference (numbers) or the count of
mismatches (strings, compared exactly; empty == empty). Then summary statistics
(DEG direction shares, program-score Spearman rho) recomputed from the
regenerated tables.

Writes ``results/caste_programs/si_tables/compare_si_tables.txt``.

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/caste_de/cnmf_gsea/compare_si_tables.py
"""

import argparse
import contextlib
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
from cnmf_gsea_common import lineage
from cnmf_gsea_paths import REPO
from sd_common import SI, de_csv

sys.path.insert(0, str(REPO / "scripts" / "deposit"))

OUT = SI / "compare_si_tables.txt"

# file name -> (generator module, takes source, row key columns)
TABLES = {
    "Supp_celltype_GO_terms_full_unfiltered.csv": (
        "sd_celltype_go",
        True,
        ["cell_type", "go_id"],
    ),
    "Supp_program_GO_terms.csv": ("sd_program_go", True, ["program", "go_id"]),
    "Supp_pseudobulk_DEGs_annotated.csv": (
        "sd_pseudobulk_degs",
        True,
        ["cell_type", "gene_id"],
    ),
    "cellranger_cellbender_output.csv": ("sd_cellranger_qc", False, ["Sample ID"]),
    "EGAPx_sample_tables.csv": ("sd_egapx_inputs", False, ["Sample"]),
}

# tables with no generator, and why (none: every table has one)
MISSING: dict[str, str] = {}


def read_saved(path: Path) -> tuple[pd.DataFrame, str]:
    """A saved CSV and a note on its md5 and encoding (BOM)."""
    raw = path.read_bytes()
    bom = "UTF-8 BOM" if raw[:3] == b"\xef\xbb\xbf" else "no BOM"
    note = f"md5 {hashlib.md5(raw).hexdigest()}, {bom}"
    return pd.read_csv(path, encoding="utf-8-sig"), note


def _num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def compare(saved: pd.DataFrame, ours: pd.DataFrame, keys: list[str]) -> list[str]:
    """Report lines: columns, keys, order, per-column differences."""
    out = [f"  rows saved {len(saved)} / ours {len(ours)}"]
    same_cols = list(saved.columns) == list(ours.columns)
    out.append(f"  columns identical (names and order): {same_cols}")
    if not same_cols:
        out.append(f"    saved: {list(saved.columns)}")
        out.append(f"    ours: {list(ours.columns)}")
    hk = saved[keys].astype(str).agg("|".join, axis=1)
    ok = ours[keys].astype(str).agg("|".join, axis=1)
    if hk.duplicated().any() or ok.duplicated().any():
        out.append(
            f"  duplicate keys: saved {hk.duplicated().sum()}, "
            f"ours {ok.duplicated().sum()}"
        )
    only_h, only_o = sorted(set(hk) - set(ok)), sorted(set(ok) - set(hk))
    out.append(
        f"  keys {keys}: shared {len(set(hk) & set(ok))}, only saved {len(only_h)}, "
        f"only ours {len(only_o)}"
    )
    for lab, ks in (("only saved", only_h), ("only ours", only_o)):
        if ks:
            out.append(f"    {lab} (first 5): {ks[:5]}")
    shared = set(hk) & set(ok)
    out.append(
        "  row order identical on shared keys: "
        f"{[k for k in hk if k in shared] == [k for k in ok if k in shared]}"
    )
    h = saved.assign(_k=hk.values).drop_duplicates("_k").set_index("_k")
    o = ours.assign(_k=ok.values).drop_duplicates("_k").set_index("_k")
    idx = sorted(shared)
    h, o = h.loc[idx], o.loc[idx]
    for c in [c for c in saved.columns if c in ours.columns and c not in keys]:
        a, b = h[c], o[c]
        if _num(a) and _num(b):
            d = (a - b).abs()
            rel = d / b.abs().where(b.abs() > 0)
            na = int((a.isna() != b.isna()).sum())
            out.append(
                f"  {c:28s} max|d| {d.max():.3g}  median|d| {d.median():.3g}  "
                f"max rel {rel.max():.3g}  n|d|>1e-6 {int((d > 1e-6).sum())}  "
                f"NA mismatch {na}"
            )
        else:
            sa = a.astype(object).where(a.notna(), "")
            sb = b.astype(object).where(b.notna(), "")
            ne = sa.astype(str) != sb.astype(str)
            line = f"  {c:28s} string mismatches {int(ne.sum())}"
            if ne.any():
                i = ne.idxmax()
                line += f"  e.g. {i}: saved {sa[i]!r} ours {sb[i]!r}"
            out.append(line)
    for c in ("NES", "stat"):
        if c in h.columns and _num(h[c]):
            out.append(f"  r({c}) saved vs ours: {h[c].corr(o[c]):.4f}")
    for c in ("FDR_q", "FDR_all_GO_library", "padj"):
        if c in saved.columns:
            sh, so = set(hk[saved[c] < 0.05]), set(ok[ours[c] < 0.05])
            out.append(
                f"  {c} < 0.05: saved {len(sh)}, ours {len(so)}, shared {len(sh & so)}"
            )
    return out


def quoted_numbers() -> list[str]:
    """Summary statistics of the caste DE tables, from the regenerated tables."""
    import sd_program_go
    import sd_pseudobulk_degs

    out = []
    for src in ("repo", "saved"):
        with contextlib.redirect_stdout(io.StringIO()):
            d = sd_pseudobulk_degs.build(src)
        glia = d[d.cell_type.str.endswith("glia") & (d.cell_type != "Other glia")]
        neur = d[d.cell_type.str.contains("Kenyon|projection neurons", regex=True)]
        q = lambda x: x.groupby("cell_type").direction.apply(  # noqa: E731
            lambda s: (s == "queen-up").mean()
        )
        out.append(
            f"  [{src}] DEGs: {len(d)} rows, {d.gene_id.nunique()} unique genes "
            "(text, photoreceptors left out: 1,502 genes); "
            "glia queen-biased mean over "
            f"{glia.cell_type.nunique()} types {q(glia).mean():.1%} (text 77%); "
            f"neurons {q(neur).mean():.1%} over {neur.cell_type.nunique()} types "
            "(text 65%)"
        )
    with contextlib.redirect_stdout(io.StringIO()):
        p = sd_program_go.build("repo")
    for prog, term, txt in [
        (20, "ADP metabolic process", "0.010"),
        (6, "pyridine-containing compound metabolic process", "0.002"),
        (12, "long-term synaptic potentiation", "0.021"),
        (12, "neurotransmitter receptor activity", "0.002"),
    ]:
        r = p[(p.program == prog) & (p.term == term)]
        v = r[["FDR_all_GO_library", "FDR_BP_only_library", "namespace"]]
        out.append(f"  program {prog} {term!r}: {v.to_dict('records')} (text {txt})")
    out += spearman_check()
    return out


def spearman_check() -> list[str]:
    """Spearman rho of programs 20, 6, 18 (gene score vs Wald statistic).

    Not in any of the five tables. Definition: ``rho_for`` in the source
    ``out/fig4_supp/scripts/clean_run_six.py`` (md5
    b785adbb43f66a642f7eaf4a9cc817fb, panel d): Spearman of the program's
    gene-spectra score against the cell type's Wald statistic over every gene with
    a statistic. Every glial / neuronal type is listed, with the
    replicated-call cell types marked ``*``.
    """
    from cnmf_gsea_paths import RUN
    from scipy.stats import spearmanr

    ss = pd.read_csv(
        RUN / "cnmf_joint/joint/joint.gene_spectra_score.k_50.dt_2_0.txt",
        sep="\t",
        index_col=0,
    ).T  # genes x programs
    ss.columns = ss.columns.astype(int)
    j = pd.read_csv(RUN / "repo" / "joint_k50_caste.csv")
    j["cell_type"] = j.cell_type.str.replace(r"\s+", " ", regex=True)
    rep = j[
        j.same & (j["mean"].abs() >= 0.5) & (j.min_abs >= 0.25) & (j.usage_pct >= 2)
    ]
    out = []
    for src in ("repo", "saved"):
        de = pd.read_csv(de_csv(src))
        de["cell_type"] = de.cell_type.str.replace(r"\s+", " ", regex=True)
        cts = sorted(c for c in de.cell_type.unique() if lineage(c))
        for p, txt in [(20, 0.37), (6, -0.23), (18, 0.31)]:
            stars = set(rep[rep.program == p].cell_type)
            res = []
            for ct in cts:
                wv = de[de.cell_type == ct].set_index("gene")["stat"].dropna()
                g = ss[p].reindex(wv.index).dropna()
                rho = spearmanr(g, wv.reindex(g.index)).statistic
                res.append(f"{ct}{'*' if ct in stars else ''} {rho:+.2f}")
            out.append(f"  [{src}] rho program {p} (text {txt}): " + "; ".join(res))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--saved-dir", type=Path, required=True, help="the five saved tables"
    )
    args = ap.parse_args()
    lines = []
    for name, (mod_name, has_source, keys) in TABLES.items():
        lines.append(f"== {name}  (generator {mod_name}.build)")
        path = args.saved_dir / name
        if not path.exists():
            lines.append(f"  saved file missing: {path}")
            continue
        saved, bom = read_saved(path)
        lines.append(f"  saved file: {path} ({bom})")
        try:
            mod = __import__(mod_name)
            runs = ("repo", "saved") if has_source else (None,)
            for src in runs:
                with contextlib.redirect_stdout(io.StringIO()):
                    ours = mod.build(src) if src else mod.build()
                lines.append(f" -- ours ({src or 'raw files'})")
                lines += compare(saved, ours, keys)
        except ImportError:
            lines.append(f"  NOT REGENERATED: {MISSING.get(name, 'no generator')}")
        except (FileNotFoundError, StopIteration) as e:
            lines.append(f"  NOT REGENERATED: {type(e).__name__}: {e}")
    lines.append("== numbers quoted in the paper")
    lines += quoted_numbers()
    text = "\n".join(lines) + "\n"
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
