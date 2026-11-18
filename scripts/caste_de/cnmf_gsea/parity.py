#!/usr/bin/env python
"""Parity of the in-repo cNMF / GSEA port with the source code and saved outputs.

Analysis design and source notebook: I. Traniello.

Three comparisons, written to ``pipeline/parity/`` (``parity_summary.txt`` plus
one CSV per table):

1. Ported vs verbatim code on the same inputs. The verbatim copy of the
   notebook cells (commit cd8d4278; outputs in ``results/caste_programs/``, one
   directory per source) against this port (``pipeline/``): the factorization
   files, every step table, for ``--source saved`` and ``--source repo``.
2. Ported code on the SAVED cNMF and DE (``--source saved``) against the saved
   tables (``cnmf_gsea_paths.SAVED``): the deterministic steps must match.
3. The repo run (``--source repo``, its own factorization and the repo's
   DESeq2 table) against the saved run: programs matched by spectra
   correlation (``program_map.csv``), the replicated program x cell-type calls,
   the label-swap null, the figure-program GO labels and the cell-type GSEA.

Run in the pipeline env after every step
(``run_pipeline.sh parity``).
"""

import hashlib
import re

import numpy as np
import pandas as pd
from cnmf_gsea_paths import (
    FIG3_INPUTS,
    FIG_PROGS,
    PARITY,
    PROGRAM_MAP,
    REPO_CNMF,
    RUN,
    SAVED,
    SAVED_CNMF,
    SAVED_DIRNAME,
    SAVED_PROGRAM_COL,
    VERBATIM,
)

from bimp_st.paths import CASTE_DE_SAVED_INPUTS

# the panel d labels from the saved run (caste_de_freeze_saved_outputs.py)
SHIPPED_LABELS = CASTE_DE_SAVED_INPUTS.resolve() / "program_labels.csv"
STEP_TABLES = [
    "cnmf_k_selection.csv",
    "joint_k40_caste.csv",
    "joint_k50_caste.csv",
    "label_swap_null.csv",
    "label_swap_stars.csv",
    "joint_k50_figure_programs_go_bp.csv.gz",
    "joint_k50_figure_programs_go_bp_cell109.csv.gz",
    "joint_k50_go_labels.csv",
    "gsea_all.csv.gz",
]
CNMF_FILES = [
    "joint/cnmf_tmp/joint.spectra.k_50.merged.df.npz",
    "joint/cnmf_tmp/joint.spectra.k_40.merged.df.npz",
    "joint/joint.overdispersed_genes.txt",
    "joint/joint.k_selection_stats.df.npz",
    "joint/joint.spectra.k_50.dt_2_0.consensus.txt",
    "joint/joint.usages.k_50.dt_2_0.consensus.txt",
    "joint/joint.gene_spectra_score.k_50.dt_2_0.txt",
    "joint/joint.usages.k_40.dt_2_0.consensus.txt",
    "joint/joint.gene_spectra_score.k_40.dt_2_0.txt",
]
# the notebook's printed panel d labels (original notebook cell 111 output)
NOTEBOOK_CELL111_LABELS = {
    18: ("Transporters, oxidoreductases, esterases†", None),
    20: ("ADP metabolic process", 0.010),
    6: ("Pyridine-containing compound metabolic process", 0.002),
    7: ("ECM, Notch, Rho signaling†", None),
    12: ("Long-term synaptic potentiation", 0.016),
}
LINES: list[str] = []


def say(s="") -> None:
    """Print and keep for parity_summary.txt."""
    print(s)
    LINES.append(str(s))


def norm(s) -> str:
    """Collapse whitespace in a cell-type label."""
    return re.sub(r"\s+", " ", str(s))


def md5(p) -> str:
    """md5 of a file, read in 4 MB blocks."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def table_diff(a: pd.DataFrame, b: pd.DataFrame) -> str:
    """Describe how two tables differ: shape, keys, max numeric difference."""
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f"shape/columns differ {a.shape} vs {b.shape}"
    out = []
    for c in a.columns:
        if a[c].dtype.kind in "fi" and b[c].dtype.kind in "fi":
            d = np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))
            if d > 0:
                out.append(f"{c} max |diff| {d:.2e}")
        elif not a[c].astype(str).map(norm).equals(b[c].astype(str).map(norm)):
            out.append(f"{c} differs")
    return "; ".join(out) or "values identical (formatting only)"


def section_ported_vs_verbatim() -> None:
    """1. Ported code against the verbatim copy, on the same inputs."""
    say("## 1. Ported code vs verbatim copy of the original cells (same inputs)")
    rows = []
    for f in CNMF_FILES:
        a, b = REPO_CNMF / f, VERBATIM / "cnmf_joint" / f
        if a.exists() and b.exists():
            rows.append(("cnmf_joint", f, md5(a) == md5(b), ""))
    for src in (SAVED_DIRNAME, "repo"):
        for t in STEP_TABLES:
            a, b = RUN / src / t, VERBATIM / src / t
            if not (a.exists() and b.exists()):
                rows.append(
                    (src, t, None, "missing: " + ("port" if b.exists() else "verbatim"))
                )
                continue
            same = md5(a) == md5(b)
            rows.append(
                (
                    src,
                    t,
                    same,
                    "" if same else table_diff(pd.read_csv(a), pd.read_csv(b)),
                )
            )
    a, b = PROGRAM_MAP, VERBATIM / "program_map.csv"
    if a.exists() and b.exists():
        rows.append(("repo", "program_map.csv", md5(a) == md5(b), ""))
    tab = pd.DataFrame(rows, columns=["run", "file", "byte_identical", "note"])
    tab.to_csv(PARITY / "ported_vs_verbatim.csv", index=False)
    say(tab.to_string(index=False))


def caste_table(path) -> pd.DataFrame:
    """A joint_k*_caste.csv, cell types whitespace-normalized, sorted."""
    t = pd.read_csv(path)
    t["cell_type"] = t.cell_type.map(norm)
    return t.sort_values(["cell_type", "program"]).reset_index(drop=True)


def replicated(t: pd.DataFrame) -> pd.DataFrame:
    """The replication rule of panel d's stars (original cell 113)."""
    m = t.same & (t["mean"].abs() >= 0.5) & (t.min_abs >= 0.25) & (t.usage_pct >= 2.0)
    return t[m]


def compare_caste(ours, saved, label) -> None:
    """Keys, flags and max numeric difference of two caste tables."""
    a, b = caste_table(ours), caste_table(saved)
    keys = a[["cell_type", "program"]].equals(b[["cell_type", "program"]])
    num = [c for c in b.columns if b[c].dtype.kind == "f" and c in a.columns]
    d = {c: float((a[c] - b[c]).abs().max()) for c in num} if keys else {}
    flags = keys and a.same.equals(b.same) and a.home.map(norm).equals(b.home.map(norm))
    diffs = ", ".join(f"{c} {v:.2e}" for c, v in d.items())
    say(
        f"{label}: rows {len(a)} vs {len(b)}, keys identical {keys}, "
        f"same/home identical {flags}, max |diff| {diffs}"
    )


def gsea_compare(ours, saved, label, out) -> None:
    """Shared terms, max |dFDR| / |dNES| and FDR < 0.05 overlap of two GSEAs."""
    a, b = pd.read_csv(ours), pd.read_csv(saved)
    for t in (a, b):
        t["cell_type"] = t.cell_type.map(norm)
    m = b.merge(
        a,
        on=["cell_type", "Term"],
        how="outer",
        suffixes=("_saved", "_ours"),
        indicator=True,
    )
    both = m[m._merge == "both"]
    d_fdr = (both["FDR q-val_saved"] - both["FDR q-val_ours"]).abs().max()
    d_nes = (both.NES_saved - both.NES_ours).abs().max()
    r_nes = np.corrcoef(both.NES_saved, both.NES_ours)[0, 1]
    key = ["cell_type", "Term"]
    sig_h = set(map(tuple, b.loc[b["FDR q-val"] < 0.05, key].values))
    sig_o = set(map(tuple, a.loc[a["FDR q-val"] < 0.05, key].values))
    say(
        f"{label}: rows {len(b)} saved / {len(a)} ours, shared {len(both)}; "
        f"max |dFDR| {d_fdr:.2e}, max |dNES| {d_nes:.2e}, r(NES) {r_nes:.4f}; "
        f"FDR<0.05 terms saved {len(sig_h)}, ours {len(sig_o)}, "
        f"shared {len(sig_h & sig_o)}"
    )
    m.to_csv(out, index=False)


def shown_labels(table, to_run) -> pd.DataFrame:
    """Panel d's GO label rows from one BP table, against the saved rows.

    ``to_run`` maps the saved program number to the table's program number.
    Exact FDR and NES differences against the saved ``joint_k50_go_labels.csv``
    (original cell 104's output), from which the frozen labels were taken
    (``caste_de_freeze_saved_outputs.py``).
    """
    saved = pd.read_csv(SAVED / "joint_k50_go_labels.csv")
    saved = saved.set_index(["program", "name"])
    rows = []
    for p in FIG_PROGS:
        t = table[(table.program == to_run[p]) & (table["FDR q-val"] < 0.05)]
        r = top_bp(t)
        row = dict(program=p, term=None, fdr=np.nan, nes=np.nan)
        if r is not None:
            row.update(term=r["name"], fdr=float(r["FDR q-val"]), nes=float(r.NES))
            if (p, r["name"]) in saved.index:
                h = saved.loc[(p, r["name"])]
                row.update(
                    saved_fdr=float(h["FDR q-val"]),
                    saved_nes=float(h.NES),
                    d_fdr=abs(row["fdr"] - float(h["FDR q-val"])),
                    d_nes=abs(row["nes"] - float(h.NES)),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def shown_label_sections(run_dir, to_run, prefix) -> None:
    """shown_labels for the cell 27 and the original cell 109 BP tables."""
    for f, what in (
        ("joint_k50_figure_programs_go_bp.csv.gz", "cell 27, 1,000 permutations"),
        ("joint_k50_figure_programs_go_bp_cell109.csv.gz", "original cell 109"),
    ):
        if not (run_dir / f).exists():  # cell 109 table: program_go.py --cell109
            continue
        t = shown_labels(pd.read_csv(run_dir / f), to_run)
        say(f"{prefix}: shown labels from {what} vs the saved rows:")
        say(t.to_string(index=False))
        t.to_csv(PARITY / f"{prefix}_shown_labels_{f.split('.')[0]}.csv", index=False)


def top_bp(t: pd.DataFrame) -> pd.Series | None:
    """Cell 30's row_label choice: lowest FDR, then highest NES, then name."""
    t = t.sort_values(["FDR q-val", "NES", "name"], ascending=[True, False, True])
    return t.iloc[0] if len(t) else None


def section_saved() -> None:
    """2. Ported code on the saved cNMF / DE against the saved tables."""
    say(
        "\n## 2. Ported code on the saved cNMF / DE (--source saved) vs the saved "
        "tables"
    )
    idir = RUN / SAVED_DIRNAME
    for k in (50, 40):
        compare_caste(
            idir / f"joint_k{k}_caste.csv",
            SAVED / f"joint_k{k}_caste.csv",
            f"joint_k{k}",
        )
    z = np.load(SAVED_CNMF / "joint/joint.k_selection_stats.df.npz", allow_pickle=True)
    ks = pd.DataFrame(z["data"], columns=z["columns"])
    ks = ks.rename(columns={"silhouette": "stability"})  # as cell 23 does
    ours = pd.read_csv(idir / "cnmf_k_selection.csv")
    same = np.array_equal(
        ks.values.astype(float), ours[list(ks.columns)].values.astype(float)
    )
    say(f"k selection identical: {same}")

    a = pd.read_csv(idir / "joint_k50_go_labels.csv")
    b = pd.read_csv(SAVED / "joint_k50_go_labels.csv")
    m = b.merge(
        a,
        on=["program", "name"],
        how="outer",
        suffixes=("_saved", "_ours"),
        indicator=True,
    )
    both = m[m._merge == "both"]
    say(
        f"joint_k50_go_labels (top 5 per program, 50 programs): rows {len(b)} saved / "
        f"{len(a)} ours, shared {len(both)}, only saved "
        f"{int((m._merge == 'left_only').sum())}, only ours "
        f"{int((m._merge == 'right_only').sum())}; max |dFDR| "
        f"{(both['FDR q-val_saved'] - both['FDR q-val_ours']).abs().max():.2e}, "
        f"max |dNES| {(both.NES_saved - both.NES_ours).abs().max():.2e}"
    )
    m.to_csv(PARITY / "saved_go_labels_port_vs_saved.csv", index=False)
    say(
        "  (expected: the saved file is the original cell 104's output, programs "
        "18, 12, 11, 10, 15 at 500 permutations; the ported cell 102 uses 1,000)"
    )
    shown_label_sections(idir, {p: p for p in FIG_PROGS}, "saved")

    c27 = pd.read_csv(idir / "joint_k50_figure_programs_go_bp.csv.gz")
    rows = []
    for p in FIG_PROGS:
        r = top_bp(c27[(c27.program == p) & (c27["FDR q-val"] < 0.05)])
        saved_label, saved_fdr = NOTEBOOK_CELL111_LABELS[p]
        rows.append(
            dict(
                program=p,
                cell27_label=r["name"] if r is not None else "(none: manual)",
                cell27_fdr=float(r["FDR q-val"]) if r is not None else np.nan,
                saved_cell111_label=saved_label,
                saved_cell111_fdr=saved_fdr,
            )
        )
    t = pd.DataFrame(rows)
    t.to_csv(PARITY / "saved_figure_labels.csv", index=False)
    say(t.to_string(index=False))

    gsea_compare(
        idir / "gsea_all.csv.gz",
        SAVED / "gsea_all.csv.gz",
        "gsea_all (saved DE)",
        PARITY / "saved_gsea_port_vs_saved.csv",
    )
    say("label-swap null (original cell 113) on the saved cNMF:")
    say(pd.read_csv(idir / "label_swap_null.csv").to_string(index=False))


def section_repo() -> None:
    """3. The repo run (own factorization, repo DESeq2) against the saved run."""
    say("\n## 3. Repo run (--source repo) vs the saved run")
    pm = pd.read_csv(PROGRAM_MAP)
    say(
        f"program map: {(pm.r_gene_score >= 0.9).sum()} / {len(pm)} programs matched "
        f"at r >= 0.9 (gene-spectra score), {(pm.r_gene_score >= 0.99).sum()} at "
        f">= 0.99; identity map {(pm[SAVED_PROGRAM_COL] == pm.repo_program).all()}"
    )
    cols = [SAVED_PROGRAM_COL, "repo_program", "r_gene_score", "r_spectra"]
    cols += ["best_other_r"]
    say(pm[pm[SAVED_PROGRAM_COL].isin(FIG_PROGS)][cols].round(4).to_string(index=False))
    to_saved = pm.set_index("repo_program")[SAVED_PROGRAM_COL]

    a = caste_table(RUN / "repo/joint_k50_caste.csv")
    a[SAVED_PROGRAM_COL] = a.program.map(to_saved)
    b = caste_table(SAVED / "joint_k50_caste.csv")
    ra, rb = replicated(a), replicated(b)
    sa = set(zip(ra.cell_type, ra[SAVED_PROGRAM_COL], strict=True))
    sb = set(zip(rb.cell_type, rb.program, strict=True))
    say(
        f"replicated program x cell-type calls: saved {len(sb)}, repo {len(sa)}, "
        f"shared {len(sa & sb)}; only saved {sorted(sb - sa)}; "
        f"only repo {sorted(sa - sb)}"
    )
    j = b.merge(
        a.drop(columns="program").rename(columns={SAVED_PROGRAM_COL: "program"}),
        on=["cell_type", "program"],
        suffixes=("_saved", "_repo"),
    )
    keys = zip(j.cell_type, j.program, strict=True)
    calls = j[[k in (sa | sb) for k in keys]]
    cols = ["cell_type", "program"] + [
        f"{c}_{s}"
        for c in ("t1", "t2", "t3", "mean", "min_abs", "usage_pct")
        for s in ("saved", "repo")
    ]
    say(calls[cols].round(2).to_string(index=False))
    j.to_csv(PARITY / "repo_caste_vs_saved.csv", index=False)
    d = {
        c: float((j[f"{c}_saved"] - j[f"{c}_repo"]).abs().max())
        for c in ("t1", "t2", "t3", "mean", "usage_pct")
    }
    say(
        "all 650 program x cell-type rows, max |saved - repo|: "
        + ", ".join(f"{c} {v:.2e}" for c, v in d.items())
    )
    say("label-swap null on the repo run:")
    say(pd.read_csv(RUN / "repo/label_swap_null.csv").to_string(index=False))

    lab = pd.read_csv(FIG3_INPUTS / "program_labels.csv")
    cell111 = NOTEBOOK_CELL111_LABELS
    lab["saved_cell111_label"] = lab.program.map(lambda p: cell111[p][0])
    lab["saved_cell111_fdr"] = lab.program.map(lambda p: cell111[p][1])
    lab["label_matches"] = lab.label == lab.saved_cell111_label
    lab["fdr_3dp_matches"] = [
        (pd.isna(f) and h is None) or (h is not None and round(f, 3) == h)
        for f, h in zip(lab.fdr, lab.saved_cell111_fdr, strict=True)
    ]
    out = ["program", "label", "fdr", "saved_cell111_label", "saved_cell111_fdr"]
    say(lab[out + ["label_matches", "fdr_3dp_matches"]].to_string(index=False))
    lab.to_csv(PARITY / "repo_figure_labels_vs_saved.csv", index=False)
    shown_label_sections(
        RUN / "repo", pm.set_index(SAVED_PROGRAM_COL).repo_program.to_dict(), "repo"
    )
    shipped = pd.read_csv(SHIPPED_LABELS).set_index("program")
    ours = lab.set_index("program").loc[shipped.index]
    same = {
        c: bool(
            ours[c].fillna("").astype(str).equals(shipped[c].fillna("").astype(str))
        )
        for c in ("label", "go_id", "manual")
    }
    d_fdr = float((ours.fdr - shipped.fdr).abs().max())
    d_nes = float((ours.nes - shipped.nes).abs().max())
    say(
        "program_labels.csv vs the first-shipped labels "
        f"({CASTE_DE_SAVED_INPUTS.relative_to(CASTE_DE_SAVED_INPUTS.parents[1])}; "
        "program 12 FDR 0.016 at 500 permutations): "
        f"identical {same}, max |dFDR| {d_fdr:.2e}, max |dNES| {d_nes:.2e}"
    )
    gsea_compare(
        RUN / "repo/gsea_all.csv.gz",
        SAVED / "gsea_all.csv.gz",
        "gsea_all (repo DE)",
        PARITY / "repo_gsea_vs_saved.csv",
    )


def main() -> None:
    PARITY.mkdir(parents=True, exist_ok=True)
    section_ported_vs_verbatim()
    section_saved()
    section_repo()
    (PARITY / "parity_summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
