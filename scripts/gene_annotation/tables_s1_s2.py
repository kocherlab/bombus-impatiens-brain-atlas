"""Supplementary Tables S1 (panel key) and S2 (snRNA-seq annotations).

Run: PYTHONPATH=src .venv/bin/python scripts/gene_annotation/tables_s1_s2.py

Writes only under results/handoff/D_tables/. Sources:
  S1: data/panel/XeniumBrainPanel_BIMP+MGEN_vers2.csv (probe list + NCBI
      descriptions), data/panel/BIMP_Xenium_Classifiers.xlsx (the only place a
      category / relevance is recorded), data/panel/np31_to_v43_mapping.csv
      (crosswalk), data/sn/BIMP_atlas_qw_combined.h5ad (presence in reference),
      BIMP_v4.3_lookup.tsv (symbols), NCBI Gene orthologs (Dmel -> B. impatiens,
      tax 7227 -> 132113) for Drosophila orthologs, FBgn IDs from NCBI Datasets.
  S2: scanpy rank_genes_groups (Wilcoxon, one-vs-rest, groupby=cell_type)
      on the combined reference.
Drosophila orthologs exist only for panel genes (NCBI LOC ids); the rest of the
v4.3 annotation has no NCBI GeneID crosswalk in the repo.
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import urllib.request

import numpy as np
import pandas as pd

from bimp_st.genes import (
    GENE_SYMBOLS,
    read_panel_crosswalk,  # noqa: E402
)
from bimp_st.paths import REPO as ROOT  # noqa: E402

OUT = ROOT / "results" / "handoff" / "D_tables"
CACHE = OUT / "cache"
SN = ROOT / "data" / "sn" / "BIMP_atlas_qw_combined.h5ad"
LOOKUP = (
    ROOT / "pipelines" / "combined_atlas" / "data" / "BIMP_v4.3_lookup.tsv"
)
ORTH_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_orthologs.gz"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2/gene/id/{}"

# Hand-curated symbols (bimp_st GENE_SYMBOLS), plus the panel's short
# vitellogenin paralog, named Vg-like to keep it apart from canonical Vgn
# (005500), which the panel lacks.
PAPER_SYMBOLS = {**GENE_SYMBOLS, "bimp_v4.3_001050": "Vg-like"}


def tex(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\^{}"),
    ]:
        s = s.replace(a, b)
    return (
        s.replace("…", r"\ldots{}")
        .replace("β", r"\textbeta{}")
        .replace("α", r"\textalpha{}")
        .replace("γ", r"\textgamma{}")
        .replace("κ", r"\textkappa{}")
    )


def protein_name(desc: str) -> str:
    m = re.search(r"\[protein=([^\]]+)\]", str(desc))
    s = m.group(1) if m else str(desc)
    return re.sub(r"\s*isoform\s+\S+$", "", s).strip()


# ---------------------------------------------------------------- orthologs
def ncbi_geneid(sym: str) -> int:
    """LOC123 -> 123; a named gene (CRY2) is resolved via NCBI Datasets."""
    if re.fullmatch(r"LOC\d+", sym):
        return int(sym[3:])
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / "ncbi_geneid_by_symbol.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    if sym not in cache:  # cached so a rerun is offline-reproducible
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol/{sym}/taxon/132113"
        out = subprocess.run(
            ["curl", "-s", "-q", "-H", "accept: application/json", url],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        cache[sym] = int(json.loads(out)["reports"][0]["gene"]["gene_id"])
        cf.write_text(json.dumps(cache))
    return int(cache[sym])


def dmel_orthologs(geneids: set[int]) -> dict[int, list[tuple[str, str]]]:
    """B. impatiens NCBI GeneID -> [(FBgn, symbol)], cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / "dmel_orthologs_panel.json"
    if cf.exists():
        raw = json.loads(cf.read_text())
        if set(map(int, raw)) >= geneids:
            return {int(k): [tuple(x) for x in v] for k, v in raw.items()}
    gz = CACHE / "gene_orthologs.gz"  # persistent cache
    if not gz.exists():
        urllib.request.urlretrieve(ORTH_URL, gz)
    pairs: dict[int, list[int]] = {}
    with gzip.open(gz, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] == "7227" and f[3] == "132113" and int(f[4]) in geneids:
                pairs.setdefault(int(f[4]), []).append(int(f[1]))
    dm = sorted({d for v in pairs.values() for d in v})
    info: dict[int, tuple[str, str]] = {}
    for i in range(0, len(dm), 50):
        chunk = dm[i : i + 50]
        url = DATASETS.format(",".join(map(str, chunk)))
        out = subprocess.run(
            ["curl", "-s", "-q", "-H", "accept: application/json", url],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for r in json.loads(out).get("reports", []):
            g = r["gene"]
            info[int(g["gene_id"])] = (
                g.get("nomenclature_authority", {}).get("identifier", ""),
                g["symbol"],
            )
    res = {b: [info[d] for d in ds if d in info] for b, ds in pairs.items()}
    cf.write_text(json.dumps({str(k): v for k, v in res.items()}))
    return res


def dmel_by_symbol(sym) -> str:
    """FBgn:symbol of the D. melanogaster gene whose symbol equals the annotation
    symbol."""
    if not isinstance(sym, str):
        return ""
    cf = CACHE / "dmel_symbol_lookup.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    if sym not in cache:
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol/{sym}/taxon/7227"
        out = subprocess.run(
            ["curl", "-s", "-q", "-H", "accept: application/json", url],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        hit = ""
        try:
            for r in json.loads(out).get("reports", []):
                g = r["gene"]
                if g["symbol"] == sym:
                    hit = (
                        f"{g.get('nomenclature_authority', {}).get('identifier', '')}"
                        f":{sym}"
                    )
        except (ValueError, KeyError):
            pass
        cache[sym] = hit
        cf.write_text(json.dumps(cache))
    return cache[sym]


# --------------------------------------------------------------------- S1
def _s1_base(var_names) -> pd.DataFrame:
    """Panel rows with NCBI-ortholog columns; no E_orthologs overlay (used by
    panel_orthologs.py to avoid a circular read)."""
    p = pd.read_csv(
        ROOT / "data/panel/XeniumBrainPanel_BIMP+MGEN_vers2.csv", encoding="utf-8-sig"
    ).dropna(subset=["BIMP_NP_Symbol"])
    p = p[["BIMP2.2_Symbol", "BIMP_NP_Symbol", "BIMP_NP_Description", "Note"]]
    cls = pd.read_excel(ROOT / "data/panel/BIMP_Xenium_Classifiers.xlsx")
    m = read_panel_crosswalk()
    lk = pd.read_csv(LOOKUP, sep="\t").set_index("gene_id")
    d = p.merge(
        cls.drop(columns="BIMP_NP_Description"), on="BIMP_NP_Symbol", how="left"
    ).merge(
        m[["np31_id", "v43_id", "loc_id"]],
        left_on="BIMP_NP_Symbol",
        right_on="np31_id",
        how="left",
    )
    assert len(d) == 101 and d.v43_id.notna().all()
    d["BIMP2.2_Symbol"] = d["BIMP2.2_Symbol"].str.strip()
    d["loc_id"] = d["loc_id"].str.strip()
    assert (d["BIMP2.2_Symbol"] == d.loc_id).all()
    d["protein"] = d.BIMP_NP_Description.map(protein_name)
    d["lookup_symbol"] = d.v43_id.map(lk["gene"])
    d["symbol"] = [
        PAPER_SYMBOLS.get(v.lower()) or (s if isinstance(s, str) else None)
        for v, s in zip(d.v43_id, d.lookup_symbol, strict=True)
    ]
    d["symbol_source"] = np.where(
        d.symbol.notna(), "paper/annotation symbol", "NCBI protein name"
    )
    d["symbol"] = d.symbol.fillna(d.protein)
    d["in_snrna_reference"] = d.v43_id.isin(var_names)
    # orthologs
    gid = d.loc_id.map(ncbi_geneid)
    orth = dmel_orthologs(set(gid))
    d["dmel_flybase_id"] = [";".join(x[0] for x in orth.get(g, [])) for g in gid]
    d["dmel_symbol"] = [";".join(x[1] for x in orth.get(g, [])) for g in gid]
    d["ncbi_geneid"] = gid
    # fatty acid synthase double probe
    d["probe_note"] = ""
    fas = d.protein.eq("fatty acid synthase")
    assert fas.sum() == 2
    d.loc[fas, "probe_note"] = "two probes, one shared Xenium gene_name"
    # Two probes carry the same Xenium gene_name; show both as numbered probes.
    for n, i in enumerate(d.loc[fas].sort_values("v43_id").index, 1):
        d.loc[i, "symbol"] = f"fatty acid synthase (probe {n})"
    d["selection_category"] = d.relevance
    for c in ("anatomical_marker", "molecular_class"):  # spelling typos in source
        d[c] = d[c].replace(
            {"serotinergic": "serotonergic", "seritonergic": "serotonergic"}
        )
    # Candidate FlyBase ID by annotation symbol (not orthology evidence).
    d["dmel_candidate_by_symbol"] = [
        "" if o else dmel_by_symbol(sy)
        for o, sy in zip(d.dmel_symbol, d.lookup_symbol, strict=True)
    ]
    d["dmel_method"] = np.where(d.dmel_symbol != "", "NCBI ortholog", "")
    d["marker_class"] = [
        ", ".join(x for x in (a, b) if isinstance(x, str))
        for a, b in zip(d.anatomical_marker, d.molecular_class, strict=True)
    ]
    d = d.sort_values("symbol", key=lambda s: s.str.lower()).reset_index(drop=True)
    cols = [
        "symbol",
        "symbol_source",
        "v43_id",
        "BIMP_NP_Symbol",
        "loc_id",
        "protein",
        "dmel_flybase_id",
        "dmel_symbol",
        "dmel_method",
        "ncbi_geneid",
        "selection_category",
        "marker_class",
        "anatomical_marker",
        "molecular_class",
        "circuit_sender",
        "circuit_receiver",
        "in_snrna_reference",
        "probe_note",
        "dmel_candidate_by_symbol",
    ]
    return d[cols]


E_ORTH = ROOT / "results" / "handoff" / "E_orthologs" / "panel_dmel_orthologs.csv"
ACCEPTED = {"NCBI ortholog", "reciprocal best hit", "two-source consensus"}


def build_s1(var_names) -> pd.DataFrame:
    """S1 with FlyBase columns restricted to accepted orthology calls: NCBI Gene
    orthologs and reciprocal best hits (panel_dmel_orthologs.csv). Non-reciprocal
    best hits are never shown."""
    d = _s1_base(var_names)
    e = pd.read_csv(E_ORTH, dtype=str, keep_default_na=False).set_index("BIMP_v4.3_ID")
    assert set(e.index) == set(d.v43_id)
    meth = d.v43_id.map(e["method"])
    ok = meth.isin(ACCEPTED)
    d["dmel_method"] = np.where(ok, meth, "")
    d["dmel_flybase_id"] = np.where(ok, d.v43_id.map(e["dmel_FBgn"]), "")
    d["dmel_symbol"] = np.where(ok, d.v43_id.map(e["dmel_symbol"]), "")
    d["dmel_candidate_by_symbol"] = ""
    return d.drop(columns="ncbi_geneid")


def s1_tex(d: pd.DataFrame) -> str:
    L = [
        "% Table S1 body. Needs \\usepackage{longtable,booktabs,array} in the "
        "preamble.",
        "% Generated by scripts/gene_annotation/tables_s1_s2.py -- do not edit by "
        "hand.",
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{@{}p{2.5cm}p{2.5cm}p{2.9cm}p{2.5cm}p{3.1cm}c@{}}",
        "\\caption{\\textbf{Xenium gene panel key (BIMP vNP3.1).} Each row is one of "
        "the "
        "101 probes. Symbols are the paper or NCBI annotation symbol where one exists; "
        "otherwise the NCBI protein name is given. \\emph{Drosophila} orthologs "
        "(FlyBase "
        "gene ID and symbol) are NCBI Gene ortholog assignments or, where marked "
        "\\textdagger, reciprocal best hits (DIAMOND blastp, e-value $\\leq10^{-10}$, "
        "$\\geq30\\%$ query coverage, best hit in both directions and not tied with a "
        "paralog, \\emph{B.~impatiens} RefSeq BIMP\\_2.2 versus "
        "\\emph{D.~melanogaster} UniProt proteome) or, where marked $\\ddagger$, "
        "calls in which at least two "
        "independent lines name the same gene and none contradicts it (Ensembl "
        "Metazoa Compara; OrthoDB v11 "
        "group with that gene as its only \\emph{Drosophila} member; reciprocal best "
        "hit under DIAMOND "
        "\\texttt{--ultra-sensitive}, e-value $\\leq10^{-5}$; or matching gene name "
        "plus significant alignment); "
        "non-reciprocal best hits and calls supported by a single line are not "
        "listed as orthologs, and blank means no ortholog call, including two genes "
        "where "
        "\\emph{Drosophila} lacks the gene (vitellogenin, \\emph{Dnmt1a}). Category "
        "and relevance "
        "are the panel-design annotations recorded for each probe; blank where none "
        "was "
        "recorded. Ref.: gene present in the snRNA-seq reference; 100 of 101 probes "
        "match, corazonin does not. Two probes (*) are annotated as fatty acid "
        "synthase and share one Xenium \\texttt{gene\\_name}; they are different "
        "genes (distinct B.~impatiens IDs) and are listed separately, so the panel has "
        "101 probes but 100 distinct gene names.\\label{tab:panel_key}}\\\\",
        "\\toprule",
        "Symbol & \\emph{B.~impatiens} ID & \\emph{Drosophila} ortholog (FlyBase) & "
        "Category & Relevance & Ref.\\\\",
        "\\midrule\\endfirsthead",
        "\\multicolumn{6}{@{}l}{\\emph{Table~\\ref{tab:panel_key} continued}}\\\\",
        "\\toprule",
        "Symbol & \\emph{B.~impatiens} ID & \\emph{Drosophila} ortholog (FlyBase) & "
        "Category & Relevance & Ref.\\\\",
        "\\midrule\\endhead",
        "\\bottomrule\\endlastfoot",
    ]
    for r in d.itertuples():
        sym = tex(r.symbol) + ("$^{*}$" if r.probe_note else "")
        ids = f"{tex(r.v43_id)}\\newline {tex(r.loc_id)}"
        if r.dmel_symbol:
            fl = "; ".join(
                f"{tex(s)} ({tex(i)})"
                for i, s in zip(
                    r.dmel_flybase_id.split(";"), r.dmel_symbol.split(";"), strict=False
                )
            )
            if r.dmel_method == "reciprocal best hit":
                fl += "$^{\\dagger}$"
            elif r.dmel_method == "two-source consensus":
                fl += "$^{\\ddagger}$"
        else:
            fl = "---"
        L.append(
            " & ".join(
                [
                    sym,
                    ids,
                    fl,
                    tex(r.marker_class),
                    tex(r.selection_category),
                    "yes" if r.in_snrna_reference else "no",
                ]
            )
            + "\\\\"
        )
    L += ["\\end{longtable}}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------- S2
def build_s2(adata, s1: pd.DataFrame):
    import scanpy as sc

    from bimp_st.cluster_labels import CELL_TYPE_ORDER, POOL_MAP

    cats = list(adata.obs["cell_type"].astype(str).unique())
    assert set(cats) == set(CELL_TYPE_ORDER) and len(cats) == 29, "29-type mismatch"
    assert set(CELL_TYPE_ORDER) == set(POOL_MAP)
    adata.obs["cell_type"] = pd.Categorical(
        adata.obs["cell_type"].astype(str), categories=CELL_TYPE_ORDER
    )
    sc.tl.rank_genes_groups(adata, "cell_type", method="wilcoxon")  # one-vs-rest
    lk = pd.read_csv(LOOKUP, sep="\t").set_index("gene_id")
    panel_ids = set(s1.v43_id)
    sym = {r.v43_id: r.symbol for r in s1.itertuples()}
    rows = []
    for ct in CELL_TYPE_ORDER:
        df = sc.get.rank_genes_groups_df(adata, ct).head(50)
        for rank, r in enumerate(df.itertuples(), 1):
            g = r.names
            gene = sym.get(g) or lk["gene"].get(g)
            desc = lk["description"].get(g)
            rows.append(
                dict(
                    cell_type=ct,
                    rank=rank,
                    gene_id=g,
                    symbol=gene if isinstance(gene, str) else "",
                    description=desc if isinstance(desc, str) else "",
                    score=r.scores,
                    logfoldchange=r.logfoldchanges,
                    pval_adj=r.pvals_adj,
                    on_panel=g in panel_ids,
                )
            )
    top = pd.DataFrame(rows)
    ct_caste = pd.crosstab(adata.obs.cell_type.astype(str), adata.obs.caste)
    s1d = s1.set_index("v43_id")
    summ = []
    for i, ct in enumerate(CELL_TYPE_ORDER, 1):
        t = top[top.cell_type == ct]
        t10 = t.head(10)

        def lab(r):
            if r.symbol:
                return r.symbol
            if r.description:
                d = r.description
                if len(d) > 30:
                    d = d[:30].rsplit(" ", 1)[0] + "…"
                return d
            return r.gene_id.replace("BIMP_v4.3_", "v4.3_")

        summ.append(
            dict(
                id=i,
                annotation=ct,
                pooled_group=POOL_MAP[ct],
                n_queen=int(ct_caste.loc[ct, "Queen"]),
                n_worker=int(ct_caste.loc[ct, "Worker"]),
                top10_markers="; ".join(lab(r) for r in t10.itertuples()),
                top10_gene_ids="; ".join(t10.gene_id),
                top10_dmel_orthologs="; ".join(
                    f"{lab(r)}={s1d.loc[r.gene_id, 'dmel_symbol'] or 'NA'}"
                    for r in t10.itertuples()
                    if r.gene_id in s1d.index
                ),
                n_panel_in_top50=int(t.on_panel.sum()),
                panel_genes_in_top50="; ".join(sym[g] for g in t[t.on_panel].gene_id),
            )
        )
    return top, pd.DataFrame(summ)


def s2_tex(s: pd.DataFrame) -> str:
    L = [
        "% Table S2 body. Needs \\usepackage{longtable,booktabs,array} in the "
        "preamble.",
        "% Generated by scripts/gene_annotation/tables_s1_s2.py -- do not edit by "
        "hand.",
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{@{}r p{2.6cm} r r p{6.8cm} r@{}}",
        "\\caption{\\textbf{snRNA-seq reference populations.} The 29 transcriptionally "
        "distinct populations of the combined \\emph{B.~impatiens} snRNA-seq reference "
        "(central brain and optic lobe), with nuclei per caste, the ten top markers "
        "(Scanpy \\texttt{rank\\_genes\\_groups}, Wilcoxon, one versus rest, ranked by "
        "score) and the number of Xenium panel genes among each population's top 50 "
        "markers. Markers are shown by annotation symbol where one exists, otherwise "
        "by "
        "NCBI product name. Full top-50 lists are in the accompanying data file. "
        "OL\\_1--OL\\_13 are unannotated optic-lobe "
        "subclusters.\\label{tab:snrna_annotations}}\\\\",
        "\\toprule",
        "ID & Population & Queen & Worker & Top 10 markers & Panel in top 50\\\\",
        "\\midrule\\endfirsthead",
        "\\multicolumn{6}{@{}l}{\\emph{Table~\\ref{tab:snrna_annotations} "
        "continued}}\\\\",
        "\\toprule",
        "ID & Population & Queen & Worker & Top 10 markers & Panel in top 50\\\\",
        "\\midrule\\endhead",
        "\\bottomrule\\endlastfoot",
    ]
    for r in s.itertuples():
        mk = "; ".join(tex(x) for x in r.top10_markers.split("; "))
        L.append(
            " & ".join(
                [
                    str(r.id),
                    tex(r.annotation),
                    f"{r.n_queen:,}".replace(",", "{,}"),
                    f"{r.n_worker:,}".replace(",", "{,}"),
                    mk,
                    str(r.n_panel_in_top50),
                ]
            )
            + "\\\\"
        )
    L += ["\\end{longtable}}", ""]
    return "\n".join(L)


def main() -> None:
    import anndata as ad

    OUT.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(SN)
    s1 = build_s1(set(adata.var_names))
    s1.to_csv(OUT / "S1_panel_key.csv", index=False)
    (OUT / "S1_panel_key.tex").write_text(s1_tex(s1))
    print(
        "S1:",
        len(s1),
        "probes; in reference:",
        int(s1.in_snrna_reference.sum()),
        "; not:",
        s1[~s1.in_snrna_reference].symbol.tolist(),
        "; with Dmel ortholog:",
        int((s1.dmel_symbol != "").sum()),
    )
    top, s2 = build_s2(adata, s1)
    top.to_csv(OUT / "snrna_markers_top50.csv", index=False)
    s2.to_csv(OUT / "S2_snrna_annotations.csv", index=False)
    (OUT / "S2_snrna_annotations.tex").write_text(s2_tex(s2))
    print(
        "S2:", len(s2), "populations; nuclei Q/W:", s2.n_queen.sum(), s2.n_worker.sum()
    )


if __name__ == "__main__":
    main()
