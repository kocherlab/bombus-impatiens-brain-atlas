#!/usr/bin/env python
"""Genome-wide Drosophila orthologs for the BIMP v4.3 genes (reciprocal best hit).

The repo's ortholog table
(``results/handoff/E_orthologs/panel_dmel_orthologs.csv``) covers the 101 panel genes
only, through RefSeq LOC ids. The DE tables are keyed by BIMP v4.3 ids, which carry no
RefSeq protein id, so this builds the v4.3 proteome and repeats the panel pipeline's
reciprocal-best-hit rule on all genes:

1. proteome: the longest CDS of every gene in ``BIMP_OGS_v4.3.gtf`` (the cellranger
   reference annotation), cut from ``BIMP_genome_v4.fasta`` and translated (standard
   code, cut at the first stop);
2. DIAMOND blastp --sensitive, e <= 1e-10, both directions, against the
   D. melanogaster proteome the panel pipeline already built (UniProt UP000000803;
   headers ``accession|FBgn|GeneID``, cached in ``results/handoff/E_orthologs/cache``,
   read only);
3. forward = best FBgn by bitscore over isoform pairs; reverse = best v4.3 gene for
   that FBgn's proteins; an ortholog call needs reciprocity, >= 30% query coverage, and
   the second-best FBgn below 0.9 of the best bitscore (0.8-0.9 is flagged
   one-to-many). Otherwise ``best hit, not reciprocal`` / ``ambiguous`` is a hit, never
   an ortholog call.

Validated against the panel table. The GTF also gives each gene's product description
and ``NCBIOrtholog`` (the A. mellifera GeneID), which keys the repo's GO annotation
(``data/go_annotation``).

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/gene_annotation/genome_dmel_orthologs.py   # DIAMOND=/path/to/diamond
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from bimp_st.site import get as site_path

REPO = Path(__file__).resolve().parents[2]
GENOME_DIR = site_path("genome_dir") / "genome"
GTF = GENOME_DIR / "BIMP_OGS_v4.3.gtf"
FASTA = GENOME_DIR / "BIMP_genome_v4.fasta"
CACHE = REPO / "results/handoff/E_orthologs/cache"
OUT = REPO / "results/round5/glia_labels"
OUTFMT = [
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "qlen",
    "slen",
    "evalue",
    "bitscore",
    "qcovhsp",
    "scovhsp",
]
EMAX, MIN_COV, PARALOG_NOTE, PARALOG_AMBIG = 1e-10, 30.0, 0.8, 0.9
CODONS = {
    a + b + c: aa
    for (a, b, c), aa in zip(
        ((a, b, c) for a in "TCAG" for b in "TCAG" for c in "TCAG"),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
        strict=False,
    )
}
COMP = str.maketrans("ACGTN", "TGCAN")


def read_fasta(path: Path) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    for line in open(path):
        if line[0] == ">":
            if name:
                seqs[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf).upper()
    return seqs


def translate(nt: str) -> str:
    aa = [CODONS.get(nt[i : i + 3], "X") for i in range(0, len(nt) - 2, 3)]
    p = "".join(aa)
    return p.split("*")[0]


def gtf_genes():
    """Gene table (description, A. mellifera GeneID) and every transcript's CDS
    segments."""
    genes, cds = {}, {}
    for line in open(GTF):
        p = line.rstrip("\n").split("\t")
        if len(p) < 9:
            continue
        a = p[8]
        g = re.search(r'gene_id "([^"]+)"', a).group(1)
        if p[2] == "gene":
            x = re.search(r'db_xref "NCBIOrtholog:(\d+)"', a)
            d = re.search(r'description "([^"]*)"', a)
            b = re.search(r'gene_biotype "([^"]+)"', a)
            genes[g] = dict(
                description=d.group(1) if d else "",
                apis_geneid=x.group(1) if x else "",
                gene_biotype=b.group(1) if b else "",
            )
        elif p[2] == "CDS":
            t = re.search(r'transcript_id "([^"]+)"', a).group(1)
            cds.setdefault((g, t), []).append(
                (p[0], int(p[3]), int(p[4]), p[6], int(p[7]) if p[7].isdigit() else 0)
            )
    return genes, cds


def build_proteome(genes, cds) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    faa = OUT / "bimp43_longest.faa"
    genome = read_fasta(FASTA)
    best = {}
    for (g, t), segs in cds.items():
        L = sum(e - s + 1 for _, s, e, _, _ in segs)
        if g not in best or L > best[g][0]:
            best[g] = (L, t, segs)
    n = 0
    with open(faa, "w") as o:
        for g, (_L, t, segs) in sorted(best.items()):
            strand = segs[0][3]
            segs = sorted(segs, key=lambda x: x[1], reverse=(strand == "-"))
            nt = ""
            for chrom, s, e, st, _ph in segs:
                piece = genome[chrom][s - 1 : e]
                nt += piece.translate(COMP)[::-1] if st == "-" else piece
            nt = nt[segs[0][4] :]
            prot = translate(nt)
            if len(prot) >= 30:
                o.write(f">{t}|{g}\n{prot}\n")
                n += 1
    print(f"proteome: {n} proteins of {len(genes)} genes -> {faa}", flush=True)
    return faa


def diamond(faa: Path) -> tuple[Path, Path]:
    dm = os.environ.get("DIAMOND") or shutil.which("diamond")
    assert dm, "set DIAMOND to a diamond binary"
    fwd, rev = OUT / "fwd.tsv", OUT / "rev.tsv"
    subprocess.run(
        [dm, "makedb", "--in", str(faa), "-d", str(OUT / "bimp43"), "--quiet"],
        check=True,
    )
    for q, d, o in (
        (faa, CACHE / "dmel", fwd),
        (CACHE / "dmel.fa", OUT / "bimp43", rev),
    ):
        subprocess.run(
            [
                dm,
                "blastp",
                "-q",
                str(q),
                "-d",
                str(d),
                "--evalue",
                str(EMAX),
                "--max-target-seqs",
                "100",
                "--sensitive",
                "--threads",
                "8",
                "--quiet",
                "-o",
                str(o),
                "--outfmt",
                "6",
                *OUTFMT,
            ],
            check=True,
        )
    return fwd, rev


def load(fn):
    t = pd.read_csv(fn, sep="\t", names=OUTFMT)
    t["qg"], t["sg"] = t.qseqid.str.split("|").str[1], t.sseqid.str.split("|").str[1]
    return t


def main() -> None:
    genes, cds = gtf_genes()
    faa = OUT / "bimp43_longest.faa"
    if not faa.exists():
        faa = build_proteome(genes, cds)
    fwd_f, rev_f = OUT / "fwd.tsv", OUT / "rev.tsv"
    if not (fwd_f.exists() and rev_f.exists()):
        diamond(faa)
    fwd, rev = load(fwd_f), load(rev_f)
    sym = {}
    for r in csv.DictReader(open(CACHE / "dmel.tsv"), delimiter="\t"):
        fb = r["FlyBase"].strip(";").split(";")[0]
        if fb.startswith("FBgn"):
            sym[fb] = r["Gene Names (primary)"] or fb
    # reverse: for each FBgn (the query side of rev), the best v4.3 gene (the subject
    # side)
    rev_best = (
        rev.sort_values("bitscore", ascending=False).groupby("qg").first().sg.to_dict()
    )
    rows = []
    for g, d in fwd.sort_values("bitscore", ascending=False).groupby("qg"):
        top = d.iloc[0]
        others = d[d.sg != top.sg]
        ratio = float(others.bitscore.iloc[0] / top.bitscore) if len(others) else 0.0
        recip = rev_best.get(top.sg) == g
        if recip and top.qcovhsp >= MIN_COV and ratio < PARALOG_AMBIG:
            call = "reciprocal best hit" + (
                " (one-to-many)" if ratio >= PARALOG_NOTE else ""
            )
        elif recip:
            call = "ambiguous"
        else:
            call = "best hit, not reciprocal"
        rows.append(
            dict(
                gene=g,
                dmel_FBgn=top.sg,
                dmel_symbol=sym.get(top.sg, ""),
                pident=top.pident,
                qcov=top.qcovhsp,
                evalue=top.evalue,
                bitscore=top.bitscore,
                second_best_ratio=ratio,
                reciprocal=recip,
                call=call,
                second_best_symbol=sym.get(others.sg.iloc[0], "")
                if len(others)
                else "",
            )
        )
    t = pd.DataFrame(rows)
    g = pd.DataFrame.from_dict(genes, orient="index").rename_axis("gene").reset_index()
    t = g.merge(t, on="gene", how="left")
    t["call"] = t["call"].fillna("no hit (e <= 1e-10)")
    prot = {ln[1:].strip().split("|")[1] for ln in open(faa) if ln[0] == ">"}
    t["has_protein"] = t.gene.isin(prot)
    # weak hint only for genes with no hit at 1e-10: best Dmel hit at e <= 1e-3, never
    # an ortholog call
    nohit = set(t.loc[t.call.str.startswith("no hit") & t.has_protein, "gene"])
    rel_f = OUT / "relaxed.tsv"
    if nohit and not rel_f.exists():
        sub = OUT / "nohit.faa"
        with open(sub, "w") as o:
            on = False
            for ln in open(faa):
                if ln[0] == ">":
                    on = ln[1:].strip().split("|")[1] in nohit
                if on:
                    o.write(ln)
        dm = os.environ.get("DIAMOND") or shutil.which("diamond")
        subprocess.run(
            [
                dm,
                "blastp",
                "-q",
                str(sub),
                "-d",
                str(CACHE / "dmel"),
                "--evalue",
                "1e-3",
                "--max-target-seqs",
                "5",
                "--sensitive",
                "--threads",
                "8",
                "--quiet",
                "-o",
                str(rel_f),
                "--outfmt",
                "6",
                *OUTFMT,
            ],
            check=True,
        )
    if rel_f.exists() and rel_f.stat().st_size:
        rl = load(rel_f).sort_values("bitscore", ascending=False).groupby("qg").first()
        t["relaxed_dmel_symbol"] = t.gene.map(rl.sg.map(sym))
        t["relaxed_evalue"] = t.gene.map(rl.evalue)
        t["relaxed_pident"] = t.gene.map(rl.pident)
    t.to_csv(OUT / "v43_dmel_rbh.csv.gz", index=False)
    print(t.call.value_counts().to_string(), flush=True)
    # validation against the panel table (RefSeq-based pipeline)
    pan = pd.read_csv(REPO / "results/handoff/E_orthologs/panel_dmel_orthologs.csv")
    m = pan.merge(
        t, left_on="BIMP_v4.3_ID", right_on="gene", suffixes=("_panel", "_v43")
    )
    both = m[
        m.dmel_FBgn_panel.notna()
        & m.dmel_FBgn_v43.notna()
        & m.call.str.startswith("reciprocal")
    ]
    agree = both.dmel_FBgn_panel == both.dmel_FBgn_v43
    print(
        f"validation: panel genes {len(pan)}, with a v4.3 RBH call {len(both)}, same "
        f"FBgn {int(agree.sum())} "
        f"({agree.mean():.1%})",
        flush=True,
    )
    m.assign(same_FBgn=m.dmel_FBgn_panel == m.dmel_FBgn_v43)[
        [
            "gene symbol",
            "BIMP_v4.3_ID",
            "dmel_symbol_panel",
            "dmel_symbol_v43",
            "dmel_FBgn_panel",
            "dmel_FBgn_v43",
            "method",
            "call",
            "same_FBgn",
        ]
    ].to_csv(OUT / "validation_vs_panel_orthologs.csv", index=False)


if __name__ == "__main__":
    main()
