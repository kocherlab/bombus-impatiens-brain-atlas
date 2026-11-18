"""Drosophila orthologs for the 101 Xenium panel genes (feeds Table S1).

Run: PYTHONPATH=src .venv/bin/python scripts/gene_annotation/panel_orthologs.py
Writes only under results/handoff/E_orthologs/.

Method per gene:
  1. NCBI Gene orthologs (gene_orthologs.gz, tax 7227 <-> 132113) -> "NCBI ortholog".
  2. Otherwise reciprocal best hit (DIAMOND blastp --sensitive, e-value <= 1e-10).
     B. impatiens: NCBI RefSeq GCF_000188095.3 (BIMP_2.2) proteins, collapsed to
     NCBI GeneID (the LOC ids). D. melanogaster: UniProt proteome UP000000803,
     collapsed to FBgn. Forward = best FBgn by bitscore over all isoform pairs;
     reverse = best B. impatiens gene for that FBgn's proteins. Reciprocal iff the
     reverse best is the query gene. Non-reciprocal -> "best hit, not reciprocal"
     (not an ortholog call; excluded from S1). No hit -> "unresolved".
  RBH is also computed for NCBI-ortholog genes as a concordance check.
Needs a `diamond` binary (env DIAMOND or on PATH); alignments are cached.
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gene_annotation.tables_s1_s2 as si  # noqa: E402

OUTD = si.ROOT / "results" / "handoff" / "E_orthologs"
C = OUTD / "cache"
B = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/188/095/GCF_000188095.3_BIMP_2.2"
UNIP = (
    "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000000803&format=tsv"
    "&fields=accession,gene_primary,xref_flybase,xref_geneid,sequence"
)
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
EMAX = 1e-10
PARALOG_NOTE = (
    0.8  # 2nd-best gene within this fraction of best bitscore: flag one-to-many
)
PARALOG_AMBIG = (
    0.9  # ... within this fraction: best hit is a near tie, not accepted as ortholog
)
MIN_COV = 30.0  # min % of the B. impatiens query covered by the alignment


def sh(cmd):
    subprocess.run(cmd, check=True)


def datasets_genes(gids: list[str]) -> dict[str, tuple[str, str]]:
    """NCBI GeneID -> (FBgn, symbol) for D. melanogaster genes; cached, never caches
    misses."""
    cf = C / "geneid_info.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    need = [g for g in gids if g and g not in cache]
    for i in range(0, len(need), 20):
        ch = need[i : i + 20]
        out = subprocess.run(
            [
                "curl",
                "-s",
                "-q",
                "-H",
                "accept: application/json",
                si.DATASETS.format(",".join(ch)),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for r in json.loads(out).get("reports", []):
            g = r["gene"]
            cache[str(g["gene_id"])] = [
                g.get("nomenclature_authority", {}).get("identifier", ""),
                g["symbol"],
            ]
    for g1 in [
        g for g in need if g not in cache
    ]:  # a chunk with one bad id returns nothing: retry singly
        out = subprocess.run(
            [
                "curl",
                "-s",
                "-q",
                "-H",
                "accept: application/json",
                si.DATASETS.format(g1),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for r in json.loads(out).get("reports", []):
            g = r["gene"]
            cache[str(g["gene_id"])] = [
                g.get("nomenclature_authority", {}).get("identifier", ""),
                g["symbol"],
            ]
    cf.write_text(json.dumps(cache))
    return {g: tuple(cache[g]) for g in gids if g in cache}


def prepare():
    C.mkdir(parents=True, exist_ok=True)
    if not (C / "bimp.fa").exists():
        sh(
            [
                "curl",
                "-sq",
                "-o",
                str(C / "bimp.faa.gz"),
                f"{B}/GCF_000188095.3_BIMP_2.2_protein.faa.gz",
            ]
        )
        sh(
            [
                "curl",
                "-sq",
                "-o",
                str(C / "bimp.gff.gz"),
                f"{B}/GCF_000188095.3_BIMP_2.2_genomic.gff.gz",
            ]
        )
        p2g = {}
        for line in gzip.open(C / "bimp.gff.gz", "rt"):
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            a = dict(x.split("=", 1) for x in f[8].split(";") if "=" in x)
            g = re.search(r"GeneID:(\d+)", a.get("Dbxref", ""))
            if a.get("protein_id") and g:
                p2g[a["protein_id"]] = g.group(1)
        keep = False
        with open(C / "bimp.fa", "w") as o:
            for line in gzip.open(C / "bimp.faa.gz", "rt"):
                if line[0] == ">":
                    pid = line[1:].split()[0]
                    keep = pid in p2g
                    if keep:
                        o.write(f">{pid}|{p2g[pid]}\n")
                elif keep:
                    o.write(line)
    if not (C / "dmel.fa").exists():
        sh(["curl", "-sq", "-o", str(C / "dmel.tsv"), UNIP])
        recs = list(csv.DictReader(open(C / "dmel.tsv"), delimiter="\t"))
        # some UniProt entries (e.g. SPARC) lack a FlyBase xref: resolve FBgn via NCBI
        miss = sorted(
            {
                r["GeneID"].strip(";").split(";")[0]
                for r in recs
                if not r["FlyBase"].startswith("FBgn") and r["GeneID"]
            }
        )
        info = datasets_genes(miss)
        with open(C / "dmel.fa", "w") as o:
            for r in recs:
                gid = r["GeneID"].strip(";").split(";")[0]
                fb = r["FlyBase"].strip(";").split(";")[0]
                if not fb.startswith("FBgn"):
                    fb = info.get(gid, ("", ""))[0]
                if fb.startswith("FBgn") and r["Sequence"]:
                    o.write(f">{r['Entry']}|{fb}|{gid}\n{r['Sequence']}\n")
    dm = os.environ.get("DIAMOND") or shutil.which("diamond")
    assert dm, "diamond binary not found (set DIAMOND)"
    for name in ("bimp", "dmel"):
        if not (C / f"{name}.dmnd").exists():
            sh(
                [
                    dm,
                    "makedb",
                    "--in",
                    str(C / f"{name}.fa"),
                    "-d",
                    str(C / name),
                    "--quiet",
                ]
            )
    for q, d, out in (("bimp", "dmel", "fwd"), ("dmel", "bimp", "rev")):
        if not (C / f"{out}.tsv").exists():
            sh(
                [
                    dm,
                    "blastp",
                    "-q",
                    str(C / f"{q}.fa"),
                    "-d",
                    str(C / d),
                    "--evalue",
                    str(EMAX),
                    "--max-target-seqs",
                    "100",
                    "--sensitive",
                    "--threads",
                    "8",
                    "--quiet",
                    "-o",
                    str(C / f"{out}.tsv"),
                    "--outfmt",
                    "6",
                    *OUTFMT,
                ]
            )


def relaxed_candidates(base) -> pd.DataFrame:
    """Sub-threshold (e<=1e-3) best hit for panel proteins; used only to annotate
    unresolved genes. Never an ortholog call."""
    f = C / "relaxed.tsv"
    if not f.exists():
        keep = set(base.gid)
        with open(C / "panel.fa", "w") as o:
            on = False
            for line in open(C / "bimp.fa"):
                if line[0] == ">":
                    on = line.strip().split("|")[1] in keep
                if on:
                    o.write(line)
        dm = os.environ.get("DIAMOND") or shutil.which("diamond")
        sh(
            [
                dm,
                "blastp",
                "-q",
                str(C / "panel.fa"),
                "-d",
                str(C / "dmel"),
                "--evalue",
                "1e-3",
                "--max-target-seqs",
                "20",
                "--sensitive",
                "--threads",
                "8",
                "--quiet",
                "-o",
                str(f),
                "--outfmt",
                "6",
                *OUTFMT,
            ]
        )
    return load("relaxed.tsv")


def load(fn):
    """Headers are acc|gene[|..]; field 1 is the gene on both sides (GeneID / FBgn)."""
    t = pd.read_csv(C / fn, sep="\t", names=OUTFMT)
    t["qg"] = t.qseqid.str.split("|").str[1]
    t["sg"] = t.sseqid.str.split("|").str[1]
    return t


def best_by_gene(t, col):
    return (
        t.sort_values("bitscore", ascending=False)
        .groupby(col, as_index=False)
        .first()
        .sort_values("bitscore", ascending=False)
        .reset_index(drop=True)
    )


def fb_symbols(fb_to_gid: dict[str, str]) -> dict[str, str]:
    info = datasets_genes([g for g in fb_to_gid.values() if g])
    return {f: info.get(g, ("", ""))[1] for f, g in fb_to_gid.items()}


# ---------------------------------------------------------------- adjudication
# A call beyond NCBI/strict-RBH is accepted only when >= 2 independent evidence lines
# name the same D. melanogaster gene and none contradicts it. Lines:
#   ncbi   NCBI Gene ortholog (direct)
#   ens    Ensembl Metazoa Compara (bombus_impatiens -> drosophila_melanogaster);
#          counts when the candidate is the only Dmel ortholog listed
#   odb    OrthoDB v11 group (Endopterygota, else Insecta) containing the gene; counts
#          when the candidate is the only Dmel member of that group
#   rrbh   relaxed reciprocal best hit: DIAMOND --ultra-sensitive, e<=RRBH_E, >=30%
#          query coverage, reciprocal, neither direction tied (<0.9) with a paralog
#   annot  annotation + alignment: the RefSeq protein name equals the Dmel gene's name
#          and the ultra-sensitive alignment to that gene has e<=1e-3
# Contradiction: the candidate's best DIAMOND bitscore is < 0.8 of the best Dmel hit's.
RRBH_E = 1e-5
CONTRADICT = 0.8
EXT = OUTD / "cache_ext"
NEW_METHOD = "two-source consensus"
NO_ORTH = "no Dmel ortholog (deliberate call)"
NO_ORTH_ALLOWED = {"vitellogenin", "Dnmt1a"}
LOWCOV = "NCBI ortholog, not corroborated (low coverage)"


def _norm(s: str) -> str:
    s = re.sub(r"low quality protein:?_?", "", str(s).lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def dmel_descriptions(fbs: set, fb2gid: dict) -> dict:
    """FBgn -> NCBI Gene description (cached)."""
    cf = EXT / "dmel_desc.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    need = [f for f in fbs if f not in cache]
    for f in need:
        gid = fb2gid.get(f) or ""
        if not gid:
            continue
        out = subprocess.run(
            [
                "curl",
                "-s",
                "-q",
                "-H",
                "accept: application/json",
                si.DATASETS.format(gid),
            ],
            capture_output=True,
            text=True,
        ).stdout
        try:
            cache[f] = json.loads(out)["reports"][0]["gene"].get("description", "")
        except (ValueError, KeyError, IndexError):
            pass
    cf.write_text(json.dumps(cache))
    return cache


def ensembl_display(fb: str) -> str:
    """FlyBase symbol of an FBgn (Ensembl lookup display_name; cached)."""
    cf = EXT / "fb_symbol.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    if fb not in cache:
        t = subprocess.run(
            [
                "curl",
                "-s",
                "-q",
                "-H",
                "Content-Type: application/json",
                f"https://rest.ensembl.org/lookup/id/{fb}?species=drosophila_melanogaster",
            ],
            capture_output=True,
            text=True,
        ).stdout
        try:
            cache[fb] = json.loads(t)["display_name"]
        except (ValueError, KeyError):
            return ""
        cf.write_text(json.dumps(cache))
    return cache[fb]


def uniprot_extra(fbs: list) -> None:
    """Dmel proteins absent from the proteome FASTA (e.g. Crz) ->
    cache_ext/extra_dmel.fa."""
    f = EXT / "extra_dmel.fa"
    have = {ln[1:].split("|")[1] for ln in open(C / "dmel.fa") if ln[0] == ">"}
    have |= (
        {ln[1:].split("|")[1] for ln in open(f) if ln[0] == ">"}
        if f.exists()
        else set()
    )
    with open(f, "a") as o:
        for fb in sorted(set(fbs) - have):
            url = (
                "https://rest.uniprot.org/uniprotkb/search?query=xref:flybase-"
                + fb
                + "&format=tsv&fields=accession,sequence&size=5"
            )
            t = (
                subprocess.run(
                    ["curl", "-s", "-q", url], capture_output=True, text=True
                )
                .stdout.strip()
                .split("\n")
            )
            for _k, line in enumerate(t[1:2]):
                acc, seq = line.split("\t")
                o.write(f">{acc}|{fb}|\n{seq}\n")


def ultra_runs(cands: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward (panel proteins -> Dmel) and reverse (candidate Dmel -> B. impatiens),
    both DIAMOND --ultra-sensitive, e<=1e-3; cached under cache_ext/."""
    dm = os.environ.get("DIAMOND") or shutil.which("diamond")
    assert dm, "diamond binary not found (set DIAMOND)"
    EXT.mkdir(exist_ok=True)
    uniprot_extra(sorted(cands))
    plus = EXT / "dmel_plus.fa"
    plus.write_text(
        open(C / "dmel.fa").read()
        + (
            open(EXT / "extra_dmel.fa").read()
            if (EXT / "extra_dmel.fa").exists()
            else ""
        )
    )
    if (
        not (EXT / "dmel_plus.dmnd").exists()
        or (EXT / "dmel_plus.dmnd").stat().st_mtime < plus.stat().st_mtime
    ):
        sh([dm, "makedb", "--in", str(plus), "-d", str(EXT / "dmel_plus"), "--quiet"])
    common = [
        "--evalue",
        "1e-3",
        "--ultra-sensitive",
        "--threads",
        "8",
        "--quiet",
        "--outfmt",
        "6",
        *OUTFMT,
    ]
    fw = EXT / "ultra_fwd.tsv"
    if not fw.exists():
        sh(
            [
                dm,
                "blastp",
                "-q",
                str(C / "panel.fa"),
                "-d",
                str(EXT / "dmel_plus"),
                "--max-target-seqs",
                "20",
                "-o",
                str(fw),
                *common,
            ]
        )
    t = pd.read_csv(fw, sep="\t", names=OUTFMT)
    t["qg"], t["sg"] = t.qseqid.str.split("|").str[1], t.sseqid.str.split("|").str[1]
    top = (
        t.sort_values("bitscore", ascending=False)
        .groupby(["qg", "sg"], as_index=False)
        .first()
        .sort_values("bitscore", ascending=False)
        .groupby("qg")
        .head(5)
    )
    want = set(top.sg) | cands
    rv = EXT / "ultra_rev.tsv"
    sig = EXT / "ultra_rev.set"
    if not rv.exists() or not sig.exists() or sig.read_text() != ",".join(sorted(want)):
        keep = False
        with open(EXT / "cand_dmel.fa", "w") as o:
            for line in open(plus):
                if line[0] == ">":
                    keep = line.strip().split("|")[1] in want
                if keep:
                    o.write(line)
        sh(
            [
                dm,
                "blastp",
                "-q",
                str(EXT / "cand_dmel.fa"),
                "-d",
                str(C / "bimp"),
                "--max-target-seqs",
                "50",
                "-o",
                str(rv),
                *common,
            ]
        )
        sig.write_text(",".join(sorted(want)))
    r = pd.read_csv(rv, sep="\t", names=OUTFMT)
    r["qg"], r["sg"] = r.qseqid.str.split("|").str[1], r.sseqid.str.split("|").str[1]
    return t, r


def adjudicate(
    df: pd.DataFrame, base: pd.DataFrame, fb2gid: dict, sym: dict
) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gene_annotation.panel_ortholog_sources as src  # noqa: E402

    locs = list(df["LOC ID"])
    ens = src.ensembl_sets(json.loads((EXT / "ensembl.json").read_text()))
    nm = src.name_map()
    odb = src.orthodb_sets(json.loads((EXT / "orthodb.json").read_text()), nm)
    apis = src.apis_sets(locs)
    fbsym = {
        v: k for k, v in reversed(list(nm.items())) if not re.match(r"CG\d+|Dmel_", k)
    }
    tdf = pd.read_csv(
        EXT / "dmel_names.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    fbsym.update(
        {
            r[4].strip(";").split(";")[0]: r[1]
            for r in tdf.itertuples(index=False)
            if r[1]
        }
    )

    # candidate FBgns from the database lines, for the reverse/ultra runs
    cands = set()
    for loc in locs:
        e = ens.get(loc) or []
        cands |= {x["fb"] for x in e}
        o = odb.get(loc)
        if o:
            for g in o["groups"]:
                cands |= g["fb"]
    fwd, rev = ultra_runs(cands)
    prot = dict(zip(base.v43_id, base.protein, strict=True))
    desc = dmel_descriptions(
        cands | set(df.dmel_FBgn.dropna().astype(str)) - {""}, fb2gid
    )

    def bgene(t, q):
        s = t[t.qg == q]
        return (
            s.sort_values("bitscore", ascending=False)
            .groupby("sg", as_index=False)
            .first()
            .sort_values("bitscore", ascending=False)
        )

    rows = []
    for r in df.to_dict("records"):
        loc, gid = r["LOC ID"], r["LOC ID"].replace("LOC", "")
        pass1 = r["method"]
        ncbi_fb = r["dmel_FBgn"] if pass1 == "NCBI ortholog" else ""
        ncbi_set = {x for x in str(ncbi_fb).split(";") if x.startswith("FBgn")}
        e = ens.get(loc)
        ens_set = None if e is None else {x["fb"] for x in e}
        o = odb.get(loc)
        og = None
        if o is not None:
            for lvl in ("33392", "50557"):
                gs = [g for g in o["groups"] if g["level"] == lvl]
                if gs:
                    og = gs[0]
                    break
        odb_set = None if og is None else set(og["fb"])
        odb_unmapped = [] if og is None else og["unmapped"]
        # relaxed RBH
        fb_ = bgene(fwd, gid)
        rr = None
        if len(fb_):
            t = fb_.iloc[0]
            rvv = bgene(rev, t.sg)
            rb = rvv.iloc[0].sg if len(rvv) else None
            f2 = fb_.iloc[1].bitscore / t.bitscore if len(fb_) > 1 else 0.0
            r2 = rvv.iloc[1].bitscore / rvv.iloc[0].bitscore if len(rvv) > 1 else 0.0
            rr = dict(
                fb=t.sg,
                pid=t.pident,
                cov=t.qcovhsp,
                ev=t.evalue,
                recip=rb == gid,
                f2=f2,
                r2=r2,
            )
        rrbh_fb = (
            rr["fb"]
            if rr
            and rr["recip"]
            and rr["ev"] <= RRBH_E
            and rr["cov"] >= MIN_COV
            and rr["f2"] < PARALOG_AMBIG
            and rr["r2"] < PARALOG_AMBIG
            else None
        )
        # candidate universe
        universe = (
            set(ncbi_set)
            | (ens_set or set())
            | (odb_set or set())
            | ({rrbh_fb} if rrbh_fb else set())
        )
        res = {}
        for x in sorted(universe):
            v = {}
            if x in ncbi_set:
                v["NCBI Gene"] = "ortholog" + (
                    "" if len(ncbi_set) == 1 else f" (one of {len(ncbi_set)})"
                )
            if ens_set and x in ens_set and len(ens_set) == 1:
                v["Ensembl Compara"] = next(h["type"] for h in e if h["fb"] == x)
            if odb_set and x in odb_set and len(odb_set) == 1 and not odb_unmapped:
                v["OrthoDB"] = og["og"]
            if rrbh_fb == x:
                v["relaxed RBH"] = (
                    f"id {rr['pid']:.0f}%, cov {rr['cov']:.0f}%, e={rr['ev']:.1e}"
                )
            # annotation + alignment
            hit = fb_[fb_.sg == x]
            if (
                len(hit)
                and hit.iloc[0].evalue <= 1e-3
                and x in desc
                and _norm(desc[x]) == _norm(prot.get(r["BIMP_v4.3_ID"], ""))
                and _norm(desc[x])
            ):
                v["annotation + alignment"] = (
                    f"name '{desc[x]}'; alignment id {hit.iloc[0].pident:.0f}% "
                    f"cov {hit.iloc[0].qcovhsp:.0f}% e={hit.iloc[0].evalue:.1e}"
                )
            ratio = (
                (float(hit.iloc[0].bitscore) / float(fb_.iloc[0].bitscore))
                if len(hit) and len(fb_)
                else None
            )
            res[x] = (v, ratio)
        good = {
            x: v
            for x, (v, ratio) in res.items()
            if len(v) >= 2
            and (
                ratio is None or ratio >= CONTRADICT or "NCBI Gene" in v and len(v) >= 3
            )
        }
        r2 = dict(r)
        r2["method_pass1"] = pass1
        r2["ensembl_dmel"] = (
            "gap" if ens_set is None else (";".join(sorted(ens_set)) or "none")
        )
        if o is None:
            r2["orthodb_dmel"] = "gap"
        elif og is None:
            r2["orthodb_dmel"] = "no group at Endopterygota/Insecta level"
        else:
            mem = ";".join(sorted(og["fb"])) + (
                " +unmapped " + ",".join(odb_unmapped) if odb_unmapped else ""
            )
            r2["orthodb_dmel"] = f"{og['og']}: {mem or 'no Dmel member'}"
        r2["relaxed_rbh"] = (
            (
                f"{rr['fb']} id{rr['pid']:.0f} cov{rr['cov']:.0f} e={rr['ev']:.1e} "
                f"{'reciprocal' if rr['recip'] else 'not reciprocal'}"
                f"{'' if rrbh_fb else ' (does not meet relaxed-RBH criteria)'}"
            )
            if rr
            else "no hit"
        )
        r2["apis_route_ncbi"] = ";".join(apis[loc]["dmel_gid"]) or (
            "apis ortholog has no Dmel pair" if apis[loc]["apis"] else "none"
        )
        r2["evidence"] = ""
        if pass1 in si.ACCEPTED:
            ev = (
                res.get(r["dmel_FBgn"].split(";")[0], ({}, None))[0]
                if r["dmel_FBgn"]
                else {}
            )
            if (
                pass1 == "NCBI ortholog"
                and float(r["coverage"] or 100) < MIN_COV
                and len(good) == 0
            ):
                r2["method"] = LOWCOV
                r2["confidence note"] += (
                    f"; moved out of S1: alignment covers {r['coverage']}% of query "
                    "and no second "
                    "independent source names the same gene "
                    f"(Ensembl: {r2['ensembl_dmel']}; OrthoDB: {r2['orthodb_dmel']})"
                )
                r2["evidence"] = "NCBI only"
            else:
                r2["evidence"] = "; ".join(f"{k}: {s}" for k, s in ev.items()) or pass1
        elif len(good) == 1:
            x, v = next(iter(good.items()))
            r2.update(
                method=NEW_METHOD,
                dmel_FBgn=x,
                dmel_symbol=ensembl_display(x) or fbsym.get(x, ""),
            )
            r2["evidence"] = "; ".join(f"{k}: {s}" for k, s in v.items())
            r2["confidence note"] = (
                f"{len(v)} independent lines agree ({', '.join(v)}); pass-1 call was "
                f"'{pass1}'. " + r["confidence note"]
            )
            if rr and rr["fb"] == x:
                r2.update(
                    {
                        "%id": round(rr["pid"], 1),
                        "coverage": round(rr["cov"], 1),
                        "evalue": f"{rr['ev']:.1e}",
                    }
                )
            r2["reciprocal"] = "yes" if rr and rr["fb"] == x and rr["recip"] else "no"
        elif len(good) > 1:
            r2["method"] = "unresolved" if pass1 == "unresolved" else pass1
            r2["evidence"] = "conflict: " + "; ".join(
                f"{x} ({', '.join(v)})" for x, v in good.items()
            )
        else:
            # deliberate no-ortholog call: OrthoDB group has the B. impatiens gene and
            # no Dmel member, and Ensembl lists no Dmel ortholog
            # Only where the biology independently supports absence (Dmel has no
            # vitellogenin: the yolk proteins Yp1-3 are lipase-family, not
            # vitellogenins; no Dnmt1: only Dnmt2/Mt2). An OrthoDB group that merely
            # lacks a Dmel member is not proof of absence (FASN, NPF), so it is not
            # used to label anything else.
            if (
                r["gene symbol"] in NO_ORTH_ALLOWED
                and og is not None
                and not og["fb"]
                and not odb_unmapped
                and ens_set is not None
                and not ens_set
                and pass1 not in si.ACCEPTED
            ):
                r2["method"] = NO_ORTH
                r2["evidence"] = (
                    f"OrthoDB {og['og']} ({og['name']}) has no D. melanogaster "
                    "member; Ensembl lists none"
                )
                r2["dmel_FBgn"] = r2["dmel_symbol"] = ""
            else:
                sup = {x: v for x, (v, _) in res.items() if v}
                r2["evidence"] = (
                    "; ".join(f"{x}: {', '.join(v)}" for x, v in sup.items())
                    or "no source names a gene"
                )
        if r2["method"] not in si.ACCEPTED | {NEW_METHOD} and r2["evidence"]:
            r2["confidence note"] = (
                r2["confidence note"] + "; other lines: " + r2["evidence"]
            )
        r2["n_independent_lines"] = (
            len(good.get(next(iter(good)), {}))
            if len(good) == 1
            else len(res.get(str(r2["dmel_FBgn"]).split(";")[0], ({}, 0))[0])
        )
        rows.append(r2)
    return pd.DataFrame(rows)


NAMED_NOTES = """\
## Decisions for the genes called out in review

* **Kr-h1** (LOC100740522, `Kruppel homolog 1`): **accepted, FBgn0266450 (Kr-h1)**. Lines: Ensembl Compara (one-to-one) +
  OrthoDB group 244393at33392 (Dmel member Kr-h1) + annotation (RefSeq name identical to the FlyBase name) with a
  Smith-Waterman alignment of the C2H2 zinc-finger array (`alignments_named.txt`: score 931 against a shuffled-sequence
  null maximum of 77, 0/500 shuffles >= observed, 192 identities over 391 aligned columns). The B. impatiens model (500 aa)
  aligns over residues 2-326 of the 845 aa Dmel protein (the zinc-finger region), while the standard DIAMOND search
  reported only a 6% query-coverage hit and no hit at e<=1e-10. NOTE: FBgn0266696 is *Svil* (Supervillin), not
  Kr-h1; Kr-h1 is FBgn0266450 (FlyBase R6.54, NCBI GeneID 33861).
* **sNPF** (LOC100748800, `short neuropeptide F precursor`): **accepted, FBgn0032840 (sNPF)**. Lines: OrthoDB group
  299742at33392 (Dmel member sNPF) + reciprocal best hit under DIAMOND `--ultra-sensitive` (56% identity over 41% of the
  query, e=3.7e-06; fails the strict e<=1e-10 / short-precursor regime only because the precursor is 135 aa). Supporting,
  not counted: identical mature-peptide core `AQRSPSLRLRFG` in the alignment (`alignments_named.txt`, score 110 vs null
  max 59).
* **NPF** (LOC100744000, `neuropeptide Y-like`): **left blank (one line only)**. No BLAST-family hit, no Ensembl Dmel
  ortholog, and the OrthoDB group (`pro-neuropeptide Y-like`) has no Dmel member. The Smith-Waterman alignment to Dmel NPF is
  above the shuffled null (66 vs max 39) but the RefSeq name (neuropeptide Y-like) does not match the FlyBase name
  (neuropeptide F), so the annotation line is not satisfied. Not called rather than inferred.
* **Crz** (LOC100746471, `corazonin`): **left blank (one line only)**. Dmel Crz is FBgn0013767 (FBgn0015233 does not
  exist); the Dmel precursor is in the search database, but the B. impatiens precursor has no hit even at e<=1e-3
  under `--ultra-sensitive` (the RefSeq model lacks the conserved pQTFQYSRGWTNamide core: alignment score 41, inside the
  shuffled null, 4/500 >= observed). OrthoDB places the gene in a `corazonin` group with Dmel Crz as its only member, but
  Ensembl lists no Dmel ortholog and the alignment line fails, so only one line supports the call.
* **vitellogenin** (LOC100745377): **no Dmel ortholog (deliberate call)**. OrthoDB group 304584at33392 (`Vitellogenin`)
  has no D. melanogaster member, Ensembl lists none, and no hit at e<=1e-3 under `--ultra-sensitive`. D. melanogaster
  lacks a vitellogenin; the yolk proteins Yp1-3 are lipase-family and are not forced as orthologs.
* **Dnmt1a** (LOC100742803): **no Dmel ortholog (deliberate call)**. OrthoDB group 407468at33392 (`DNA cytosine-5
  methyltransferase`) has no Dmel member, Ensembl lists none; the only sub-threshold hit (Kdm2) covers 4% of the query.
  D. melanogaster has Dnmt2 (Mt2) only.
* **HSFBP1** (LOC100740425): **accepted, FBgn0032429 (CG5446)**. Ensembl Compara (one-to-one) + OrthoDB group
  341239at33392 + reciprocal best hit under `--ultra-sensitive` (69% identity over 96% of the query, e=9.9e-08; the pass-1
  threshold e<=1e-10 is hard to reach for such a short protein).
* **repo** (LOC100741401, NCBI ortholog FBgn0011701, homeodomain-only alignment covering 17% of the query): **kept**.
  The second line is OrthoDB group 313377at33392 in which repo is the only Dmel member. Ensembl lists no ortholog.
  Because the alignment is homeodomain-only, this is the weakest call in the table.
* **Moved out / not upgraded.** Single-line candidates are not called: for the non-reciprocal and paralog-tied cases,
  Ensembl and OrthoDB mostly disagree with each other or list several Dmel genes (e.g. kainate 2, Tret1, tramtrack, Hex70a,
  Idgf4, Jheh1), which is exactly the paralog ambiguity the pass-1 rule flagged. OrthoDB groups that merely lack a Dmel
  member (FASN probe 2, apolipoprotein D, nAChR alpha-10-like, NPF) are *not* read as absence of an ortholog.
* **nAChRb2**: sources agree on FBgn0031261 (nAChRbeta3); the panel symbol says beta2, so the symbol/ortholog name
  discordance is reported, not hidden.
"""  # noqa: E501 -- Markdown written verbatim to the report; wrapping changes its text


def write_evidence(df: pd.DataFrame) -> None:
    vc = df.method.value_counts()
    acc = int(df.method.isin(si.ACCEPTED).sum())
    L = [
        "# Drosophila orthologs of the 101 panel genes: evidence",
        "",
        f"**{acc} of 101 probes carry an accepted ortholog** (NCBI Gene ortholog "
        f"{vc.get('NCBI ortholog', 0)}; strict "
        f"reciprocal best hit {vc.get('reciprocal best hit', 0)}; two-source "
        f"consensus {vc.get(NEW_METHOD, 0)}). "
        "Pass 1 (NCBI + strict RBH) gave 68. Two genes are deliberate no-ortholog "
        "calls "
        f"({vc.get(NO_ORTH, 0)}); the remaining {101 - acc - vc.get(NO_ORTH, 0)} are "
        "left blank.",
        "",
        "Evidence lines (a call beyond NCBI / strict RBH needs >= 2 that name the "
        "same gene and none contradicting it, "
        f"i.e. candidate best DIAMOND bitscore >= {CONTRADICT} of the best Dmel hit):",
        "",
        "1. NCBI Gene ortholog. 2. Ensembl Metazoa Compara (rest.ensembl.org, "
        "bombus_impatiens -> drosophila_melanogaster); "
        "counts only if the candidate is the sole Dmel ortholog listed. 3. OrthoDB "
        "v11 group (Endopterygota, else Insecta) "
        "containing the gene; counts only if the candidate is the sole Dmel member "
        "(second annotation set 7227_1 ignored; "
        "members resolved to FBgn via UniProt/Ensembl, unresolvable members block "
        "'sole'). 4. Relaxed RBH: DIAMOND "
        f"`--ultra-sensitive`, e<={RRBH_E:g}, >={MIN_COV:.0f}% query coverage, "
        "reciprocal, neither direction tied with a paralog "
        f"(<{PARALOG_AMBIG}). 5. Annotation + alignment: RefSeq protein name equals "
        "the Dmel gene's NCBI name and the "
        "ultra-sensitive alignment has e<=1e-3 (Kr-h1 and sNPF additionally have the "
        "Smith-Waterman alignments in "
        "`alignments_named.txt`). The Apis route (B. impatiens -> A. mellifera -> "
        "Dmel through NCBI Gene orthologs) was "
        "queried and returns no Dmel gene for any of the 101 genes (NCBI lists too "
        "few Apis-Dmel pairs), so it "
        "never counted.",
        "",
        NAMED_NOTES,
        "## Per-gene evidence (all 101 probes)",
        "",
        "| gene | B. impatiens | Dmel call | method | pass 1 | evidence / other lines "
        "| Ensembl | OrthoDB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples(index=False):
        call = (
            f"{r.dmel_symbol} ({r.dmel_FBgn})"
            if r.dmel_FBgn and r.method in si.ACCEPTED
            else "blank"
        )
        L.append(
            f"| {r[0]} | {r[2]} | {call} | {r.method} | {r.method_pass1} | "
            f"{r.evidence} | "
            f"{r.ensembl_dmel} | {r.orthodb_dmel} |"
        )
    (OUTD / "ortholog_evidence.md").write_text("\n".join(L) + "\n")


def main():
    prepare()
    fwd, rev = load("fwd.tsv"), load("rev.tsv")
    base = si._s1_base(set())
    base["gid"] = base.ncbi_geneid.astype(str)
    relaxed = relaxed_candidates(base)
    fb2gid = {}
    for line in open(C / "dmel.fa"):
        if line[0] == ">":
            _, fb, gid = line[1:].strip().split("|")
            fb2gid[fb] = gid

    rows = []
    for r in base.itertuples():
        x = dict(
            symbol=r.symbol,
            v43=r.v43_id,
            loc=r.loc_id,
            gid=r.gid,
            ncbi_fb=r.dmel_flybase_id,
            ncbi_sym=r.dmel_symbol,
            top=None,
        )
        f = fwd[fwd.qg == r.gid]
        if not f.empty:
            best = best_by_gene(f, "sg")
            top = best.iloc[0]
            rvg = best_by_gene(
                rev[rev.qg == top.sg], "sg"
            )  # best B. impatiens genes for this FBgn
            x.update(
                top=top.sg,
                pid=top.pident,
                cov=top.qcovhsp,
                ev=top.evalue,
                fwd2=best.iloc[1].bitscore / top.bitscore if len(best) > 1 else 0.0,
                fwd2fb=best.iloc[1].sg if len(best) > 1 else "",
                rbest=rvg.iloc[0].sg if len(rvg) else None,
                rev2=rvg.iloc[1].bitscore / rvg.iloc[0].bitscore
                if len(rvg) > 1
                else 0.0,
                rev2g=rvg.iloc[1].sg if len(rvg) > 1 else "",
                hits=best.set_index("sg"),
            )
        rows.append(x)

    share = defaultdict(list)
    for x in rows:
        if x["top"]:
            share[x["top"]].append(x["symbol"])
    for x in rows:
        if not x["top"]:
            rl = relaxed[relaxed.qg == x["gid"]]
            x["relaxed"] = best_by_gene(rl, "sg").iloc[0] if len(rl) else None
    fbs = (
        {x["top"] for x in rows if x["top"]}
        | {f for x in rows for f in x["ncbi_fb"].split(";") if f}
        | {x["fwd2fb"] for x in rows if x.get("fwd2fb")}
        | {x["relaxed"].sg for x in rows if x.get("relaxed") is not None}
    )
    sym = fb_symbols({f: fb2gid.get(f, "") for f in fbs})

    out = []
    for x in rows:
        o = {"gene symbol": x["symbol"], "BIMP_v4.3_ID": x["v43"], "LOC ID": x["loc"]}
        recip = bool(x["top"]) and x["rbest"] == x["gid"]
        if x["ncbi_fb"]:
            fbl, syl = x["ncbi_fb"].split(";"), x["ncbi_sym"].split(";")
            hit = (
                x["hits"].loc[fbl[0]]
                if x["top"] and fbl[0] in x["hits"].index
                else None
            )
            note = []
            if len(fbl) > 1:
                note.append(f"NCBI lists {len(fbl)} Dmel orthologs (one-to-many)")
            if not x["top"]:
                note.append("no blastp hit at e<=1e-10; NCBI call retained")
            elif x["top"] in fbl:
                note.append(
                    "own blastp best hit agrees with NCBI call"
                    + ("; reciprocal" if recip else "; not reciprocal at protein level")
                )
            else:
                note.append(
                    f"own blastp best hit differs ({sym.get(x['top'], '')} "
                    f"{x['top']}); "
                    "NCBI call retained as accepted orthology (check)"
                )
            if hit is not None and hit.qcovhsp < MIN_COV:
                note.append(
                    f"low alignment coverage ({hit.qcovhsp:.0f}% of query; "
                    "domain-level similarity only)"
                )
            o.update(
                {
                    "dmel_FBgn": ";".join(fbl),
                    "dmel_symbol": ";".join(syl),
                    "method": "NCBI ortholog",
                    "%id": round(hit.pident, 1) if hit is not None else "",
                    "coverage": round(hit.qcovhsp, 1) if hit is not None else "",
                    "evalue": f"{hit.evalue:.1e}" if hit is not None else "",
                    "reciprocal": ("yes" if recip and x["top"] == fbl[0] else "no")
                    if x["top"]
                    else "n/a",
                    "confidence note": "; ".join(note),
                }
            )
        elif not x["top"]:
            rl = x["relaxed"]
            why = "no D. melanogaster blastp hit at e<=1e-10"
            if rl is not None:
                why += (
                    f"; best sub-threshold hit (not accepted) {sym.get(rl.sg, '')} "
                    f"{rl.sg}, "
                    f"e={rl.evalue:.1e}, {rl.pident:.0f}% id over {rl.qcovhsp:.0f}% "
                    "of query"
                )
            o.update(
                {
                    "dmel_FBgn": "",
                    "dmel_symbol": "",
                    "method": "unresolved",
                    "%id": "",
                    "coverage": "",
                    "evalue": "",
                    "reciprocal": "n/a",
                    "confidence note": why,
                }
            )
        else:
            note = []
            if not recip:
                note.append(f"reverse best hit is GeneID {x['rbest']}, not this gene")
            if x["fwd2"] >= PARALOG_NOTE:
                note.append(
                    f"close Dmel paralog {sym.get(x['fwd2fb'], '')} ({x['fwd2']:.2f} "
                    "of top bitscore): one-to-many"
                )
            if x["rev2"] >= PARALOG_NOTE:
                note.append(
                    f"close B. impatiens paralog GeneID {x['rev2g']} in reverse "
                    f"search ({x['rev2']:.2f})"
                )
            if len(share[x["top"]]) > 1:
                note.append(
                    "same Dmel best hit as panel genes: "
                    + ", ".join(s for s in share[x["top"]] if s != x["symbol"])
                )
            strong = x["pid"] >= 40 and x["cov"] >= 70
            tier = (
                "high"
                if recip and strong and not note
                else ("moderate" if recip else "low")
            )
            note.append(f"{tier} (id {x['pid']:.0f}%, cov {x['cov']:.0f}%)")
            method = "reciprocal best hit" if recip else "best hit, not reciprocal"
            if recip and x["fwd2"] >= PARALOG_AMBIG:
                method = "best hit, ambiguous paralog"
                note.insert(
                    0,
                    f"reciprocal but near-tied with paralog (>= {PARALOG_AMBIG:.1f} "
                    "of top bitscore); "
                    "not accepted as ortholog",
                )
            elif recip and x["cov"] < MIN_COV:
                method = "best hit, low coverage"
                note.insert(
                    0,
                    f"reciprocal but covers <{MIN_COV:.0f}% of query; not accepted as "
                    "ortholog",
                )
            o.update(
                {
                    "dmel_FBgn": x["top"],
                    "dmel_symbol": sym.get(x["top"], ""),
                    "method": method,
                    "%id": round(x["pid"], 1),
                    "coverage": round(x["cov"], 1),
                    "evalue": f"{x['ev']:.1e}",
                    "reciprocal": "yes" if recip else "no",
                    "confidence note": "; ".join(note),
                }
            )
        out.append(o)

    df = pd.DataFrame(out).sort_values("gene symbol", key=lambda s: s.str.lower())
    df = adjudicate(df, base, fb2gid, sym)
    df.to_csv(OUTD / "panel_dmel_orthologs.csv", index=False)
    write_evidence(df)
    n_ok = int(df.method.isin(si.ACCEPTED).sum())
    (OUTD / "summary_by_method.txt").write_text(
        df.method.value_counts().to_string()
        + f"\n\naccepted (shown in S1): {n_ok} of {len(df)}\n"
    )
    print(df.method.value_counts().to_string(), "\naccepted:", n_ok)


if __name__ == "__main__":
    main()
