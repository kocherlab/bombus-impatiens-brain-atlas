"""Genome-wide Drosophila orthologs + search names for every BIMP_v4.3 gene.

Writes data/panel/bimp_v43_gene_names.csv, which export_data.py (via gene_names.py)
folds into the web atlas manifest so genes are findable by bee name, fly ortholog
symbol, fly synonyms and common names (mblk-1, Met/gce, ...).

Same method as scripts/gene_annotation/panel_orthologs.py, extended from the 100 panel
genes to the whole OGS, and keyed directly on BIMP_v4.3 proteins rather than the NCBI
BIMP_2.2 LOC proteome:

  * B. impatiens: BIMP_OGS_v4.3 CDS translated with gffread from the reference GTF +
    genome (the Cell Ranger reference, <genome_dir>/genome/, site key genome_dir).
    Internal stops -> X.
  * D. melanogaster: UniProt proteome UP000000803, headers ACC|FBgn|GeneID (the
    handoff cache's dmel.fa), plus a UniProt TSV with gene_primary, gene_synonym,
    protein_name, xref_flybase, reviewed for names.
  * DIAMOND v2.2.8 blastp --sensitive -e 1e-5 -k 25, both directions.

Per bee gene (best over isoform pairs, by bitscore):
  rbh                      forward best fly gene's reverse best is this gene
  rbh_one_to_many          rbh, and further fly genes' reverse best is also this gene
                           (e.g. Met + gce for the single bee Met)
  best_hit_not_reciprocal  NOT an ortholog call; recorded, not used for aliases
  none                     no hit at e <= 1e-10
confidence: high (>= 40% id over >= 50% of the bee protein), moderate (>= 30% coverage),
"domain only" (< 30%: a short conserved domain, e.g. Mblk-1/E93's 69-aa Pipsqueak motif
at 91% id; DIAMOND reports one HSP per pair, so coverage is a lower bound).

Rebuild (inputs live outside the repo; see data/panel/README_gene_names.md):
  diamond blastp -q bimp.fa -d dmel -o fwd.tsv --sensitive -e 1e-5 -k 25 \
      --outfmt 6 qseqid sseqid pident length qlen slen evalue bitscore qcovhsp scovhsp
  diamond blastp -q dmel.fa -d bimp -o rev.tsv  (same options)
  python3 viewer/tools/gene_orthologs.py --gtf BIMP_OGS_v4.3.gtf \
      --lookup BIMP_lookup.csv --fwd fwd.tsv --rev rev.tsv --dmel-fa dmel.fa \
      --dmel-names dmel_pn.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/panel/bimp_v43_gene_names.csv"
EMAX = 1e-10
MIN_COV = 30.0
NEAR_TIE = 0.9
MAX_SYN = 8

# Names people search by that neither the bee annotation nor fly orthology supplies.
# Keyed by v4.3 id; "common" is |-separated. "symbol"/"desc" override the bee annotation
# (which is kept as an alias).
CURATED = {
    # NCBI calls this "circadian locomoter output cycles protein kaput", but it is the
    # reciprocal best hit of fly gce (268 bits, 41% id) and Met's own best bee hit
    # (207 bits); fly Clk's reciprocal partner is 002563 (488 bits, 43% id). The bee
    # has the single pre-duplication Met/gce gene: the JH receptor.
    "BIMP_v4.3_002131": dict(
        symbol="Met",
        desc="Methoprene-tolerant, the juvenile hormone receptor (fly Met/gce "
        "ortholog; "
        "NCBI mislabels it circadian locomoter output cycles protein kaput)",
        common="Met|gce|Met/gce|JH receptor|juvenile hormone receptor|"
        "methoprene-tolerant|germ cell-expressed|kaput",
    ),
    "BIMP_v4.3_002563": dict(symbol="Clk", common="CLOCK"),
    # honeybee Mblk-1 (mushroom body large-type Kenyon cell-specific protein 1) is E93
    "BIMP_v4.3_008845": dict(
        symbol="Mblk-1",
        common="mblk-1|mblk1|Mblk|E93|Eip93F|"
        "mushroom body large-type Kenyon cell-specific protein 1",
    ),
    "BIMP_v4.3_005433": dict(common="Kr-h1|Krh1|krueppel homolog 1"),
    "BIMP_v4.3_005319": dict(symbol="Jhamt", common="Jhamt|JH acid methyltransferase"),
    "BIMP_v4.3_002440": dict(symbol="Jhe", common="Jhe|juvenile hormone esterase"),
    "BIMP_v4.3_003656": dict(symbol="Jhe", common="Jhe|juvenile hormone esterase"),
    "BIMP_v4.3_009948": dict(symbol="Jhe", common="Jhe|juvenile hormone esterase"),
    # RBH is fly Jheh2; Jheh1 and Jheh3 (a fly tandem triplication) are co-orthologs
    "BIMP_v4.3_002733": dict(
        symbol="Jheh",
        common="Jheh|Jheh1|Jheh2|Jheh3|juvenile hormone epoxide hydrolase",
    ),
    "BIMP_v4.3_002958": dict(common="InR|InR1|insulin receptor"),
    "BIMP_v4.3_001273": dict(common="InR|InR2|insulin receptor"),
    "BIMP_v4.3_002771": dict(common="Jhbp|takeout|to|JH binding protein"),
    "BIMP_v4.3_002610": dict(common="Jhbp|takeout|to|JH binding protein"),
    "BIMP_v4.3_000471": dict(symbol="Ats", common="Ats|AT|allatotropin"),
    "BIMP_v4.3_009248": dict(
        symbol="AstA", common="AstA|Ast-A|Ast|allatostatin A|allatostatin"
    ),
    "BIMP_v4.3_009257": dict(common="Tk|tachykinin|tachykinins"),
    "BIMP_v4.3_005730": dict(common="sNPF|short neuropeptide F"),
    "BIMP_v4.3_007673": dict(common="Hr38|hormone receptor 38"),
    # 005500 (1,772 aa) is the full-length vitellogenin; 001050/001051 (~318 aa) are
    # short
    # vitellogenin-like ORFs (001050 is the panel's "vitellogenin"), so they keep their
    # names
    "BIMP_v4.3_005500": dict(symbol="Vg", common="Vg|Vgn|vitellogenin"),
    "BIMP_v4.3_001050": dict(common="Vg|vitellogenin"),
    "BIMP_v4.3_001051": dict(common="Vg|vitellogenin"),
    "BIMP_v4.3_001668": dict(common="Vg|vitellogenin"),
    "BIMP_v4.3_010935": dict(
        symbol="Vgr", common="Vgr|yl|yolkless|vitellogenin receptor"
    ),
}

_JUNK = re.compile(
    r"^(CG\d|CR\d|FBgn|anon|l\(|BcDNA|EG:|Dmel|lincRNA|GH\d|EP\d|BG:|DS\d|DmelCG)", re.I
)


def clean_syns(raw: str, primary: str) -> list[str]:
    out, seen = [], {primary.lower()}
    for s in raw.split():
        if len(s) < 2 or len(s) > 12 or _JUNK.match(s) or any(c in s for c in ":\\/;"):
            continue
        if re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", s) or re.fullmatch(r"Dm[A-Z].*", s):
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= MAX_SYN:
            break
    return out


def protein_name(pn: str) -> str:
    s = re.sub(r"\s*\(.*$", "", pn)  # first clause only
    s = re.sub(r",?\s*isoform\b.*$", "", s)
    return s.strip()


def strip_isoform(s: str) -> str:
    return re.sub(r"\s*isoform\s+[\w-]+$", "", s).strip()


def load_dmel(names_tsv, dmel_fa):
    """FBgn -> (symbol, protein name, [synonyms]), preferring reviewed entries.

    Joined on the UniProt accession in dmel.fa's ACC|FBgn headers, not on UniProt's own
    FlyBase xref: some FBgns in the cached proteome have since been merged or lack the
    xref (e.g. Jheh1, A1ZBF2), and keying on the xref silently drops their names."""
    acc_fb = {}
    for line in open(dmel_fa):
        if line[0] == ">":
            acc, fb = line[1:].split("|")[:2]
            acc_fb[acc] = fb
    best = {}
    for r in csv.DictReader(open(names_tsv), delimiter="\t"):
        fb = acc_fb.get(r["Entry"])
        sym = r["Gene Names (primary)"].strip().replace("Dmel\\", "")
        if not fb or not sym:
            continue
        rank = r["Reviewed"] == "reviewed"
        if fb in best and best[fb][0] >= rank:
            continue
        best[fb] = (
            rank,
            sym,
            protein_name(r["Protein names"]),
            clean_syns(r["Gene Names (synonym)"], sym),
        )
    return {fb: v[1:] for fb, v in best.items()}


def gene_hits(path, qgene, sgene):
    """query gene -> subject gene -> best pair stats (bits, pident, qcov, evalue)."""
    hits = defaultdict(dict)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        q, s = qgene(f[0]), sgene(f[1])
        bits, ev = float(f[7]), float(f[6])
        if bits > hits[q].get(s, (0.0,))[0]:
            hits[q][s] = (bits, float(f[2]), float(f[8]), ev)
    return hits


def main():
    ap = argparse.ArgumentParser()
    for a in ("gtf", "lookup", "fwd", "rev", "dmel_names", "dmel_fa"):
        ap.add_argument("--" + a.replace("_", "-"), required=True, type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    bee = {}
    for r in csv.DictReader(open(a.lookup), delimiter="\t"):
        bee[r["gene_id"]] = dict(
            symbol=r["gene"].strip(),
            desc=r["description"].strip().strip('"'),
            biotype=r["gene_biotype"],
        )
    for line in open(a.gtf):  # GTF product where the lookup has no description
        f = line.split("\t")
        if len(f) == 9 and f[2] in ("CDS", "transcript"):  # transcript: lncRNA products
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
            m = re.search(r'product "([^"]+)"', f[8])
            bt = re.search(r'transcript_biotype "([^"]+)"', f[8])
            b = bee.setdefault(gid, dict(symbol="", desc="", biotype=""))
            b["biotype"] = b["biotype"] or (
                "protein_coding" if f[2] == "CDS" else bt.group(1) if bt else ""
            )
            if m and not b["desc"]:
                b["desc"] = strip_isoform(
                    re.sub(r", transcript variant \w+$", "", m.group(1))
                )
    dmel = load_dmel(a.dmel_names, a.dmel_fa)
    fwd = gene_hits(a.fwd, lambda x: x.split("|")[1], lambda x: x.split("|")[1])
    rev = gene_hits(a.rev, lambda x: x.split("|")[1], lambda x: x.split("|")[1])

    def ok(h):
        return h[3] <= EMAX

    rev_best = {}
    for fb, hs in rev.items():
        good = {g: h for g, h in hs.items() if h[3] <= EMAX}
        if good:
            rev_best[fb] = max(good, key=lambda g: good[g][0])

    rows = []
    for gid in sorted(bee):
        b, cur = bee[gid], CURATED.get(gid, {})
        hs = {fb: h for fb, h in fwd.get(gid, {}).items() if ok(h)}
        method, fb, h, co, note = "none", "", None, [], ""
        if hs:
            ranked = sorted(hs, key=lambda x: -hs[x][0])
            fb, h = ranked[0], hs[ranked[0]]
            if rev_best.get(fb) == gid:
                co = [x for x in ranked[1:] if rev_best.get(x) == gid]
                method = "rbh_one_to_many" if co else "rbh"
            else:
                method = "best_hit_not_reciprocal"
                rb = rev_best.get(fb)
                note = f"fly gene's best bee hit is {rb}" if rb else ""
            if (
                len(ranked) > 1
                and hs[ranked[1]][0] >= NEAR_TIE * h[0]
                and ranked[1] not in co
            ):
                note = "; ".join(
                    x
                    for x in (
                        note,
                        f"near tie with {dmel.get(ranked[1], (ranked[1],))[0]}",
                    )
                    if x
                )
        orth = method.startswith("rbh")
        if orth:
            conf = (
                "high"
                if h[1] >= 40 and h[2] >= 50
                else "moderate"
                if h[2] >= MIN_COV
                else "domain only"
            )
        else:
            conf = "low" if h else ""
        sym_d, pname, syns = dmel.get(fb, ("", "", []))
        co_syms = [dmel.get(x, (x,))[0] for x in co]

        aliases = []
        if orth:
            aliases += [sym_d] + co_syms + syns + [pname] + [fb] + co
            for x in co:
                aliases += dmel.get(x, ("", "", []))[2] + [dmel.get(x, ("", "", []))[1]]
        common = cur.get("common", "")
        if b["symbol"]:
            aliases.append(b["symbol"])
        if (
            cur.get("symbol")
            and b["desc"]
            and not b["desc"].lower().startswith("uncharacterized")
        ):
            aliases.append(b["desc"])  # the overridden annotation stays searchable
        seen, al = set(), []
        for x in aliases:
            if x and x.lower() not in seen:
                seen.add(x.lower())
                al.append(x)
        rows.append(
            {
                "bimp_id": gid,
                "bee_symbol": b["symbol"],
                "bee_desc": b["desc"],
                "biotype": b["biotype"],
                "symbol": cur.get("symbol", ""),
                "desc": cur.get("desc", ""),
                "dmel_symbol": sym_d if fb else "",
                "dmel_fbgn": fb,
                "dmel_name": pname,
                "dmel_coorthologs": ";".join(co_syms),
                "method": method,
                "confidence": conf,
                "pident": f"{h[1]:.1f}" if h else "",
                "qcov": f"{h[2]:.1f}" if h else "",
                "evalue": f"{h[3]:.1e}" if h else "",
                "bitscore": f"{h[0]:.0f}" if h else "",
                "note": note,
                "common_names": common,
                "aliases": "|".join(al),
            }
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter

    c = Counter(r["method"] for r in rows if r["biotype"] == "protein_coding")
    print(f"wrote {a.out} ({len(rows)} genes); protein-coding methods: {dict(c)}")


if __name__ == "__main__":
    main()
