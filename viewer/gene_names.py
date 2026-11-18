"""Bee names, fly orthologs and search aliases for the atlas's gene list.

Reads data/panel/bimp_v43_gene_names.csv (built by tools/gene_orthologs.py) and fills
each manifest gene's `symbol`, `desc`, `aliases` (search text only) and `fly` (the fly
ortholog symbol(s), when there is a reciprocal-best-hit call). export_data.py calls
annotate() on every export; run this file directly to patch an already-exported
pages/data/manifest.json (and its inferred_meta.json sidecar) in place:

    python3 viewer/gene_names.py [pages/data]
    python3 viewer/gene_names.py --check Met gce mblk-1 ...   # resolve a gene list

Labels: a curated name wins (BIMP_v4.3_002131 is Met/gce, which NCBI labels CLOCK).
Otherwise measured (panel) genes keep the export's label and description, which the
figure scripts share. Inferred genes, which the export leaves unnamed, take the bee
annotation's symbol, else the reciprocal-best-hit fly symbol, else the shortened bee
description; a gene the bee annotation leaves unnamed says it is labelled by its fly
ortholog.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data/panel/bimp_v43_gene_names.csv"
_JUNK_SYM = re.compile(r"^(BcDNA|anon|l\(|EG:)")  # fly symbols too opaque for a label
_UNNAMED = re.compile(r"^(uncharacterized|putative uncharacterized|hypothetical)", re.I)


def _short(s: str) -> str:
    s = re.sub(r"\s*isoform\s+[\w-]+$", "", s)
    s = re.sub(r"\s*[_,]\s*$", "", s).strip()
    return s[:25].rstrip() + "…" if len(s) > 26 else s


def load(path: Path = CSV) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {r["bimp_id"].lower(): r for r in csv.DictReader(fh)}


_MT = {
    "nd": "NADH dehydrogenase subunit ",
    "cox": "cytochrome c oxidase subunit ",
    "atp": "ATP synthase subunit ",
    "cytb": "cytochrome b",
    "rrns": "12S ribosomal RNA",
    "rrnl": "16S ribosomal RNA",
}


def _mito(g: dict) -> None:
    """Mitochondrial genome genes (mt-nd3, mt-trna-met2, ...): the OGS has no
    product."""
    part = g["id"].lower()[3:]
    if part.startswith("trna-"):
        aa = part[5:].capitalize()
        g["symbol"], g["desc"] = f"mt-tRNA-{aa}", f"mitochondrial tRNA-{aa}"
        return
    m = re.match(r"(nd|cox|atp|cytb|rrns|rrnl)(\w*)$", part)
    if m:
        g["symbol"] = "mt-" + part.upper()
        g["desc"] = "mitochondrial " + _MT[m.group(1)] + m.group(2).upper()
        g["aliases"] = ["mt:" + part.upper(), "mitochondrial"]


def annotate(genes: list[dict], table: dict | None = None) -> list[dict]:
    """Fill symbol/desc/aliases/fly on manifest gene dicts, in place. Idempotent."""
    table = load() if table is None else table
    for g in genes:
        if g["id"].lower().startswith("mt-"):
            _mito(g)
            continue
        r = table.get(g["id"].lower())
        if not r:
            continue
        orth = r["method"].startswith("rbh")
        fly = (
            "/".join(
                x for x in [r["dmel_symbol"]] + r["dmel_coorthologs"].split(";") if x
            )
            if orth
            else ""
        )
        bee_desc = (
            r["bee_desc"] if r["bee_desc"] and not _UNNAMED.match(r["bee_desc"]) else ""
        )
        fly_label = (
            r["dmel_symbol"] if orth and not _JUNK_SYM.match(r["dmel_symbol"]) else ""
        )
        # closest fly gene when it is not an ortholog call: said in the description only
        near = (
            f"closest fly gene {r['dmel_symbol']} (not "
            f"reciprocal, {float(r['pident']):.0f}% identity)"
            if r["method"] == "best_hit_not_reciprocal" and r["dmel_symbol"]
            else ""
        )
        if r["symbol"]:
            g["symbol"] = r["symbol"]
        elif g.get("kind") == "inferred" and not g.get("symbol"):
            # last resorts are what the annotation says it is ("uncharacterized lncRNA",
            # "pseudogene"), so no row is a bare id; the id is shown beneath every name
            g["symbol"] = (
                r["bee_symbol"]
                or fly_label
                or (_short(bee_desc) if bee_desc else "")
                or _short(r["bee_desc"])
                or r["biotype"].replace("_", " ")
            ) or None
        if r["desc"]:
            g["desc"] = r["desc"]
        elif g.get("kind") == "inferred" and not g.get("desc"):
            if bee_desc:
                g["desc"] = bee_desc
            elif orth:
                g["desc"] = f"unnamed in the bee annotation; fly ortholog {fly}" + (
                    f" ({r['dmel_name']})" if r["dmel_name"] else ""
                )
            else:
                g["desc"] = "; ".join(x for x in (r["bee_desc"], near) if x) or None
        al = [x for x in r["common_names"].split("|") + r["aliases"].split("|") if x]
        seen = {(g.get("symbol") or "").lower()}
        g["aliases"] = [x for x in al if not (x.lower() in seen or seen.add(x.lower()))]
        if fly:
            g["fly"] = fly
        else:
            g.pop("fly", None)
    return genes


def patch(data_dir: Path) -> None:
    table = load()
    assert table, f"no gene-name table at {CSV}"
    for name in ("manifest.json", "inferred_meta.json"):
        p = data_dir / name
        if not p.exists():
            continue
        obj = json.loads(p.read_text())
        genes = obj["genes"] if isinstance(obj, dict) else obj
        annotate(genes, table)
        p.write_text(json.dumps(obj))
        named = sum(1 for g in genes if g.get("symbol"))
        print(
            f"{p}: {len(genes)} genes, {named} named, "
            f"{sum(1 for g in genes if g.get('fly'))} with a fly "
            f"ortholog, {p.stat().st_size / 1e6:.2f} MB"
        )


def check(names: list[str]) -> int:
    """Print the gene(s) each name resolves to by exact symbol/alias/fly match; return
    misses."""
    genes = annotate([{"id": k, "kind": "inferred"} for k in load()])
    miss = 0
    for n in names:
        t = n.lower()
        hits = [
            g
            for g in genes
            if t
            in {
                x.lower()
                for x in [g.get("symbol") or ""]
                + g.get("aliases", [])
                + (g.get("fly") or "").split("/")
            }
        ]
        miss += not hits
        print(
            f"{n:16s} "
            + (
                "; ".join(
                    f"{g['id']} {g.get('symbol')} (fly {g.get('fly') or '-'})"
                    for g in hits[:4]
                )
                or "NOT FOUND"
            )
        )
    return miss


if __name__ == "__main__":
    # python3 gene_names.py [data_dir]          patch an exported manifest in place
    # python3 gene_names.py --check Met gce ... resolve names (e.g. a gene list) to
    # genes
    if sys.argv[1:2] == ["--check"]:
        sys.exit(check(sys.argv[2:]))
    patch(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "pages/data")
