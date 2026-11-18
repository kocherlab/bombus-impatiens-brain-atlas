"""Second and third independent orthology sources for the 101 panel genes (feeds
Table S1).

Fetches, per B. impatiens panel gene (NCBI GeneID = LOC number):
  * Ensembl Metazoa Compara (rest.ensembl.org,
    bombus_impatiens -> drosophila_melanogaster)
  * OrthoDB v11 (data.orthodb.org): every Endopterygota / Insecta orthologous group that
    contains the gene, with its D. melanogaster members.
Responses are cached verbatim under results/handoff/E_orthologs/cache_ext/ (never
overwritten); a failed request is retried and, if it still fails, recorded as ERR so a
gap is visible rather than silently an empty ortholog list.

Run: PYTHONPATH=src .venv/bin/python scripts/gene_annotation/panel_ortholog_sources.py
Writes only under results/handoff/E_orthologs/.
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "handoff" / "E_orthologs"
X = R / "cache_ext"
LEVELS = (33392, 50557)  # Endopterygota, Insecta (Dmel and Bombus both inside)


def get(u: str, hdr: dict | None = None, tries: int = 4) -> str:
    err = None
    for i in range(tries):
        # curl, not urllib: the venv's Python has no usable CA bundle here
        cmd = ["curl", "-s", "-q", "-f", "-m", "90", u]
        for k, v in (hdr or {}).items():
            cmd += ["-H", f"{k}: {v}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        err = f"curl exit {r.returncode}"
        time.sleep(4 * (i + 1))
    return "ERR " + str(err)


def fetch_all(locs: list[str]) -> tuple[dict, dict]:
    X.mkdir(parents=True, exist_ok=True)
    ef, of = X / "ensembl.json", X / "orthodb.json"
    ens = json.loads(ef.read_text()) if ef.exists() else {}
    odb = json.loads(of.read_text()) if of.exists() else {}
    from concurrent.futures import ThreadPoolExecutor

    def one(loc):
        gid = loc.replace("LOC", "")
        if loc not in ens or ens[loc].startswith("ERR"):
            ens[loc] = get(
                f"https://rest.ensembl.org/homology/id/bombus_impatiens/{loc}"
                "?target_species=drosophila_melanogaster;type=orthologues;sequence=none",
                {"Content-Type": "application/json"},
            )
        if loc not in odb or odb[loc].get("err"):
            rec: dict = {}
            for lvl in LEVELS:
                s = get(
                    "https://data.orthodb.org/current/search?"
                    f"query={gid}&level={lvl}&species=132113&take=50"
                )
                try:
                    ogs = json.loads(s).get("data", [])
                except Exception:  # noqa: BLE001
                    ogs, rec["err"] = [], s[:100]
                for og in ogs:
                    t = get(
                        f"https://data.orthodb.org/current/tab?id={og}&species=7227,132113"
                    )
                    if t.startswith("ERR"):
                        rec["err"] = t[:100]
                        continue
                    rows = [x.split("\t") for x in t.strip().split("\n")[1:]]
                    # keep only groups that really contain this gene (search is a
                    # text query)
                    if any(r[3].startswith("132113") and r[5] == gid for r in rows):
                        rec.setdefault(str(lvl), []).append({"og": og, "rows": rows})
            odb[loc] = rec

    todo = list(locs)
    with ThreadPoolExecutor(3) as ex:
        for i in range(0, len(todo), 16):  # checkpoint every batch
            list(ex.map(one, todo[i : i + 16]))
            ef.write_text(json.dumps(ens))
            of.write_text(json.dumps(odb))
            print(f"{min(i + 16, len(todo))}/{len(todo)}", flush=True)
    return ens, odb


def name_map() -> dict[str, str]:
    """Dmel gene token (primary symbol, CG number, synonym) -> FBgn, from the UniProt
    proteome. Primary symbols take precedence over synonyms; a synonym shared by two
    genes is dropped."""
    t = pd.read_csv(X / "dmel_names.tsv", sep="\t", dtype=str, keep_default_na=False)
    prim: dict[str, str] = {}
    syn: dict[str, set] = {}
    for r in t.itertuples(index=False):
        fb = r[4].strip(";").split(";")[0]
        if not fb.startswith("FBgn"):
            continue
        for tok in [r[1]] + [x.strip() for x in re.split(r"[;,]", r[2])]:
            if tok:
                prim.setdefault(tok, fb)
        # synonyms are whitespace-separated, and CG numbers appear as 'Dmel\CG5446'
        for tok in re.split(r"[\s;,]+", r[3]):
            tok = tok.replace("Dmel\\", "")
            if tok:
                syn.setdefault(tok, set()).add(fb)
    m = {k: next(iter(v)) for k, v in syn.items() if len(v) == 1}
    m.update(prim)
    return m


def ensembl_sets(ens: dict[str, str]) -> dict[str, list[dict] | None]:
    """LOC -> [{fb, type, pid}]; a failed request -> None (a gap, not 'no ortholog')."""
    out: dict = {}
    for loc, txt in ens.items():
        if txt.startswith("ERR"):
            out[loc] = None
            continue
        rows = []
        data = json.loads(txt)["data"]
        if (
            not data
        ):  # gene absent from the Compara release: no tree, so no evidence either way
            out[loc] = None
            continue
        for h in data[0]["homologies"]:
            rows.append(
                dict(
                    fb=h["target"]["id"],
                    type=h["type"],
                    pid=h["target"].get("perc_id"),
                    qpid=h["source"].get("perc_id"),
                    level=h.get("taxonomy_level"),
                )
            )
        out[loc] = rows
    return out


def ensembl_symbol(tok: str) -> str:
    """Dmel symbol / CG number -> FBgn via Ensembl lookup (cached); '' if it does not
    resolve."""
    cf = X / "token_fb.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    if tok not in cache:
        t = get(
            "https://rest.ensembl.org/lookup/symbol/drosophila_melanogaster/"
            + tok.replace(" ", "%20"),
            {"Content-Type": "application/json"},
            tries=2,
        )
        try:
            j = json.loads(t)
            cache[tok] = j["id"] if str(j.get("id", "")).startswith("FBgn") else ""
        except (ValueError, KeyError):
            cache[tok] = ""
        cf.write_text(json.dumps(cache))
    return cache[tok]


def orthodb_sets(odb: dict, nm: dict[str, str]) -> dict:
    """LOC -> {'groups': [{og, level, fb:set(FBgn), unmapped:[tokens], n_bimp, name}]}
    for every Endopterygota/Insecta group containing the gene; None on a failed
    request."""
    out: dict = {}
    for loc, rec in odb.items():
        if rec.get("err"):
            out[loc] = None
            continue
        res = {"groups": []}
        for lvl in ("33392", "50557"):
            for g in rec.get(lvl, []):
                fb, un, nb = set(), [], 0
                for r in g["rows"]:
                    if r[3].startswith("132113"):
                        nb += 1
                    elif (
                        r[3] == "7227_0"
                    ):  # 7227_1 is a second, redundant Dmel annotation set
                        tok = r[5][len("Dmel_") :] if r[5].startswith("Dmel_") else r[5]
                        hit = nm.get(tok) or ensembl_symbol(tok)
                        if hit:
                            fb.add(hit)
                        else:
                            un.append(r[5])
                res["groups"].append(
                    dict(
                        og=g["og"],
                        level=lvl,
                        fb=fb,
                        unmapped=un,
                        n_bimp=nb,
                        name=g["rows"][0][1],
                    )
                )
        out[loc] = res
    return out


def apis_sets(locs: list[str]) -> dict[str, dict]:
    """NCBI Gene orthologs routed through Apis mellifera (tax 7460):
    B. impatiens GeneID -> Apis GeneID(s) -> D. melanogaster GeneID(s)."""
    want = {x.replace("LOC", "") for x in locs}
    b2a: dict[str, set] = {}
    a2d: dict[str, set] = {}
    with gzip.open(X / "gene_orthologs.gz", "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5 or f[2] != "Ortholog":
                continue
            t1, g1, t2, g2 = f[0], f[1], f[3], f[4]
            if {t1, t2} == {"132113", "7460"}:
                b, a = (g1, g2) if t1 == "132113" else (g2, g1)
                if b in want:
                    b2a.setdefault(b, set()).add(a)
            elif {t1, t2} == {"7227", "7460"}:
                d_, a = (g1, g2) if t1 == "7227" else (g2, g1)
                a2d.setdefault(a, set()).add(d_)
    out = {}
    for x in locs:
        g = x.replace("LOC", "")
        ap = b2a.get(g, set())
        out[x] = dict(
            apis=sorted(ap),
            dmel_gid=sorted({d_ for a in ap for d_ in a2d.get(a, set())}),
        )
    return out


if __name__ == "__main__":
    d = pd.read_csv(R / "panel_dmel_orthologs.csv", dtype=str)
    e, o = fetch_all(list(d["LOC ID"]))
    print(
        sum(v.startswith("ERR") for v in e.values()),
        "ensembl errors;",
        sum(bool(v.get("err")) for v in o.values()),
        "orthodb errors",
    )
