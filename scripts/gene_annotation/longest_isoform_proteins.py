#!/usr/bin/env python
"""Longest protein isoform per gene from the released BIMP v4.3 protein set.

The input to the annotation BUSCO (Methods, "Genome annotation": BUSCO v6.0.0,
hymenoptera_odb12). Every record of BIMP_OGS_v4.3_pep.fa is an EGAPx
(Gnomon) model named <gene>-R<n>; the MitoHiFi genes carry no protein here.
Per gene the longest sequence is kept (ties: the first in file order).

Input   <genome_release>/BIMP_OGS_v4.3_pep.fa
        (release of 2025-07-18; md5 27e9c5029e8f42051d1e9429326f5730)
Output  <out>/BIMP_OGS_v4.3_pep.longest.fa and <out>/longest_isoform.json
        (input md5, records read, genes kept, output md5)

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/gene_annotation/longest_isoform_proteins.py \\
        --out results/genome/annotation_busco
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from bimp_st.site import get as site_path

PEP = site_path("genome_release") / "BIMP_OGS_v4.3_pep.fa"
PEP_MD5 = "27e9c5029e8f42051d1e9429326f5730"
ISOFORM = re.compile(r"^(?P<gene>.+)-R\d+$")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_fasta(path):
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pep", type=Path, default=PEP)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    in_md5 = md5(args.pep)
    if args.pep == PEP and in_md5 != PEP_MD5:
        raise SystemExit(f"{args.pep}: md5 {in_md5} != released {PEP_MD5}")

    best: dict[str, tuple[str, str]] = {}
    n = 0
    for name, seq in read_fasta(args.pep):
        n += 1
        m = ISOFORM.match(name)
        if not m:
            raise SystemExit(f"unexpected protein id {name!r}")
        gene = m["gene"]
        if gene not in best or len(seq) > len(best[gene][1]):
            best[gene] = (name, seq)

    args.out.mkdir(parents=True, exist_ok=True)
    fa = args.out / "BIMP_OGS_v4.3_pep.longest.fa"
    with open(fa, "w") as fh:
        for _, (name, seq) in best.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")
    meta = {
        "input": str(args.pep),
        "input_md5": in_md5,
        "n_proteins": n,
        "n_genes": len(best),
        "output": str(fa),
        "output_md5": md5(fa),
    }
    (args.out / "longest_isoform.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
