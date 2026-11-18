#!/usr/bin/env python
"""Assembly statistics for the BIMP_v4 genome, recomputed from the FASTA.

Computes total length, scaffold count, scaffold N50, longest scaffold, and the
values shown with the snail plot (fig:genome_snail: N90, auN, GC/AT/N
composition). The snail plot itself
is a static blobtoolkit image, so this script recomputes every assembly
number from the sequence the gene models were built on.

Input
  The BIMP_v4 assembly, the FASTA the deposit staging links
  (deposit_staging/ncbi/assembly/BIMP_genome_v4.fasta):
  <genome_dir>/genome/BIMP_genome_v4.fasta (site key ``genome_dir``,
  :mod:`bimp_st.site`).
  Override with --fasta.
  Optionally, a BUSCO short_summary (.json, or the .txt one-liner) with
  --busco; its complete / single / duplicated / fragmented / missing counts,
  their percentages of the lineage size, and the lineage size are copied into
  the summary. scripts/gene_annotation/run_genome_busco.sh runs BUSCO and then this.

Definitions
  Every FASTA record counts as one scaffold (the mitochondrion included,
  as in the snail plot's "total 19"). N50/N90: length L such that scaffolds
  of length >= L hold >= 50%/90% of the total. auN = sum(L^2) / sum(L).
  GC and AT are percentages of the A/C/G/T bases (case-insensitive, soft
  masking ignored); N is the percentage of all bases that are N.
  Contigs are maximal non-N runs and gaps (scaffolding joins) maximal N runs;
  the chrom_* keys restrict to the BIMP_chr_<n> records (Methods: 18
  chromosome-length scaffolds, 8.6-24.7 Mb, 26 contigs, 8 joins, 100 bp gaps).
  The FASTA is md5-identical (a7ff89e8d7d97027d5038a5c2ac98dd6) to the
  decompressed lab release GenomeReleases/BIMP_v4.3/BIMP_genome_v4.fasta.gz.

Output
  results/handoff/genome_stats/summary.json (flat name -> value) and
  scaffolds.tsv (name, length, GC count, AT count, N count).

Run
    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/gene_annotation/genome_assembly_stats.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from bimp_st.site import lookup as site_lookup

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "handoff" / "genome_stats"
FASTA = site_lookup("genome_dir")
FASTA = Path(FASTA) / "genome/BIMP_genome_v4.fasta" if FASTA else None


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


def nx(lengths, frac):
    total, run = sum(lengths), 0
    for L in sorted(lengths, reverse=True):
        run += L
        if run >= frac * total:
            return L
    raise ValueError("empty assembly")


def busco_summary(path):
    """Parse a BUSCO short_summary (.json from v5.x, or the .txt one-liner)."""
    p = Path(path)
    if p.suffix == ".json":
        # The JSON's "percentage" fields are rounded to 0.1; the counts give
        # the two-decimal values the snail plot prints.
        d = json.loads(p.read_text())
        r = d["results"]
        n = int(r["n_markers"])
        out = {
            "busco_lineage": d["lineage_dataset"]["name"],
            "busco_n": n,
            "busco_version": d.get("versions", {}).get("busco"),
        }
        for key, field in (
            ("complete", "Complete BUSCOs"),
            ("single", "Single copy BUSCOs"),
            ("dup", "Multi copy BUSCOs"),
            ("frag", "Fragmented BUSCOs"),
            ("missing", "Missing BUSCOs"),
        ):
            out[f"busco_{key}_n"] = int(r[field])
            out[f"busco_{key}_pct"] = 100 * int(r[field]) / n
        return out
    t = p.read_text()
    m = re.search(
        r"C:([\d.]+)%\[S:([\d.]+)%,D:([\d.]+)%\],F:([\d.]+)%,M:([\d.]+)%,n:(\d+)", t
    )
    lin = re.search(r"lineage dataset is: (\S+)", t)
    return {
        "busco_lineage": lin.group(1) if lin else None,
        "busco_complete_pct": float(m.group(1)),
        "busco_single_pct": float(m.group(2)),
        "busco_dup_pct": float(m.group(3)),
        "busco_frag_pct": float(m.group(4)),
        "busco_missing_pct": float(m.group(5)),
        "busco_n": int(m.group(6)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=FASTA, required=FASTA is None)
    ap.add_argument("--busco", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, gaps = [], []
    for name, seq in read_fasta(args.fasta):
        s = seq.upper()
        gc = s.count("G") + s.count("C")
        at = s.count("A") + s.count("T")
        n = s.count("N")
        g = [len(m.group()) for m in re.finditer(r"N+", s)]
        gaps += g
        contigs = sum(1 for m in re.finditer(r"[^N]+", s))
        rows.append((name, len(s), gc, at, n, contigs, len(g)))
    with open(OUT / "scaffolds.tsv", "w") as fh:
        fh.write("name\tlength\tgc\tat\tn\tcontigs\tgaps\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    # Chromosome-length scaffolds (their length range, contigs, joins and gap
    # size, as the Methods give them): the BIMP_chr_<n> records, contigs =
    # maximal non-N runs, joins = N runs.
    chrom = [r for r in rows if re.fullmatch(r"BIMP_chr_\d+", r[0])]
    L = [r[1] for r in rows]
    total = sum(L)
    gc, at, n = (sum(r[i] for r in rows) for i in (2, 3, 4))
    S = {
        "fasta": str(args.fasta),
        "fasta_md5": hashlib.md5(Path(args.fasta).read_bytes()).hexdigest(),
        "n_scaffolds": len(L),
        "total_bp": total,
        "total_mb": total / 1e6,
        "n50_mb": nx(L, 0.5) / 1e6,
        "n90_mb": nx(L, 0.9) / 1e6,
        "aun_mb": sum(x * x for x in L) / total / 1e6,
        "longest_mb": max(L) / 1e6,
        "longest_name": rows[L.index(max(L))][0],
        "gc_pct": 100 * gc / (gc + at),
        "at_pct": 100 * at / (gc + at),
        "n_pct": 100 * n / total,
        "n_bases_other": total - gc - at - n,
        "n_contigs": sum(r[5] for r in rows),
        "n_gaps": len(gaps),
        "gap_lengths": sorted(set(gaps)),
        "n_chrom_scaffolds": len(chrom),
        "chrom_min_mb": min(r[1] for r in chrom) / 1e6,
        "chrom_max_mb": max(r[1] for r in chrom) / 1e6,
        "chrom_contigs": sum(r[5] for r in chrom),
        "chrom_gaps": sum(r[6] for r in chrom),
    }
    if args.busco:
        S["busco_source"] = str(args.busco)
        S.update(busco_summary(args.busco))
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
