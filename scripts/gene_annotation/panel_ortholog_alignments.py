"""Smith-Waterman alignments (BLOSUM62, gap 11/1) of the B. impatiens precursors /
short proteins that the BLAST-family search cannot place, against their candidate
D. melanogaster genes, with a shuffled-sequence null (500 shuffles of the Dmel sequence,
fixed seed). Used as the alignment half of the annotation + alignment line for Kr-h1,
sNPF, NPF and Crz (Table S1 evidence).

Needs Biopython (not in the project .venv): run with it on PYTHONPATH, e.g.
  PYTHONPATH=src:/path/to/biopython .venv/bin/python \\
      scripts/gene_annotation/panel_ortholog_alignments.py
Writes results/handoff/E_orthologs/alignments_named.txt only.
"""

from __future__ import annotations

import random
from pathlib import Path

from Bio import Align
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "handoff" / "E_orthologs"
C = R / "cache"
# (panel symbol, B. impatiens GeneID, Dmel FBgn, Dmel symbol, motif note)
PAIRS = [
    ("Kr-h1", "100740522", "FBgn0266450", "Kr-h1", "C2H2 zinc-finger array"),
    (
        "sNPF",
        "100748800",
        "FBgn0032840",
        "sNPF",
        "sNPF mature peptides (RSPSLRLRFamide)",
    ),
    (
        "NPF",
        "100744000",
        "FBgn0027109",
        "NPF",
        "NPF mature peptide (C-terminal RxRFamide)",
    ),
    (
        "Crz",
        "100746471",
        "FBgn0013767",
        "Crz",
        "corazonin mature peptide pQTFQYSRGWTNamide",
    ),
]


def seqs(path: Path) -> dict[str, str]:
    out, cur = {}, None
    for line in open(path):
        if line[0] == ">":
            cur = line[1:].strip()
            out[cur] = ""
        elif cur:
            out[cur] += line.strip()
    return out


def main() -> None:
    bimp, dmel = seqs(C / "bimp.fa"), seqs(C / "dmel.fa")
    dmel.update(seqs(R / "cache_ext" / "extra_dmel.fa"))
    al = Align.PairwiseAligner(
        mode="local",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-11,
        extend_gap_score=-1,
    )
    rng = random.Random(1)
    lines = []
    for sym, gid, fb, dsym, motif in PAIRS:
        b = max((v for k, v in bimp.items() if k.endswith("|" + gid)), key=len)
        d = max((v for k, v in dmel.items() if f"|{fb}|" in k), key=len)
        a = al.align(b, d)[0]
        null = []
        for _ in range(500):
            sh = list(d)
            rng.shuffle(sh)
            null.append(al.score(b, "".join(sh)))
        n_ge = sum(x >= a.score for x in null)
        ident = a.counts().identities
        q0, q1 = a.aligned[0][0][0], a.aligned[0][-1][1]
        lines.append(
            f"## {sym} (GeneID {gid}, {len(b)} aa) vs {dsym} ({fb}, {len(d)} aa); "
            f"{motif}\n"
            f"score {a.score:.0f}; shuffled-Dmel null max {max(null):.0f}, mean "
            f"{sum(null) / len(null):.0f}; "
            f"{n_ge}/500 shuffles >= observed; identities {ident}/{a.shape[1]} "
            "aligned columns; "
            f"B. impatiens residues {q0 + 1}-{q1}\n{a}\n"
        )
    (R / "alignments_named.txt").write_text("\n".join(lines))
    print("\n".join(ln.split("\n")[0] + "\n" + ln.split("\n")[1] for ln in lines))


if __name__ == "__main__":
    main()
