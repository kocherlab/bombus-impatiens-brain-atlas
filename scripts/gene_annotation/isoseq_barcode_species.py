"""Which Iso-Seq barcodes of run r84046_20250214_212632 are B. impatiens.

Share of FLNC reads per barcode sharing >= 5 exact 25-mers (either strand) with a
transcriptome: the first 2,000 records of each SMRT Link flnc-<i>.bam (BGZF,
parsed directly; no samtools needed). Run once per transcriptome:

    PYTHONPATH=src .venv/bin/python \\
        scripts/gene_annotation/isoseq_barcode_species.py [TRANSCRIPTS.fa]

Default BIMP v4.3; the Megalopta genalis control is
GenomeReleases/official_release_v3.3/MGEN/MGEN_OGS_v3.3_trans.fa. Result:
barcodes 1-2 are B. impatiens, 3-7 M. genalis. 24 GB, ~20 min
on one core: run through Slurm.
"""

import gzip
import struct
import sys

from bimp_st.site import get as site_path  # noqa: E402

D = str(site_path("isoseq_raw") / "Isoseq_analysis_13887")
BIMP = str(site_path("genome_release") / "BIMP_OGS_v4.3_trans.fa")
TRANS = sys.argv[1] if len(sys.argv) > 1 else BIMP
N, K = 2000, 25
NT = "=ACMGRSVTWYHKDBN"
RC = str.maketrans("ACGT", "TGCA")


def reads(path, n):
    f = gzip.open(path)
    assert f.read(4) == b"BAM\1"
    lt = struct.unpack("<i", f.read(4))[0]
    f.read(lt)
    nref = struct.unpack("<i", f.read(4))[0]
    for _ in range(nref):
        ln = struct.unpack("<i", f.read(4))[0]
        f.read(ln + 4)
    out = []
    while len(out) < n:
        b = f.read(4)
        if len(b) < 4:
            break
        bs = struct.unpack("<i", b)[0]
        rec = f.read(bs)
        l_rn = rec[8]
        n_cig = struct.unpack("<H", rec[12:14])[0]
        l_seq = struct.unpack("<i", rec[16:20])[0]
        off = 32 + l_rn + 4 * n_cig
        packed = rec[off : off + (l_seq + 1) // 2]
        s = "".join(NT[x >> 4] + NT[x & 15] for x in packed)[:l_seq]
        out.append(s)
    return out


samples = {}
for i in range(1, 8):
    samples[i] = reads(f"{D}/flnc-{i}.bam", N)
    print(i, len(samples[i]), "reads", file=sys.stderr)

# k-mers of the reads (both strands) -> set of (sample, read) owners
owner = {}
for i, rs in samples.items():
    for j, s in enumerate(rs):
        for t in (s, s.translate(RC)[::-1]):
            for p in range(0, len(t) - K + 1, 5):
                owner.setdefault(t[p : p + K], set()).add((i, j))
print("kmers", len(owner), file=sys.stderr)

hits = {}
seq = []


def scan(s):
    for p in range(len(s) - K + 1):
        o = owner.get(s[p : p + K])
        if o:
            for key in o:
                hits[key] = hits.get(key, 0) + 1


with open(TRANS) as fh:
    for line in fh:
        if line.startswith(">"):
            if seq:
                scan("".join(seq).upper())
            seq = []
        else:
            seq.append(line.strip())
    if seq:
        scan("".join(seq).upper())

print("barcode\tn_reads\tfrac_reads_ge5_kmer_hits")
for i, rs in samples.items():
    ok = sum(1 for j in range(len(rs)) if hits.get((i, j), 0) >= 5)
    print(f"{i}\t{len(rs)}\t{ok / len(rs):.3f}")
