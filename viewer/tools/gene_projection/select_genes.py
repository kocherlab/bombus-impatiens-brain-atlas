"""Choose the reference genes to project onto the atlas.

Step 2 of the atlas gene projection (see README.md). Rule:

  * every reference gene detected (raw count > 0) in >= MIN_TYPE_FRAC (1%) of the
    nuclei of at least one of the 29 `cell_type` groups (`max_type_frac`), plus
  * every neuropeptide precursor in BIMPv4.3_neuropeptides_MALDI.tsv, plus
  * every Xenium panel gene (data/panel/np31_to_v43_mapping.csv),

restricted to genes present in the reference, in reference order. The per-type
floor (not a whole-reference one) is the point: allatostatin A (BIMP_v4.3_009248)
is rare over all nuclei but common in its top type, and the earlier HVG + 5%-overall
rule dropped it. The list is checked in as genes_atlas_v2.txt.

    PYTHONPATH=src:scripts .venv/bin/python \
        viewer/tools/gene_projection/select_genes.py [STATS_CSV] [OUT_TXT]
"""

import sys
from pathlib import Path

import pandas as pd

from bimp_st.paths import REPO, RESULTS

MIN_TYPE_FRAC = 0.01
NEUROPEPTIDES = REPO / "data/panel/BIMPv4.3_neuropeptides_MALDI.tsv"
PANEL = REPO / "data/panel/np31_to_v43_mapping.csv"
CHECK = {
    "BIMP_v4.3_009248": "allatostatin A",
    "BIMP_v4.3_000471": "allatotropin",
    "BIMP_v4.3_009257": "tachykinin",
}

stats = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else RESULTS / "gene_projection" / "ref_gene_stats.csv"
)
out = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else RESULTS / "gene_projection" / "genes_atlas_v2.txt"
)

d = pd.read_csv(stats, index_col=0)
npd = pd.read_csv(NEUROPEPTIDES, sep="\t")
neuropeptides = set(npd.gene_id.str.replace(r"-R\d+$", "", regex=True)) & set(d.index)
panel = set(pd.read_csv(PANEL).v43_id.dropna()) & set(d.index)
detected = set(d.index[d.max_type_frac >= MIN_TYPE_FRAC])
genes = [g for g in d.index if g in detected | neuropeptides | panel]

print(
    f"max_type_frac >= {MIN_TYPE_FRAC}: {len(detected)}; + neuropeptide precursors "
    f"{len(neuropeptides - detected)}; + "
    f"panel {len(panel - detected - neuropeptides)}; "
    f"total {len(genes)}"
)
for g, name in CHECK.items():
    assert g in genes, f"{name} ({g}) missing from the gene list"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(genes) + "\n")
print("written", out)
