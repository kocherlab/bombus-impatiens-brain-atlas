# `bimp_v43_gene_names.csv` — bee names, fly orthologs, search aliases

One row per BIMP_v4.3 gene (11,046: every gene in the OGS v4.3 lookup and GTF). The web
atlas reads it through `viewer/gene_names.py`, so every gene is findable by its bee
name, its *D. melanogaster* ortholog symbol and synonyms, and common names (mblk-1,
Met/gce, ...). Built 2026-10-18 by `viewer/tools/gene_orthologs.py`; that file's
docstring has the method and the rebuild commands.

## Columns

| column | meaning |
|---|---|
| `bimp_id` | BIMP_v4.3 gene id |
| `bee_symbol`, `bee_desc`, `biotype` | OGS v4.3 lookup (`BIMP_lookup.csv`); `bee_desc` falls back to the GTF `product` |
| `symbol`, `desc` | curated overrides (`CURATED` in the builder); empty = use the bee annotation |
| `dmel_symbol`, `dmel_fbgn`, `dmel_name` | forward best fly gene (UniProt primary symbol, FlyBase id, protein name) |
| `dmel_coorthologs` | further fly genes whose reverse best hit is this bee gene (one-to-many) |
| `method` | `rbh`, `rbh_one_to_many`, `best_hit_not_reciprocal` (not an ortholog call), `none` |
| `confidence` | `high` (≥40 % id over ≥50 % of the bee protein), `moderate` (≥30 % coverage), `domain only` (<30 %), `low` (non-reciprocal) |
| `pident`, `qcov`, `evalue`, `bitscore` | best isoform pair for the forward best hit |
| `note` | near ties (second fly gene ≥0.9 of the top bitscore) and where a non-reciprocal hit's fly gene points instead |
| `common_names` | curated search names, `|`-separated |
| `aliases` | everything searchable, `|`-separated |

Protein-coding yield: 5,613 `rbh` + 1,034 `rbh_one_to_many`, 1,285 non-reciprocal, 2,355 none.

## Inputs (not in the repo)

- **Bee proteins**: CDS from `BIMP_OGS_v4.3.gtf` + `BIMP_genome_v4.fasta`, translated with
  gffread 0.12.9 (32,374 transcripts; internal stops → X). Source: the snRNA-seq
  cellranger reference, `<snrna_source>/src/reference_genome/`
  (`genome/` and `tables/BIMP_lookup.csv`).
- **Fly proteins**: UniProt proteome UP000000803, the `dmel.fa` cached by
  `scripts/gene_annotation/panel_orthologs.py` (`results/handoff/E_orthologs/cache/`).
  Names joined by UniProt accession from a UniProt TSV
  (`accession,gene_primary,gene_synonym,protein_name,xref_flybase,reviewed`) fetched 2026-10-18.
- DIAMOND 2.2.8, `blastp --sensitive -e 1e-5 -k 25`, both directions; calls need e ≤ 1e-10.

## Curated calls worth knowing

- **BIMP_v4.3_002131 is Met/gce (the JH receptor), not CLOCK.** NCBI (and so the panel map
  and the old atlas label) calls it "circadian locomoter output cycles protein kaput".
  It is the reciprocal best hit of fly *gce* (268 bits, 41 % id) and the best bee hit of
  fly *Met* (207 bits); its best Clk hit is 110 bits. Fly *Clk*'s reciprocal partner is
  **BIMP_v4.3_002563** (488 bits, 43 % id, 79 % coverage). The bee has the single
  pre-duplication Met/gce gene.
- **BIMP_v4.3_008845 is Mblk-1 / E93** (fly Eip93F): reciprocal, but over the 69-aa
  Pipsqueak domain only (91 % id), hence `domain only`.
- **Vitellogenin**: 005500 (1,772 aa) is full-length Vg; the panel's "vitellogenin",
  001050, and its neighbour 001051 are ~318-aa ORFs.
- No confident hit for the short peptide precursors AstC, Mip/Ast-B, Tk or allatotropin
  (Drosophila has no allatotropin); they are found by bee name and curated aliases.
