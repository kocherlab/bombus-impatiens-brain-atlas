# Analyses downstream of the atlas

Two Snakemake workflows that read the finished atlas objects and write into
`results/`. Both are run from this directory with `-s`:

| File | What it runs |
|---|---|
| `region_relabel.smk` (wrapper: `run_region_relabel.sh`) | recomputes the per-cell region flags from the newest annotation set in `data/annotations/<brain>/`, then every analysis and figure that depends on the region set |
| `atlas_analyses.smk` | the supplementary analyses: 17-type vocabulary, regional diversity and region prediction, panel concordance, Kenyon-cell threshold sweep, snRNA KC subclusters, joint embedding, Leiden stability, antennal-lobe checks, tables S1/S2 |

```bash
SMK="uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake"
$SMK -s region_relabel.smk -n --profile ../../config/profiles/local
$SMK -s atlas_analyses.smk -n --profile ../../config/profiles/local
```

The relabel is kept out of the whole-brain pipeline on purpose: `label` is the
root of that DAG, and re-running it would rebuild the stochastic alignment.

In `region_relabel.smk`, declared inputs are the annotation GeoJSONs, the
per-section crops (`verify`, `crops`), the relabelled atlases and each step's
upstream outputs. Other objects the scripts read through `bimp_st.paths` (the
aligned per-brain atlas in `join`, and any further object an analysis or figure
script loads) are not declared, so replacing one does not schedule a rerun:
force the affected rules with `-R <rule>`.

## Region relabel

The failure modes below are easy to reintroduce; this section records them.

### What the relabel does

Four polygons on worker section `0068401` R1, classed `lobula`, are antennal
lobes and are reclassified. `v3/` (in `data/annotations/<brain>/`) is the working
set: the original `v1/` with `al` / `ellipsoid_body` / `fan_shaped_body`
appended, plus that one correction. It rebuilds byte-for-byte from the tracked
`v1/` and `v2/`, so `v2/` stays untouched as the pre-relabel record; `v4/` is v3
deduplicated with every polygon name standardised.

`verify` passes EXACT MATCH on the six original regions in both brains: the
only movement in the original six is the deliberate relabel.

The evidence, and the one section that is flagged but deliberately NOT
changed, are in `scripts/regions/relabel_worker_r1_antennal_lobe.py`.

### Coverage decision worth knowing about

Worker `0068401` R4 is recorded as **untraced** for the antennal lobe rather
than absent, so its cells leave the antennal-lobe
denominator. R3 carries eight antennal-lobe polygons and R5 four, and a
structure cannot leave the block and re-enter it.

It is one enumerated exception with its evidence in
`nine_region_common.UNTRACED_SECTIONS`, deliberately not an inferred rule: the
rule that would catch it in general assumes `Region_N` is serial anatomical
order, which is not established here. Add entries there rather than
generalising.

The ellipsoid and fan-shaped bodies on that slide are partial too (one and two
of its eight sections) but are descriptive-only, so they are disclosed
rather than masked.

### Failure modes — do not reintroduce

**Mixing two vintages of the same flag.** Taking the six original
`in_<region>` flags from an atlas object, where `labeling.py` bakes them in,
and only the newer regions from the crop sidecars is safe only while the six
are immutable, and they are not. Ask `attach_region_flags` for the whole
vocabulary you intend to use, so every flag in a row comes from one place.

**A figure that does not move when its inputs do.** A figure reading frozen
flags re-renders byte-identically after a relabel. Treat an unexpected MATCH
as a finding, exactly like an unexpected DIFFER.

**Caches keyed on data but not on code.** Two analyses cache prepared arrays
containing the region label. A data-mtime check is not enough on its own:
editing the code that fills a cache leaves every data mtime untouched. Both
caches include their own mtime and `nine_region_common.py`'s.

### Re-running

```bash
pipelines/analyses/run_region_relabel.sh
```

A full run is 14 steps, about an hour. Afterwards run
`scripts/paper/verify_manuscript_figures.py` and refresh any registered figure
that DIFFERs.
