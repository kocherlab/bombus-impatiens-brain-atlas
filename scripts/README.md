# scripts/

Analysis, figure and table scripts, one folder per topic.
Run from the repo root:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/<folder>/<script>.py
```

Folders are namespace packages: one script imports another as
`from kenyon.fig_kc_subtypes import ...` (and the registry as
`from paper.manuscript_figure_registry import ...`), so `scripts/` must be on the path
(the Snakefiles and `generate_all_figures.sh` set it). Shared helpers live in
`src/bimp_st/`; the repo root is `bimp_st.paths.REPO`.

Prefixes say what a file produces: `fig_<slug>` a main figure (copied to
`figures/main/fig_<slug>` of the manuscript repository), `si_` a supplementary figure,
`table_` an SI table, `run_` a SLURM launcher. Figures are written by slug:
`results/figures/main/<slug>/fig_<slug>.*` and `results/figures/si/figS_<slug>.*`
(analysis tables keep their older `results/` directories). The single map of
item -> label -> generator -> results path -> manuscript path is
`paper/manuscript_figure_registry.py`. Verification of the quoted numbers is not
part of this release.

Pipelines: `scripts/paper/Snakefile` rebuilds every registered figure;
`pipelines/analyses/atlas_analyses.smk` runs the supplementary atlas analyses;
`pipelines/analyses/region_relabel.smk` the region relabel and its dependents.
Scripts no registered item uses are not part of this release.

## paper/: the registry, integrity checks and build tools

- `verify_manuscript_figures.py`: every registered figure, table and Supplementary Data file against its source (any DIFFER fails); registry coverage; Supplementary Data list in first-citation order (`--manuscript DIR` or `BIMP_MANUSCRIPT` for another checkout).
- `check_data_provenance.py`: static lint for superseded objects and other data traps; `--baseline provenance_baseline.txt` compares findings with the reviewed key list.
- `manuscript_text.py`: shared helpers (printed text without comments or `\iffalse` blocks, printed numerals, PDF-date-blind md5).
- `ci.sh`: the checks CI runs (ruff, unit tests, figure registry, data provenance), runnable locally.
- `check_gene_symbol_sync.py`: viewer and `fig:caste_de` gene symbols agree.
- `check_manuscript_layout.py`: no figure or caption overruns the page.
- `check_tangram_column_naming.py`: checks that Tangram posterior columns are named by cell-type identity, not by position.
- `manuscript_figure_registry.py`: every registered figure, table and Supplementary Data item: slug, LaTeX label, generator, results path, manuscript path (used by `Snakefile`, the verifier, the sync and the export).
- `sync_manuscript_figures.py`: copies results/ into the manuscript by the registry's map (dry run by default; `--apply`).
- `build_manuscript_scratch.py`: builds the PDF in a scratch directory.
- `split_tall_si_figures.py`: cuts the lettered lower rows (A-C) of the tall Xenium QC render into `figS_xenium_qc`.
- `make_figure_map.py`: writes the figure map in the top-level README.md from the registry (`--check`: fail if stale; a unit test runs it).
- `Snakefile`: the build target (generators, each script an input, then sync dry run and checks). `generate_all_figures.sh` is a thin wrapper around it.

## atlas/: references, the spatial atlas, alignment, `fig:atlas_overview`

- `build_snrna_reference.py`: combined central-brain + optic-lobe snRNA reference for Tangram.
- `build_snrna_reference_no_optic_kc.py`: the same reference without optic-lobe Kenyon cells.
- `snrna_reference_joint_umap.py`: joint embedding of the combined reference.
- `cluster_spatial_atlas.py`: rebuilds `combined_clustered.h5ad`, keeping the Leiden resolution whose cluster count is closest to a target of 31 (r = 2.8, 32 clusters).
- `spatial_leiden_resolution.py`: finds the Leiden resolution that gives 31 spatial clusters.
- `build_region_stitched_labels.py`: builds `paths.CANONICAL_STITCHED`.
- `run_mosaicfield_alignment.sh`: MOSAICField nonlinear alignment of all queen sections.
- `mosaicfield_stitch.py`: stitches the chunked MOSAICField run into the 3D cache `si_mosaicfield.py` reads (run order in its docstring).
- `paste3_registration_accuracy.py`: accuracy of the PASTE3 3D reconstruction.
- `fig_atlas_overview.py`: **`fig:atlas_overview`**, the whole-brain atlas.
- `atlas_overview_callout_box.py`: finds the queen KC-IC callout cube hard-coded in `fig:atlas_overview`.
- `si_atlas_quality.py`: SI, does Xenium recover the snRNA reference structure.
- `si_leiden_stability.py`: SI, Leiden resolution stability under resampling (`ref|seed|aggregate|figure`).
- `si_mosaicfield.py`: SI, MOSAICField alignment.
- `si_panel_concordance.py`: SI, Xenium panel vs snRNA expression concordance.
- `si_snrna_panel_umap.py`: SI, snRNA UMAPs on the Xenium probe set only.
- `fig_xenium_celltype_overlay.py`: SI, Xenium morphology with cell assignments, channels and KC populations.
- `si_xenium_run_qc.py`: SI, Xenium run QC and negative controls.
- `snrna_xenium_joint_embedding.py`: snRNA vs Xenium on the same genes, separate and joint UMAPs.

## identity/: Tangram mapping, the cluster vote and where labels land

- `run_tangram_remap_grid.sh`, `run_tangram_remap_grid_fixed_epochs.sh`: one cell of the Tangram re-map grid (the canonical `g2_R2_s42` configuration and its seed replicates; the first also takes the epoch count, block mass and data scale as arguments, the second fixes 500 epochs).
- `run_tangram_remap_no_optic_kc.sh`: the same grid without optic-lobe Kenyon cells.
- `run_tangram_seed_replicate.sh`: an independent replicate with an explicit seed.
- `score_tangram_remap_grid.py`: scores the re-map grid against gates fixed before mapping.
- `tangram_seed_stability.py`: is the canonical mapping reproducible across seeds.
- `panel_mapping_ceiling.py`: the ceiling on what the 101-gene panel can carry.
- `noncircular_validation.py`: Tangram-independent validation battery.
- `run_loocv_tangram2_0af3056.sh`: leave-one-out gene imputation at the pinned tangram2 commit; `loocv_null.py` gives its nulls.
- `cluster_correspondence_scaling.py`: is the cluster-correspondence result an artefact of unscaled centroids.
- `prior_distortion_diagnostic.py`, `small_kc_attractor_diagnostic.py`: diagnostics the mapping pipeline points to when its identity check fails.
- `viewer_imputed_gene_rule.py`: recomputes the viewer projection's gene rule.
- `si_reference_composition.py`: SI, reference composition by library.
- `si_tangram_confidence.py`: SI, label-transfer confidence.
- `si_tangram_validation.py`: SI, mapping validation.
- `celltype_region_sankey.py`: cell type → region Sankey (shared helper).
- `celltype_anatomy_sankey.py`: do transferred cell types land in their expected anatomy.
- `cluster_vote_null.py`: is vote stability from the clusters or from aggregation.
- `cluster_vote_purity.py`: per-cluster vote decisions and purity.
- `cluster_vote_stats.py`: what the vote calls "Other", and how often it differs from the per-cell label.
- `fine_vote_rule.py`: canonical-17 cluster labelling without a giant Mixed block.
- `vote_scheme_comparison.py`: pool-then-vote vs vote-over-17 (the basis of the `fig:caste_comparison` vocabulary).
- `vote_stability_decomposition.py`: cross-seed label stability, voting vs anatomy.
- `percell_region_composition.py`: cell-type × region composition at the cell level.
- `optic_glia_vote_diagnostic.py`: why optic-neuropil cells vote Glia without anatomy.
- `label_spatial_coherence.py`: are transferred labels spatially coherent.
- `label_anatomy_placement.py`: where the canonical Kenyon and optic labels sit relative to anatomy.
- `section_raw_celltypes.py`: one section coloured by the raw Tangram call.
- `drop_generic_optic_kc_entry.py`: what changes when the generic optic-lobe Kenyon entry is dropped.
- `si_label_sensitivity.py`: SI, caste composition under different labelling schemes.
- `si_percell_vs_vote.py`: SI, per-cell argmax vs the cluster vote.
- `si_probe_clusters.py`: SI, sections coloured by probe-only Leiden clusters.

## regions/: annotation, relabel, region prediction, diversity, `fig:caste_comparison`

- `build_annotations_v3.py`, `build_annotations_v4.py`: rebuild `data/annotations/<brain>/v3` / `v4`.
- `relabel_worker_r1_antennal_lobe.py`: reclassifies worker 0068401 R1's `lobula` polygons as antennal lobe.
- `relabel_nine_regions.py`: per-cell region membership (the nine-region pipeline's `verify`/`crops`/`join`).
- `region_coverage.py`: coverage and per-region counts (`results/nine_region/coverage.json`).
- `region_prediction.py`: does local cell-type composition identify region (shared by other scripts).
- `region_prediction_3d.py`: does the 3D volume improve region prediction.
- `region_prediction_foldwise_leiden.py`: leakage control with fold-wise Leiden; `run_region_prediction_foldwise_leiden.sh` runs it.
- `si_region_prediction.py`: SI, region prediction.
- `region_diversity.py`: per-region cell-type diversity (shared with `fig:caste_comparison` and the Hill SI figure).
- `si_hill_diversity.py`: SI, Hill-number diversity profile and beta diversity.
- `si_exploded_regions.py`: SI, exploded view of the annotated regions.
- `region_structure.py`: region-structure analysis (no registered item uses it), run by the `region_structure` rule of `pipelines/analyses/atlas_analyses.smk`.
- `antennal_lobe_matched_range.py`: antennal lobe on a matched anterior-posterior range, cross-brain transfer.
- `antennal_lobe_polygon_adequacy.py`: are the worker 0069041 antennal-lobe polygons under-drawn.
- `celltype_3d_adjacency.py`: which cell types are 3D neighbours.
- `celltype_spatial_breadth.py`: how spatially spread each cell type is.
- `caste_comparison_sccoda.py`: composition analysis step (scCODA and the CSVs in `results/figures/fig2/` the figure reads; shared with `caste_de/snrna_controls.py`).
- `caste_comparison_tables.py`: composition at two levels, by caste.
- `region_composition.py`: per-region composition on the exclusive `region_of` cell set and the regional vs caste contrast (`bimp_st.composition`) into `results/region_composition/`, read by panel C.
- `fig_caste_comparison.py`: **`fig:caste_comparison`**, caste differences in composition.
- `caste_vs_section_variation.py`: caste difference relative to section-to-section variation.

## caste_de/: `fig:caste_de`, snRNA caste DE, regional DE

All snRNA caste DE uses one model, `~ trial + caste` (design by
I. Traniello), at every level. Run order (Slurm; DESeq2 steps in `.venv-deseq2`,
the rest in `.venv`, both with `PYTHONPATH=src:scripts`):

1. `pseudobulk_deseq2.py` and `fine_type_deseq2.py`: CellBender-count pseudobulks
   per class and per fine type (inputs only).
2. `caste_de_blocked_all.py` (`.venv-deseq2`; recipe `src/bimp_st/caste_de.py`,
   whole trial pairs): whole brain, pooled classes and fine types into
   `bimp_st.paths.CASTE_DE_FIT_DIR` (`results/caste_de/three_trial_pairs/`:
   `de_{coarse,fine}_blocked.csv`, `deg_counts.csv`, `focal_genes.csv`,
   `per_trial_signs.csv`); checks unchanged types against the two-trial-pair fit.
3. `parity_saved_de.py`: fine types fitted on the same samples against the saved
   analysis's `de_all.csv.gz` (`bimp_st.paths.CASTE_DE_PARITY_SAVED`).
4. `glia_composition_confound.py`, `de_without_fat_body_nuclei.py --stage all` (`.venv-deseq2`).
5. `freeze_cellbender_ambient.py` once, then `snrna_controls.py --stage pathway`
   (and `contamination`).
6. `table_glia_top10_degs.py`; `si_caste_de_volcanoes.py --stage all` (Supplementary Fig
   caste_de_volcanoes); `regional_caste_benchmark.py` then `regional_caste_benchmark_offset.py`.
7. `si_deg_concordance.py` and `BIMP_FIG_SUBSAMPLE_FRACTION=1
   neuropeptides/fig_neuropeptide_spatial.py` (figures to `results/figures/`, as
   `manuscript_figure_registry.FIGURES` records).

- `src/bimp_st/caste_de.py`: the one snRNA caste-DE recipe (`~ trial + caste`, its gene and sample filters, whole trial pairs), focal gene IDs and the per-trial sign check; every refitting script imports it.
- `caste_de_blocked_all.py`: **the snRNA caste DE** (step 2).
- `freeze_cellbender_ambient.py`: copies the CellBender ambient profiles into `bimp_st.paths.CASTE_DE_AMBIENT_DIR` with md5s.
- `parity_saved_de.py`: checks the fine types against the saved `de_all.csv.gz`.
- `caste_de_freeze_saved_outputs.py`: freezes the saved joint cNMF (k = 50) and GSEA label outputs into `bimp_st.paths.CASTE_DE_SAVED_INPUTS` with provenance (`fig_caste_de.py` reads them only with `--saved-inputs`, for comparison).
- `fig_caste_de.py`: **`fig:caste_de`**, panels a-d; panel d reads the in-repo cNMF / GSEA pipeline (`results/caste_programs/pipeline/fig3_inputs/`).
- `cnmf_gsea/`: the joint cNMF (k = 50), program caste test, label-swap null and GSEA (analysis by I. Traniello, ported from the original notebook cells): `fig:caste_de` panel d inputs and three Supplementary Data tables, with parity checks against the original code and saved outputs. Run the whole pipeline, then the figure, with one Slurm wrapper: `bash scripts/caste_de/cnmf_gsea/run_pipeline.sh all` (freeze -> inputs -> prepare -> factorize -> caste -> map -> null, go, gsea -> fig3inputs -> parity, figure); stages and sources in `cnmf_gsea/README.md`.
- `table_glia_top10_degs.py`: top 10 pooled-glia DEGs (CSV; written as Supplementary Data `glia_caste_de_top10.csv` by `deposit/supplementary_data_csvs.py`).
- `glia_composition_confound.py`: is pooled-glia DE explained by subtype composition.
- `vgn_fat_body_origin.py`: where the queen Vgn reads come from.
- `de_without_fat_body_nuclei.py`: caste DE refit without the 13 fat-body nuclei.
- `snrna_controls.py`: scCODA reference sensitivity, contamination and JH/ecdysone pathway controls.
- `si_caste_de_volcanoes.py`: Supplementary figure `fig:caste_de_volcanoes`.
- `si_deg_concordance.py`: SI, spatial vs snRNA DE concordance and its direction counts.
- `si_regional_degs.py`: SI, per-region density, caste effect sizes (no P value) and markers; `--only degs` redraws `fig:regional_volcanoes`. Composer over `regional_degs_data.py` (loading, region DE) and `regional_degs_panels.py` (drawing).
- `regional_deg_table.py`: the regional caste effect table (Supplementary Data `regional_caste_de.csv`).
- `regional_caste_benchmark.py`, `regional_caste_benchmark_offset.py`: per-region caste contrasts against within-brain variation, and their robustness to an offset.

Not a registered item: `si_deg_recurrence.py` (recurrence of the blocked-fit DEGs
across lists; its tables feed `snrna_controls.py`).

## neuropeptides/: `fig:neuropeptide_spatial`, Tk and sNPF

- `fig_neuropeptide_spatial.py`: **`fig:neuropeptide_spatial`** and `figS_ap_gradient` (shared helpers for the Tk scripts). Composer over `neuropeptide_spatial_data.py` and `neuropeptide_spatial_panels.py`.
- `si_neuropeptide_landscape.py`: SI, neuropeptide caste-bias landscape.
- `si_tk_multimodal.py`: SI, Tk in microCT, Xenium and MALDI.
- `tk_3d_spillover.py`: Tk optic-lobe spillover in 3D; `si_tk_3d_spillover.py` draws it with `tk_3d_render.py`.
- `tk_shift_battery.py`: the Tk compartment-shift battery on the nine-region flags.
- `tk_detection_sensitivity.py`: panel-wide specificity and detection controls for the Tk shift.
- `tk_optic_carryover.py`: is the queen optic-lobe Tk signal Kenyon-cell carryover.

## kenyon/: `fig:kc_subtypes` and Kenyon-cell projections

- `fig_kc_subtypes.py`: **`fig:kc_subtypes`**. Composer over `kc_subtypes_data.py` (subzones, lobes, snRNA tables) and `kc_subtypes_panels.py` (drawing).
- `snrna_kc_subclusters.py`: `fig:kc_subtypes` A / `figS_snuc_kc_subclusters`.
- `large_type_kc_subclusters.py`: subclusters the reference's Large-type Kenyon cells.
- `project_large_type_subclusters.py`: projects them onto the Kenyon compartments (`run_project_large_type_subclusters.sh`).
- `project_kc_populations.py`: five-population restricted projection, `fig:kc_subtypes` C (`run_project_kc_populations.sh`).
- `project_kc_populations_class_gated.py`: the same gated by predicted class (`run_project_kc_populations_class_gated.sh`).
- `project_kc_populations_optic_control.py`: optic-neuropil negative control (`run_project_kc_populations_optic_control.sh`); `kc_populations_optic_control_stats.py` compares effect sizes.
- `project_kc_populations_with_optic.py`: five-population reclassification with the optic neuropils in the query.
- `large_type_projection_anatomy_test.py`: anatomy test on the restored projections.
- `large_type_subcluster_coda.py`: CoDA of Large-type subcluster composition by caste.
- `restricted_projection_numbers.py`: restricted-projection numbers for the SI null panel.
- `kc_subcluster_separability.py`: are the three Large-type subclusters distinct.
- `kc_compartment_ceiling.py`: the compartment-prediction ceiling for `fig:kc_subtypes`.
- `other_kenyon_cells.py`: is "Other Kenyon cells" a population or a bin.
- `forced_simplex_controls.py`: controls on the forced three-subcluster projection.
- `snpf_middle_type_control.py`, `snpf_spillover_controls.py`: is the queen sNPF zone excess Middle-type cells or spillover.
- `table_mb_ol_kc_labels.py`: SI figure and table of Kenyon labels in mushroom body and optic neuropils.
- `si_kc_threshold.py`: what counts as a Kenyon cell (Supp. Methods numbers; no figure ships).
- `si_kc_projection_null.py`: SI, restricted projection vs null.
- `si_kc_restricted_confidence.py`: SI, confidence of the two restricted runs.

## subcellular/: `fig:subcellular` and `fig:focal_genes`

- `si_subcellular_summary.py`: **`figS_subcellular_regions`**, localization across regions from the bento and ELLA tables; run order of `bento_regroup.py`, `bento_regions.py`, `ella_section.py`, the controls and the summaries in `README_ella_bento.md` (`regroup_common.py` is their shared code).
- `si_bento_pergene.py`: per-gene localization statistics (imported by `si_subcellular_summary.py`); `bento_pergene_stats.py` builds their inputs.
- `si_focal_genes.py`: SI, focal-gene spatial expression (Vg-like, yl, Kr-h1).

## gene_annotation/: Tables S1/S2, orthologs, genome

- `panel_orthologs.py`, `panel_ortholog_sources.py`, `panel_ortholog_alignments.py`: Drosophila orthologs of the panel genes from three sources, and their alignments.
- `tables_s1_s2.py`: Tables S1 (panel key) and S2 (snRNA annotations).
- `genome_dmel_orthologs.py`: genome-wide reciprocal-best-hit orthologs.
- `genome_assembly_stats.py`: BIMP_v4 assembly statistics (scaffolds, contigs, joins); `run_genome_busco.sh` adds a genome-mode BUSCO.
- `longest_isoform_proteins.py`, `run_annotation_busco.sh`: BUSCO 6.0.0 of the v4.3 annotation (Methods); env `config/envs/busco.yml`.
- `isoseq_barcode_species.py`: which Iso-Seq barcodes are *B. impatiens* (k-mer check).
- `Snakefile`: the genome rules (assembly and EGAPx provenance: Methods, "Genome assembly" and "Genome annotation").
- `snrna_markers_top500.py`: top-500 markers per snRNA cell type.
- `spatial_markers_top500.py`: top-500 genes per anatomical region of the Xenium atlas, measured vs inferred.

## deposit/: public data release

- `build_staging.py`: builds the deposit staging tree.
- `export_public_objects.py`: writes release-ready atlas objects into it.
- `sd_cellranger_qc.py`: the snRNA library QC table it includes.
- `sd_egapx_inputs.py`: the EGAPx short-read and Iso-Seq input tables, from the run's `run_params.yaml` (`data/genome/egapx_run_params.yaml`).
- `supplementary_data_csvs.py`: every Supplementary Data CSV (`<slug>.csv`, in the first-citation order of `manuscript_figure_registry.SUPPLEMENTARY_DATA_ITEMS`; LaTeX numbers them) plus `SUPPLEMENTARY_DATA.md`, into `results/supplementary_data/`; `sync_manuscript_figures.py --apply` copies them to the manuscript (byte-checked by `verify_manuscript_figures.py`).
- `export_numbered_si.py`: journal-ready numbered copies (`Figure_N`, `Supplementary_Figure_SN`, `Supplementary_Table_SN`, `Supplementary_Data_N_<slug>.csv`) and an index, from a built manuscript's `.aux`, into the gitignored `export/`.
