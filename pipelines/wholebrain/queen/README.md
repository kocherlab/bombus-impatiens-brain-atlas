# Whole-Brain Pipeline (Brain 1 — Queen)

Reconstructs a 3D transcriptomic atlas from serial Xenium sections of *Bombus impatiens* brain.

## Pipeline

```
Raw Xenium → h5ad crops → anatomical labeling (GeoJSON) → PASTE3 alignment → Leiden clustering
```

The Snakemake rules and the analysis code are shared with the other brain and
live in [`../`](../README.md); this directory holds the
brain's data, `config.yaml` (its settings) and `labeling.py` (its
ANNOTATION_MAPPING).

```
label → align → cluster ─┬→ visualize_3d
                          └→ render_video
```

## Scripts here

| File | Purpose | CLI |
|------|---------|-----|
| `convert_single_region.py` | Raw Xenium to h5ad crops (this brain's crop boxes; not a pipeline rule) | `python convert_single_region.py <region>` |
| `compositional_analysis.py` | Aitchison CoDA helpers (`clr`), imported by `scripts/identity/noncircular_validation.py` | Typer: `analyze`, `validate`, `compare-individuals` |
| `run_mosaicfield.py` | Chunked MOSAICField alignment behind the SI figure (`scripts/atlas/run_mosaicfield_alignment.sh`) | Typer |
| `labeling.py` | ANNOTATION_MAPPING: which annotation GeoJSON covers which crop (data only) | read by `../labeling_core.py` and `bimp_st.nine_region_common` |

## Usage

```bash
# Full pipeline via Snakemake, from this directory
uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake --profile ../../../config/profiles/slurm

# Or step by step, from this directory
PYTHONPATH=../../../src ../../../.venv/bin/python ../analysis_workflow.py label-all
PYTHONPATH=../../../src ../../../.venv/bin/python ../analysis_workflow.py align
PYTHONPATH=../../../src ../../../.venv/bin/python ../analysis_workflow.py cluster
```

## Output

| File | Description |
|------|-------------|
| `data/aligned.h5ad` | 3D-aligned cells with x, y, z coordinates |
| `data/aligned_clustered.h5ad` | + Leiden clusters |
| `figures/brain_3d.html` | Interactive 3D visualization |
| `figures/brain_rotation.mp4` | Rotating 3D video |

The canonical Tangram mapping is the combined run (`../../combined_atlas/`).

## Data

- 2 slides (0040240, 0040243), 8 imaging regions each
- 2 crops per region (A = anterior, B = posterior)
- ~30 usable sections, ~500k total cells after QC
- Annotations: GeoJSON polygons in `data/annotations/<brain>/v1/` (repository root)
