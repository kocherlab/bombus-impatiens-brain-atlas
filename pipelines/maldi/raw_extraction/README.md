# maldi — imzML MSI region & ion-image pipeline

Tools for working with SCiLS-exported imzML mass-spectrometry imaging data
(Princeton samples `princeton_80524`, `princeton_80642`): extract per-pixel
peptide ion images, and hand-draw + name anatomical regions in napari.

## Layout

```
raw_extraction/
├── slurm_extract.py      # parallel per-pixel extraction: peptide windows + TIC -> parquet/h5/PNG
├── extract_80524.sh      # SLURM batch wrapper (stage .ibd to node-local disk, then extract)
├── build_cache.py        # parse the 1.2 GB .imzML XML once -> compact offset table
├── export_draw_image.py  # build native-res multi-channel TIFF (TIC + peptides) for napari
├── seg_grid.py           # auto-segment sections -> label TIFF + names JSON (the figure's path)
├── import_regions.py     # map labels -> pixels: pixel_regions parquet, region stats, crops
├── regionlib.py          # pure functions (label->pixel, stats, crop); unit-tested
├── test_regionlib.py
├── cache/                # offsets_<sample>.parquet (offset table), cache_<sample>.npz
├── data/                 # raw .7z + extracted .ibd/.imzML (gitignored, ~290 GB)
└── ion_out/              # generated ion images, feature tables (gitignored)
```

## Key facts about the data

- **Centroided / processed** imzML: each pixel stores a peak list (m/z + intensity),
  not a continuous spectrum. You can only read intensities at *picked* peaks.
- **No named regions** in the imzML (they lived in the SCiLS `.slx`); draw them here.
- Pixel grid 2484 × 1013, 20 µm square pixels, 533,563 on-tissue pixels (80524).
- Targets: sNPF v1 @ m/z 974.5891, Tk2 @ m/z 993.4757. Mass accuracy scatters
  ~±20 ppm, so ±15 ppm is tight; ±30 ppm captures more. Tolerance is configurable.

## Usage

```bash
source .venv/bin/activate

# 1. Extract ion images (sNPF, Tk2) + TIC for a sample (SLURM)
bash ../../../config/sbatch.sh extract_80524.sh 15   # 15 = ppm tolerance
#   -> ion_out/ionimg_princeton_80524_ppm15.png   (sNPF + Tk2 by x,y)
#   -> ion_out/features_princeton_80524_ppm15.parquet   (per-pixel, cached)
#   -> ion_out/ionimg_princeton_80524_ppm15.h5          (arrays incl. TIC)

# 2. Build the drawing image, then draw regions LOCALLY in napari
python export_draw_image.py ion_out/features_princeton_80524_ppm15.parquet \
    princeton_80524 ion_out/draw_princeton_80524.tif ion_out/draw_princeton_80524.json
#   (the figure's regions come from seg_grid.py instead)
#   -> labels_draw_princeton_80524.tif + regions_draw_princeton_80524.json

# 3. Import the drawn regions
python import_regions.py ion_out/features_princeton_80524_ppm15.parquet \
    labels_draw_princeton_80524.tif regions_draw_princeton_80524.json \
    princeton_80524 ion_out/regions
#   -> pixel_regions / region_stats parquet + per-region cropped images
```

## Setup and provenance (bimp-st Fig. neuropeptide_spatial)

`scripts/neuropeptides/fig_neuropeptide_spatial.py` reads
`ion_out/aligned_panel_princeton_{80524,80642}.h5ad`. `ion_out/`, `data/`,
`cache/` and `.venv` are gitignored and live only in the checkout that ran
the pipeline (site key `data_repo`, `bimp_st.site`).

- **Worktree or second checkout:** link them, never copy:

  ```bash
  M=$(PYTHONPATH=src .venv/bin/python -m bimp_st.site data_repo)/pipelines/maldi/raw_extraction
  ln -s $M/ion_out pipelines/maldi/raw_extraction/ion_out
  ```

- **Check the inputs:** `ion_out.md5` (tracked) holds the md5 of every file the
  figure reads and of their upstream inputs (panel parquets, per-section slices,
  offset caches, channel specs), plus the sizes of the raw imzML/ibd files.
  `cd pipelines/maldi/raw_extraction && md5sum -c ion_out.md5`
  (the `#` header lines are reported as improperly formatted; every file line
  must say OK).
- **Rebuild from raw** (`data` -> the SCiLS imzML export, site key `maldi_raw`;
  `.venv` -> an environment with the packages its scripts import):
  1. `python build_cache.py data/Princeton_imzmls/princeton_<id>.imzML cache/...`
     (offset tables);
  2. `bash config/sbatch.sh pipelines/maldi/raw_extraction/extract_panel_<id>.sh` -> `ion_out/panel_princeton_<id>.parquet`;
  3. `seg_grid.py` (section regions) then `make_slices.py` ->
     `ion_out/slices_princeton_<id>/slice_###.h5ad`;
  4. `bash config/sbatch.sh pipelines/maldi/raw_extraction/align_panel_<id>.sh` (PASTE3, `maldi_alignment.py align --method both
     --seed 5`) -> `ion_out/aligned_panel_princeton_<id>.h5ad`.
  A rebuild is checked against `ion_out.md5`.
