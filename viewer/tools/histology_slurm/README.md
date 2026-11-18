# Histology export (SLURM)

Recreates the atlas viewer's histology layer from the Xenium data on the cluster:
the DAPI overview, the three other stains, and the high-resolution tile pyramid.
Everything runs through Slurm on CPU (`bash config/sbatch.sh`; partition and QOS in
`config/profiles/slurm/sbatch.env`). Nothing runs on the login node except submission,
`squeue` and `rsync`.

## Inputs (gitignored, in the main checkout)

| what | where |
|---|---|
| atlas object the viewer is exported from | `results/tangram/<TANGRAM_REL>` (read from `viewer/export_data.py`) |
| per-crop h5ads, with `uns/image` = DAPI, boundary, 18S, αSMA/vimentin | `pipelines/wholebrain/{queen,worker}/data/output-XETG00151__*.h5ad` |
| PASTE3-aligned DAPI stacks (~20 GB each) | `pipelines/wholebrain/{queen,worker}/figures/alignment_validation/aligned_images.tif` |

Submit from the repo root. In a worktree that lacks these, set
`DATA_REPO` to the checkout that holds them. All
settings, and how to override them, are in `env.sh`.

## Run

```bash
bash viewer/tools/histology_slurm/submit_all.sh
```

That submits three jobs as a dependency chain:

| step | script | Slurm | produces in `viewer/pages/data/` | time |
|---|---|---|---|---|
| 1 | `01_histology.sbatch` → `export_histology.py` (+ `histo_refit.py` for the worker) | cpu, short, 4 cpu, 16 GB, 55 min | `histo/`: `manifest.json`, `registration.json`, `{queen,worker}/NN.webp` + `NN-s.webp` | ~16 min |
| 2 | `02_stains.sbatch` → `export_stains.py` (after 1) | cpu, short, 4 cpu, 24 GB, 55 min | `histo_stains/`: `{caste}/NN-{boundary,rna18s,smavim}.webp` + `-s`, `stains.json`, `stains_check.json` | ~16 min |
| 3 | `03_tiles.sbatch` → `export_tiles.py`, array 0 = queen, 1 = worker (after 1, alongside 2) | cpu, short, 4 cpu, 16 GB, 55 min per task | `histo_tiles/`: `{caste}/{ii}/{stain}/{level}/{tx}_{ty}.webp`, `tiles.json`, `tiles_check.json` | ~45 min per brain |

Logs go to `viewer/tools/histology_slurm/logs/`. To redo one step, submit its
script alone. To redo one brain's tiles, run `BIMP_SBATCH_EXTRA=--array=1 bash config/sbatch.sh viewer/tools/histology_slurm/03_tiles.sbatch`
(1 = worker). It merges into `tiles.json` under a file lock and keeps the other
brain's entries.

## Expected outputs

- `histo/`: **30 queen + 35 worker** sections.
- `histo_stains/`: 3 stains × 65 sections × (large + small) = 390 images.
  `stains_check.json` should show `rigid_p95_px` ≈ 0 and `dapi_shift_px` [0, 0] for
  every section. That is the independent check that the DAPI layer is registered.
- `histo_tiles/`: **17,128 tiles**, about 335 MB: 4 stains × 3 levels. Level 0 is
  about 0.8–0.9 µm/px for the queen and 0.6–0.7 µm/px for the worker. Level 2 has the overview's own size.
  `tiles_check.json` has every section's `rigid_p95_px` ≈ 0.

## Copy back and merge

From the machine that builds the site. Use `-rlt`, not `-a`, which can fail with
"unable to escalate dir mode":

```bash
for d in histo histo_stains histo_tiles; do
  rsync -rlt <cluster>:<checkout>/viewer/pages/data/$d/ viewer/pages/data/$d/
done
python3 viewer/tools/histology_slurm/merge_manifest.py viewer/pages/data
```

`merge_manifest.py` (stdlib, seconds) adds the `stains` and `tiles` indexes to
`histo/manifest.json`. It keeps a tile host the deploy step already set, then
checks that every referenced file exists and prints the counts above. It exits
non-zero if anything is missing.

## Deploy

The tiles are served from Cloudflare (`https://bimp-st-tiles.kocherlab.workers.dev`),
not from GitHub Pages. Uploading them, and pointing `tiles.path` at that host, is
done by the deploy script in `viewer/deploy/`. See its README.

## Optional step 4: 8× DAPI tiles

`04_tiles_hq.sbatch` writes `histo_tiles_hq/` (DAPI only, ~0.44 µm per pixel queen, ~0.34 worker), roughly 19–20k
files, which `tools/shard_hq_tiles.py` splits into Cloudflare-sized shards. Create `logs/` first
(`mkdir -p viewer/tools/histology_slurm/logs`; submission fails in 0 s without it). The full procedure,
including merge, shard, check and upload, is at the end of `deploy/README.md`.
