# Rebuilding and deploying the web atlas

The live site is **https://kocherlab.github.io/bombus-impatiens-brain-atlas/**. It is served from three places:

| What | Where | Size | Put there by |
|---|---|---|---|
| the page, cells (`base*.bin`), manifests, histology sections and stains, drawings, fonts | `gh-pages` branch → GitHub Pages | ~70 MB | `deploy/gh-pages.sh` |
| per-gene files (9,710: 100 measured + 9,610 inferred) | Cloudflare Worker `bimp-st-genes` → https://bimp-st-genes.kocherlab.workers.dev | ~3.3 GB | `deploy/cloudflare.sh genes` |
| high-resolution histology tiles (both brains, 4 stains, ~0.85 µm/px) | Cloudflare Worker `bimp-st-tiles` → https://bimp-st-tiles.kocherlab.workers.dev | ~350 MB, 17k files | `deploy/cloudflare.sh tiles` |

GitHub Pages caps a site at 1 GB, which is why the genes and tiles live on Cloudflare (free plan:
static assets, no bandwidth cap, at most 20,000 files of at most 25 MiB per Worker). Only the
copies of the manifests on `gh-pages` point at Cloudflare; the local `pages/data/` keeps relative
paths, so a local build works offline. `pages/data/` itself is gitignored on every source branch.

## Rebuild, in order

Steps 1–4 make data and run on a cluster through Slurm; nothing heavier than `ls`, `squeue` or
`rsync` runs on the login node. Jobs are submitted with `bash config/sbatch.sh`, which takes the
partition and QOS for each job class from `config/profiles/slurm/sbatch.env`. Each folder's
README has the exact submission lines.

1. **Cells and measured genes** — `export_data.py --pages` writes `pages/data/base.bin`, the lite
   and hero tiers' source, the 100 measured genes and `manifest.json`. Cell types are the
   expression-only cluster vote on the canonical Tangram run (`bimp_st.paths`).
2. **Inferred genes** — `tools/gene_projection/` (README there): project the canonical Tangram
   mapping onto the snRNA reference, keep genes detected in ≥1% of cells of at least one snRNA
   cell type (plus every panel and neuropeptide gene), and export them as lazy per-gene files.
3. **Gene names** — `python3 viewer/gene_names.py viewer/pages/data` fills each gene's bee
   name, fly ortholog and search aliases from `data/panel/bimp_v43_gene_names.csv` (built by
   `tools/gene_orthologs.py`; see `data/panel/README_gene_names.md`). `export_data.py` does this
   itself; run it by hand after copying a manifest from elsewhere.
4. **Histology** — `tools/histology_slurm/` (README there): the DAPI section stack, the stain
   composites, then the tile pyramid (a Slurm array, one task per brain).

Copy results back with `rsync -rlt` (not `-a`: the cluster's directory modes fail on the Mac), and
copy the gene folder on its own: a whole-`pages/data` sync also brings back the cluster's
`manifest.json` and wipes the names from step 3.

5. **Build** — `node viewer/build-pages.mjs` writes `pages/index.html`, the hero tier
   (`base-hero.bin`, every queen cell of the lite tier) and copies `assets/`.
6. **Check locally** —
   ```bash
   node viewer/tools/serve.mjs viewer/pages        # http://127.0.0.1:8765/
   node viewer/test/smoke-site.mjs                     # home, every cell, histology, six named genes
   ```

## Deploy

```bash
npx wrangler login                       # once per machine (Cloudflare account; subdomain "kocherlab")
viewer/deploy/cloudflare.sh all      # or: tiles | genes — only when that data changed
viewer/deploy/gh-pages.sh            # builds, commits to a temporary gh-pages worktree, stops
viewer/deploy/gh-pages.sh --push     # …and pushes; the site updates in a minute or two
node viewer/test/smoke-site.mjs https://kocherlab.github.io/bombus-impatiens-brain-atlas/
```

`cloudflare.sh` refuses to upload genes unless every file the manifest lists is present and
non-empty. `gh-pages.sh` names the source commit in its commit message.

## Things that have bitten

- **Never run `wrangler` from `viewer`.** With no config file it "auto-configures" the
  folder (writes `wrangler.jsonc`, adds itself to `package.json`) and starts deploying all of
  `pages/` as a Worker. `cloudflare.sh` runs it from a temporary staging folder instead.
- **A root-owned `~/.npm` cache** breaks `npx wrangler` with `EACCES`/`EEXIST`; set
  `NPM_CONFIG_CACHE` to a folder you own, or `sudo chown -R $(id -u) ~/.npm`.
- **Cross-origin timing hides status codes**: in the browser's resource timing, a Cloudflare tile
  or gene shows status 0 unless the server sends `Timing-Allow-Origin`. That is not a failure;
  fetch it directly to check.
- **The live gene count is a checked number**: a number check reads the deployed
  manifest, so changing the inferred set changes a checked value.

## Higher-resolution tiles: measured sizes and Cloudflare limits

Size estimates from `export_tiles.py --estimate` (sections 3 and 14 of each brain, 3 levels, 512 px WebP;
Slurm job, ~1.5 h, writes nothing but the estimate). Today's tiles are `--upscale 4`.

| `--upscale` | µm/px queen / worker | DAPI only | all four stains | tiles per section (L0/L1/L2) |
|---|---|---|---|---|
| 4 (deployed) | 0.68 / ~0.68 | 30 / 33 MB | 152 / 181 MB | worker 52 / 15 / 4 |
| 8 | 0.44 / 0.34 | 68 / 76 MB | 405 / 512 MB | queen 153–159 / 43–47 / 13–14, worker 181–183 / 52 / 15 |

8× costs about three times the bytes and over three times the files of 4×. Bytes are not the constraint; the free-plan
**20,000 files per Worker** is: `bimp-st-tiles` already holds ~17k, so any higher-resolution set needs
its own Worker (extrapolated, not measured: DAPI-only 8× is roughly 19–20k files, so two Workers to be
safe; all stains at 8× or native resolution is several Workers). The exporter stops at 8×; native
Xenium (0.2125 µm/px) would need `--upscale 16` added to its `choices`.

The export runs on Slurm (`tools/histology_slurm/03_tiles.sbatch`), but the cluster holds no
Cloudflare credentials and no `wrangler`, so the upload (`cloudflare.sh`) is run from a machine with
`npx wrangler login`, or on a compute node with an API token in `CLOUDFLARE_API_TOKEN`. The account's
plan cannot be read from here; `npx wrangler whoami` and the dashboard show it.

### Building and deploying the 8× DAPI tiles (not yet deployed; the export had not finished when this was written)

Each step is a separate command; none of this touches the live `bimp-st-tiles` Worker or the 4× tiles.

1. **Export** (Slurm, one task per brain, own folder `histo_tiles_hq/`, day class, 64 GB):
   `mkdir -p viewer/tools/histology_slurm/logs && bash config/sbatch.sh viewer/tools/histology_slurm/04_tiles_hq.sbatch`
   (from the repo root; in a worktree set `DATA_REPO` and `DATA_OUT` as for the other steps). It runs
   `export_tiles.py --upscale 8 --stains dapi --url-dir histo_tiles_hq`.
2. **Merge and shard** after copying `histo/`, `histo_stains/`, `histo_tiles/` and `histo_tiles_hq/` into `pages/data/`:
   `python3 viewer/tools/histology_slurm/merge_manifest.py` then
   `python3 viewer/tools/shard_hq_tiles.py`. The second cuts whole sections into shards of at most
   19,000 files (`histo_tiles_hq_shards/hq<n>/`) and gives each section its own tile `path` on
   `https://bimp-st-tiles-hq<n>.kocherlab.workers.dev/`. Run them in that order: `merge_manifest.py`
   rewrites every section's tiles back to the 4× set.
3. **Check locally** (`serve.mjs`, `smoke-site.mjs`) before any upload. Sections with HQ tiles show only
   DAPI at full resolution; the other stains stay at overview resolution there.
4. **Upload**: `viewer/deploy/cloudflare.sh tiles-hq` creates one Worker per shard. It needs either
   `npx wrangler login`, or `CLOUDFLARE_API_TOKEN` (the "Edit Cloudflare Workers" token template) and
   `CLOUDFLARE_ACCOUNT_ID` in the environment; it is not part of `all`. wrangler 4.86 installs and
   dry-runs on the cluster (`--dry-run` needs no login).
5. **Publish**: `viewer/deploy/gh-pages.sh --push`. Per-section HQ paths are already absolute URLs
   and are kept as they are.
