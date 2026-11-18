# viewer — 3D brain-atlas viewer

A 3D point-cloud viewer of the *Bombus impatiens* spatial-transcriptomics atlas
(1.3M cells, queen + worker). Two builds from one codebase.

**To rebuild the data or deploy the site, see [`deploy/README.md`](deploy/README.md)** (cluster
exports, Cloudflare hosting for genes and tiles, the gh-pages script, the smoke test).

- **Live site (all 1.3M cells):** https://kocherlab.github.io/bombus-impatiens-brain-atlas/ —
  `build-pages.mjs`; data fetched from `pages/data/` (base.bin + 100k lite tier +
  lazy genes), rendered with the **raw WebGL2** renderer (`gl.POINTS`) — far faster
  and smoother than deck.gl's instanced quads at 1.3M points. Deployed from the
  `gh-pages` branch.
- **Single-file page (180k sample):** `build.mjs`; all data and code inlined, no
  external requests (runs under a strict CSP). Same raw WebGL2 renderer.

## What it does

- **The 1.3M-cell atlas** (live site) or a **180k caste-balanced sample** of it
  (single-file page), reassembled into a
  rotatable 3D volume (drag to orbit, scroll to zoom, auto-orbit on load).
- **Color by** — cell type, with a **labeling toggle**: *Canonical* (the
  canonical expression-only cluster-vote coarse classes — 7 + Mixed; the default) or
  *Subtypes* (a per-cluster majority vote of the 29 subtypes — spatially
  coherent, unlike the noisy per-cell argmax). Both avoid the per-cell
  salt-and-pepper; the raw per-cell call is shown in the inspector only. Palette
  grouped so *hue = anatomy*,
  caste (queen gold / worker teal, co-registered overlay), Tangram mapping
  confidence (magma), or **gene expression** — a picker searchable by symbol or
  protein name (e.g. "dopamine", "vitellogenin"); every gene carries its protein
  description, and the focal genes (sNPF, Kr-h1, oct-β2R, …) are starred.
- **Filter** — per-caste (queen / worker / both; defaults to queen alone; **"both"
  renders the two brains side-by-side**, not overlaid); **multi-select anatomical
  regions** (the eight hand-labeled masks —
  calyx inner/neck/outer compartments, lamina/medulla/lobula, antennal lobe and
  central complex — grouped by mushroom body / optic lobe / antennal lobe /
  central complex, select any combination); grouped cell-type legend
  with click-to-toggle and hover-to-isolate.
- **Inspect** — click any cell (radius GPU color-picking, so small points on a
  dense cloud are easy to hit) → type, caste, confidence, region membership, and
  the active gene's expression; the picked cell stays highlighted.
- **Progressive load** — the file is laid out so the manifest + a 100k-cell lite
  tier + all code come first (~0.9 MB) and the heavy 16 MB full blob is *last*. The
  browser stream-parses, paints the 100k lite view, and stays interactive while the
  full 180k-cell + 100-gene tier keeps downloading; it then hot-swaps in and is
  cached in IndexedDB so return visits skip the decode. So first paint is ~0.9 MB in,
  not 17 MB — even though the artifact is a single self-contained file.
- Sliders for point size, opacity, and **Z exaggeration** — the stack defaults to
  its **true 10 µm slice spacing** (1×; the brain is ~340 µm thick vs ~4 mm wide, a
  thin sheet), and the slider exaggerates depth for legibility. x/y/z share one µm
  scale so proportions are physical.
- Cell types with **0 cells** (subtypes that never win a cluster; empty canonical
  classes like Photoreceptors/Hemocytes under the vote) are omitted from the legend.

## Architecture

No bundler (the npm registry is unavailable in this environment), so the source
is a set of plain IIFE modules that publish to a single `window.BIMP` namespace
and load in dependency order. `build.mjs` inlines them by concatenation — a
reliable string join with no transformation — so **what the tests load is byte-
for-byte what ships**.

```
src/
  styles.css      body.html          # chrome + DOM shell
  palette.js      # anatomy→hue families, caste colors
  mat4.js         # perspective / lookAt / multiply
  shaders.js      # GLSL ES 3.0 (point cloud + GPU pick pass)
  camera.js       # orbit camera
  decode.js       # base64 → gzip → typed-array views
  state.js        # single view-state store + derived counts
  renderer.js     # GL program, buffers, draw, pick FBO
  ui.js           # panels: color-by, legend, gene picker, inspector
  boot.js         # decode → upload → build UI → input → RAF loop + selfTest()
  data.js         # GENERATED payload (positions + categories + genes + manifest)
build.mjs         # src/* → dist/bimp_brain.html   (the artifact)
export_data.py    # h5ad → src/data.js
test/
  run.mjs         # headless self-test (drives installed Chromium, no npm)
  shots.mjs       # per-mode screenshots for visual QA
```

## Build & test

```bash
# 1. regenerate the data payload from the atlas (once, or on data change)
PYTHONPATH=src .venv/bin/python viewer/export_data.py --sample 200000 --genes 24

# 2. build the self-contained artifact  → viewer/dist/bimp_brain.html
node viewer/build.mjs

# 3. headless test: renders with software WebGL, checks pixels + GPU picking +
#    gene recolor, fails on any console error. Writes test/last-screenshot.png.
node viewer/test/run.mjs --build

# 4. visual QA: one screenshot per color mode → test/shot-*.png
node viewer/test/shots.mjs --build
```

### The test framework, without npm

The npm registry is blocked here, so there is no Vitest/Playwright install.
Instead `test/run.mjs` drives the Chromium that ships in the Playwright browser
cache directly via its CLI (`--use-angle=swiftshader` for software WebGL,
`--dump-dom` + `--screenshot`). The page exposes `BIMP.selfTest()`, which runs a
real GL pass — renders, reads back pixels, GPU-picks the center cell, recolors by
gene — and writes a JSON verdict into the DOM; the harness parses it and exits
non-zero on failure. This caught e.g. a vertex/fragment `int`-precision mismatch
that never surfaces from static syntax checks.

To iterate: edit a `src/*` module, `node build.mjs`, open `dist/bimp_brain.html`
(it is a valid standalone page) or re-run the tests.

## Deploying / updating the artifact

`dist/bimp_brain.html` is the deliverable: host it as a static file anywhere. It is regenerated deterministically from the
atlas (`export_data.py` uses `SEED=0`) and resolves the h5ad from the main
checkout automatically when run inside a git worktree.

## Caveats

- The single-file page shows a 180k sample for payload/perf; the per-type **"cells mapped"**
  counts shown in the UI are the *exact full-atlas* totals from the manifest.
- **Cell-type labels:** the raw `tangram_cell_type` column is a per-cell argmax
  that the 100-gene Xenium panel cannot resolve for Kenyon cell subtypes (the six
  subtypes are ~0.9 correlated on the panel). The viewer shows the expression-only
  hierarchical cluster vote (`bimp_st.cluster_labels.hierarchical_vote`, the
  labelling of every figure): a cluster takes a fine type where the vote decides one,
  else its pooled group, else Mixed; Kenyon subtypes are held at "Kenyon cells".
  "Atlas types" is the vote as written, "Groups" its seven pooled groups. The raw
  per-cell argmax is exposed only as an inspector field.
- Gene names come from two sources: full protein descriptions for all 100 panel
  genes from `data/panel/np31_to_v43_mapping.csv`, plus hand-curated short symbols
  for the 9 focal genes (sNPF, Kr-h1, oct-β2R, …). Keep those in sync with
  `src/bimp_st/genes.py`'s `GENE_SYMBOLS`, which is the superset;
  `scripts/paper/check_gene_symbol_sync.py` checks the shared ids. The exporter always
  includes the focal genes, then fills up to `--genes` by descending variance; each
  entry stores `symbol` (short) + `desc` (full protein) so the picker is searchable
  by either. There are no HGNC-style symbols — these are *B. impatiens* v4.3
  gene-model proteins.
- Caste contrast is pseudoreplicated (1 queen + 1 worker brain) — surfaced in the
  intro copy.
- Uses combined `tangram_cell_type` labels (not `_stitched`), the canonical
  labelling (`bimp_st.paths.CANONICAL_TANGRAM`).
