/* The home page's section figure: every section of both brains as small multiples,
 * after the serial-section grid of fig:atlas_overview panel C (scripts/atlas/fig_atlas_overview.py), but drawn
 * from this atlas's own data rather than copied from that figure.
 *
 * Reads pages/data/manifest.json + base.bin, takes every cell of each brain (queen, then worker),
 * groups them by section (one z level per section; never by Xenium capture region, which holds
 * several), and stipples each section into one tile of a single horizontal WebP strip,
 * assets/sections.webp. Each cell is a fine ink dot at partial strength, so density
 * reads as tone, like the hero. Orientation is the atlas's front view: the renderer's world is (y, -x)
 * with y up, so on screen stored y runs right and stored x runs down (dorsal up). All sections
 * of both brains share one frame and one scale (the export puts both castes on the same µm
 * scale), so the queen's larger brain is honestly larger; sections are already aligned, so
 * none is rotated on its own.
 *
 * It also rewrites the tile grid between the "sections" markers in src/body.html: each tile is
 * a window onto the strip (one shared, lazily loaded image), and the depth labels (µm, as the
 * atlas's section rail gives them) are real
 * text in the page's own fonts. Needs ImageMagick (`magick`) for the WebP encode.
 *
 *   node viewer/tools/sections-figure.mjs [tile-width-px] [webp-quality]
 */
import { readFileSync, writeFileSync, mkdtempSync, rmSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const APP = join(dirname(fileURLToPath(import.meta.url)), "..");
const TW = +(process.argv[2] || 176); // tile width in image px: 2x the ~85 CSS px it shows at
const QUALITY = +(process.argv[3] || 75);
const BRAINS = [[0, "Queen"], [1, "Worker"]]; // manifest castes: ["queen", "worker"]
const PAD = 3;
const ALPHA = +(process.env.ALPHA || 0.15); // one dot's ink; overlapping dots build tone
const BLUR = +(process.env.BLUR || 0.5); // px
const DOT = +(process.env.DOT || 1.4); // dot diameter in image px (~0.7 CSS px at 2x)
const INK = 0x0b; // #0b0b0a, as grey

const M = JSON.parse(readFileSync(join(APP, "pages", "data", "manifest.json"), "utf8"));
let buf = readFileSync(join(APP, "pages", "data", M.base));
if (buf[0] === 0x1f && buf[1] === 0x8b) buf = gunzipSync(buf);
const col = (k, T) => { const [o, len] = M.offsets[k]; return new T(buf.buffer.slice(buf.byteOffset + o, buf.byteOffset + o + len)); };
const X = col("x", Int16Array), Y = col("y", Int16Array), Z = col("z", Int16Array), C = col("caste", Uint8Array);

// group by brain and section; the viewer draws world +z as anterior, so z descending runs
// front to back
const bySec = BRAINS.map(() => new Map()), us = BRAINS.map(() => []), vs = BRAINS.map(() => []);
for (let i = 0; i < M.n; i++) {
  const m = bySec[C[i]];
  if (!m) continue;
  us[C[i]].push(Y[i]); vs[C[i]].push(X[i]);
  (m.get(Z[i]) || m.set(Z[i], []).get(Z[i])).push(i);
}
// one extent for every tile of both brains: the union of each brain's own 0.1st-99.9th percentile range, so
// a few stray cells don't shrink every tile, and neither brain (the queen's is the larger) is
// trimmed by more than a thousandth of its cells a side
const q = (a, p) => a.sort((x, y) => x - y)[Math.floor(p * (a.length - 1))];
const lo = (a) => Math.min(...a.map((x) => q(x, 0.001))), hi = (a) => Math.max(...a.map((x) => q(x, 0.999)));
const u0 = lo(us), u1 = hi(us), v0 = lo(vs), v1 = hi(vs);
const s = (TW - 2 * PAD) / (u1 - u0);
const TH = Math.ceil((v1 - v0) * s + 2 * PAD);
// Labels are depth in µm, exactly as the atlas's section rail gives it (histology.js depthUm):
// (z - the brain's first z) x frame.scale_um, z normalised to +-1 as the viewer holds it. Tiles
// read top down like the rail, deepest section first and the first cut (0 µm) last.
const HM = join(APP, "pages", "data", "histo", "manifest.json");
const SCALE = existsSync(HM) ? JSON.parse(readFileSync(HM, "utf8")).frame?.scale_um : null;
const tiles = BRAINS.flatMap((_, b) => {
  const zs = [...bySec[b].keys()].sort((x, y) => y - x), z0 = zs[zs.length - 1];
  return zs.map((z, j) => ({ b, z, um: SCALE ? Math.round(((z - z0) / 32767) * SCALE) : (zs.length - 1 - j) * 10 }));
});
const N = tiles.length, W = TW * N;
if (W > 16383) throw new Error(`strip ${W} px wide exceeds WebP's 16383 px limit; pass a smaller tile width`);

// coverage per pixel: each dot is a small box of DOT px, area-weighted onto the pixel grid, and
// overlapping dots combine as layers of ink: ink = 1 - (1 - ALPHA)^coverage
const cov = new Float32Array(W * TH), r = DOT / 2;
let cells = 0;
tiles.forEach(({ b, z }, k) => {
  for (const i of bySec[b].get(z)) {
    if (Y[i] < u0 || Y[i] > u1 || X[i] < v0 || X[i] > v1) continue;
    const cx = k * TW + PAD + (Y[i] - u0) * s, cy = PAD + (X[i] - v0) * s;
    const x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r;
    for (let py = Math.floor(y0); py < y1; py++) {
      const wy = Math.min(y1, py + 1) - Math.max(y0, py);
      if (py < 0 || py >= TH || wy <= 0) continue;
      for (let px = Math.floor(x0); px < x1; px++) {
        const wx = Math.min(x1, px + 1) - Math.max(x0, px);
        if (wx > 0) cov[py * W + px] += wx * wy;
      }
    }
    cells++;
  }
});
// Encoded opaque, ink on white, as one grey channel: lossy WebP keeps stipple far smaller than
// a noisy alpha channel (which WebP stores near-losslessly). The page lays it on the page background with
// mix-blend-mode: multiply, so the white drops out. A light blur takes off the per-pixel grain
// the eye cannot resolve at 2x and that would otherwise cost most of the bytes.
const grey = Buffer.alloc(W * TH), keep = Math.log(1 - ALPHA);
for (let p = 0; p < W * TH; p++) {
  const a = 1 - Math.exp(keep * cov[p]);
  grey[p] = Math.round(255 - a * (255 - INK));
}
const tmp = mkdtempSync(join(tmpdir(), "sections-"));
writeFileSync(join(tmp, "strip.gray"), grey);
const out = join(APP, "assets", "sections.webp");
execFileSync("magick", ["-size", `${W}x${TH}`, "-depth", "8", `gray:${join(tmp, "strip.gray")}`,
  "-blur", `0x${BLUR}`, "-quality", String(QUALITY), "-define", "webp:method=6", out]);
rmSync(tmp, { recursive: true });

// the page: one labelled group per brain, each a list of windows onto the one strip
const img = `<img src="assets/sections.webp" width="${W}" height="${TH}" alt="" loading="lazy" decoding="async">`;
const I = "\n          ";
const groups = BRAINS.map(([, name], b) => {
  const mine = tiles.map((t, k) => ({ ...t, k })).filter((t) => t.b === b);
  const li = mine.map(({ k, um }) => `  <li><div class="t" style="--k:${k}">${img}</div><span class="num">${um}&nbsp;µm</span></li>`);
  return `<div class="grp">${I}  <h3 class="sc">${name} · <span class="num">${mine.length}</span> sections</h3>` +
    // the picture role sits on a wrapper: role="img" is not allowed on an <ol>, and it would orphan the <li>s
    `${I}  <div role="img" aria-label="The ${name.toLowerCase()} brain's ${mine.length} sections top down, by depth in µm, each a stipple of its cells"><ol>` +
    `${I}  ${li.join(`${I}  `)}${I}  </ol></div>${I}</div>`;
});
const block = `<div class="sec-grid" style="--n:${N};--ar:${TW}/${TH}">${I}${groups.join(I)}${I}</div>`;
const bodyPath = join(APP, "src", "body.html");
const body = readFileSync(bodyPath, "utf8");
const re = /(<!-- sections:start[^>]*-->)[\s\S]*?(<!-- sections:end -->)/;
if (!re.test(body)) throw new Error("src/body.html has no <!-- sections:start --> … <!-- sections:end --> block");
writeFileSync(bodyPath, body.replace(re, `$1${I}${block}${I}$2`));
const kb = readFileSync(out).length / 1024;
console.log(`${BRAINS.map(([, n], b) => `${n} ${bySec[b].size}`).join(" · ")} sections · ${cells} cells · ` +
  `tile ${TW}×${TH} px · strip ${W}×${TH} · sections.webp ${kb.toFixed(0)} KB`);
