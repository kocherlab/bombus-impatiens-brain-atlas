/* Does the hover survive a real pointer moving over the real atlas?
 *
 * test/hover-hold.mjs unit-tests the hold; this drives the BUILT artifact so the wiring is
 * covered too. A pointer is walked slowly across the cloud while s.hover is sampled every
 * frame. Picking runs at ~11 Hz and misses whenever the ray lands between cells, so before
 * the hold every one of those misses blanked the highlight, the tooltip and the UMAP marker
 * for a frame or three — visible as blinking.
 *
 * Asserts the pointer keeps finding cells, and that the hover is not dropped repeatedly
 * while it does.
 *
 *   node viewer/test/hover-blink.mjs
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ART = join(APP, "dist", "bimp_brain.html");
const TMP = join(process.env.BIMP_TEST_TMP || "/tmp", "tmp");
const DRIVER = join(TMP, "hover-blink.html");

const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
if (!CHROMIUM) { console.error("Chromium not found in the Playwright cache"); process.exit(1); }
if (!existsSync(ART)) { console.error("✗ dist not built — run: node viewer/build.mjs"); process.exit(1); }

// The artifact drives itself: append a probe script that walks the pointer and records
// s.hover per frame. Written next to the artifact's own markup so file:// stays happy.
const art = readFileSync(ART, "utf8");
const probe = `
<script>
(async () => {
 try {
  // Timers, not rAF: under a virtual-time budget rAF is tied to real frame production, and
  // each frame here is 180k points through swiftshader — waiting on it starves the probe.
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const s = BIMP.state.s, R = BIMP.renderer;
  const cv = document.getElementById('gl');
  for (let i = 0; i < 300 && !(R && R.N > 0); i++) await sleep(50);
  R.setProgress(1);
  s.spin = false;                       // hold the view still: this tests picking, not motion
  await sleep(300);
  const r = cv.getBoundingClientRect();
  const cx = r.width / 2, cy = r.height / 2;
  // Find where the cloud actually IS. Anything off the cloud legitimately hovers nothing,
  // so the walk has to stay on cells or the test measures the wrong thing.
  const onCell = [], offCell = [];
  for (let a = 0; a < 12; a++) {
    const t = a / 12 * Math.PI * 2, rad = 20 + (a % 4) * 30;
    const x = Math.round(cx + Math.cos(t) * rad), y = Math.round(cy + Math.sin(t) * rad * 0.7);
    if (R.pick(x, y, 4) >= 0) onCell.push([x, y]);
  }
  // ...and somewhere the ray hits nothing. A pick between two cells returns exactly this,
  // and it is what used to blank the highlight mid-hover.
  for (let a = 0; a < 8 && offCell.length < 2; a++) {
    const x = 12 + a * 6, y = 12 + a * 4;
    if (R.pick(x, y, 4) < 0) offCell.push([x, y]);
  }
  const move = (x, y) => cv.dispatchEvent(new PointerEvent('pointermove',
    { clientX: r.left + x, clientY: r.top + y, bubbles: true, pointerId: 1, pointerType: 'mouse', isPrimary: true }));
  const seq = [];
  // Walk cells, but dip through a spot with nothing under it every so often — a pointer
  // skimming a gap. The dip is far shorter than the hold, so the hover must ride through it.
  const excursions = [];
  if (onCell.length >= 5 && offCell.length >= 1) {
    for (let i = 0; i < 120; i++) {
      const phase = i % 10;
      const gap = phase >= 8;                                // 2 samples of 10 land on nothing
      const p = gap ? offCell[0] : onCell[Math.floor(i / 10) % onCell.length];
      move(p[0], p[1]);
      await sleep(30);
      seq.push(s.hover);
      excursions.push(gap);
    }
  }
  let hits = 0, drops = 0, gapDrops = 0, gapCount = 0;
  for (let i = 0; i < seq.length; i++) {
    if (seq[i] >= 0) hits++;
    else if (i > 0 && seq[i - 1] >= 0) drops++;      // a run of -1 that follows a live hover
    // Count the gaps, and how many of them blanked the hover. Headless frames are far
    // slower than 60 fps, so the hold can also expire while the pointer sits ON a cell —
    // that is an artefact of this environment, not the bug, so score the gaps specifically.
    if (excursions[i] && !excursions[i - 1]) gapCount++;
    if (excursions[i] && seq[i] < 0) gapDrops++;
  }
  const out = document.createElement('pre'); out.id = 'probe';
  out.textContent = JSON.stringify({ frames: seq.length, hits, drops, gapDrops, gapCount,
    distinct: new Set(seq.filter((v) => v >= 0)).size, onCell: onCell.length,
    offCell: offCell.length, gaps: excursions.filter(Boolean).length,
    diag: { N: R.N, progress: R.getProgress(), rect: [Math.round(r.width), Math.round(r.height)] } });
  document.body.appendChild(out);
 } catch (err) {
  const out = document.createElement('pre'); out.id = 'probe';
  out.textContent = JSON.stringify({ error: String(err && err.stack || err) });
  document.body.appendChild(out);
 }
})();
</script>`;
writeFileSync(DRIVER, art + probe);

const dump = execFileSync(CHROMIUM, [
  "--headless", "--disable-gpu", "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
  "--no-sandbox", "--window-size=900,700", "--virtual-time-budget=40000",
  "--dump-dom", "file://" + DRIVER + "?nofull",
], { encoding: "utf8", maxBuffer: 1 << 28 });

const m = dump.match(/<pre id="probe">([\s\S]*?)<\/pre>/);
if (!m) { console.error("✗ probe never reported — page failed to load?\n" + dump.slice(0, 2000)); process.exit(1); }
const r = JSON.parse(m[1]);

let fail = 0;
const check = (name, ok, detail) => { console.log(`${ok ? "✓" : "✗"} ${name}  ${detail}`); if (!ok) fail++; };

console.log("— hover over the real atlas —");
console.log("  diag " + JSON.stringify(r.diag));
// non-vacuous: the walk has to be over real cells, and has to visit several of them
check("the walk had both cells and gaps to cross", r.onCell >= 5 && r.offCell >= 1 && r.gaps > 10,
  `${r.onCell}/12 positions on cells, ${r.gaps} samples over a gap`);
check("the hover was live often enough to be droppable", r.hits > r.frames * 0.4,
  `${r.hits}/${r.frames} frames had a live hover, ${r.distinct} distinct cells`);
check("a brief gap does not blank the hover",
  r.gapDrops <= 2, `${r.gapDrops} of ${r.gaps} gap samples blanked it, over ${r.gapCount} gaps ` +
  `(${r.drops} total drops incl. hold expiry on slow headless frames)`);

console.log(fail ? `\n${fail} check(s) failed` : "\nhover stays put");
process.exit(fail ? 1 : 0);
