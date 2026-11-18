/* Does the live UMAP marker actually follow the pointer?
 *
 * Drives the PAGES build, because that is what is deployed and it is the only build that
 * carries per-cell UMAP coordinates (the artifact's full tier omits them, so its Show-UMAP
 * toggle has nothing to show). Serves pages/ over http and injects a probe that walks a
 * pointer across the cloud, sampling BOTH umaps:
 *
 *   #umap-live  — the standalone panel, which is supposed to track hover
 *   #umap-mini  — the inspector's copy, which historically only redrew on click
 *
 * The marker is located by reading the canvas back and taking the centroid of the pure-white
 * pixels it is drawn with, so this measures what is ON SCREEN, not an internal variable.
 *
 *   node viewer/test/umap-follow.mjs
 */
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ROOT = join(APP, "pages");
const SHOT = join(HERE, "umap-follow.png");
const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
const TYPES = { ".html": "text/html", ".json": "application/json", ".bin": "application/octet-stream" };

const PROBE = `
<script>
(async () => {
 const out = (o) => { const p = document.createElement('pre'); p.id='probe'; p.textContent = JSON.stringify(o); document.body.appendChild(p); };
 try {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const s = BIMP.state.s, R = BIMP.renderer;
  const cv = document.getElementById('gl');
  await BIMP.ready;                       // full tier live → umapx present if it is exported at all
  for (let i = 0; i < 400 && !(R && R.N > 0); i++) await sleep(50);
  R.setProgress(1);
  s.spin = false;                         // this measures picking, not motion
  await sleep(400);

  const D = BIMP.data || {};
  const hasUmap = !!(D.umapx && D.umapy);
  // Non-vacuity for the coords themselves: a constant array would make a marker that
  // "correctly" never moves, and the test would pass while the panel looked broken.
  let spread = 0;
  if (hasUmap) {
    let lo = 255, hi = 0;
    for (let i = 0; i < D.n; i += Math.max(1, Math.floor(D.n / 5000))) { const v = D.umapx[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    spread = hi - lo;
  }

  // turn the standalone panel on the way a visitor does
  document.getElementById('tg-umap').click();
  await sleep(200);
  const panelShown = document.getElementById('umap-panel').classList.contains('show');

  // centroid of the white marker pixels on a umap canvas (the cloud behind it is dim colour)
  function markerAt(id) {
    const c = document.getElementById(id);
    if (!c || !c.width) return null;
    const g = c.getContext('2d');
    let d; try { d = g.getImageData(0, 0, c.width, c.height).data; } catch (e) { return null; }
    let sx = 0, sy = 0, k = 0;
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
      const o = (y * c.width + x) * 4;
      if (d[o] > 230 && d[o+1] > 230 && d[o+2] > 230 && d[o+3] > 200) { sx += x; sy += y; k++; }
    }
    return k ? [Math.round(sx / k), Math.round(sy / k), k] : null;
  }

  const r = cv.getBoundingClientRect();
  const cx = r.width / 2, cy = r.height / 2;
  const onCell = [];
  for (let a = 0; a < 24 && onCell.length < 8; a++) {
    const t = a / 24 * Math.PI * 2, rad = 40 + (a % 5) * 45;
    const x = Math.round(cx + Math.cos(t) * rad), y = Math.round(cy + Math.sin(t) * rad * 0.7);
    if (R.pick(x, y, 4) >= 0) onCell.push([x, y]);
  }
  const move = (x, y) => cv.dispatchEvent(new PointerEvent('pointermove',
    { clientX: r.left + x, clientY: r.top + y, bubbles: true, pointerId: 1, pointerType: 'mouse', isPrimary: true }));

  // Click one cell first, so the inspector (and its #umap-mini) is open and has something
  // to be stale against — that is exactly the state a visitor is in when they say the UMAP
  // "doesn't update as I hover".
  if (onCell.length) {
    const [x0, y0] = onCell[0];
    BIMP.ui.inspect(R.pick(x0, y0, 4));
    await sleep(250);
  }
  const miniAtStart = markerAt('umap-mini');

  const live = [], mini = [], hovers = [], labels = [];
  for (let i = 0; i < 40 && onCell.length >= 4; i++) {
    const p = onCell[i % onCell.length];
    move(p[0], p[1]);
    await sleep(120);                     // > the 90 ms pick cadence
    hovers.push(s.hover);
    labels.push((document.getElementById('umap-live-lab') || {}).textContent || '');
    live.push(markerAt('umap-live'));
    mini.push(markerAt('umap-mini'));
  }
  const key = (m) => m ? m[0] + ',' + m[1] : 'none';
  out({
    hasUmap, spread, panelShown, n: D.n,
    hoverDistinct: new Set(hovers.filter((v) => v >= 0)).size,
    hoverLive: hovers.filter((v) => v >= 0).length, frames: hovers.length,
    liveDistinct: new Set(live.map(key)).size, liveDrawn: live.filter(Boolean).length,
    miniDistinct: new Set(mini.map(key)).size, miniDrawn: mini.filter(Boolean).length,
    miniStart: key(miniAtStart),
    labelDistinct: new Set(labels).size,
    livePositions: live.map(key).slice(0, 12), miniPositions: mini.map(key).slice(0, 12),
  });
 } catch (err) { out({ error: String(err && err.stack || err) }); }
})();
</script>`;

async function main() {
  if (!CHROMIUM) { console.error("✗ Chromium not found"); process.exit(1); }
  if (!existsSync(join(ROOT, "index.html"))) { console.error("✗ pages/index.html missing (run build-pages.mjs)"); process.exit(1); }

  const server = createServer((req, res) => {
    let p = normalize(decodeURIComponent(req.url.split("?")[0])).replace(/^(\.\.[/\\])+/, "");
    if (p === "/" || p === "\\") p = "/index.html";
    const file = join(ROOT, p);
    if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    const ext = file.slice(file.lastIndexOf("."));
    // Inject the probe into the served page rather than editing the built file — the thing
    // under test stays byte-identical to what gets deployed.
    if (ext === ".html") {
      // BIMP_INDEX swaps in an alternate page (negative controls) while still serving the
      // real pages/data behind it.
      res.writeHead(200, { "content-type": "text/html" });
      res.end(readFileSync(process.env.BIMP_INDEX || file, "utf8") + PROBE);
      return;
    }
    res.writeHead(200, { "content-type": TYPES[ext] || "application/octet-stream" });
    res.end(readFileSync(file));
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;

  let dom = "";
  try {
    dom = await new Promise((resolve, reject) => {
      const child = spawn(CHROMIUM, [
        "--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
        "--window-size=1280,860", "--virtual-time-budget=90000",
        "--screenshot=" + SHOT, "--dump-dom", `http://127.0.0.1:${port}/`,
      ], { stdio: ["ignore", "pipe", "pipe"] });
      const buf = [];
      child.stdout.on("data", (c) => buf.push(c));
      child.stderr.resume();
      child.on("error", reject);
      child.on("close", () => resolve(Buffer.concat(buf).toString("utf8")));
    });
  } finally { server.close(); }

  const m = dom.match(/<pre id="probe">([\s\S]*?)<\/pre>/);
  if (!m) { console.error("✗ probe never reported\n" + dom.slice(0, 1500)); process.exit(1); }
  const r = JSON.parse(m[1]);
  if (r.error) { console.error("✗ probe threw: " + r.error); process.exit(1); }

  let fail = 0;
  const check = (name, ok, detail) => { console.log(`${ok ? "✓" : "✗"} ${name}  ${detail}`); if (!ok) fail++; };
  console.log("— live UMAP follows the pointer —");
  console.log("  " + JSON.stringify(r) + "\n");

  check("the deployed build carries UMAP coordinates", r.hasUmap && r.spread > 40, `hasUmap=${r.hasUmap} spread=${r.spread}/255`);
  check("the panel opens", r.panelShown, `#umap-panel .show = ${r.panelShown}`);
  check("the walk hovered several different cells", r.hoverDistinct >= 4,
    `${r.hoverLive}/${r.frames} frames live, ${r.hoverDistinct} distinct`);
  check("the live marker moves with the pointer", r.liveDistinct >= 4,
    `${r.liveDistinct} distinct marker positions, drawn on ${r.liveDrawn}/${r.frames} samples`);
  check("the inspector's mini UMAP also follows", r.miniDistinct >= 4,
    `${r.miniDistinct} distinct positions (started at ${r.miniStart})`);

  console.log(fail ? `\n${fail} check(s) failed` : "\nboth umaps track the pointer");
  process.exit(fail ? 1 : 0);
}
main();
