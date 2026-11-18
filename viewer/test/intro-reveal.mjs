/* Both intro doors must actually reveal the atlas. Tests the PAGES build.
 *
 * In fetch mode (what GitHub Pages serves) the cloud is uploaded and pinned at
 * progress 0 — deliberately, so the build-up plays on-screen instead of behind the
 * intro overlay. Only dismissIntro() → maybeReveal() starts it. A button that tears
 * the intro down its own way therefore leaves a PERFECTLY EMPTY CANVAS, which is
 * an intro in which the brains could not be seen.
 *
 * This must run against pages/, NOT dist/: the single-file artifact takes the inline
 * path, which never pins progress to 0 and so cannot reproduce the bug at all. An
 * earlier version of this test pointed at dist/ and passed while the site was broken.
 *
 * Serves viewer/pages over loopback, then drives both doors:
 *   • "▸ Take the guided tour"  (#enter)
 *   • "skip to the atlas →"     (#enter-tour)
 *
 * Runs with ?nofull&gate: the lite tier (software GL cannot drive 1.3M at any usable
 * speed) but with the real reveal gating intact, so the doors are genuinely tested.
 * Asserts the build STARTS — progress leaves 0 and pixels light up — not that it finishes.
 *
 *   node viewer/test/intro-reveal.mjs
 */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { dirname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ROOT = process.env.BIMP_PAGES || join(APP, "pages");

const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
if (!CHROMIUM) { console.error("Chromium not found in the Playwright cache"); process.exit(1); }
for (const f of ["index.html", "data/manifest.json", "data/base.bin"]) {
  if (!existsSync(join(ROOT, f))) { console.error(`missing ${f} under ${ROOT} — run: node viewer/build-pages.mjs`); process.exit(1); }
}

// The probe is injected by the server at /probe-<door>.html, so nothing temporary is
// written into the repo and the page is byte-identical to index.html otherwise.
const probe = (door) => `
<script>
// Force the ANIMATED reveal: headless reports prefers-reduced-motion:reduce, which
// short-circuits straight to progress 1 and would hide the very bug this test exists for.
const realMM = window.matchMedia.bind(window);
window.matchMedia = (q) => (/prefers-reduced-motion/.test(q)
  ? { matches: false, media: q, addListener(){}, removeListener(){}, addEventListener(){}, removeEventListener(){} }
  : realMM(q));
window.addEventListener('load', () => {
  const samples = [], errs = [];
  const say = () => { document.title = 'REVEAL ' + JSON.stringify({ door: '${door}', samples, errs: errs.slice(0,3) }); };
  window.addEventListener('error', (e) => { errs.push(String(e.message)); say(); });
  let clicked = false;
  const wait = setInterval(() => {
    const b = document.getElementById('${door}');
    if (!(window.BIMP && BIMP.renderer && BIMP.renderer.gl && BIMP.data && BIMP.data.n && b)) return;
    if (b.id === 'enter' && b.disabled) return;      // waits for the full tier to be live
    clearInterval(wait);
    clicked = true;
    try { b.click(); } catch (e) { errs.push('click threw: ' + (e && (e.stack || e.message))); }
    let n = 0;
    const poll = setInterval(() => {
      const R = BIMP.renderer, gl = R.gl;
      R.render();   // the context has no preserveDrawingBuffer — read in the same tick as a draw
      const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w*h*4);
      gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,px);
      let lit = 0;
      for (let i = 0; i < px.length; i += 4) if (px[i]>20||px[i+1]>24||px[i+2]>30) lit++;
      samples.push([R.getProgress ? +R.getProgress().toFixed(3) : -1, lit]);
      say();
      // keep going until the reveal front is well underway, so the pixel check is
      // measuring a running build rather than its first few cells
      if (++n >= 14 || (R.getProgress && R.getProgress() >= 0.45)) clearInterval(poll);
    }, 700);
  }, 150);
  setTimeout(() => { if (!clicked) { errs.push('never became clickable'); say(); } }, 12000);
});
</script>`;

// Served by python3 -m http.server: a hand-rolled node server wedged after serving the
// first page here, and the point of this test is the app, not the plumbing. The probe is
// written into pages/ as a temp file and removed again in the finally below.
const PORT = 8300 + (process.pid % 400);
const tmpFiles = [];
for (const door of (process.env.BIMP_DOORS ? process.env.BIMP_DOORS.split(",") : ["enter", "enter-tour"])) {
  const f = join(ROOT, `probe-${door}.html`);
  writeFileSync(f, readFileSync(join(ROOT, "index.html"), "utf8") + probe(door));
  tmpFiles.push(f);
}
const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"],
  { cwd: ROOT, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));

const fails = [];
const DOORS = process.env.BIMP_DOORS ? process.env.BIMP_DOORS.split(",") : ["enter", "enter-tour"];
try {
  for (const door of DOORS) {
    const dom = execFileSync(CHROMIUM, ["--headless", "--disable-gpu", "--no-sandbox",
      "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--virtual-time-budget=32000",
      "--window-size=420,320", "--dump-dom", `http://127.0.0.1:${PORT}/probe-${door}.html?nofull&gate`],
      { encoding: "utf8", maxBuffer: 1 << 26, stdio: ["ignore", "pipe", "ignore"] });
    const m = dom.match(/REVEAL (\{.*?\})<\/title>/s);
    const label = door === "enter" ? "guided tour" : "skip to atlas";
    if (!m) { fails.push(`${label}: probe never reported`); continue; }
    const { samples, errs } = JSON.parse(m[1].replace(/&quot;/g, '"'));
    const maxProg = samples.length ? Math.max(...samples.map((s) => s[0])) : 0;
    const maxLit = samples.length ? Math.max(...samples.map((s) => s[1])) : 0;
    console.log(`  ${label.padEnd(14)} progress peak ${maxProg}   lit peak ${maxLit}px   ` +
      `[${samples.map((s) => s[0] + ":" + s[1]).join(" ")}]`);
    if (errs.length) fails.push(`${label}: page errors ${JSON.stringify(errs)}`);
    if (!(maxProg > 0)) fails.push(`${label}: the build never started — progress stayed 0, canvas EMPTY`);
    if (!(maxLit > 20)) fails.push(`${label}: the build ran to ${maxProg} but only ${maxLit} pixels ever lit`);
  }
} finally {
  server.kill("SIGKILL");
  for (const f of tmpFiles) { try { unlinkSync(f); } catch {} }
}

if (fails.length) { for (const f of fails) console.error("✗ " + f); process.exit(1); }
console.log("✓ both intro doors reveal the atlas (build starts, pixels light up)");
process.exit(0);
