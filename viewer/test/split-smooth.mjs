/* Is the queen→both slide EASED, or does it start and stop dead?
 *
 * The split is the viewer's most conspicuous motion: every point travels sideways while the
 * camera pulls back. The camera has always flown through easeInOutCubic. The slide did not
 * — uMorph went to the shader raw, and the morph clock advances linearly — so the brains
 * moved at constant speed and stopped instantly underneath a camera that was easing. Two
 * motions with different velocity profiles, one with no ease-in or ease-out at all, is what
 * reads as a jerk.
 *
 * Measuring it is fiddly for two reasons, both learned the hard way here:
 *
 *   - Wall-clock sampling cannot resolve a 550 ms morph in this environment. Under a
 *     virtual-time budget the app's own frame loop pumps the clock faster than any sleep in
 *     the probe, and the whole morph collapses into about three samples. So don't measure
 *     against time: read the raw morph clock in the SAME frame as the picture, and compare
 *     the two to each other.
 *   - The camera flight must be OFF. With it on, the pull-back moves the centre of mass far
 *     more than the slide does — measured, the centroid first ran backwards and then
 *     overshot the final displacement. Setting the caste filter on the state
 *     directly gives the same morph with a still camera.
 *
 * With the camera still, the centre of mass of one half is a clean proxy for how far the
 * slide has got, and the only question is which curve it follows against the clock. At
 * clock 0.25 a linear mix is 25% of the way across; easeInOutCubic is 6%.
 *
 *   node viewer/test/split-smooth.mjs
 *   BIMP_ART=/path/to/other.html node ...    # a comparison build
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ART = process.env.BIMP_ART || join(APP, "dist", "bimp_brain.html");
const TMP = join(process.env.BIMP_TEST_TMP || "/tmp", "tmp");
const DRIVER = join(TMP, "split-smooth.html");
const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
if (!CHROMIUM) { console.error("Chromium not found"); process.exit(1); }
if (!existsSync(ART)) { console.error("✗ not built — run: node viewer/build.mjs"); process.exit(1); }

const probe = `
<script>
(async () => {
 const out = (o) => { const p=document.createElement('pre'); p.id='probe'; p.textContent=JSON.stringify(o); document.body.appendChild(p); };
 try {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const s = BIMP.state.s, R = BIMP.renderer;
  const cv = document.getElementById('gl');
  for (let i=0;i<300 && !(R && R.N>0); i++) await sleep(50);
  R.setProgress(1);
  s.spin = false;                       // the idle orbit would move the centroid too
  await sleep(500);

  const gl = R.gl, W = cv.width, H = cv.height;
  const px = new Uint8Array(W * H * 4);
  function frame() {
    // Render and read in the SAME task: WebGL discards the drawing buffer once composited,
    // so a read that waits for the next tick comes back all zeros.
    R.render();
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let lit = 0, litL = 0, sxL = 0;
    for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) {
      const o = (y * W + x) * 4;
      if (px[o] + px[o+1] + px[o+2] > 46) {
        lit++;
        // Track the LEFT half only. The two brains move apart symmetrically, so the overall
        // centroid barely shifts while one side's moves by the full half-separation.
        if (x < W / 2) { litL++; sxL += x; }
      }
    }
    return { lit, cxL: litL ? sxL / litL : 0, litL,
             lod: R.getLod ? R.getLod() : 1, m: R.getMorph ? R.getMorph() : 1 };
  }

  const before = frame();
  // Set the state directly rather than clicking the Caste button: that also starts a 550 ms
  // camera flyTo, and the pull-back swamps the slide in any pixel measurement.
  s.casteFilter = -1;
  const seq = [];
  for (let i = 0; i < 80; i++) { const f = frame(); seq.push(f); if (f.m >= 1 && i > 4) break; await sleep(0); }
  await sleep(1500);
  const after = frame();
  out({ before, after, seq });
 } catch (e) { out({ error: String(e && e.stack || e) }); }
})();
</script>`;

writeFileSync(DRIVER, readFileSync(ART, "utf8") + probe);
let dump = "";
try {
  dump = execFileSync(CHROMIUM, ["--headless", "--disable-gpu", "--use-gl=swiftshader",
    "--enable-unsafe-swiftshader", "--no-sandbox", "--window-size=900,700",
    "--virtual-time-budget=90000", "--dump-dom", "file://" + DRIVER + "?nofull"],
    { encoding: "utf8", maxBuffer: 1 << 28 });
} catch (e) { dump = (e.stdout || "").toString(); }

const m = dump.match(/<pre id="probe">([\s\S]*?)<\/pre>/);
if (!m) { console.error("✗ probe never reported\n" + dump.slice(0, 1500)); process.exit(1); }
const r = JSON.parse(m[1]);
if (r.error) { console.error("✗ probe threw: " + r.error); process.exit(1); }

const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
const x0 = r.before.cxL, x1 = r.after.cxL, span = x1 - x0;
const pairs = r.seq.filter((f) => f.m > 0.02 && f.m < 0.98)
  .map((f) => ({ m: f.m, frac: (f.cxL - x0) / span }));
const err = (fn) => pairs.length ? pairs.reduce((a, p) => a + Math.abs(p.frac - fn(p.m)), 0) / pairs.length : NaN;
const errEased = err(ease), errLinear = err((u) => u);
const lits = r.seq.map((f) => f.lit);
const minLit = Math.min(...lits);
// biggest single-sample advance across the slide — a lurch shows up here
let jump = 0;
for (let i = 1; i < pairs.length; i++) jump = Math.max(jump, Math.abs(pairs[i].frac - pairs[i - 1].frac));

let fail = 0;
const check = (name, ok, detail) => { console.log(`${ok ? "✓" : "✗"} ${name}  ${detail}`); if (!ok) fail++; };
console.log("— queen → both, camera held still —");
console.log(`  left-half centroid ${x0.toFixed(1)} → ${x1.toFixed(1)} px (span ${span.toFixed(1)})`);
console.log("  clock→where " + pairs.map((p) => p.m.toFixed(2) + "→" + p.frac.toFixed(2)).join(" ") + "\n");

check("the split actually happened", Math.abs(span) > 12, `the left half's centroid moved ${span.toFixed(1)} px`);
check("no frame loses a big slice of the cloud", minLit >= r.before.lit * 0.75,
  `lit dipped to ${minLit}, started at ${r.before.lit}`);
// non-vacuity: with nothing sampled mid-morph the two curves cannot be told apart at all
check("the slide was caught mid-flight", pairs.length >= 4,
  `${pairs.length} samples with the clock strictly between 0 and 1`);
check("the slide is eased, not linear", errEased < errLinear * 0.6,
  `mean |error| vs easeInOutCubic ${errEased.toFixed(3)}, vs linear ${errLinear.toFixed(3)} ` +
  `— the camera flies eased over the same 550 ms, so a linear slide travels out of step with it`);
check("and it never lurches", jump <= 0.45, `worst single-sample advance ${(jump * 100).toFixed(0)}% of the slide`);

console.log(fail ? `\n${fail} check(s) failed` : "\nthe split eases in and out");
process.exit(fail ? 1 : 0);
