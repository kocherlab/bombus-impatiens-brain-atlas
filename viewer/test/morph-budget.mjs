/* What else is competing for the frame while a transition plays?
 *
 * A morph is the most expensive thing this viewer draws: for 0.55 s the vertex shader
 * evaluates BOTH view states for every point (shaders.js) — and until now nothing told the
 * rest of the app that the view was busy, because the CAMERA never moves during a morph.
 * So two things landed on exactly those frames:
 *
 *   1. hover picking — a whole extra 1.3M-point scene pass, every 90 ms
 *   2. full point density — the motion LOD watches camera speed, and saw none
 *
 * This asserts both are now stood down, and — the part that is easy to get wrong — that
 * they are stood down ONLY when they should be. Thinning a stationary colour dissolve
 * would be visible as sparkle, so the LOD drop is expected on a caste split (every point
 * moves) and NOT on a colour-mode change (nothing moves).
 *
 *   node viewer/test/morph-budget.mjs
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ART = process.env.BIMP_ART || join(APP, "dist", "bimp_brain.html");
const TMP = join(process.env.BIMP_TEST_TMP || "/tmp", "tmp");
const DRIVER = join(TMP, "morph-budget.html");
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
  s.spin = false;                       // a spinning camera would drive the LOD by itself
  await sleep(400);

  // Count pick passes by wrapping R.pick. The frame loop calls it through this same
  // reference, so the counter sees exactly what the loop spends.
  let picks = 0;
  const realPick = R.pick.bind(R);
  R.pick = (...a) => { picks++; return realPick(...a); };

  const r = cv.getBoundingClientRect();
  // Keep a pointer moving over the cloud: boot.js clears its pending pick after each one,
  // so a genuinely stationary pointer would stop asking and the counter would read zero
  // for reasons that have nothing to do with the morph.
  const move = (x, y) => cv.dispatchEvent(new PointerEvent('pointermove',
    { clientX: r.left + x, clientY: r.top + y, bubbles: true, pointerId: 1, pointerType: 'mouse', isPrimary: true }));

  // sample the LOD and the pick counter across a window, moving the pointer throughout
  async function watch(ms) {
    const lods = []; const p0 = picks;
    for (let t = 0; t < ms; t += 40) {
      move(r.width/2 + (t%80?3:-3), r.height/2);   // keep asking for a pick
      await sleep(40);
      lods.push(R.getLod ? R.getLod() : 1);
    }
    // atFloor: how much of the window sat AT the thinning floor. Min alone cannot tell a
    // camera flight (which dips through the floor and climbs straight back out) from a
    // morph pinning the cloud there for its whole duration — and only the second is a bug.
    const atFloor = lods.filter((v) => v <= 0.62).length / lods.length;
    return { picks: picks - p0, lodMin: Math.min(...lods), lodMax: Math.max(...lods),
             atFloor: +atFloor.toFixed(2), n: lods.length };
  }

  // --- baseline: settled view, nothing morphing. Picking must be alive here, or every
  //     "no picks during a morph" result below is vacuous.
  const idle = await watch(700);

  // --- a caste split: every point slides across the screen
  document.querySelector('#caste-seg [data-v="both"]').click();
  const split = await watch(520);        // inside MORPH_SECS (0.55 s)
  await sleep(700);                      // let it settle + the LOD ease back up
  const afterSplit = await watch(700);

  // --- a colour-mode change: the cloud is stationary, only the colours dissolve
  document.querySelector('#mode-seg [data-mode="conf"]').click();
  const recolor = await watch(520);
  await sleep(700);

  out({ idle, split, afterSplit, recolor });
 } catch (e) { out({ error: String(e && e.stack || e) }); }
})();
</script>`;

writeFileSync(DRIVER, readFileSync(ART, "utf8") + probe);
let dump = "";
try {
  dump = execFileSync(CHROMIUM, ["--headless", "--disable-gpu", "--use-gl=swiftshader",
    "--enable-unsafe-swiftshader", "--no-sandbox", "--window-size=1000,760",
    "--virtual-time-budget=60000", "--dump-dom", "file://" + DRIVER + "?nofull"],
    { encoding: "utf8", maxBuffer: 1 << 28 });
} catch (e) { dump = (e.stdout || "").toString(); }

const m = dump.match(/<pre id="probe">([\s\S]*?)<\/pre>/);
if (!m) { console.error("✗ probe never reported\n" + dump.slice(0, 1500)); process.exit(1); }
const r = JSON.parse(m[1]);
if (r.error) { console.error("✗ probe threw: " + r.error); process.exit(1); }

let fail = 0;
const check = (name, ok, detail) => { console.log(`${ok ? "✓" : "✗"} ${name}  ${detail}`); if (!ok) fail++; };
console.log("— what competes with a transition —");
console.log("  " + JSON.stringify(r) + "\n");

// non-vacuity first: if the pointer never provoked a pick when the view WAS settled,
// then "no picks during the morph" would be true of a viewer with picking ripped out.
check("picking is alive on a settled view", r.idle.picks > 0, `${r.idle.picks} picks in ${r.idle.n} samples`);
check("picking resumes after the morph ends", r.afterSplit.picks > 0, `${r.afterSplit.picks} picks`);
check("no pick pass lands during a caste split", r.split.picks === 0, `${r.split.picks} picks during the morph`);
check("no pick pass lands during a colour dissolve", r.recolor.picks === 0, `${r.recolor.picks} picks during the morph`);
// The LOD half. A morph must NOT pin the cloud to the thinning floor. That was tried:
// holding LOD_MIN for the length of a caste split measured as a 40% density drop at the
// click and a snap back as the brains landed — two brightness lurches around the very
// motion it was meant to smooth (test/split-smooth.mjs). Only the camera may thin the
// cloud, and it does so along its own eased flight.
check("a morph does not pin the cloud to the thinning floor", r.split.atFloor <= 0.35,
  `${(r.split.atFloor * 100).toFixed(0)}% of the split sat at the floor ` +
  `(the camera's pull-back may dip through it; a morph holding it there is the lurch that was reverted)`);
check("and it is back to full density once things land", r.afterSplit.lodMax >= 0.99,
  `lod returned to ${r.afterSplit.lodMax.toFixed(2)}`);
check("a stationary colour dissolve is NOT thinned", r.recolor.lodMin >= 0.99,
  `lod stayed at ${r.recolor.lodMin.toFixed(2)} (thinning a still cloud reads as sparkle)`);

console.log(fail ? `\n${fail} check(s) failed` : "\ntransitions get the frame to themselves");
process.exit(fail ? 1 : 0);
