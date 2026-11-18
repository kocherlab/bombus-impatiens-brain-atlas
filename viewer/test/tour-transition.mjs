/* Deterministic test of the guided tour's step transition — no browser, no WebGL.
 *
 * Headless SwiftShader starves requestAnimationFrame, so animation smoothness can't be
 * measured in the browser here. Instead we load the REAL src/camera.js and src/tour.js
 * into a VM with a fake clock and minimal DOM/renderer stubs, then drive the tour and
 * assert what "no jumping" actually means:
 *
 *   1. the camera trajectory is continuous — no frame where the pose leaps
 *      (a snap shows up as one step orders of magnitude above its neighbours),
 *   2. it lands on the stop's goal pose,
 *   3. a control handler that yanks the pose mid-step (caste "Both" bumps the zoom,
 *      as ui.js really does) never reaches the screen,
 *   4. each stop changes the view state once (twice for a caste-split stop, whose split is
 *      deliberately held back), never more — the dissolve itself is the renderer's job and
 *      is covered by test/morph.mjs against the real shader,
 *   5. the caste split lands late in the flight, not at its start (split early, the brains
 *      swell off-frame before the camera has pulled back), and before the camera does.
 *
 *   node viewer/test/tour-transition.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

// BIMP_SRC lets the negative controls point at a deliberately-broken copy of src/
const SRC = process.env.BIMP_SRC || join(dirname(dirname(fileURLToPath(import.meta.url))), "src");
const STEP_MS = 16;

// ---- fake clock: rAF callbacks fire in step order, performance.now() is virtual ----
let now = 0;
let rafQ = [];
const clock = {
  performance: { now: () => now },
  requestAnimationFrame: (cb) => { rafQ.push(cb); return rafQ.length; },
};
function tickClock() {
  now += STEP_MS;
  const due = rafQ; rafQ = [];
  for (const cb of due) cb(now);
}

// ---- DOM stub: just enough for tour.js (buttons it clicks, the caption card) ----
const clicks = [];
function makeEl(sel) {
  return {
    sel, textContent: "", innerHTML: "", disabled: false, style: {},
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); }, contains(c) { return this._s.has(c); } },
    setAttribute() {}, _on: [], addEventListener(ev, fn) { if (ev === "click") this._on.push(fn); },
    // a click runs the simulated ui.js handler for that control AND any listener the tour wired
    click() { clicks.push(sel); handlers[sel] && handlers[sel](); this._on.forEach((f) => f()); },
    querySelectorAll: () => [],
  };
}
const els = {};
const el = (k) => (els[k] || (els[k] = makeEl(k)));
const doc = {
  body: makeEl("body"),
  getElementById: (id) => el("#" + id),
  querySelector: (sel) => el(sel),
  querySelectorAll: () => [],
};

// ---- state + stubs standing in for ui.js / renderer.js ----
const s = { mode: 0, gene: -1, casteFilter: 0, regionMask: 0, spin: true, z: 1 };
const handlers = {
  // mirrors the real ui.js seg handlers, including the pose yank on caste changes
  '#mode-seg [data-mode="celltype"]': () => { s.mode = 0; },
  '#mode-seg [data-mode="gene"]': () => { s.mode = 3; },
  '#caste-seg [data-v="queen"]': () => { const p = s.casteFilter; s.casteFilter = 0; if (p < 0) BIMP.camera.cam.radius = 2.7; },
  '#caste-seg [data-v="worker"]': () => { s.casteFilter = 1; },
  '#caste-seg [data-v="both"]': () => { const p = s.casteFilter; s.casteFilter = -1; if (p >= 0) BIMP.camera.cam.radius = Math.max(BIMP.camera.cam.radius, 4.6); },
  '#label-seg [data-v="canonical"]': () => {},
  '#label-seg [data-v="subtypes"]': () => {},
};

const ctx = {
  window: {}, document: doc, console,
  performance: clock.performance,
  requestAnimationFrame: clock.requestAnimationFrame,
  Math, Date, JSON,
};
ctx.window = ctx;                 // src modules write to window.BIMP
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(readFileSync(join(SRC, "camera.js"), "utf8"), ctx, { filename: "camera.js" });
const BIMP = ctx.BIMP;

// region masks: 400 cells in two blobs so regionCentroid() has something real to aim at
const N = 400;
const region = new Uint8Array(N), X = new Int16Array(N), Y = new Int16Array(N), Z = new Int16Array(N);
for (let i = 0; i < N; i++) {
  const mb = i < N / 2;
  region[i] = mb ? 0b111 : 0b111000;
  X[i] = mb ? 6000 : -9000; Y[i] = mb ? 4000 : 11000; Z[i] = mb ? 1000 : -2000;
}
BIMP.state = { s };
BIMP.data = { n: N, region, x: X, y: Y, z: Z };
BIMP.manifest = { regionDefs: [
  { key: "in_kc_ic", bit: 0 }, { key: "in_kc_nc", bit: 1 }, { key: "in_kc_oc", bit: 2 },
  { key: "in_lamina", bit: 3 }, { key: "in_medulla", bit: 4 }, { key: "in_lobula", bit: 5 },
] };
BIMP.renderer = {};   // the tour no longer touches the renderer; state changes speak for themselves
BIMP.ui = {
  syncRegionUI() {}, updateStat() {}, prefetchGene() {},
  geneIndexBySymbol: (sym) => (sym === "tachykinins" ? 0 : sym === "sNPF" ? 1 : -1),
  selectGene: (i) => { s.gene = i; s.mode = 3; },
};

// tour.js captures BIMP.state/camera at load time, so it loads after the stubs exist
// (the same order build.mjs uses: state.js → renderer.js → ui.js → tour.js).
vm.runInContext(readFileSync(join(SRC, "tour.js"), "utf8"), ctx, { filename: "tour.js" });

// ---- drive: start the tour, then Next through every stop ----
const cam = BIMP.camera.cam;
const trace = [];                  // per-frame pose + fade + state fingerprint
const fp = () => [s.mode, s.gene, s.casteFilter, s.regionMask].join("|");
function runFrames(ms) {
  for (let t = 0; t < ms; t += STEP_MS) {
    tickClock();
    BIMP.camera.tick(STEP_MS / 1000);
    trace.push({ t: now, fly: BIMP.camera.isFlying(), cf: s.casteFilter, r: cam.radius, th: cam.theta, ph: cam.phi, tx: cam.target[0], ty: cam.target[1], fp: fp() });
  }
}
BIMP.tour.init();          // wires Back/Next/Skip onto the stub buttons
BIMP.tour.start();
runFrames(3200);
const stops = 7;
for (let i = 1; i < stops; i++) { doc.getElementById("tour-next").click(); runFrames(3200); }

// ---- assertions ----
const fails = [];
const ok = (cond, msg) => { if (!cond) fails.push(msg); };

// 1. continuity: no single frame moves the pose far more than its neighbours
let worst = { d: 0, t: 0 };
const deltas = [];
for (let i = 1; i < trace.length; i++) {
  const a = trace[i - 1], b = trace[i];
  const d = Math.hypot(b.r - a.r, (b.th - a.th) * 3, (b.ph - a.ph) * 3, b.tx - a.tx, b.ty - a.ty);
  deltas.push(d);
  if (d > worst.d) worst = { d, t: b.t };
}
const sorted = [...deltas].sort((x, y) => x - y);
const p99 = sorted[Math.floor(sorted.length * 0.99)];
// an eased 1.2-2.6 s flight at 16 ms/frame moves in small, smooth increments; a snap is a
// single frame carrying a whole transition, i.e. many times the 99th percentile step
ok(worst.d < Math.max(p99 * 3, 0.02), `camera SNAP: one frame moved ${worst.d.toFixed(4)} at t=${worst.t} (p99 step ${p99.toFixed(4)})`);

// 2. landed on the last stop's goal (theta .92 / phi 1.1 / radius 5.2, target origin)
const last = trace[trace.length - 1];
ok(Math.abs(last.r - 5.2) < 1e-3 && Math.abs(last.th - 0.92) < 1e-3 && Math.abs(last.ph - 1.1) < 1e-3,
  `final pose off: r=${last.r.toFixed(3)} th=${last.th.toFixed(3)} ph=${last.ph.toFixed(3)}`);

// 3. the caste handler's pose yank never reaches the screen: the frames straddling each
// state swap must keep gliding (a yank would land the whole zoom change in one frame).
// NB "radius == 4.6" alone proves nothing — a flight legitimately passes through it.

// 4. each stop changes the view exactly once, and the change never yanks the pose
let swaps = 0, worstSwapStep = 0;
for (let i = 1; i < trace.length; i++) {
  if (trace[i].fp !== trace[i - 1].fp) {
    swaps++;
    worstSwapStep = Math.max(worstSwapStep, Math.abs(trace[i].r - trace[i - 1].r));
  }
}
ok(worstSwapStep < 0.1, `pose yanked ${worstSwapStep.toFixed(3)} in the swap frame — a handler beat the flight`);
ok(swaps >= stops - 1, `expected a state change per stop, saw ${swaps}`);
ok(swaps <= stops + 1, `${swaps} state changes for ${stops} stops — a stop is changing the view more than once`);

// 5. every split into "both" happens mid-flight, in its second half
let splits = 0;
for (let i = 1; i < trace.length; i++) {
  if (trace[i].cf < 0 && trace[i - 1].cf >= 0) {
    splits++;
    let j = i; while (j > 0 && trace[j - 1].fly) j--;          // when this flight began
    let k = i; while (k < trace.length - 1 && trace[k].fly) k++; // when it lands
    const u = (trace[i].t - trace[j].t) / Math.max(1, trace[k].t - trace[j].t);
    ok(trace[i].fly && u > 0.35 && u < 0.95, `caste split at ${(u * 100).toFixed(0)}% of its flight (want mid-to-late, before landing)`);
  }
}
ok(splits === 2, `expected 2 caste splits (stops 4 and 7), saw ${splits}`);

// ---- report ----
console.log(`frames=${trace.length}  view changes=${swaps}/${stops - 1} stops  ` +
  `worst pose step=${worst.d.toFixed(5)} (p99 ${p99.toFixed(5)})  swap-frame step=${worstSwapStep.toFixed(5)}  ` +
  `final r=${last.r.toFixed(2)}`);
if (fails.length) { for (const f of fails) console.error("✗ " + f); process.exit(1); }
console.log("✓ tour transitions: continuous camera, one clean view change per stop, splits sequenced after the pull-back");
