// Perf harness: headless Chrome over CDP (Node's built-in WebSocket).
//   node perf.mjs <label> [port]
// Prints JSON per scenario: rendered frames/s (R.render calls), rAF frame-time p50/p95,
// main-thread TaskDuration per second, CPU-seconds per second summed over all Chrome
// processes, JS heap, GPU-bound ms per render (render+finish), buffer estimate, network.
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { join } from "node:path";

const LABEL = process.argv[2] || "run";
const PORT = +(process.argv[3] || 9333);
const BASE = process.env.URL || "http://127.0.0.1:8765/";
// CHROME=/path/to/chrome overrides the default (Google Chrome on macOS).
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const prof = mkdtempSync(join(process.env.TMPDIR || "/tmp", "bimpperf-"));
const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`,
  "--window-size=1440,900", "--force-device-scale-factor=2", "--no-first-run",
  "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
  ...(process.env.SW ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : ["--use-angle=metal", "--enable-gpu"]), "--ignore-gpu-blocklist", "about:blank"], { stdio: "ignore" });
setInterval(() => {}, 1000);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
process.on("exit", () => { try { chrome.kill("SIGKILL"); } catch (e) { /* gone */ } });

async function j(path) { for (let i = 0; i < 100; i++) { try { return await (await fetch(`http://127.0.0.1:${PORT}${path}`)).json(); } catch (e) { await sleep(100); } } throw new Error("no cdp"); }
function client(url) {
  const ws = new WebSocket(url); let id = 0; const wait = new Map(); const evs = [];
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && wait.has(d.id)) { const w = wait.get(d.id); wait.delete(d.id); d.error ? w.rej(new Error(JSON.stringify(d.error))) : w.res(d.result); } else evs.push(d); };
  const ready = new Promise((r) => { ws.onopen = r; });
  const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; wait.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
  return { ready, send, evs, close: () => ws.close() };
}
const ver = await j("/json/version");
const B = client(ver.webSocketDebuggerUrl); await B.ready;
const tabs = await j("/json/list");
const P = client(tabs.find((t) => t.type === "page").webSocketDebuggerUrl); await P.ready;
await P.send("Page.enable"); await P.send("Runtime.enable"); await P.send("Performance.enable"); await P.send("Network.enable");
const ev = async (expr) => { const r = await P.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 400)); return r.result.value; };
import { execSync } from "node:child_process";
function cpu() { // CPU seconds over the whole Chrome process tree (browser, renderer, GPU)
  const rows = execSync("ps -A -o pid=,ppid=,time=").toString().trim().split("\n").map((l) => l.trim().split(/\s+/));
  const kids = new Map(); rows.forEach(([p, pp]) => { if (!kids.has(pp)) kids.set(pp, []); kids.get(pp).push(p); });
  const tree = new Set([String(chrome.pid)]); const st = [String(chrome.pid)]; while (st.length) { const p = st.pop(); for (const k of kids.get(p) || []) if (!tree.has(k)) { tree.add(k); st.push(k); } }
  const sec = (t) => { const [a, b] = t.split("."); const parts = a.split(":").map(Number); let v = 0; parts.forEach((x) => { v = v * 60 + x; }); return v + (b ? +("0." + b) : 0); };
  return rows.filter(([p]) => tree.has(p)).reduce((a, r) => a + sec(r[2]), 0); }
function rssMB() { const rows = execSync("ps -A -o pid=,rss=").toString().trim().split("\n").map((l) => l.trim().split(/\s+/)); const all = new Map(rows); let t = 0; const r2 = execSync("ps -A -o pid=,ppid=").toString().trim().split("\n").map((l) => l.trim().split(/\s+/)); const tree = new Set([String(chrome.pid)]); let ch = true; while (ch) { ch = false; for (const [p, pp] of r2) if (tree.has(pp) && !tree.has(p)) { tree.add(p); ch = true; } } for (const p of tree) t += +(all.get(p) || 0); return Math.round(t / 1024); }
async function metrics() { const r = await P.send("Performance.getMetrics"); const o = {}; r.metrics.forEach((m) => { o[m.name] = m.value; }); return o; }

// in-page probe: wrap R.render to count, rAF intervals
const PROBE = `(() => { if (window.__pp) return; const R = BIMP.renderer; const P = window.__pp = { renders: 0, ft: [] };
  const real = R.render; R.render = function () { P.renders++; return real.apply(this, arguments); };
  let last = 0; const loop = (t) => { if (last) P.ft.push(t - last); last = t; requestAnimationFrame(loop); }; requestAnimationFrame(loop); })()`;
async function window_(name, secs, during) {
  await ev(PROBE); await ev("__pp.renders = 0; __pp.ft = []; 0");
  const c0 = await cpu(), m0 = await metrics(), t0 = Date.now();
  if (during) await during(secs); else await sleep(secs * 1000);
  const c1 = await cpu(), m1 = await metrics(), dt = (Date.now() - t0) / 1000;
  const r = await ev("(() => { const f = __pp.ft.slice().sort((a,b)=>a-b); const q = (p) => f.length ? +f[Math.floor(p*(f.length-1))].toFixed(1) : null; return { renders: __pp.renders, raf: f.length, p50: q(0.5), p95: q(0.95) }; })()");
  const o = { scenario: name, rendersPerSec: +(r.renders / dt).toFixed(1), rafPerSec: +(r.raf / dt).toFixed(1), ftP50: r.p50, ftP95: r.p95,
    cpuPerSec: +((c1 - c0) / dt).toFixed(2), mainTaskPerSec: +((m1.TaskDuration - m0.TaskDuration) / dt).toFixed(3), rssMB: rssMB() };
  console.log(JSON.stringify(o));
  if (process.env.DIAG) console.log('EXC', name, JSON.stringify(P.evs.filter((e) => e.method === 'Runtime.exceptionThrown' || (e.method === 'Runtime.consoleAPICalled' && e.params.type === 'error')).map((e) => JSON.stringify(e.params).slice(0, 700))));
  if (process.env.DIAG) console.log('DIAG', name, JSON.stringify(await ev("(() => { const R = BIMP.renderer, gl = R.gl, cv = gl.canvas, L = R.look, cs = getComputedStyle(cv); const c = document.elementFromPoint(innerWidth/2, innerHeight/2); return { cw: cv.width, ch: cv.height, ccw: cv.clientWidth, cch: cv.clientHeight, prog: R.getProgress(), N: R.N, alpha: L.alpha, ink: L.ink, rect: L.rect, bg: L.bg, sizeMul: L.sizeMul, view: document.documentElement.dataset.view, vis: cs.visibility, op: cs.opacity, disp: cs.display, top: c && (c.id || c.className), err: gl.getError(), dpr: R.getDpr(), lod: R.getLod(), morph: R.getMorph(), s: { mode: BIMP.state.s.mode, opacity: BIMP.state.s.opacity, size: BIMP.state.s.size, caste: BIMP.state.s.casteFilter, vis0: BIMP.state.s.visible[0] } }; })()")));
  if (process.env.SHOT) { const { writeFileSync } = await import('node:fs'); const r = await P.send('Page.captureScreenshot', { format: 'png' }); writeFileSync(`${process.env.SHOT}-${name.replace(/[^a-z0-9]+/gi, '_')}.png`, Buffer.from(r.data, 'base64')); }
  return o;
}
// GPU-bound cost of one render at the current state: render + gl.finish, median of 15
const GPU = `(() => { const R = BIMP.renderer, gl = R.gl; const t = []; for (let i = 0; i < 5; i++) { const a = performance.now(); R.render(); gl.finish(); t.push(performance.now() - a); } t.sort((a,b)=>a-b); return { gpuMsPerRender: +t[2].toFixed(1), canvas: gl.canvas.width + 'x' + gl.canvas.height, N: R.N, aa: gl.getContextAttributes().antialias, samples: gl.getParameter(gl.SAMPLES) }; })()`;
const DRAG = (secs) => ev(`new Promise((res) => { const cv = document.getElementById('gl'), r = cv.getBoundingClientRect(); const x0 = r.left + r.width/2, y0 = r.top + r.height/2;
  const pe = (t, x, y) => cv.dispatchEvent(new PointerEvent(t, { clientX: x, clientY: y, bubbles: true, pointerId: 7, pointerType: 'mouse', isPrimary: true, button: 0, buttons: t==='pointerup'?0:1 }));
  pe('pointerdown', x0, y0); let k = 0; const end = performance.now() + ${secs * 1000};
  const go = () => { k++; pe('pointermove', x0 + 120*Math.sin(k/20), y0 + 40*Math.cos(k/30)); if (performance.now() < end) requestAnimationFrame(go); else { pe('pointerup', x0, y0); res(k); } }; requestAnimationFrame(go); })`);
const HOVER = (secs) => ev(`new Promise((res) => { const cv = document.getElementById('gl'), r = cv.getBoundingClientRect(); let k = 0; const end = performance.now() + ${secs * 1000};
  const go = () => { k++; cv.dispatchEvent(new PointerEvent('pointermove', { clientX: r.left + r.width/2 + 30*Math.sin(k/15), clientY: r.top + r.height/2, bubbles: true, pointerId: 9, pointerType: 'mouse' })); if (performance.now() < end) requestAnimationFrame(go); else res(k); }; requestAnimationFrame(go); })`);


const SWEEP = process.argv[4] === "sweep";
const GPU2 = (dpr) => `(() => { Object.defineProperty(window, 'devicePixelRatio', { value: ${dpr}, configurable: true }); const R = BIMP.renderer, gl = R.gl; R.resize(gl.canvas); BIMP.state.s.spin = false;
  const px = new Uint8Array(4); const one = () => { R.render(); gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px); };
  one(); one(); const t = []; for (let i = 0; i < 15; i++) { const a = performance.now(); one(); t.push(performance.now() - a); } t.sort((a,b)=>a-b);
  return { dpr: ${dpr}, canvas: gl.canvas.width + 'x' + gl.canvas.height, msPerRender: +t[7].toFixed(1) }; })()`;
if (SWEEP) {
  if (process.env.NOAA) await P.send("Page.addScriptToEvaluateOnNewDocument", { source: "const g0 = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function (t, o) { if (t === 'webgl2') o = Object.assign({}, o, { antialias: false }); return g0.call(this, t, o); };" });
  await P.send("Page.navigate", { url: BASE + "?nointro" });
  for (let i = 0; i < 240; i++) { await sleep(500); try { const st = await ev("JSON.stringify([!!window.BIMP, !!(window.BIMP&&BIMP.fullReady), window.BIMP&&BIMP.renderer&&BIMP.renderer.N, window.BIMP&&BIMP.renderer&&BIMP.renderer.getProgress()])"); if (i % 10 == 0) console.log(i, st); if (await ev("!!(window.BIMP && BIMP.fullReady && BIMP.renderer.N > 1e6 && BIMP.renderer.getProgress() > 0.999)")) break; } catch (e) { console.log('err', e.message.slice(0,100)); } }
  console.log('ready', await ev('BIMP.renderer.N')); await sleep(2000);
  for (const d of [2, 1.5, 1.25, 1]) console.log(JSON.stringify(await ev(GPU2(d))));
  // hover-pick cost: one pick
  console.log(JSON.stringify(await ev(`(() => { const R = BIMP.renderer; const t = []; for (let i = 0; i < 6; i++) { const a = performance.now(); R.pick(700, 400, 4); t.push(performance.now() - a); } t.sort((a,b)=>a-b); return { pickMs: +t[3].toFixed(1) }; })()`)));
  process.exit(0);
}

if (process.argv[4] === "checks") {
  await P.send("Page.navigate", { url: BASE + "?nointro" });
  for (let i = 0; i < 240; i++) { await sleep(500); try { if (await ev("!!(window.BIMP && BIMP.fullReady && BIMP.renderer.N > 1e6 && BIMP.renderer.getProgress() > 0.999)")) break; } catch (e) { /* loading */ } }
  await ev(PROBE); await ev("BIMP.state.s.spin = false; document.getElementById('tg-spin') && document.getElementById('tg-spin').setAttribute('aria-pressed','false'); 0");
  const st = await ev("BIMP.selfTest ? BIMP.selfTest().then(r => ({ ok: r.ok, fails: r.checks.filter(c => !c.pass).map(c => c.name + ' ' + c.info) })) : 'none'");
  console.log("selfTest", JSON.stringify(st));
  const after = async (label, js, wait = 1500) => { await sleep(2500); await ev("__pp.renders = 0; 0"); await ev(js); await sleep(wait); const n = await ev("__pp.renders"); console.log(label.padEnd(34), "renders after:", n); };
  await after("nothing (asleep)", "0");
  await after("click Gene mode", "document.querySelector('#mode-seg [data-mode=gene]').click(); 0");
  await after("click Type mode", "document.querySelector('#mode-seg [data-mode=celltype]').click(); 0");
  await after("click Worker caste", "document.querySelector('#caste-seg [data-v=worker]').click(); 0");
  await after("click Queen caste", "document.querySelector('#caste-seg [data-v=queen]').click(); 0");
  await after("state change w/o event (s.size)", "BIMP.state.s.size = 3.2; 0");
  await after("camera.reset via R key", "window.dispatchEvent(new KeyboardEvent('keydown', { key: 'r' })); 0");
  await after("camera.flyTo (no event)", "BIMP.camera.flyTo({ theta: 2.0 }, 600); 0");
  await after("wheel zoom", "document.getElementById('gl').dispatchEvent(new WheelEvent('wheel', { deltaY: 100, deltaMode: 1, bubbles: true, cancelable: true })); 0");
  await after("spin toggle on", "document.getElementById('tg-spin').click(); 0", 1000);
  await ev("document.getElementById('tg-spin').click(); 0");
  await after("click w/ unlisted state", "document.body.addEventListener('click', () => { BIMP.state.s.zzz = 1; }, { once: true }); document.body.click(); 0");
  await after("render throws once, then spin", "(() => { const R = BIMP.renderer, r1 = R.render; let thrown = false; R.render = function () { if (!thrown) { thrown = true; throw new Error('injected'); } return r1.apply(this, arguments); }; BIMP.state.s.size = 2.9; setTimeout(() => { BIMP.state.s.spin = true; }, 100); return 0; })()", 3000);
  await ev("BIMP.state.s.spin = false; 0");
  await after("tour start", "BIMP.tour && BIMP.tour.start ? (BIMP.tour.start(), 0) : 0", 2000);
  console.log("hud fps:", await ev("document.getElementById('hud-fps').textContent"));
  process.exit(0);
}
if (process.env.AAON) await P.send("Page.addScriptToEvaluateOnNewDocument", { source: "const g1 = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function (t, o) { if (t === 'webgl2') o = Object.assign({}, o, { antialias: true }); return g1.call(this, t, o); };" });
const out = { label: LABEL, scen: [] };
// ---- home page ----
await P.send("Page.navigate", { url: BASE });
let homeOk = false; for (let i = 0; i < 40 && !homeOk; i++) { await sleep(500); homeOk = await ev("!!(window.BIMP && BIMP.renderer && BIMP.renderer.N > 0)").catch(() => false); }
await sleep(3000);
if (homeOk) out.scen.push(await window_("home hero idle", 5)); else console.log("home: renderer never loaded");
// ---- atlas ----
await P.send("Network.clearBrowserCache");
let bytes = 0; const reqs = [];
const evStart = P.evs.length;
await P.send("Page.navigate", { url: BASE + "?nointro" });
for (let i = 0; i < 240; i++) { await sleep(500); try { if (await ev("!!(window.BIMP && BIMP.fullReady && BIMP.renderer.N > 1e6 && BIMP.renderer.getProgress() > 0.999)")) break; } catch (e) { /* loading */ } }
if (process.env.FORCE) await ev('BIMP.renderer.needsFrame = () => true; 0');
await sleep(2500);
for (const e of P.evs.slice(evStart)) if (e.method === "Network.loadingFinished") bytes += e.params.encodedDataLength;
out.netMB = +(bytes / 1048576).toFixed(1);
console.log(JSON.stringify({ netMBOnEnter: out.netMB }));
console.log(JSON.stringify({ gl: await ev("(() => { const gl = BIMP.renderer.gl; const d = gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'n/a'; })()") }));
out.gpu = await ev(GPU2(2)); console.log(JSON.stringify(out.gpu));
await ev("BIMP.state.s.spin = true; 0");
out.scen.push(await window_("atlas idle, spin on", 5));
await ev("BIMP.state.s.spin = false; 0"); await sleep(800);
out.scen.push(await window_("atlas idle, spin off", 5));
console.log(JSON.stringify({ anims: await ev("document.getAnimations().map(a => (a.animationName || a.constructor.name) + ':' + (a.effect && a.effect.target ? (a.effect.target.id || a.effect.target.className) : '') + ':' + a.playState).slice(0, 20)") }));
await sleep(5000);
out.scen.push(await window_("atlas idle, spin off, +5s", 5));
if (process.env.ONLYIDLE) process.exit(0);
out.scen.push(await window_("atlas hover moving", 4, HOVER));
out.scen.push(await window_("atlas dragging", 4, DRAG));
console.log(JSON.stringify({ dprAfterDrag: await ev("BIMP.renderer.getDpr ? BIMP.renderer.getDpr() : -1"), canvas: await ev("BIMP.renderer.gl.canvas.width") }));
await ev("document.querySelector('#mode-seg [data-mode=gene]').click(); 0").catch(()=>0); await sleep(4000);
out.scen.push(await window_("gene mode, spin off", 5));
await ev("document.querySelector('#mode-seg [data-mode=celltype]').click(); 0"); await sleep(1000);
const hav = await ev("!!(BIMP.histology && BIMP.histology.available)"); console.log(JSON.stringify({ hav, h: await ev("JSON.stringify({ has: !!BIMP.histology, keys: BIMP.histology ? Object.keys(BIMP.histology) : 0 })") }));
if (hav) {
  await ev("BIMP.histology.setOn(true); 0"); for (let i = 0; i < 60; i++) { await sleep(500); const st = await ev("BIMP.histology.stats()"); if (i > 6 && st.queued === 0 && st.loading === 0) break; } await sleep(1500);
  out.scen.push(await window_("histology on, spin off", 5));
  if (process.env.INSTR) await ev((await import("node:fs")).readFileSync("instr.js", "utf8"));
  await ev("BIMP.histology.enter2D(); 0"); await sleep(3000);
  console.log(JSON.stringify(await ev("(() => { const o = {}; try { o.anim = BIMP.histology.isAnimating(); } catch (e) { o.animErr = String(e.stack).slice(0, 300); } try { o.nf = BIMP.renderer.needsFrame(); } catch (e) { o.nfErr = String(e.stack).slice(0, 300); } return o; })()")));
  console.log(JSON.stringify(P.evs.filter((e) => e.method === "Runtime.exceptionThrown").slice(0, 3).map((e) => JSON.stringify(e.params.exceptionDetails).slice(0, 600))));
  if (process.env.INSTR) console.log(JSON.stringify({ px2d: await ev("(() => { const gl = BIMP.renderer.gl; const p = new Uint8Array(4 * 400); gl.readPixels(gl.canvas.width/2 - 200, gl.canvas.height/2, 400, 1, gl.RGBA, gl.UNSIGNED_BYTE, p); let lit = 0; for (let i = 0; i < p.length; i += 4) if (p[i] + p[i+1] + p[i+2] > 90) lit++; return { lit, renders: __pp.renders, err: gl.getError(), lost: gl.isContextLost(), cam: JSON.stringify(BIMP.camera.cam), slab: BIMP.state.s.slab }; })()") }));
  out.scen.push(await window_("2D section view", 5));
  console.log(JSON.stringify({ histo: await ev("BIMP.histology.stats()") }));
  await ev("BIMP.histology.exit2D(); BIMP.histology.setOn(false); 0");
}
console.log(JSON.stringify({ gl: await ev("(() => { const gl = BIMP.renderer.gl; const d = gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'n/a'; })()") }));
P.close(); B.close(); chrome.kill("SIGKILL"); process.exit(0);
