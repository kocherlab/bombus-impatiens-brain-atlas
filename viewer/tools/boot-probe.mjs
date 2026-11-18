// Boot / loading probe: load a page N times in headless Chrome and report, per load,
// time to first paint of cells and to "fully revealed", long tasks, peak JS heap, the
// worst rAF gaps, and every exception or console error.
//   node tools/boot-probe.mjs [--url=...] [--n=5] [--cpu=4] [--mobile] [--net=slow4g|fast3g]
//                             [--sw] [--port=9500] [--wait=60]
// Needs the preview server on 127.0.0.1:8765 (node tools/serve.mjs pages).
import { spawn, execSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { join } from "node:path";

const A = Object.fromEntries(process.argv.slice(2).map((a) => { const s = a.replace(/^--/, ""), i = s.indexOf("="); return i < 0 ? [s, true] : [s.slice(0, i), s.slice(i + 1)]; }));
const URL0 = A.url || "http://127.0.0.1:8765/?nointro";
const N = +(A.n || 5), PORT = +(A.port || 9500), WAIT = +(A.wait || 60) * 1000;
// CHROME=/path/to/chrome overrides the default (Google Chrome on macOS).
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const prof = mkdtempSync(join(process.env.TMPDIR || "/tmp", "bimpperf-"));
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`,
  A.mobile ? "--window-size=390,844" : "--window-size=1440,900", "--no-first-run",
  ...(A.sw ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : ["--use-angle=metal", "--enable-gpu"]), "about:blank"], { stdio: "ignore" });
process.on("exit", () => { try { chrome.kill("SIGKILL"); } catch (e) { /* gone */ } });
setInterval(() => {}, 1000);
// resident memory of the whole Chrome tree (renderer + GPU + browser), MB: counts ArrayBuffers,
// which the JS heap metric does not
function rssMB() {
  const rows = execSync("ps -A -o pid=,ppid=,rss=").toString().trim().split("\n").map((l) => l.trim().split(/\s+/));
  const tree = new Set([String(chrome.pid)]); let ch = true;
  while (ch) { ch = false; for (const [p, pp] of rows) if (tree.has(pp) && !tree.has(p)) { tree.add(p); ch = true; } }
  return Math.round(rows.filter(([p]) => tree.has(p)).reduce((a, r) => a + +r[2], 0) / 1024);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function j(path) { for (let i = 0; i < 100; i++) { try { return await (await fetch(`http://127.0.0.1:${PORT}${path}`)).json(); } catch (e) { await sleep(100); } } throw new Error("no cdp"); }
function client(url) {
  const ws = new WebSocket(url); let id = 0; const wait = new Map(); const evs = [];
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && wait.has(d.id)) { const w = wait.get(d.id); wait.delete(d.id); d.error ? w.rej(new Error(JSON.stringify(d.error))) : w.res(d.result); } else evs.push(d); };
  const ready = new Promise((r) => { ws.onopen = r; });
  const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; wait.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
  return { ready, send, evs };
}
const tabs = await j("/json/list");
const P = client(tabs.find((t) => t.type === "page").webSocketDebuggerUrl); await P.ready;
await P.send("Page.enable"); await P.send("Runtime.enable"); await P.send("Performance.enable"); await P.send("Network.enable");
const ev = async (expr) => { const r = await P.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 300)); return r.result.value; };
if (A.mobile) await P.send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 3, mobile: true });
if (A.mobile) await P.send("Emulation.setTouchEmulationEnabled", { enabled: true, maxTouchPoints: 5 });
if (A.cpu) await P.send("Emulation.setCPUThrottlingRate", { rate: +A.cpu });
const NET = { slow4g: { latency: 150, downloadThroughput: 1.6e6 / 8 * 1.0, uploadThroughput: 750e3 / 8 }, mid: { latency: 40, downloadThroughput: 2.5e6, uploadThroughput: 1e6 }, fast3g: { latency: 563, downloadThroughput: 1.44e6 / 8, uploadThroughput: 675e3 / 8 } };
if (A.net) await P.send("Network.emulateNetworkConditions", { offline: false, ...NET[A.net] });
await P.send("Network.setCacheDisabled", { cacheDisabled: !A.cache });
if (A.block) await P.send("Network.setBlockedURLs", { urls: String(A.block).split(",") });
if (A.nowebgl) await P.send("Page.addScriptToEvaluateOnNewDocument", { source: "const g0 = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function (t, o) { return t === 'webgl2' ? null : g0.call(this, t, o); };" });
// in-page recorder, installed before any page script
await P.send("Page.addScriptToEvaluateOnNewDocument", { source: `(() => {
  const L = window.__boot = { long: [], ft: [], firstCells: 0, revealed: 0, fullReady: 0 };
  try { new PerformanceObserver((l) => l.getEntries().forEach((e) => L.long.push([Math.round(e.startTime), Math.round(e.duration)]))).observe({ type: 'longtask', buffered: true }); } catch (e) {}
  let last = 0; const loop = (t) => { if (last) L.ft.push([Math.round(t), +(t - last).toFixed(1)]); last = t;
    const B = window.BIMP, R = B && B.renderer;
    if (R && R.getProgress && !L.firstCells && R.N > 0 && R.getProgress() > 0.02) L.firstCells = Math.round(t);
    if (R && R.getProgress && !L.revealed && B.fullReady && R.getProgress() > 0.999) L.revealed = Math.round(t);
    if (B && B.fullReady && !L.fullReady) L.fullReady = Math.round(t);
    requestAnimationFrame(loop); }; requestAnimationFrame(loop);
})()` });

const rows = [];
for (let k = 0; k < N; k++) {
  const e0 = P.evs.length;
  if (A.profile === "load") { await P.send("Profiler.enable"); await P.send("Profiler.setSamplingInterval", { interval: 200 }); await P.send("Profiler.start"); }
  await P.send("Page.navigate", { url: URL0 + (URL0.includes("?") ? "&" : "?") + "r=" + k });
  let peak = 0, rssPeak = 0, done = false; const t0 = Date.now();
  while (Date.now() - t0 < WAIT) {
    await sleep(250);
    try {
      const m = await P.send("Performance.getMetrics"); const h = m.metrics.find((x) => x.name === "JSHeapUsedSize"); if (h && h.value > peak) peak = h.value; rssPeak = Math.max(rssPeak, rssMB());
      if (await ev("!!(window.__boot && __boot.revealed)")) { done = true; break; }
    } catch (e) { /* navigating */ }
  }
  await sleep(1500);
  const b = await ev("JSON.stringify({ firstCells: __boot.firstCells, revealed: __boot.revealed, fullReady: __boot.fullReady, long: __boot.long, gaps: __boot.ft.filter((f) => f[1] > 50).slice(0, 30), loop: window.BIMP && BIMP._loop ? BIMP._loop() : null, N: window.BIMP && BIMP.renderer ? BIMP.renderer.N : 0 })").then(JSON.parse).catch((e) => ({ err: e.message }));
  let after = null;
  if (A.profile && A.profile !== "load") { await P.send("Profiler.enable"); await P.send("Profiler.setSamplingInterval", { interval: 200 }); await P.send("Profiler.start"); }
  if (A.after) { try { after = await ev(A.after); } catch (e) { after = "ERR " + e.message.slice(0, 200); } }
  if (A.profile) {
    const { profile } = await P.send("Profiler.stop");
    const byId = new Map(profile.nodes.map((n) => [n.id, n])), self = new Map();
    const dts = profile.timeDeltas; let t = profile.startTime;
    // only samples in the last 1.6 s (the window after the click)
    const end = profile.endTime, from = A.profile === "load" || A.profile === "all" ? 0 : end - 1.6e6;
    profile.samples.forEach((id, i) => { t += dts[i]; if (t < from) return; const n = byId.get(id); const cf = n.callFrame; const k = (cf.functionName || "(anon)") + " " + (cf.url || "").split("/").pop() + ":" + cf.lineNumber; self.set(k, (self.get(k) || 0) + (dts[i + 1] || 0) / 1000); });
    console.log("PROFILE", JSON.stringify([...self].sort((a, b) => b[1] - a[1]).slice(0, 18).map(([k, v]) => k + " " + v.toFixed(1) + "ms")));
  }
  if (A.shot) { const { writeFileSync } = await import("node:fs"); const r = await P.send("Page.captureScreenshot", { format: "png" }); writeFileSync(`${A.shot}-${k}.png`, Buffer.from(r.data, "base64")); }
  const exc = P.evs.slice(e0).filter((e) => e.method === "Runtime.exceptionThrown" || (e.method === "Runtime.consoleAPICalled" && e.params.type === "error"))
    .map((e) => e.method === "Runtime.exceptionThrown" ? "EXC " + (e.params.exceptionDetails.exception ? e.params.exceptionDetails.exception.description : e.params.exceptionDetails.text).slice(0, 400) : "ERR " + e.params.args.map((a) => a.value || a.description || "").join(" ").slice(0, 400));
  let bytes = 0; for (const e of P.evs.slice(e0)) if (e.method === "Network.loadingFinished") bytes += e.params.encodedDataLength;
  if (A.reqs) { const url = new Map(); for (const e of P.evs.slice(e0)) if (e.method === "Network.requestWillBeSent") url.set(e.params.requestId, e.params.request.url);
    const L = []; for (const e of P.evs.slice(e0)) if (e.method === "Network.loadingFinished") L.push([Math.round(e.params.encodedDataLength / 1024), (url.get(e.params.requestId) || "").replace(/^https?:\/\/[^/]+/, "").slice(0, 80)]);
    console.log("REQS", JSON.stringify(L.sort((a, b) => b[0] - a[0]).slice(0, 25))); }
  const longOver200 = (b.long || []).filter((l) => l[1] > 200);
  const row = { k, done, firstCellsMs: b.firstCells, fullReadyMs: b.fullReady, revealedMs: b.revealed, peakHeapMB: +(peak / 1048576).toFixed(0), rssPeakMB: rssPeak, netMB: +(bytes / 1048576).toFixed(1),
    longTasks: (b.long || []).length, longOver200: longOver200.map((l) => l[1]), worstGap: Math.max(0, ...(b.gaps || []).map((g) => g[1])), loop: b.loop, N: b.N, after, exc };
  rows.push(row); console.log(JSON.stringify(row));
}
const ok = rows.filter((r) => r.done).length;
console.log(JSON.stringify({ summary: { loads: N, revealed: ok, stuck: N - ok, withExceptions: rows.filter((r) => r.exc.length).length } }));
process.exit(0);
