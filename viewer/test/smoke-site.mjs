// Smoke test for a built or deployed site: the home page settles, the atlas loads every cell,
// histology switches on, and named genes load from wherever the manifest says (Cloudflare on
// the live site). Exits non-zero on any failure or page error.
//
//   node viewer/test/smoke-site.mjs                                  # http://127.0.0.1:8765/
//   node viewer/test/smoke-site.mjs https://kocherlab.github.io/bombus-impatiens-brain-atlas/
//
// Serve pages/ locally with:  node viewer/tools/serve.mjs viewer/pages
// Starts its own headless Chrome (CHROME=/path/to/chrome to override) and stops it after.
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = (process.argv[2] || "http://127.0.0.1:8765/").replace(/\/?$/, "/");
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const GENES = ["AstA", "Ats", "Met", "Mblk-1", "Kr-h1", "sNPF"];   // allatostatin, allatotropin, the JH receptor, markers
const PORT = 9300 + Math.floor(Math.random() * 500);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const prof = mkdtempSync(join(tmpdir(), "bimp-smoke-"));
const chrome = spawn(CHROME, ["--headless=new", "--use-angle=metal", "--hide-scrollbars", `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`, "about:blank"], { stdio: "ignore" });
const done = (code) => { chrome.kill(); try { rmSync(prof, { recursive: true, force: true }); } catch {} process.exit(code); };
let t;
for (let i = 0; i < 40 && !t; i++) { await sleep(250); try { t = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" })).json(); } catch {} }
if (!t) { console.log("chrome did not start"); done(2); }
const ws = new WebSocket(t.webSocketDebuggerUrl); await new Promise((r) => (ws.onopen = r));
let id = 0; const pend = new Map(); const errs = [];
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); }
  if (m.method === "Runtime.exceptionThrown") errs.push(m.params.exceptionDetails.exception?.description?.split("\n")[0]);
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") errs.push("console: " + m.params.args.map((a) => a.value || a.description).join(" ").slice(0, 160)); };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (x) => (await send("Runtime.evaluate", { expression: x, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send("Runtime.enable");
const fail = [];
for (const [w, h, mob] of [[1440, 900, false], [390, 844, true]]) {
  await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: mob ? 3 : 2, mobile: mob });   // high-density screens, where CSS and pixel sizes differ
  await send("Page.navigate", { url: BASE }); await sleep(4000);
  const home = await ev("({ready: document.documentElement.classList.contains('ready'), steps: document.querySelectorAll('#intro .story .step').length})");
  if (!home?.ready) fail.push(`${w}: home page never became ready`);
  // the brain's on-page copy must sit in its box at its size (a canvas shown at its pixel size ran off the page on 2x+ screens)
  const mir = await ev(`(()=>{const m=document.querySelector('.hero-mirror'),b=document.getElementById('hero-gl');if(!m||!b)return null;const r=m.getBoundingClientRect(),q=b.getBoundingClientRect();return {ratio:+(r.width/q.width).toFixed(2), off:Math.round(Math.abs((r.left+r.width/2)-(q.left+q.width/2)))}})()`);
  if (mir && (mir.ratio > 1.4 || mir.off > 4)) fail.push(`${w}: home brain copy mis-sized or off-centre ${JSON.stringify(mir)}`);
  // every visible link and button on the first screen must be what a click there actually hits
  const covered = await ev(`[...document.querySelectorAll('#masthead a,#intro .hero a,#intro .hero button')].filter(b=>{const r=b.getBoundingClientRect();
    if(!r.width||r.bottom<0||r.top>innerHeight)return false;const t=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);return !(t===b||b.contains(t));}).map(b=>b.textContent.trim().slice(0,24))`);
  if (covered?.length) fail.push(`${w}: covered, so unclickable: ${covered.join(", ")}`);
  await send("Page.navigate", { url: BASE + "?atlas" });
  await sleep(300); await ev("performance.setResourceTimingBufferSize(20000)");   // the tiles alone overflow the default 250 entries, hiding later gene fetches
  for (let i = 0; i < 90; i++) { await sleep(500); if (await ev("!!(window.BIMP && BIMP.fullReady)")) break; }
  await sleep(1500);
  const atlas = await ev("(()=>{try{return {n: BIMP.renderer.N, total: BIMP.manifest.nTotal, genes: BIMP.manifest.genes.length, tier: BIMP.tier}}catch(e){return {err: e.message}}})()");
  // phones may run the 100k lite tier by design; desktops must have every cell
  if (!mob && atlas?.n !== atlas?.total) fail.push(`${w}: ${atlas?.n} cells drawn of ${atlas?.total}`);
  const histo = await ev("(async()=>{try{BIMP.histo.setOn(true); await new Promise(r=>setTimeout(r,2500)); return BIMP.histo.isOn()}catch(e){return 'ERR '+e.message}})()");
  if (histo !== true) fail.push(`${w}: histology did not switch on (${histo})`);
  const genes = [];
  for (const g of GENES) {
    const r = await ev(`(async()=>{const i=BIMP.ui.geneIndexBySymbol?BIMP.ui.geneIndexBySymbol(${JSON.stringify(g)}):-1; if(i<0) return 'not found';
      const f=BIMP.manifest.genes[i].file; BIMP.ui.selectGene(i); await new Promise(r=>setTimeout(r,2500));
      const e=performance.getEntriesByType('resource').filter(x=>x.name.endsWith(f)).pop(); return e ? 'loaded '+(e.responseStatus||'(cross-origin)') : 'no request'})()`);
    genes.push(`${g}: ${r}`); if (!/^loaded/.test(r) || /\b(4|5)\d\d\b/.test(r)) fail.push(`${w}: gene ${g} ${r}`);
  }
  console.log(w, JSON.stringify(home), JSON.stringify(atlas), "histology", histo, "\n   ", genes.join(" · "));
}
if (errs.length) fail.push(...errs.slice(0, 8).map((e) => "page error: " + e));
console.log(fail.length ? "FAIL\n  " + fail.join("\n  ") : "ok");
done(fail.length ? 1 : 0);
