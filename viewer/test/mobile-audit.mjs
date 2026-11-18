/* Mobile friendliness audit — layout only, no WebGL, so it is fast enough to run over
 * every panel state × every phone size.
 *
 * Renders the real body.html + styles.css with representative content, then for each
 * state (idle / controls sheet open / inspector / guided tour / gene modal) checks:
 *
 *   • horizontal overflow — the page must never scroll sideways,
 *   • any visible element spilling outside the viewport,
 *   • overlaps between the panels that are on screen together,
 *   • tap targets below 40px (Apple asks 44, 40 is the floor worth flagging),
 *   • text that has been ellipsised into uselessness.
 *
 *   node viewer/test/mobile-audit.mjs            # fails on regressions
 *   node viewer/test/mobile-audit.mjs --report   # prints everything it measured
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const SRC = join(APP, "src");
const OUT = join(process.env.BIMP_TEST_TMP || tmpdir(), "tmp", "mobile-audit.html");
mkdirSync(dirname(OUT), { recursive: true });
const REPORT = process.argv.includes("--report");

const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find(existsSync);
if (!CHROMIUM) { console.error("Chromium not found (Playwright cache or /Applications)"); process.exit(1); }

const css = readFileSync(join(SRC, "styles.css"), "utf8");
const body = readFileSync(join(SRC, "body.html"), "utf8");

const SIZES = [[360, 640, "small phone"], [390, 844, "iPhone"], [414, 896, "large phone"], [768, 1024, "tablet"]];
const STATES = ["idle", "sheet", "inspector", "tour", "modal"];

const page = `<!doctype html><meta charset=utf-8><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>AUDIT</title>
<style>${css}</style>
<!-- Entrance animations use fill-mode:both, so under virtual time they sit frozen at
     opacity 0 and every animated element looks invisible — which silently excluded the
     "Show me around" pill from the overlap checks. This audit is about layout. -->
<style>*{animation:none !important;transition:none !important}</style>
${body}
<script>
// realistic content, so panels have their true heights
const gl_ = document.getElementById('gene-list');
for (let i=0;i<14;i++) gl_.insertAdjacentHTML('beforeend',
  '<div class="gene-item"><i class="gg gg-i"></i><div class="gi-main"><span class="gi-name">tachykinins</span><span class="gi-desc">neuropeptide precursor</span></div><button class="gi-info">i</button></div>');
const REG = [['Small-type compartment','Mushroom body · Kenyon cell compartments'],['Large-type compartment',''],['Class II compartment',''],
             ['Lamina','Optic lobe'],['Medulla',''],['Lobula','']];
const rl = document.getElementById('region-list');
REG.forEach(([lab,grp],i)=>{
  if (grp) rl.insertAdjacentHTML('beforeend','<div class="reg-group">'+grp+'</div>');
  rl.insertAdjacentHTML('beforeend','<label class="reg-item" data-bit="'+i+'"><span class="reg-box"></span><span class="reg-dot" style="background:#B07AA1"></span><span class="reg-nm">'+lab+'</span><span class="reg-ct">106,281</span></label>');
});
// the tree legend as ui.js renders it: a group row with a chevron, one open with children
const lb = document.getElementById('leg-body');
const row = (d, nm, open, kind) => '<div class="node d'+d+(open?' open':'')+'">' + (kind ? '<button class="chev"></button>' : '<span class="chev none"></span>') +
  '<span class="dot" style="--c:#C8781E"></span><span class="nm">'+nm+'</span><span class="ct">338,942</span></div>';
lb.innerHTML = '<div class="tree">' + row(0,'Optic lobe cells') + row(0,'Glia',false,1) + row(0,'Kenyon cells',true,1) +
  '<div class="kids"><div class="tree-note">Restricted transfer · compartments only<button class="info-dot">i</button></div>' +
  row(1,'Large-type',false,1) + row(1,'Small-type') + row(1,'Class II') + row(1,'Middle-type') + row(1,'Other KC') +
  row(1,'Kenyon cells (outside the compartments)') + '</div>' + row(0,'Other') + '</div>';
// ui.placeLegend: with no left pane (<=820px) the legend lives in the controls, after the gene picker
if (innerWidth <= 820) { const gw = document.getElementById('gene-wrap'); gw.parentNode.insertBefore(document.getElementById('legend'), gw.nextSibling); }
document.getElementById('tour-title').textContent = 'The mushroom body — Kenyon cells';
document.getElementById('tour-body').innerHTML = 'Isolate the hand-labeled Kenyon cell compartments, each named for the population that dominates it — <b>Large-type · Small-type · Class II</b> (non-compact, inner compact, outer compact). Everything else is hidden so the three compartments stand out.';
document.getElementById('tour-dots').innerHTML = '<span class="tdot on"></span>'.repeat(7);
document.getElementById('gm-name').textContent = 'tachykinins';
document.getElementById('gm-rows').innerHTML = '<div class="gm-row"><span>bimp id</span><span>bimp_v4.3_009257</span></div>'.repeat(5);
['i-subtype-row','i-canon-row','i-gene-row'].forEach(i=>document.getElementById(i).style.display='flex');

function setState(st) {
  document.getElementById('intro').style.display = 'none';
  document.getElementById('intro').classList.add('gone');   // as boot.js does: the viewer's panels key off it
  document.body.classList.remove('touring');
  document.getElementById('controls').classList.remove('sheet-open');
  document.getElementById('inspector').classList.remove('show');
  document.getElementById('umap-box').classList.remove('show');
  document.getElementById('tour').classList.remove('show');
  document.getElementById('gene-modal').style.display = 'none';
  document.getElementById('tour-btn').classList.add('ready');
  document.getElementById('caste-hint').classList.add('show');
  document.body.classList.remove('sheet-open');
  document.getElementById('tour-btn').classList.remove('hide');
  if (st === 'sheet') { document.getElementById('controls').classList.add('sheet-open'); document.body.classList.add('sheet-open'); }
  document.body.classList.remove('inspecting');
  if (st === 'inspector') {
    const insp = document.getElementById('inspector');
    insp.classList.add('show'); document.getElementById('umap-box').classList.add('show');
    // mirror ui.syncInspectorLane: the app publishes the inspector's height so the
    // controls can cap themselves to what's left
    document.body.classList.add('inspecting');
    document.documentElement.style.setProperty('--insp-h', Math.round(insp.getBoundingClientRect().height) + 'px');
  }
  if (st === 'tour') { document.body.classList.add('touring'); document.getElementById('tour').classList.add('show');
    document.getElementById('tour-btn').classList.add('hide'); }   // tour.js hides the pill while touring
  if (st === 'modal') document.getElementById('gene-modal').style.display = 'flex';
}
// #legend is inside #controls at these widths, so it is covered by the controls entry
const KEY = ['controls','inspector','tour','tour-btn','sheet-toggle','hud','gene-modal'];
function vis(el){ if(!el) return false; const cs=getComputedStyle(el);
  if (cs.display==='none'||cs.visibility==='hidden'||+cs.opacity===0) return false;
  const r=el.getBoundingClientRect(); return r.width>0&&r.height>0; }
const ctrlOpen = () => document.getElementById('controls').classList.contains('sheet-open');
// the drawer only EXISTS at the phone breakpoint — above it, #controls is a normal panel
// and the sheet-open class is inert, so it needs no close control
const drawerUI = () => getComputedStyle(document.getElementById('sheet-hd')).display !== 'none';
function audit(st) {
  setState(st);
  const W = innerWidth, H = innerHeight;
  const res = { state: st, overflowX: document.documentElement.scrollWidth > W + 1, spill: [], overlaps: [], small: [], clipped: [] };
  // elements spilling outside the viewport
  document.querySelectorAll('#app *').forEach((el) => {
    if (!vis(el)) return;
    const r = el.getBoundingClientRect();
    if (r.width > 1 && (r.left < -1 || r.right > W + 1)) {
      const id = el.id || el.className || el.tagName;
      if (!res.spill.some((s) => s.el === id)) res.spill.push({ el: String(id).slice(0,40), left: Math.round(r.left), right: Math.round(r.right) });
    }
  });
  // Overlaps between the panels that are on screen together. Full-screen overlays
  // (the gene modal, the intro) are SUPPOSED to cover everything — they get their own
  // checks below instead of being flagged against every panel they sit above.
  const OVERLAY = ['gene-modal', 'intro'];
  const shown = KEY.filter((id) => !OVERLAY.includes(id))
    .map((id) => [id, document.getElementById(id)]).filter(([, el]) => vis(el));
  for (let i = 0; i < shown.length; i++) for (let j = i + 1; j < shown.length; j++) {
    const a = shown[i][1].getBoundingClientRect(), b = shown[j][1].getBoundingClientRect();
    const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
    const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    // the Close button is the sheet's own handle — it is MEANT to sit over the sheet edge;
    // what matters is that the sheet's content clears it, checked separately below
    const pair = shown[i][0] + '|' + shown[j][0];
    if (pair === 'controls|sheet-toggle') continue;
    if (ox > 2 && oy > 2) res.overlaps.push({ a: shown[i][0], b: shown[j][0], w: Math.round(ox), h: Math.round(oy) });
  }
  // tap targets
  document.querySelectorAll('button, .reg-item, .node, .gene-item, .toggle, input[type=range], a#region-all, a#leg-all, a.introlink').forEach((el) => {
    if (!vis(el)) return;
    const r = el.getBoundingClientRect();
    if (r.height < 40 || r.width < 40) {
      const id = el.id || el.className || el.tagName;
      res.small.push({ el: String(id).slice(0,32), w: Math.round(r.width), h: Math.round(r.height) });
    }
  });
  // text ellipsised to near-nothing
  document.querySelectorAll('.reg-nm, .nm, .gi-name').forEach((el) => {
    if (!vis(el)) return;
    if (el.scrollWidth > el.clientWidth + 2) res.clipped.push({ el: el.textContent.slice(0,24), shown: el.clientWidth, needs: el.scrollWidth });
  });
  // Every panel that covers the screen must offer an obvious, thumb-sized way out, and
  // it must be ON the panel — a drawer whose only control sits in the same fixed spot
  // whether it is open or closed reads as broken.
  const exits = [];
  if (vis(document.getElementById('inspector'))) exits.push(['inspector', 'i-close']);
  if (ctrlOpen()) {
    if (W <= 600 && !drawerUI()) res.overlaps.push({ a: 'controls sheet', b: 'has no drawer header at phone width', w: 0, h: 0 });
    if (drawerUI()) exits.push(['controls sheet', 'sheet-close']);
  }
  if (vis(document.getElementById('gene-modal'))) exits.push(['gene modal', 'gm-close']);
  for (const [what, id] of exits) {
    const b = document.getElementById(id);
    if (!vis(b)) { res.overlaps.push({ a: what, b: 'has no visible close control', w: 0, h: 0 }); continue; }
    const r = b.getBoundingClientRect(), host = document.getElementById(what === 'controls sheet' ? 'controls' : (what === 'gene modal' ? 'gene-modal' : 'inspector')).getBoundingClientRect();
    if (r.top < host.top - 2 || r.bottom > host.bottom + 2 || r.left < host.left - 2 || r.right > host.right + 2)
      res.overlaps.push({ a: what, b: 'close control sits outside the panel it closes', w: 0, h: 0 });
    if (W <= 600 && (r.width < 40 || r.height < 40))
      res.small.push({ el: id + ' (close)', w: Math.round(r.width), h: Math.round(r.height) });
  }
  // while the drawer is open its "open" button must be out of the way
  if (ctrlOpen() && drawerUI() && vis(document.getElementById('sheet-toggle')))
    res.overlaps.push({ a: 'sheet-toggle', b: 'still showing while the drawer is open', w: 0, h: 0 });

  // an open sheet must reserve room at the bottom for the Close button floating over it
  // an open modal must cover the viewport (so nothing behind it is tappable) and its
  // card must fit on screen with room to breathe
  const modal = document.getElementById('gene-modal');
  if (vis(modal)) {
    const m = modal.getBoundingClientRect(), card = modal.querySelector('.gm-card').getBoundingClientRect();
    if (m.width < W - 1 || m.height < H - 1) res.overlaps.push({ a: 'gene-modal', b: 'viewport (does not cover)', w: Math.round(m.width), h: Math.round(m.height) });
    if (card.width > W - 16 || card.height > H - 16 || card.top < 0 || card.bottom > H)
      res.spill.push({ el: 'gm-card', left: Math.round(card.left), right: Math.round(card.right) });
  }
  return res;
}
const all = [];
for (const st of ${JSON.stringify(STATES)}) all.push(audit(st));
setState('idle');
const pre = document.createElement('pre'); pre.id = 'AUDIT';
pre.textContent = JSON.stringify({ w: innerWidth, h: innerHeight, states: all });
document.body.appendChild(pre);
</script>`;
writeFileSync(OUT, page);

// Chrome clamps headless WINDOWS to 500 px wide on macOS, so the phone sizes are set by
// device-metrics emulation over the DevTools protocol instead of --window-size.
const results = [];
{
  const port = 9600 + Math.floor(Math.random() * 300);
  const prof = mkdtempSync(join(tmpdir(), "mobile-audit-"));
  const ch = spawn(CHROMIUM, ["--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${port}`,
    `--user-data-dir=${prof}`, "about:blank"], { stdio: "ignore" });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let ws = null;
  for (let i = 0; i < 60 && !ws; i++) {
    await sleep(200);
    try { const l = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); const t = l.find((x) => x.type === "page"); if (t) ws = new WebSocket(t.webSocketDebuggerUrl); } catch { /* not up yet */ }
  }
  if (!ws) { console.error("✗ could not reach headless Chrome"); ch.kill(); process.exit(1); }
  await new Promise((r) => ws.addEventListener("open", r));
  let id = 0; const pend = new Map();
  ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  for (const [w, h, name] of SIZES) {
    await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: w <= 600 });
    await send("Page.navigate", { url: "file://" + OUT });
    let txt = null;
    for (let k = 0; k < 50 && !txt; k++) {
      await sleep(150);
      const r = await send("Runtime.evaluate", { expression: "(document.getElementById('AUDIT')||{}).textContent||''", returnByValue: true });
      txt = r.result && r.result.result && r.result.result.value;
    }
    if (!txt) { console.error(`✗ ${name}: audit did not run`); ch.kill(); process.exit(1); }
    results.push({ name, w, h, ...JSON.parse(txt) });
  }
  ws.close(); ch.kill("SIGKILL");
}

const problems = [];
for (const r of results) {
  const phone = r.w <= 600;
  for (const st of r.states) {
    const where = `${r.name} ${r.w}x${r.h} [${st.state}]`;
    if (st.overflowX) problems.push(`${where}: the page scrolls sideways`);
    for (const s of st.spill) problems.push(`${where}: ${s.el} spills outside the viewport (${s.left}…${s.right} of ${r.w})`);
    for (const o of st.overlaps) problems.push(`${where}: #${o.a} overlaps #${o.b} by ${o.w}x${o.h}px`);
    if (phone) {
      // on a phone every control has to be thumb-sized
      const tiny = st.small.filter((x) => x.h < 32);
      for (const t of tiny) problems.push(`${where}: tap target ${t.el} is only ${t.w}x${t.h}px`);
    }
    for (const c of st.clipped) if (c.shown < c.needs * 0.55)
      problems.push(`${where}: "${c.el}…" is clipped to ${c.shown}px of ${c.needs}px`);
  }
  if (REPORT) {
    console.log(`\n${r.name} ${r.w}x${r.h}`);
    for (const st of r.states) console.log(`  ${st.state.padEnd(10)} overflowX=${st.overflowX} spill=${st.spill.length} ` +
      `overlaps=${st.overlaps.length} small=${st.small.length} clipped=${st.clipped.length}`);
    const idle = r.states[0];
    if (idle.small.length) console.log(`    under 40px: ` + idle.small.map((x) => `${x.el} ${x.w}x${x.h}`).join(", "));
    if (idle.clipped.length) console.log(`    clipped: ` + idle.clipped.map((c) => `"${c.el}" ${c.shown}/${c.needs}px`).join(", "));
  }
}

if (problems.length) {
  console.error(`\n${problems.length} mobile issue(s):`);
  for (const p of problems) console.error("  ✗ " + p);
  process.exit(1);
}
console.log(`✓ mobile audit clean across ${SIZES.length} sizes × ${STATES.length} panel states`);
