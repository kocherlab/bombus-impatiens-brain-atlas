/* The "Cells expressing" readout: does it print the truth?
 *
 * Drives the PAGES build (all 1,300,192 cells) and reads the two percentages straight off
 * the DOM, then compares them against values computed independently from the exported
 * files by a script that shares no code with the viewer — gzip → offsets → count, in Node.
 * A readout that agreed with itself would prove nothing; this makes the browser and a
 * second implementation agree on the same number.
 *
 * Also checks the two things the readout is FOR:
 *   - it tracks the region mask (the mushroom body gives a different answer than the brain)
 *   - it tracks the colour-window floor (which is how "expressing" is defined)
 *
 *   node viewer/test/caste-prevalence.mjs
 */
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { gunzipSync } from "node:zlib";
import { readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ROOT = join(APP, "pages");
const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
const TYPES = { ".html": "text/html", ".json": "application/json", ".bin": "application/octet-stream" };

const GENE = "sNPF";   // measured, sparse, with a caste difference in detection

// ---- the independent implementation -------------------------------------------------
function truth(M) {
  const base = gunzipSync(readFileSync(join(ROOT, "data", M.base)));
  const o = M.offsets, n = M.n;
  const caste = new Uint8Array(base.buffer, base.byteOffset + o.caste[0], n);
  const region = new Uint8Array(base.buffer, base.byteOffset + o.region[0], n);
  const gi = M.genes.findIndex((g) => g.symbol === GENE && g.kind === "measured");
  if (gi < 0) throw new Error("gene " + GENE + " not in the manifest");
  const g = M.genes[gi];
  const raw = gunzipSync(readFileSync(join(ROOT, "data", g.file)));
  let v;
  if (g.bits === 4) {                       // packed 2 cells/byte, expanded ×17 like the viewer
    v = new Uint8Array(n);
    for (let i = 0; i < n; i++) v[i] = (i & 1 ? (raw[i >> 1] & 15) : (raw[i >> 1] >> 4)) * 17;
  } else v = new Uint8Array(raw.buffer, raw.byteOffset, n);

  const kcBits = M.regionDefs.filter((d) => d.key.startsWith("in_kc_")).reduce((m, d) => m | (1 << d.bit), 0);
  const count = (mask, thr) => {
    let dQ = 0, eQ = 0, dW = 0, eW = 0;
    for (let i = 0; i < n; i++) {
      if (mask !== 0 && (region[i] & mask) === 0) continue;
      if (caste[i] === 0) { dQ++; if (v[i] > thr) eQ++; }
      else { dW++; if (v[i] > thr) eW++; }
    }
    return { q: dQ ? 100 * eQ / dQ : 0, w: dW ? 100 * eW / dW : 0, dQ, dW };
  };
  // Pick the floor FROM the data rather than guessing one. A fixed 0.35 looked reasonable
  // and turned out to be vacuous: this gene has no cell at all between 1 and 89, so the
  // percentages were identical either side of it and the check proved nothing. Take the
  // median of the nonzero values instead — by construction that floor removes about half
  // the expressing cells, so a readout ignoring the floor cannot possibly still agree.
  const nz = [];
  for (let i = 0; i < n; i += 7) if (v[i] > 0) nz.push(v[i]);
  nz.sort((a, b) => a - b);
  const med = nz.length ? nz[nz.length >> 1] : 128;
  const lo = Math.round((med / 255) / 0.005) * 0.005;   // land on a slider step exactly
  return { gi, kcBits, lo, all: count(0, 0), kc: count(kcBits, 0), floor: count(0, lo * 255) };
}

const PROBE = (gi, kcBits, lo) => `
<script>
(async () => {
 const out = (o) => { const p=document.createElement('pre'); p.id='probe'; p.textContent=JSON.stringify(o); document.body.appendChild(p); };
 try {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const s = BIMP.state.s, R = BIMP.renderer, ui = BIMP.ui;
  await BIMP.ready;
  for (let i=0;i<400 && !(R && R.N>0); i++) await sleep(50);
  R.setProgress(1); s.spin = false;
  document.querySelector('#mode-seg [data-mode="gene"]').click();
  ui.selectGene(${gi});
  for (let i=0;i<200 && !(BIMP.data.genes && BIMP.data.genes[${gi}]); i++) await sleep(100);
  await sleep(700);                       // past the 90 ms debounce
  const read = () => ({ shown: document.getElementById('cb-caste').classList.contains('show'),
    q: document.getElementById('cbc-q').textContent, w: document.getElementById('cbc-w').textContent,
    note: document.getElementById('cbc-note').textContent,
    qbar: document.getElementById('cbc-q-bar').style.width, wbar: document.getElementById('cbc-w-bar').style.width });
  const all = read();

  s.regionMask = ${kcBits}; ui.updateStat();      // mushroom body only
  await sleep(700);
  const kc = read();

  s.regionMask = 0; ui.updateStat();
  await sleep(700);
  const back = read();

  // raise the colour floor: "expressing" is defined against it, so the numbers must fall
  const lo = document.getElementById('cb-lo');
  lo.value = ${lo}; lo.dispatchEvent(new Event('input', { bubbles: true }));
  await sleep(700);
  const floored = read();

  document.querySelector('#mode-seg [data-mode="celltype"]').click();
  await sleep(400);
  const hidden = document.getElementById('cb-caste').classList.contains('show');
  out({ all, kc, back, floored, hiddenOutsideGeneMode: !hidden });
 } catch (e) { out({ error: String(e && e.stack || e) }); }
})();
</script>`;

async function main() {
  if (!CHROMIUM) { console.error("✗ Chromium not found"); process.exit(1); }
  if (!existsSync(join(ROOT, "index.html"))) { console.error("✗ pages/index.html missing"); process.exit(1); }
  const M = JSON.parse(readFileSync(join(ROOT, "data", "manifest.json"), "utf8"));
  const T = truth(M);
  const probe = PROBE(T.gi, T.kcBits, T.lo);

  const server = createServer((req, res) => {
    let p = normalize(decodeURIComponent(req.url.split("?")[0])).replace(/^(\.\.[/\\])+/, "");
    if (p === "/" || p === "\\") p = "/index.html";
    const file = join(ROOT, p);
    if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    const ext = file.slice(file.lastIndexOf("."));
    // BIMP_INDEX swaps in an alternate page while still serving this pages/data — that is
    // how the negative controls run a deliberately broken viewer over the real 1.3M cells.
    if (ext === ".html") {
      const src = process.env.BIMP_INDEX || file;
      res.writeHead(200, { "content-type": "text/html" });
      res.end(readFileSync(src, "utf8") + probe);
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
      const c = spawn(CHROMIUM, ["--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
        "--window-size=1280,860", "--virtual-time-budget=120000", "--dump-dom", `http://127.0.0.1:${port}/`],
        { stdio: ["ignore", "pipe", "pipe"] });
      const b = []; c.stdout.on("data", (x) => b.push(x)); c.stderr.resume();
      c.on("error", reject); c.on("close", () => resolve(Buffer.concat(b).toString("utf8")));
    });
  } finally { server.close(); }

  const m = dom.match(/<pre id="probe">([\s\S]*?)<\/pre>/);
  if (!m) { console.error("✗ probe never reported\n" + dom.slice(0, 1500)); process.exit(1); }
  const r = JSON.parse(m[1]);
  if (r.error) { console.error("✗ probe threw: " + r.error); process.exit(1); }

  let fail = 0;
  const check = (name, ok, detail) => { console.log(`${ok ? "✓" : "✗"} ${name}  ${detail}`); if (!ok) fail++; };
  const num = (t) => parseFloat(t);
  const near = (a, b) => Math.abs(a - b) <= 0.15;      // the readout prints one decimal

  console.log(`— "cells expressing", ${GENE}, over ${M.n.toLocaleString()} cells —`);
  console.log("  browser " + JSON.stringify(r) + "\n  node    " + JSON.stringify(T).slice(0, 400) + "\n");

  check("the readout appears in gene mode", r.all.shown, `shown=${r.all.shown}`);
  check("it hides outside gene mode", r.hiddenOutsideGeneMode, `hidden=${r.hiddenOutsideGeneMode}`);
  // non-vacuity: a gene whose castes matched would make every comparison below trivially true
  check("this gene really does differ between the castes", Math.abs(T.all.q - T.all.w) > 2,
    `node says queen ${T.all.q.toFixed(1)}% vs worker ${T.all.w.toFixed(1)}%`);
  check("whole-brain percentages match an independent count",
    near(num(r.all.q), T.all.q) && near(num(r.all.w), T.all.w),
    `browser ${r.all.q}/${r.all.w} vs node ${T.all.q.toFixed(1)}%/${T.all.w.toFixed(1)}%`);
  check("the mushroom-body mask changes the answer, and correctly",
    near(num(r.kc.q), T.kc.q) && near(num(r.kc.w), T.kc.w) && Math.abs(T.kc.q - T.all.q) > 1,
    `browser ${r.kc.q}/${r.kc.w} vs node ${T.kc.q.toFixed(1)}%/${T.kc.w.toFixed(1)}%`);
  check("clearing the mask restores the whole-brain answer",
    near(num(r.back.q), T.all.q) && near(num(r.back.w), T.all.w), `${r.back.q}/${r.back.w}`);
  check("raising the colour floor narrows 'expressing', and correctly",
    near(num(r.floored.q), T.floor.q) && near(num(r.floored.w), T.floor.w) && T.floor.q < T.all.q,
    `browser ${r.floored.q}/${r.floored.w} vs node ${T.floor.q.toFixed(1)}%/${T.floor.w.toFixed(1)}%`);
  check("the bars are drawn and the larger share fills its track",
    /%$/.test(r.all.qbar) && /%$/.test(r.all.wbar) &&
    (T.all.q > T.all.w ? parseFloat(r.all.qbar) === 100 : parseFloat(r.all.wbar) === 100),
    `queen ${r.all.qbar}, worker ${r.all.wbar}`);
  check("the note states the direction", /×|same share/.test(r.all.note), JSON.stringify(r.all.note));

  console.log(fail ? `\n${fail} check(s) failed` : "\nthe readout tells the truth");
  process.exit(fail ? 1 : 0);
}
main();
