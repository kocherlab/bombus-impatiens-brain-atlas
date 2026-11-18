/* Histology layer: the Xenium stain images of every section, as textured quads at each
   section's depth, under the cells — plus the section rail and the 2D section view.
   Pages build only (build-pages.mjs); the single-file artifact never includes this module,
   so BIMP.histology is undefined there and the renderer's one-line hook (start of drawScene,
   non-pick pass) is a no-op. draw() leaves the GL state as it found it: program, VAO,
   textures, blend, depth mask.

   Data: data/histo/manifest.json (tools/export_histology.py, histo_refit.py) —
     {version, castes: {queen: {sections: [{i, z, bounds:[x0,y0,x1,y1], file, small, w, h,
      stains?: [keys]}]}, worker: …}, frame: {scale_um, z_mid_um, …},
      stains?: [{key, label, path, small}]}
   bounds and z are in export_data.py's normalised data frame (the cells' int16/32767), before
   the renderer's (x,y)→(y,−x) turn; image column 0 sits at x0, image ROW 0 at y0. `stains`
   lists the stains in menu order, first = default; paths are templates over {caste} and {ii}
   (the section's i, two digits), relative to the manifest. No `stains` key means DAPI only, at
   each section's file/small. A section's own `stains` names the ones it has (absent = all).

   Several stains can be on at once. One stain is drawn in greyscale (the page's light neutral
   on the dark stage): a single channel reads best as plain intensity, and colour would only
   suggest a comparison that is not there. Two or more are composited in one pass, each in its
   own fluorescence colour, added and clamped — the way a multichannel image is read.

   Loading is on intent. Once the atlas is idle the shown brain's small images of the first
   stain are fetched into the HTTP cache (not on save-data or 2g/3g). Hovering, focusing or
   touching the Histology switch, a stain or the 2D switch starts decoding them
   (createImageBitmap, off-thread) and uploading one-byte R8 textures, nearest the current
   section first, a couple per frame within a time budget. Switched on, the small images show
   at once; large ones replace them, nearest first, one in flight, only where a section covers
   more screen than its small image holds, and never on phones. Off releases every texture.

   Blending is MAX on a dark stage (a maximum-intensity projection: thirty near-coplanar
   sections never saturate and need no sorting) and MIN on a light one (the same projection as
   ink). The stage is read from the clear colour each frame. */
window.BIMP = window.BIMP || {};
BIMP.histology = (function () {
  const CASTES = ["queen", "worker"];
  const SPLIT = 1.15;                      // fallback only: the renderer's getSplit() is the truth
  const MAXS = 4;                          // stains composited in one pass
  // Sections whose image is known to be misregistered against its cells, by manifest `i` per
  // caste: left out rather than shown misaligned. A manifest may also carry its own
  // {"exclude": {"worker": [i, ...]}}; both lists apply.
  //
  // Empty: the per-section registration (histo_refit.py) places every section's cells on its
  // image (checked per section: no residual shift or scale).
  const EXCLUDE = { queen: [], worker: [] };
  // Composite colours, in the usual fluorescence assignments: nuclei blue, membrane green,
  // RNA red, interior protein yellow. All four are pure-hue and bright, so they stay apart from
  // one another on the dark stage, and they sit under the cells rather than mixing with the
  // cell palette (teal / magenta / orange / slate), whose hues they avoid where it matters.
  const STAIN_COL = { dapi: "#4F86FF", boundary: "#3FE07A", rna18s: "#FF4F4F", smavim: "#FFD23F" };
  const SPARE_COL = ["#4F86FF", "#3FE07A", "#FF4F4F", "#FFD23F"];
  // Display levels per stain: black point, white point, gamma (lo / hi / g).
  // The export stretches each stain to its brain's 0.5th–99.7th percentile range, so over the tissue the
  // 99.8th percentile sits at ~0.98 for every stain (measured, every 4th section of both brains)
  // and the white point stays at the top: lowering it is what clipped DAPI's nuclei to flat
  // white specks. Black points sit just above the empty background, so faint neuropil signal
  // is not crushed; a gamma below 1 lifts dim nuclei and texture; the shader's soft shoulder
  // keeps the bright end from clipping when the strength (gain) goes up. Boundary keeps a
  // higher floor: its tissue median sits above the background, a diffuse haze rather than structure.
  const LEVELS = {
    dapi: { lo: 0.01, hi: 0.98, g: 0.75 },
    boundary: { lo: 0.05, hi: 0.98, g: 0.85 },
    rna18s: { lo: 0.01, hi: 0.98, g: 0.8 },
    smavim: { lo: 0.01, hi: 0.98, g: 0.7 },
  };
  const levelsOf = (k) => LEVELS[k] || { lo: 0.03, hi: 0.85, g: 0.9 };
  const shortName = (st) => (st.short || st.label.replace(/\s*\(.*$/, ""));
  const PHONE = typeof matchMedia !== "undefined" && matchMedia("(max-width:600px)").matches;
  const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aUV;
uniform mat4 uMVP;
uniform vec3 uO, uU, uV;                   // quad origin and its two edge vectors (world)
uniform vec4 uUV;                          // texture window: offset xy, scale zw (a tile's own cell)
out vec2 vUV;
void main(){ vUV=uUV.xy+aUV*uUV.zw; gl_Position=uMVP*vec4(uO+aUV.x*uU+aUV.y*uV,1.0); }`;
  // One pass over up to four stains: colour × intensity, summed and clamped (dark stage), or
  // the same absorbed from the page (light stage). A program per number of stains on, so a
  // single stain samples one texture, not four (thirty overlapping quads make fill the cost).
  function fsFor(n) {
    const J = [...Array(n).keys()];
    return `#version 300 es
precision mediump float;
in vec2 vUV;
${J.map((j) => `uniform sampler2D uT${j};`).join("\n")}
uniform vec3 uCol[4];
uniform float uInt[4], uLo[4], uHi[4], uGam[4];
uniform float uLight, uFade;
out vec4 o;
// Display transform per stain: black point, white point, gamma, gain, then a soft shoulder
// (1 - e^-ax, normalised so 1 maps to 1) instead of a hard clip, so bright nuclei keep their
// mid-tones and gain above 1 rolls off rather than flattening to white.
float lev(float v,int j){
  float x=pow(max((v-uLo[j])/max(uHi[j]-uLo[j],1e-3),0.0),uGam[j])*uInt[j];
  return min((1.0-exp(-1.8*x))/(1.0-exp(-1.8)),1.0);
}
void main(){
  ${J.map((j) => `float l${j}=lev(texture(uT${j},vUV).r,${j});`).join(" ")}
  // Stains add WITHIN a section; sections stack with premultiplied "over" (blend ONE,
  // ONE_MINUS_SRC_ALPHA), each covering by its own signal × the section's opacity (uFade), so
  // thirty stacked sections layer instead of summing to white.
  float m=${J.map((j) => `l${j}`).reduce((a, b) => `max(${a},${b})`)};
  if(m<0.004) discard;   // outside the tissue: no blend, no write
  float al=min(m,1.0)*uFade;
  if(uLight>0.5){
    vec3 a=${J.map((j) => `(1.0-uCol[${j}])*l${j}`).join("+")};
    o=vec4(clamp(1.0-a/max(m,1e-3),0.0,1.0)*al,al);
  } else {
    vec3 c=min(${J.map((j) => `uCol[${j}]*l${j}`).join("+")},vec3(1.0));
    o=vec4(c/max(m,1e-3)*al,al);
  }
}`;
  }

  let gl = null, progs = [], vao = null, vbo = null, black = null, rprog = null, lightCache = null;
  let manifest = null, base = "", on = false, opacity = 1.0;
  // What is shown. `slotK` is the 3D Section slider: the k-th shown section of EACH visible
  // caste (0 = all). `selC[ci]` is a pick per brain (a manifest `i`, or null = that brain's
  // whole stack); the two brains are picked independently. `lead` is the brain picked last
  // (2D arrow keys and slider drive it). `matchDepth` (off by default) makes a pick pull the
  // other brain to its section nearest in depth.
  let slotK = 0, selC = [null, null], lead = 0, matchDepth = false;
  let active = [];                     // stain keys on, in menu order
  const sInt = {};                     // per-stain intensity
  // Stain changes crossfade: each stain has a weight easing to 1 when on and 0 when off (an
  // added stain holds at 0 until its images for the sections on show have arrived, so it never
  // pops in), and a one-stain greyscale look eases into the composite's colours (cmix).
  const sw = {};
  let cmix = 0, drawK = [];
  let mode2d = false, saved3d = null;  // the 2D section view, and the 3D state it came from
  const listeners = [];
  // for layer toggles elsewhere (atlas-ui): called with the new state after every change
  function onChange(cb) { if (typeof cb === "function") listeners.push(cb); return () => { const i = listeners.indexOf(cb); if (i >= 0) listeners.splice(i, 1); }; }
  const emit = () => listeners.forEach((cb) => { try { cb(on); } catch (e) { /* a listener's problem, not ours */ } });
  // one entry per section that can show: {caste, ci, k, s, a, tx: {stainKey: {small, large,
  // pending, failed, farSince}}}
  let secs = [];
  let abort = null, upQ = [], loading = 0, largeBusy = false, warming = false, warmTimer = 0;
  const fadeC = [0, 0];              // per-caste visibility, eased
  let lastT = 0, lastLodT = 0, lastDrawn = [], lodTimer = 0, lodView = null;
  function sameView(m, W, H) {
    if (!lodView || lodView.w !== W || lodView.h !== H) return false;
    for (let i = 0; i < 16; i++) if (Math.abs(m[i] - lodView[i]) > 1e-6) return false;
    return true;
  }   // lastDrawn: [ci, i] per quad, for tests
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const $ = (id) => document.getElementById(id);
  // The renderer draws on demand: anything that changes what this layer shows asks it for a
  // frame (an image decoded off-thread arrives between frames, so it must ask too).
  const inv = () => { const R = BIMP.renderer; if (R && R.invalidate) R.invalidate(); };
  let pumpDirty = true;                 // something changed that may need loading

  // ---- manifest -------------------------------------------------------------------------
  async function init(src) {
    if (base || !src) return;          // once; fetch mode only
    base = src + "histo/";
    try {
      const r = await fetch(base + "manifest.json");
      if (!r.ok) return;
      const m = await r.json();
      if (!m || !m.castes) return;
      manifest = m;
    } catch (e) { return; }          // no histology deployed: no control, silently
    stains().forEach((st) => { sInt[st.key] = 1; });
    active = [stains()[0].key];
    readUrl();
    buildControl();
    view2d.init();
    intent();
    idlePrefetch();
    const P = new URLSearchParams(location.search);
    if (P.get("histo") === "1") setOn(true);
    if (P.get("view") === "2d") view2d.enter();
    emit();                            // `available` flipped
  }
  // the stains on offer, in menu order; DAPI alone when the manifest names none
  function stains() {
    const L = manifest && Array.isArray(manifest.stains) && manifest.stains.length ? manifest.stains : null;
    return L || [{ key: "dapi", label: "DAPI" }];
  }
  const stainBy = (k) => stains().find((x) => x.key === k);
  function colOf(k) {
    if (STAIN_COL[k]) return STAIN_COL[k];
    return SPARE_COL[stains().findIndex((x) => x.key === k) % SPARE_COL.length];
  }
  const hexRgb = (h) => [1, 3, 5].map((j) => parseInt(h.slice(j, j + 2), 16) / 255);
  const pad2 = (n) => String(n).padStart(2, "0");
  function fmtPath(tpl, c, sec) {
    return tpl.replace(/\{caste\}/g, c).replace(/\{ii\}|\{i:02d\}/g, pad2(sec.i)).replace(/\{i\}/g, String(sec.i));
  }
  // image path for a stain (DAPI's are the section's own file/small); relative to histo/
  function fileOf(key, c, sec, small) {
    const st = stainBy(key) || stains()[0];
    if (!st.path) return small ? (sec.small || sec.file) : sec.file;
    return fmtPath(small ? (st.small || st.path) : st.path, c, sec);
  }
  const hasStain = (sec, key) => !Array.isArray(sec.stains) || sec.stains.includes(key);
  // the sections of a caste front to back, and which of them the layer can show
  function order(c) { const L = manifest.castes[c]; return L ? L.sections.slice().sort((a, b) => a.z - b.z) : []; }
  function excluded(c) { return new Set([...(EXCLUDE[c] || []), ...((manifest.exclude || {})[c] || [])]); }
  function shown(c) { const ex = excluded(c); return order(c).filter((s) => !ex.has(s.i) && active.some((k) => hasStain(s, k))); }
  // why a section has no image: misregistered, or none of the stains on
  function offReason(c, sec) {
    if (excluded(c).has(sec.i)) return "not aligned yet";
    if (!active.some((k) => hasStain(sec, k))) return "no image in " + active.map((k) => stainBy(k).label).join(" / ");
    return null;
  }
  // Depth of a section in µm from the first section of its brain, from the real z level
  // (frame.scale_um turns the viewer's normalised z back into µm; levels sit 10 µm apart).
  function depthUm(c, sec) {
    const f = manifest.frame, L = order(c);
    if (!L.length || !sec) return null;
    if (f && f.scale_um) return Math.round((sec.z - L[0].z) * f.scale_um);
    return L.indexOf(sec) * 10;     // no frame: the nominal 10 µm spacing
  }

  // ---- URL state ---------------------------------------------------------------------------
  // stains=dapi,boundary · sint=boundary:1.4 · sq / sw = picked section (manifest i) per brain
  // · match=1 · view=2d · histo=1. Other parameters on the URL are left alone.
  function readUrl() {
    const P = new URLSearchParams(location.search);
    if (P.has("stains")) {
      const ks = P.get("stains").split(",").filter((k) => stainBy(k));
      if (ks.length) active = stains().map((x) => x.key).filter((k) => ks.includes(k)).slice(0, MAXS);
    } else if (P.has("stain") && stainBy(P.get("stain"))) active = [P.get("stain")];
    if (P.has("sint")) P.get("sint").split(",").forEach((kv) => { const [k, v] = kv.split(":"); if (stainBy(k) && +v >= 0) sInt[k] = Math.min(2, +v); });
    if (P.has("histo_op")) opacity = Math.max(0.05, Math.min(2, parseFloat(P.get("histo_op")) || 1));
    ["sq", "sw"].forEach((p, ci) => {
      if (!P.has(p)) return;
      const i = +P.get(p), L = manifest.castes[CASTES[ci]];
      if (L && L.sections.some((x) => x.i === i)) { selC[ci] = i; lead = ci; }
    });
    matchDepth = P.get("match") === "1";
  }
  // through boot.js's BIMP.url (batched replaceState; null removes), or the same by hand
  function urlSet(k, v) {
    if (BIMP.url && BIMP.url.set) { BIMP.url.set(k, v); return; }
    try {
      const P = new URLSearchParams(location.search);
      if (v == null || v === false || v === "") P.delete(k); else P.set(k, v === true ? "" : String(v));
      const q = P.toString();
      history.replaceState(history.state, "", location.pathname + (q ? "?" + q : "") + location.hash);
    } catch (e) { /* history unavailable: nothing to keep */ }
  }
  function writeUrl() {
    if (!manifest) return;
    urlSet("histo", on ? "1" : null);
    urlSet("stain", null);
    urlSet("stains", active.join(",") === stains()[0].key ? null : active.join(","));
    urlSet("sint", active.filter((k) => sInt[k] !== 1).map((k) => k + ":" + sInt[k]).join(",") || null);
    urlSet("histo_op", opacity !== 1 ? String(opacity) : null);
    urlSet("sq", selC[0] != null ? String(selC[0]) : null);
    urlSet("sw", selC[1] != null ? String(selC[1]) : null);
    urlSet("match", matchDepth ? "1" : null);
    urlSet("view", mode2d ? "2d" : null);
    // the caste is ui.js's state, but boot.js only READS caste= and nothing wrote it, so a
    // copied Both + 2D link came back queen-only; written here until ui.js does it itself
    const f = BIMP.state.s.casteFilter;
    urlSet("caste", f < 0 ? "both" : f === 1 ? "worker" : null);
    inv();
  }

  // ---- panel controls ----------------------------------------------------------------------
  // Mounted into the layer row's settings slot (#layers-histo, atlas-ui: the Histology switch
  // itself is theirs, through setOn). Without that slot (older page), a group of our own with
  // its own switch goes before Reset view.
  function buildControl() {
    if ($("histo-more")) return;
    let mount = $("layers-histo"), own = false;
    if (!mount) {
      const anchor = $("btn-reset"); if (!anchor) return;
      mount = document.createElement("div"); mount.className = "grp"; mount.id = "histo-grp";
      mount.innerHTML = `<div class="toggle" id="tg-histo" role="button" tabindex="0" aria-pressed="false" ` +
        `title="The stained tissue of each section, under its cells"><span>Histology</span><div class="sw"></div></div>`;
      anchor.parentNode.insertBefore(mount, anchor);
      own = true;
    }
    const more = document.createElement("div");
    more.id = "histo-more"; more.hidden = !on;
    const rows = stains().map((st) =>
      `<div class="hs-row" data-key="${st.key}">` +
      `<button type="button" class="hs-chip" data-key="${st.key}" aria-pressed="false" title="${st.label}">` +
      `<i class="hs-sw"></i><span class="hs-lab">${st.label}</span></button>` +
      `<input class="hs-int" type="range" min="0" max="2" step="0.05" value="${sInt[st.key]}" aria-label="${st.label} intensity" hidden>` +
      `</div>`).join("");
    more.innerHTML =
      // one disclosure, styled like the panel's View section: a summary line naming the stains
      // shown, with their colour dots; open it to pick and mix
      `<details class="hs-disc" id="hs-disc"><summary class="sec-h"><span class="hs-t">Stains</span><span class="hs-sum"></span></summary>` +
      `<div class="hs-list">${rows}</div><div class="hs-hint"></div></details>` +
      `<div class="slider" id="sl-histo-op"><div class="row"><span>Strength</span><span class="val">${opacity.toFixed(2)}</span></div>` +
      `<input type="range" min="0.1" max="2" step="0.05" value="${opacity}" aria-label="Histology strength"></div>` +
      `<div class="slider" id="sl-histo-sec" title="Show one section at a time, front to back"><div class="row"><span>Section</span><span class="val">all</span></div>` +
      `<input type="range" min="0" max="1" step="1" value="0" aria-label="Histology section"></div>` +
      `<div class="toggle" id="tg-hothers" role="button" tabindex="0" aria-pressed="false" ` +
      `title="With a section picked in 3D, keep the rest of the stack faintly visible"><span>Show other sections</span><div class="sw"></div></div>` +
      `<div class="toggle" id="tg-hmatch" role="button" tabindex="0" aria-pressed="false" hidden ` +
      `title="Keep the queen and worker sections at the same depth"><span>Match depth</span><div class="sw"></div></div>` +
      `<div id="histo-note"></div>`;
    mount.appendChild(more);
    if (own) {
      const tg = $("tg-histo");
      tg.addEventListener("click", () => setOn(!on));
      tg.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); setOn(!on); } });
    }
    // open: phones start folded; desktop starts open the first time this visit, folded after;
    // a link may say otherwise (stainsui=open)
    const disc = more.querySelector("#hs-disc");
    let seen = false; try { seen = sessionStorage.getItem("bimp.hsSeen") === "1"; } catch (e) { /* storage off */ }
    const Pu = new URLSearchParams(location.search);
    const dflt = !PHONE && !seen;
    disc.open = Pu.has("stainsui") ? Pu.get("stainsui") === "open" : dflt;
    // setting .open above fires "toggle" too (later, asynchronously): only a state that differs
    // from the default goes into the URL, so plain links stay plain
    disc.addEventListener("toggle", () => {
      try { sessionStorage.setItem("bimp.hsSeen", "1"); } catch (e) { /* storage off */ }
      urlSet("stainsui", disc.open === dflt ? null : (disc.open ? "open" : "closed"));
    });
    more.querySelector(".hs-list").addEventListener("click", (e) => {
      const b = e.target.closest(".hs-chip"); if (b) toggleStain(b.dataset.key);
    });
    more.querySelectorAll(".hs-int").forEach((inp) => inp.addEventListener("input", (e) => {
      const k = e.target.closest(".hs-row").dataset.key;
      sInt[k] = Math.max(0, Math.min(2, parseFloat(e.target.value) || 0)); fill(e.target); writeUrl(); inv();
    }));
    const to = more.querySelector("#tg-hothers");
    const flipO = () => { showOthers = !showOthers; to.setAttribute("aria-pressed", String(showOthers)); urlSet("hothers", showOthers ? "1" : null); inv(); };
    to.addEventListener("click", flipO);
    to.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); flipO(); } });
    if (new URLSearchParams(location.search).get("hothers") === "1") flipO();
    const tm = more.querySelector("#tg-hmatch");
    tm.addEventListener("click", () => setMatch(!matchDepth));
    tm.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); setMatch(!matchDepth); } });
    more.querySelector("#sl-histo-op input").addEventListener("input", (e) => setOpacity(parseFloat(e.target.value)));
    more.querySelector("#sl-histo-sec input").addEventListener("input", (e) => setSection(parseInt(e.target.value, 10)));
    if (manifest.synthetic) more.querySelector("#histo-note").textContent = "Synthetic test images, not real histology.";
    syncStainUI(); setOpacity(opacity); syncSlider();
  }
  // the track's filled part, as ui.js's slider() draws it
  function fill(inp) { inp.style.setProperty("--fill", ((inp.value - inp.min) / (inp.max - inp.min) * 100) + "%"); }
  const composite = () => active.length > 1;
  function syncStainUI() {
    const more = $("histo-more"); if (!more) return;
    more.querySelectorAll(".hs-row").forEach((row) => {
      const k = row.dataset.key, onK = active.includes(k);
      const b = row.querySelector(".hs-chip"), inp = row.querySelector(".hs-int");
      b.setAttribute("aria-pressed", String(onK));
      // swatch = what is drawn: the stain's colour in a composite, grey when it is alone
      b.querySelector(".hs-sw").style.background = onK && !composite() ? "var(--ink)" : colOf(k);
      inp.hidden = !onK || !composite();          // per-stain intensity only matters when mixing
      inp.value = sInt[k]; fill(inp);
    });
    const hint = more.querySelector(".hs-hint");
    if (hint) hint.textContent = stains().length > 1 ? (composite() ? "Overlaid in colour; each has its own strength." : "Pick several to overlay them in colour.") : "";
    const sum = more.querySelector(".hs-sum");
    if (sum) sum.innerHTML = active.map((k) =>
      `<span class="hs-pill"><i style="background:${composite() ? colOf(k) : "var(--ink)"}"></i>${shortName(stainBy(k))}</span>`).join("");
  }
  function toggleStain(k) {
    if (!stainBy(k)) return;
    if (active.includes(k)) { if (active.length === 1) return; active = active.filter((x) => x !== k); }
    else { if (active.length >= MAXS) return; active = stains().map((x) => x.key).filter((x) => x === k || active.includes(x)); }
    syncStainUI(); rail.rebuild(); syncSlider(); writeUrl();
    pumpDirty = true; if (secs.length) pump();
  }
  function setStains(keys) {
    const ks = (keys || []).filter((k) => stainBy(k));
    if (!ks.length) return;
    active = stains().map((x) => x.key).filter((k) => ks.includes(k)).slice(0, MAXS);
    syncStainUI(); rail.rebuild(); syncSlider(); writeUrl();
    pumpDirty = true; if (secs.length) pump();
  }
  function setOpacity(v) {
    if (!(v >= 0)) return;
    opacity = Math.max(0.05, Math.min(2, v));   // above 1 is gain: real DAPI is mostly dark
    const el = $("sl-histo-op");
    if (el) { el.querySelector(".val").textContent = opacity.toFixed(2); const i = el.querySelector("input"); i.value = opacity; fill(i); }
    writeUrl();
  }
  // the brain the 2D slider and arrow keys drive
  function leadCaste() { const f = BIMP.state.s.casteFilter; return f >= 0 ? f : lead; }
  function syncSlider() {
    const el = $("sl-histo-sec"); if (!el) return;
    const inp = el.querySelector("input");
    let v = slotK, lab = slotK ? String(slotK) : "all", n;
    if (mode2d) {
      // 2D: the slider walks the lead brain's whole stack, like the rail
      const ci = leadCaste(), c = CASTES[ci], L = order(c);
      n = L.length; inp.min = "1";
      const sec = secOf(ci);
      v = sec ? L.indexOf(sec) + 1 : 1;
      lab = sec ? depthUm(c, sec) + " µm" : "—";
    } else {
      n = Math.max(1, ...CASTES.map((c) => shown(c).length)); inp.min = "0";
      const picked = CASTES.map((c, ci) => (selC[ci] != null && casteVisible(ci) ? [c, secOf(ci)] : null)).filter(Boolean);
      if (picked.length) { v = 0; lab = picked.map(([c, sec]) => depthUm(c, sec) + " µm").join(" · "); }
    }
    inp.max = String(n);
    inp.value = String(v); fill(inp);
    el.querySelector(".val").textContent = lab;
  }
  // Section slider: in 3D the k-th shown section of each visible brain (0 = all); in 2D the
  // k-th section of the lead brain's stack, exactly as if picked in the rail
  function setSection(k) {
    k = k | 0;
    if (mode2d) { const ci = leadCaste(), L = order(CASTES[ci]); if (L.length) select(ci, L[Math.max(0, Math.min(L.length - 1, k - 1))].i); return; }
    slotK = k; selC = [null, null];
    syncSlider(); rail.mark(); writeUrl();
  }
  function nearestByDepth(ci, um) {
    const c = CASTES[ci]; let best = null, bd = Infinity;
    order(c).forEach((x) => { const d = Math.abs(depthUm(c, x) - um); if (d < bd) { bd = d; best = x; } });
    return best;
  }
  // one section of one brain; the other brain is untouched unless "match depth" is on
  function select(ci, i, follow) {
    selC[ci] = i; lead = ci; slotK = 0;
    if ((follow || matchDepth) && BIMP.state.s.casteFilter < 0) {
      const o = nearestByDepth(1 - ci, depthUm(CASTES[ci], secOf(ci)));
      if (o) selC[1 - ci] = o.i;
    }
    syncSlider(); rail.mark(); view2d.slab(); writeUrl();
    pumpDirty = true; pump();
  }
  function clearCaste(ci) { selC[ci] = null; syncSlider(); rail.mark(); writeUrl(); }
  function setMatch(v) {
    matchDepth = !!v; rail.mark(); writeUrl();
    if (matchDepth) { const ci = leadCaste(); if (secOf(ci)) select(ci, selC[ci]); }
  }
  function setOn(v) {
    v = !!v;
    if (!manifest || v === on) return;
    on = v;
    const tg = $("tg-histo"), more = $("histo-more");
    if (tg) tg.setAttribute("aria-pressed", String(on));
    if (more) more.hidden = !on;
    if (on) { warming = false; clearTimeout(warmTimer); start(); }
    else { stop(); if (!mode2d) { selC = [null, null]; slotK = 0; syncSlider(); } }   // in 2D the section stays: it drives the cells
    cellsOver(on);
    if (on) { try { if (!sessionStorage.getItem("bimp.hsSeen")) setTimeout(() => sessionStorage.setItem("bimp.hsSeen", "1"), 0); } catch (e) { /* storage off */ } }
    rail.update(); writeUrl(); pumpDirty = true; inv();
    emit();
  }

  // Cells over histology: with the layer on, the cell dots would compete with the nuclei under
  // them, so the cells' opacity drops to CELLS_OVER — but only if the reader has not set it
  // themselves, and it goes back when the layer goes off (again only if they left it alone).
  // Moved through the panel's own Opacity slider, so the panel shows the truth.
  const CELLS_DEFAULT = 0.85, CELLS_OVER = 0.5;
  let dimmed = false;
  function cellsOver(v) {
    const s = BIMP.state.s, inp = document.querySelector("#sl-op input");
    const setOp = (x) => {
      if (inp) { inp.value = String(x); inp.dispatchEvent(new Event("input", { bubbles: true })); }
      else s.opacity = x;
    };
    if (v && !dimmed && Math.abs(s.opacity - CELLS_DEFAULT) < 1e-3) { setOp(CELLS_OVER); dimmed = true; }
    else if (!v && dimmed) { if (Math.abs(s.opacity - CELLS_OVER) < 1e-3) setOp(CELLS_DEFAULT); dimmed = false; }
  }

  // ---- loading ------------------------------------------------------------------------
  // The per-section entries, made once and kept while the layer is on (or being warmed).
  function ensureSecs() {
    if (secs.length) return;
    if (!abort) abort = new AbortController();
    CASTES.forEach((c, ci) => {
      const ex = excluded(c);
      order(c).filter((s) => !ex.has(s.i)).forEach((s, j) =>
        secs.push({ caste: c, ci, k: j + 1, s, a: 0, tx: {} }));
    });
  }
  function start() { ensureSecs(); pump(); }
  function stop() {
    if (abort) abort.abort();
    abort = null;
    upQ.forEach((u) => u.bmp.close()); upQ = [];
    if (gl) secs.forEach((q) => Object.values(q.tx).forEach((t) => { if (t.small) gl.deleteTexture(t.small.t); if (t.large) gl.deleteTexture(t.large.t); }));
    secs = []; loading = 0; largeBusy = false; warming = false;
    dropTiles();
  }
  // Intent: the reader is about to turn histology on (or switch to 2D). Decode and upload the
  // small images now; if they never do, let them go again after a while.
  function warm() {
    if (!manifest || on) return;
    warming = true;
    ensureSecs(); pumpDirty = true; pump();
    clearTimeout(warmTimer);
    warmTimer = setTimeout(() => { if (!on && !mode2d) stop(); }, 30000);
  }
  function casteVisible(ci) { const f = BIMP.state.s.casteFilter; return f < 0 || f === ci; }
  // how far a section is from where the reader is looking: its brain's pick, else the middle
  // (focus per brain computed once per pass, not per comparison)
  let focusK = [1, 1];
  function refocus() {
    focusK = [0, 1].map((ci) => {
      const L = secs.filter((x) => x.ci === ci);
      return selC[ci] != null ? (L.find((x) => x.s.i === selC[ci]) || {}).k || 1 : (L.length + 1) / 2;
    });
  }
  function prio(q) { return Math.abs(q.k - focusK[q.ci]) + (casteVisible(q.ci) ? 0 : 1000); }
  // keep at most 4 small fetch+decodes in flight, nearest first, every stain on
  function pump() {
    if (!(on || warming || mode2d) || !secs.length) return;
    if (loading >= 4) return;
    pumpDirty = false;
    refocus();
    const want = [];
    secs.forEach((q) => {
      if (!casteVisible(q.ci)) return;
      active.forEach((k) => {
        if (!hasStain(q.s, k)) return;
        const t = q.tx[k] || (q.tx[k] = { small: null, large: null, pending: false, failed: false, farSince: 0 });
        if (!t.small && !t.pending && !t.failed) want.push([prio(q), q, k, t]);
      });
    });
    want.sort((a, b) => a[0] - b[0]);
    for (const [, q, k, t] of want) {
      if (loading >= 4) break;
      t.pending = true; loading++;
      const run = abort;
      fetchBitmap(fileOf(k, q.caste, q.s, true)).then((bmp) => {
        if (run !== abort) { if (bmp) bmp.close(); return; }   // a fetch from before the last off
        loading--;
        // stays pending until drainUploads has made the texture, so it is fetched once
        if (bmp) { upQ.push({ q, k, bmp, large: false }); inv(); }
        else { t.pending = false; t.failed = true; }          // a missing image is not retried
        pump();
      });
    }
  }
  async function fetchBitmap(file) {
    const sig = abort && abort.signal;
    try {
      // tile templates may name another host outright (the tiles are served apart from the site)
      const r = await fetch(/^https?:\/\//.test(file) ? file : base + file, { signal: sig });
      if (!r.ok) return null;
      const blob = await r.blob();
      if (sig && sig.aborted) return null;
      // decoded off the main thread; premultiplying makes any alpha-masked area read as 0
      const bmp = await createImageBitmap(blob, { premultiplyAlpha: "premultiply", colorSpaceConversion: "none" });
      if (sig && sig.aborted) { bmp.close(); return null; }
      return bmp;
    } catch (e) { return null; }
  }
  function upload(bmp, mip) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, bmp.width, bmp.height, 0, gl.RED, gl.UNSIGNED_BYTE, bmp);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    if (mip) gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    const out = { t, w: bmp.width, h: bmp.height };
    bmp.close();
    return out;
  }
  // uploads within a per-frame budget (count and time), so warming or switching on never
  // costs a long frame
  function drainUploads() {
    const t0 = now();
    for (let n = 0; n < 3 && tUpQ.length && now() - t0 < 4; n++) {
      const { ent, bmp } = tUpQ.shift();
      ent.t = upload(bmp, true); ent.pending = false; ent.a = 0;
    }
    for (let n = 0; n < 3 && upQ.length && now() - t0 < 4; n++) {
      const { q, k, bmp, large } = upQ.shift();
      const t = q.tx[k]; if (!t) { bmp.close(); continue; }
      if (large) { if (t.large) gl.deleteTexture(t.large.t); t.large = upload(bmp, true); }
      else t.small = upload(bmp, true);
      t.pending = false;
    }
  }
  // Idle prefetch: once the atlas has settled, put the shown brain's small images of the first
  // stain into the HTTP cache (no decode), two at a time. Skipped when the reader asked to save
  // data or the connection is slow.
  // Readers on the lite tier (phones, weak devices), with Save-Data on, or on 2G/3G get no
  // idle prefetch: their histology loads on the first press of its control, and only the small
  // images (no large tier, no finer tiles) on the lite tier.
  const lite = () => BIMP.tier === "lite";
  function frugal() {
    const cn = navigator.connection || {};
    return lite() || !!cn.saveData || /(^|-)2g|3g/.test(cn.effectiveType || "");
  }
  function idlePrefetch() {
    if (frugal()) return;
    // never in the cells' way: wait until the atlas has fully revealed, then idle time
    const revealed = () => BIMP.fullReady && (!BIMP.renderer || !BIMP.renderer.getProgress || BIMP.renderer.getProgress() >= 1);
    const ric = window.requestIdleCallback || ((cb) => setTimeout(cb, 1500));
    const wait = () => (revealed() ? ric(go, { timeout: 5000 }) : setTimeout(wait, 700));
    const go = () => {
      if (on || frugal()) return;
      const f = BIMP.state.s.casteFilter, cis = f < 0 ? [0, 1] : [f];
      const urls = [];
      cis.forEach((ci) => order(CASTES[ci]).forEach((s) => { if (hasStain(s, active[0])) urls.push(base + fileOf(active[0], CASTES[ci], s, true)); }));
      prefetchUrls(urls, 2);
    };
    wait();
  }
  const prefetched = new Set();
  function prefetchUrls(urls, par) {
    const todo = urls.filter((u) => !prefetched.has(u));
    todo.forEach((u) => prefetched.add(u));
    let i = 0;
    const next = () => {
      if (i >= todo.length) return;
      const u = todo[i++];
      fetch(u, { priority: "low" }).then((r) => r.blob()).catch(() => {}).then(next);
    };
    for (let j = 0; j < par; j++) next();
  }
  // a stain hovered in the menu: its small set into the HTTP cache
  function prefetchStain(k) {
    if (!manifest || !stainBy(k)) return;
    const f = BIMP.state.s.casteFilter, cis = f < 0 ? [0, 1] : [f];
    const urls = [];
    cis.forEach((ci) => order(CASTES[ci]).forEach((s) => { if (hasStain(s, k)) urls.push(base + fileOf(k, CASTES[ci], s, true)); }));
    prefetchUrls(urls, 3);
  }
  function intent() {
    const sel = '#ly-histo, #tg-histo, #view-mode button[data-v="2d"], .hs-chip';
    const hit = (e) => {
      const t = e.target && e.target.closest && e.target.closest(sel);
      if (!t) return;
      if (t.classList.contains("hs-chip")) { prefetchStain(t.dataset.key); if (on) return; }
      warm();
    };
    ["pointerover", "focusin"].forEach((ev) => document.addEventListener(ev, hit, true));
    document.addEventListener("touchstart", hit, { capture: true, passive: true });
  }

  // ---- geometry -----------------------------------------------------------------------
  // Mirror the point vertex shader: data (x,y,z) → world (y, −x, z·depth) + caste split.
  // A section may instead carry `corners` [[x,y] at (col 0,row 0), [x,y] at (last col edge,
  // row 0), [x,y] at (col 0, last row edge)] (outer edges, data frame): any affine, so a
  // per-section rotation needs no resampling on the export side.
  function quad(sec, zMul, off) {
    const ox = off ? off[0] : 0, oy = off ? off[1] : 0;
    const z = sec.z * zMul, K = sec.corners;
    if (K) {
      return {
        o: [K[0][1] + ox, -K[0][0] + oy, z],
        u: [K[1][1] - K[0][1], -(K[1][0] - K[0][0]), 0],
        v: [K[2][1] - K[0][1], -(K[2][0] - K[0][0]), 0],
      };
    }
    const [x0, y0, x1, y1] = sec.bounds;
    return {
      o: [y0 + ox, -x0 + oy, z],         // texel (0,0): column 0 at x0, row 0 at y0
      u: [0, -(x1 - x0), 0],             // along image columns (+x data → −y world)
      v: [y1 - y0, 0, 0],                // along image rows    (+y data → +x world)
    };
  }
  // The caste split as the cells have it: renderer.getSplit() gives {mag, dir} (side by side on
  // desktop, dir [1,0]; stacked on a portrait phone, dir [0,-1]); queen at −mag·dir, worker at +.
  function splitOf(f, ci) {
    if (f >= 0) return [0, 0];
    const R = BIMP.renderer, sp = R && R.getSplit ? R.getSplit() : { mag: SPLIT, dir: [1, 0] };
    const k = (ci === 0 ? -1 : 1) * sp.mag;
    return [k * sp.dir[0], k * sp.dir[1]];
  }
  const mix2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  // on-screen size of a quad in device pixels (its longest projected edge)
  function screenPx(mvp, g, W, H) {
    const P = (a, b) => {
      const x = g.o[0] + a * g.u[0] + b * g.v[0], y = g.o[1] + a * g.u[1] + b * g.v[1], z = g.o[2] + a * g.u[2] + b * g.v[2];
      const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
      if (cw <= 1e-4) return null;
      return [((mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12]) / cw) * W / 2, ((mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13]) / cw) * H / 2];
    };
    const p0 = P(0, 0), pu = P(1, 0), pv = P(0, 1);
    if (!p0 || !pu || !pv) return Infinity;
    return Math.max(Math.hypot(pu[0] - p0[0], pu[1] - p0[1]), Math.hypot(pv[0] - p0[0], pv[1] - p0[1]));
  }
  // Large image wanted while a section spans more pixels than its small texture holds;
  // nearest the reader's section first, one in flight. Hysteresis both ways, and a large one
  // is let go only after it has been too small (or not drawn) for a few seconds. Phones keep
  // to the small images.
  function lodPass(mvp, items, W, H, t) {
    const drawn = new Set(items.map((it) => it.q));
    for (const q of secs) {
      for (const k in q.tx) {
        const x = q.tx[k];
        if (!x.large || (drawn.has(q) && active.includes(k))) continue;
        if (!x.farSince) x.farSince = t;
        else if (t - x.farSince > 3000) { gl.deleteTexture(x.large.t); x.large = null; x.farSince = 0; }
      }
    }
    refocus();
    const byNear = items.slice().sort((a, b) => prio(a.q) - prio(b.q));
    for (const it of byNear) {
      const q = it.q, px = screenPx(mvp, it.g, W, H);
      for (const k of active) {
        const x = q.tx[k]; if (!x || !x.small) continue;
        const sw = Math.max(x.small.w, x.small.h);
        const big = fileOf(k, q.caste, q.s, false), lil = fileOf(k, q.caste, q.s, true);
        if (!x.large) {
          if (!PHONE && !lite() && px > sw * 1.4 && !largeBusy && !x.pending && big !== lil) {
            largeBusy = true; x.pending = true;
            const run = abort;
            fetchBitmap(big).then((bmp) => {
              if (run !== abort) { if (bmp) bmp.close(); return; }
              largeBusy = false;
              if (bmp && on) { upQ.push({ q, k, bmp, large: true }); inv(); }
              else { x.pending = false; if (bmp) bmp.close(); }
            });
          }
          x.farSince = 0;
        } else if (px < sw * 1.1) {
          if (!x.farSince) x.farSince = t;
          else if (t - x.farSince > 3000) { gl.deleteTexture(x.large.t); x.large = null; x.farSince = 0; }
        } else x.farSince = 0;
      }
    }
  }

  // ---- high-resolution tiles ------------------------------------------------------------
  // Optional pyramid per section (manifest "tiles": {path, tile}; per section tiles: {stains,
  // levels: [{l, w, h, nx, ny, have, stains?}]}; level 0 finest, the last the overview's own
  // size). Used only for a section shown at full strength (2D, or picked in 3D): the coarsest
  // level is its base, drawn over the overview and fading in, after which the overview is
  // dropped (the tiles share one per-brain stretch, the overview does not); the finest level
  // whose size is no more than ~1.5 screen pixels per texel is drawn over that, tile by tile
  // as they arrive. Only tiles on screen are fetched, nearest the centre first, three at a
  // time, within a texture budget; least recently used ones are let go.
  const TILE_BUDGET = PHONE ? 24 : 96;
  const tcache = new Map();             // key → {t, a, used, pending, failed}
  let tLoading = 0, lastPlan = null;
  const tUpQ = [];
  const tkey = (q, k, l, tx, ty) => `${q.ci}/${q.s.i}/${k}/${l}/${tx}_${ty}`;
  function tilesOf(sec) { return manifest && manifest.tiles && sec.tiles && Array.isArray(sec.tiles.levels) && sec.tiles.levels.length ? sec.tiles : null; }
  function tStainOK(T, lv, k) { const L = lv.stains || T.stains; return !L || L.includes(k); }
  function tPath(q, k, l, tx, ty) {
    return (q.s.tiles.path || manifest.tiles.path).replace(/\{caste\}/g, q.caste).replace(/\{ii\}/g, pad2(q.s.i))
      .replace(/\{key\}|\{stain\}/g, k).replace(/\{l\}/g, String(l)).replace(/\{tx\}|\{col\}/g, String(tx)).replace(/\{ty\}|\{row\}/g, String(ty));
  }
  function subQuad(g, lv, tx, ty, size) {
    const u0 = tx * size / lv.w, u1 = Math.min((tx + 1) * size, lv.w) / lv.w;
    const v0 = ty * size / lv.h, v1 = Math.min((ty + 1) * size, lv.h) / lv.h;
    return {
      o: [0, 1, 2].map((c) => g.o[c] + u0 * g.u[c] + v0 * g.v[c]),
      u: g.u.map((x) => x * (u1 - u0)), v: g.v.map((x) => x * (v1 - v0)),
    };
  }
  // With "overlap": n, a tile image holds n px more on each side that has a neighbour (never
  // past the level's edges), so linear filtering has real neighbours instead of a clamped edge.
  // The quad stays the 512-px grid cell; this is the window of the image that cell occupies.
  function cellUV(lv, tx, ty, size) {
    const ov = (manifest.tiles && manifest.tiles.overlap) || 0;
    const ax = (t, n) => {
      const c0 = t * size, c1 = Math.min((t + 1) * size, n);
      const p0 = Math.max(0, c0 - ov), p1 = Math.min(n, (t + 1) * size + ov);
      return [(c0 - p0) / (p1 - p0), (c1 - c0) / (p1 - p0)];
    };
    const [ox, sx] = ax(tx, lv.w), [oy, sy] = ax(ty, lv.h);
    return [ox, oy, sx, sy];
  }
  // on screen? and how far from the centre (NDC)
  function ndcBox(mvp, g) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [a, b] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const x = g.o[0] + a * g.u[0] + b * g.v[0], y = g.o[1] + a * g.u[1] + b * g.v[1], z = g.o[2] + a * g.u[2] + b * g.v[2];
      const w = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
      if (w <= 1e-4) return { d: 0 };
      const nx = (mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12]) / w, ny = (mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13]) / w;
      x0 = Math.min(x0, nx); x1 = Math.max(x1, nx); y0 = Math.min(y0, ny); y1 = Math.max(y1, ny);
    }
    if (x1 < -1 || x0 > 1 || y1 < -1 || y0 > 1) return null;
    return { d: Math.hypot((x0 + x1) / 2, (y0 + y1) / 2) };
  }
  // what to draw and what to fetch for one full-strength section
  function tilePlan(it, mvp, W, H, t, dt) {
    const T = tilesOf(it.q.s); if (!T) return null;
    const size = manifest.tiles.tile || manifest.tiles.size || 512;
    const lvs = T.levels.slice().sort((a, b) => a.l - b.l), base = lvs[lvs.length - 1];
    const px = screenPx(mvp, it.g, W, H);
    // the coarsest level that still gives at least ~one texel per screen pixel (the finest when
    // zoomed past it). The lite tier keeps to the base in 3D; in 2D, one section at a time, it
    // gets the finer levels too: the tile budget (24 on phones) is what bounds memory there
    let fine = base;
    if (!lite() || mode2d) { fine = lvs[0]; for (let j = lvs.length - 1; j >= 0; j--) if (Math.max(lvs[j].w, lvs[j].h) >= px * 1.05) { fine = lvs[j]; break; } }   // never magnified while a finer level exists
    const draws = [], want = [];
    lastPlan = { px: Math.round(px), fine: fine.l, levels: lvs.map((v) => [v.l, v.w, v.h]) };
    let baseAll = true, any = false;
    for (const lv of fine === base ? [base] : [base, fine]) {
      for (let ty = 0; ty < lv.ny; ty++) for (let tx = 0; tx < lv.nx; tx++) {
        if (lv.have && lv.have[ty * lv.nx + tx] !== "1") continue;
        const g = subQuad(it.g, lv, tx, ty, size), vis = ndcBox(mvp, g);
        if (!vis) continue;
        const ks = drawK.filter((k) => tStainOK(T, lv, k));
        let a = 1, ready = true;
        const texs = drawK.map((k) => {
          if (!ks.includes(k)) return null;
          const key = tkey(it.q, k, lv.l, tx, ty), e = tcache.get(key);
          want.push({ key, q: it.q, k, l: lv.l, tx, ty, d: vis.d - (lv === base ? 10 : 0) });
          if (e && e.t) { e.used = t; e.a = Math.min(1, e.a + (REDUCED ? 1 : dt / 0.25)); a = Math.min(a, e.a); return e.t; }
          ready = false; return null;
        });
        if (lv === base && (!ready || a < 1)) baseAll = false;
        if (texs.some(Boolean)) { draws.push({ g, texs, a, base: lv === base, uv: cellUV(lv, tx, ty, size) }); any = true; }
      }
    }
    return { draws, want, skipOverview: any && baseAll };
  }
  function tileRequests(want, t) {
    want.sort((a, b) => a.d - b.d);
    const keep = new Set(want.slice(0, TILE_BUDGET).map((w) => w.key));
    for (const w of want.slice(0, TILE_BUDGET)) {
      if (tLoading >= 3) break;
      const e = tcache.get(w.key);
      if (e && (e.t || e.pending || e.failed)) continue;
      const ent = e || { t: null, a: 0, used: t, pending: false, failed: false };
      tcache.set(w.key, ent);
      ent.pending = true; tLoading++;
      const run = abort;
      fetchBitmap(tPath(w.q, w.k, w.l, w.tx, w.ty)).then((bmp) => {
        if (run !== abort) { if (bmp) bmp.close(); return; }
        tLoading--;
        if (bmp && tcache.get(w.key) === ent) { tUpQ.push({ ent, bmp }); inv(); }
        else { ent.pending = false; ent.failed = !bmp; if (bmp) bmp.close(); }
      });
    }
    // over budget: let the least recently used tiles go (never the ones wanted now)
    if (tcache.size > TILE_BUDGET) {
      const old = [...tcache.entries()].filter(([k, e]) => !keep.has(k) && !e.pending).sort((a, b) => a[1].used - b[1].used);
      for (const [k, e] of old.slice(0, tcache.size - TILE_BUDGET)) { if (e.t && gl) gl.deleteTexture(e.t.t); tcache.delete(k); }
    }
  }
  function dropTiles() {
    tUpQ.forEach((u) => u.bmp.close()); tUpQ.length = 0;
    if (gl) tcache.forEach((e) => { if (e.t) gl.deleteTexture(e.t.t); });
    tcache.clear(); tLoading = 0;
  }

  // ---- caste-split clock ----------------------------------------------------------------
  // The cells slide between split positions over the renderer's 0.55 s morph with an
  // ease-in-out cubic; the sections follow on their own copy of that clock (same duration,
  // same curve), so the renderer hook needs no more than the matrices.
  const MORPH_SECS = 0.55;
  const REDUCED = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  let curCaste = null, prevCaste = null, morphT = 1;
  function syncSplit(dt) {
    const f = BIMP.state.s.casteFilter;
    if (curCaste === null) { curCaste = prevCaste = f; morphT = 1; }
    else if (f !== curCaste) { prevCaste = curCaste; curCaste = f; morphT = 0; }
    if (morphT < 1) morphT = REDUCED ? 1 : Math.min(1, morphT + dt / MORPH_SECS);
    return { morph: ease(morphT), prevCaste };
  }
  function kMax(ci) { let m = 0; for (const q of secs) if (q.ci === ci && q.k > m) m = q.k; return m; }
  // How opaque a section should be (0 = not drawn). In 2D exactly the section whose cells are
  // shown (the same manifest entry the slab is cut from), opaque, nothing else. In 3D a picked
  // section is opaque and the rest of its brain hidden, or faint with "show other sections";
  // with nothing picked the whole stack shows at a low opacity, so it reads as layered tissue
  // rather than a wall (tied to Strength, so turning that up thickens the stack).

  // ~30 near-coincident sections cover each pixel, so a layer's own opacity must be low for the
  // stack to stay see-through (0.09 → ~0.9 cover from 30 layers); a composite is thinned by √n
  // because n stain colours add toward white within each layer.
  const stackAlpha = () => Math.max(0.03, Math.min(0.4, 0.09 * opacity / Math.sqrt(Math.max(1, Object.values(sw).reduce((a, b) => a + b, 0)))));
  function wanted(q) {
    if (mode2d && !casteVisible(q.ci)) return 0;          // 2D: a brain off screen draws nothing, not even fading
    if (selC[q.ci] != null) return q.s.i === selC[q.ci] ? 1 : (!mode2d && showOthers ? 0.6 * stackAlpha() : 0);
    if (mode2d) return 0;
    if (slotK) return q.k === Math.min(slotK, kMax(q.ci)) ? 1 : (showOthers ? 0.6 * stackAlpha() : 0);
    return stackAlpha();
  }
  let showOthers = false;

  // ---- draw (renderer drawScene, before the cells, never in the pick pass) -------------
  // Every section eases its own alpha toward 1 when it should show and has an image, else 0,
  // over XFADE: switching sections crossfades instead of cutting, and a section whose image
  // arrives fades in. In 2D a brain off screen drops out at once.
  const XFADE = 0.15;
  let lastCaste = null;
  function stageLight() {
    // The renderer clears to look.bg in the atlas (its own stage colour), so read that rather
    // than asking GL for the clear colour every frame (a synchronous query stalls Chrome).
    const R = BIMP.renderer, b = R && R.look && R.look.bg;
    if (b) return 0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2] > 0.5;
    if (lightCache == null) { const cc = gl.getParameter(gl.COLOR_CLEAR_VALUE); lightCache = 0.2126 * cc[0] + 0.7152 * cc[1] + 0.0722 * cc[2] > 0.5; }
    return lightCache;
  }
  function easeStains(dt) {
    const k1 = REDUCED ? 1 : dt / 0.25;
    for (const st of stains()) {
      const k = st.key, want = active.includes(k) ? 1 : 0, cur = sw[k] || 0;
      if (want && cur < 1) {
        const ready = secs.every((q) => !(q.at > 0) || !hasStain(q.s, k) || (q.tx[k] && (q.tx[k].small || q.tx[k].failed)));
        if (ready || !cur) sw[k] = ready ? Math.min(1, cur + k1) : cur;
      } else if (!want && cur > 0) sw[k] = Math.max(0, cur - k1);
    }
    // the first stain shown is on at once (nothing to crossfade from)
    if (active.length === 1 && !stains().some((st) => (sw[st.key] || 0) > 0)) sw[active[0]] = 1;
    const target = composite() ? 1 : 0;
    cmix = cmix < target ? Math.min(target, cmix + k1) : Math.max(target, cmix - k1);
    drawK = stains().map((st) => st.key).filter((k) => (sw[k] || 0) > 0.001).slice(0, MAXS);
    if (!drawK.length) drawK = active.slice(0, 1);
  }
  const stainsSettled = () => stains().every((st) => (sw[st.key] || 0) === (active.includes(st.key) ? 1 : 0)) && cmix === (composite() ? 1 : 0);
  // ---- the home page's story: one real section under the cut ---------------------------
  // The home page's last story step thins the hero to one section; this draws that section's
  // DAPI image under it, fading with the cut (homeK, set by boot.js's story every frame). It
  // is separate from the layer: no panel, no rail, no URL, the same-origin overview image
  // only, fetched on intent (prepHome, once the story is on its way) and decoded off-thread.
  const home = { k: 0, sec: null, tex: null, bmp: null, pending: false };
  async function prepHome(zWant) {
    if (!base) await init(window.__BIMP_SRC__);
    if (!manifest) return null;
    const L = order("queen");
    let best = null, bd = Infinity;
    L.forEach((x) => { const d = Math.abs(x.z - zWant); if (d < bd) { bd = d; best = x; } });
    if (!best) return null;
    if (home.sec !== best) {
      home.sec = best;
      if (home.tex && gl) { gl.deleteTexture(home.tex.t); home.tex = null; }
      if (!home.pending) {
        home.pending = true;
        const bmp = await fetchBitmap(fileOf("dapi", "queen", best, false));
        home.pending = false;
        if (bmp && home.sec === best) { home.bmp = bmp; inv(); } else if (bmp) bmp.close();
      }
    }
    return best.z;
  }
  function dropHome() {
    clearTimeout(home.idle); home.idle = 0;
    if (home.tex && gl) gl.deleteTexture(home.tex.t);
    if (home.bmp) home.bmp.close();
    home.tex = null; home.bmp = null; home.sec = null; home.k = 0;
  }
  function setHome(k) {
    k = Math.max(0, Math.min(1, k || 0));
    if (k === home.k) return;
    home.k = k; inv();
    // nothing held once it is gone: at once in the atlas, after a pause at home (so going back
    // and forth between the story's stops does not refetch)
    clearTimeout(home.idle); home.idle = 0;
    if (!k) {
      if (document.documentElement.dataset.view === "atlas") dropHome();
      else home.idle = setTimeout(() => { if (!home.k) dropHome(); }, 15000);
    }
  }
  // the story's API: warm on intent, then draw at an alpha that follows the cut
  function warmHome(z) { return prepHome(z).catch(() => null); }
  function homeSection(z, alpha) {
    if (alpha > 0 && (!home.sec || home.z !== z) && !home.pending) { home.z = z; prepHome(z).catch(() => {}); }
    setHome(alpha);
  }
  function drawHome(glx, mvp) {
    if (!gl) setupGL(glx);
    if (home.bmp) { home.tex = upload(home.bmp, true); home.bmp = null; }
    if (!home.tex || !home.sec || home.k <= 0.002) return false;
    const P = progs[1] || (progs[1] = makeProg(1));
    const g = quad(home.sec, BIMP.state.s.z, [0, 0]);
    gl.useProgram(P.prog);
    gl.uniformMatrix4fv(P.mvp, false, mvp);
    gl.uniform1f(P.light, 0);
    gl.uniform3fv(P.col, new Float32Array([0.97, 0.975, 0.965, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    const L = levelsOf("dapi");
    gl.uniform1fv(P.lo, new Float32Array([L.lo, 0, 0, 0])); gl.uniform1fv(P.hi, new Float32Array([L.hi, 1, 1, 1]));
    gl.uniform1fv(P.gam, new Float32Array([L.g, 1, 1, 1]));
    gl.uniform1fv(P.int, new Float32Array([1, 0, 0, 0]));
    gl.uniform1f(P.fade, home.k);
    gl.uniform3fv(P.o, g.o); gl.uniform3fv(P.u, g.u); gl.uniform3fv(P.v, g.v);
    gl.uniform4fv(P.uv, [0, 0, 1, 1]);
    gl.bindVertexArray(vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, home.tex.t);
    gl.depthMask(false);
    gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(true);
    return true;
  }

  function draw(glx, mvp) {
    const t = now(), dt = lastT ? Math.min((t - lastT) / 1000, 0.05) : 0; lastT = t;
    // the home page's section (home.k > 0 only while the story's last step shows it)
    // the home section never survives into the atlas, whatever the caller left it at
    if ((home.k > 0 || home.tex || home.bmp) && document.documentElement.dataset.view === "atlas") dropHome();
    if (home.k > 0 || home.bmp) {
      const prev = glx.getParameter(glx.CURRENT_PROGRAM);
      drawHome(glx, mvp); glx.useProgram(prev);
      if (!on) return true;
    }
    const sp = syncSplit(dt);
    rail.sync(); view2d.sync(); view2d.tick(dt);
    if ((warming || on) && !gl) setupGL(glx);
    if (gl && (upQ.length || tUpQ.length)) drainUploads();       // warming uploads run before the layer is on
    if (!on || !secs.length) { lastDrawn = []; return false; }
    // the home page draws its hero with this renderer in ink; the layer belongs to the atlas
    const R = BIMP.renderer;
    if (R && R.look && R.look.ink > 0.01) return false;
    easeStains(dt);
    const s = BIMP.state.s;
    if (s.casteFilter !== lastCaste) { lastCaste = s.casteFilter; pumpDirty = true; writeUrl(); }
    if (pumpDirty) pump();                    // a caste that just became visible starts loading
    for (let ci = 0; ci < 2; ci++) {
      const want = casteVisible(ci) ? 1 : 0;
      fadeC[ci] += (want - fadeC[ci]) * Math.min(1, dt / 0.25);   // lands with the 0.55 s cell morph
      if (Math.abs(want - fadeC[ci]) < 1e-3) fadeC[ci] = want;
    }
    const items = [];
    const step = REDUCED ? 1 : dt / XFADE;
    for (const q of secs) {
      const texs = drawK.map((k) => { const x = q.tx[k]; return x ? (x.large || x.small) : null; });
      const has = texs.some(Boolean);
      q.at = has ? wanted(q) : 0;
      if (mode2d && !casteVisible(q.ci)) q.a = 0;
      else q.a = q.a < q.at ? Math.min(q.at, q.a + step) : Math.max(q.at, q.a - step);
      if (!has || q.a <= 0.001) continue;
      const vis = fadeC[q.ci];
      if (vis <= 0.002) continue;
      const prev = sp.prevCaste == null ? s.casteFilter : sp.prevCaste;
      // leaving caste fades where it stood; arriving or staying caste travels with its cells
      const sx = casteVisible(q.ci)
        ? (prev < 0 || prev === q.ci ? mix2(splitOf(prev, q.ci), splitOf(s.casteFilter, q.ci), sp.morph) : splitOf(s.casteFilter, q.ci))
        : splitOf(prev, q.ci);
      const g = quad(q.s, s.z, sx);
      // depth of the section's centre, for back-to-front order
      const cx = g.o[0] + g.u[0] / 2 + g.v[0] / 2, cy = g.o[1] + g.u[1] / 2 + g.v[1] / 2, cz = g.o[2];
      items.push({ q, texs, g, a: vis * q.a, w: mvp[3] * cx + mvp[7] * cy + mvp[11] * cz + mvp[15] });
    }
    items.sort((a, b) => b.w - a.w);          // farthest first: "over" needs back-to-front
    // high-resolution tiles for sections shown at full strength
    if (manifest.tiles) {
      const want = [];
      for (const it of items) {
        if (it.q.at < 1) continue;
        it.plan = tilePlan(it, mvp, gl.drawingBufferWidth, gl.drawingBufferHeight, t, dt);
        if (it.plan) want.push(...it.plan.want);
      }
      if (want.length) tileRequests(want, t);
    }
    lastDrawn = items.filter((it) => it.q.at > 0).map((it) => [it.q.ci, it.q.s.i]);
    const W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
    if (t - lastLodT > 250) {
      lastLodT = t; lodPass(mvp, items.filter((it) => it.q.at > 0), W, H, t);
      lodView = Float32Array.from(mvp); lodView.w = W; lodView.h = H;
    }
    // Frames are drawn on demand: if the throttle skipped the large-image pass while the view
    // had changed since the last one, ask for one more frame just after it opens, so a view that
    // has stopped still gets its large images. Only when changed: an unconditional follow-up
    // would itself be a frame inside the next window and keep the atlas drawing forever.
    else if (!lodTimer && !sameView(mvp, W, H)) lodTimer = setTimeout(() => { lodTimer = 0; inv(); }, 270 - (t - lastLodT));
    if (!items.length) return false;
    // The renderer's state at this hook is known (render() set it, drawScene rebinds its own VAO
    // after us), so it is restored by value, not read back with getParameter each frame.
    if (!rprog) rprog = gl.getParameter(gl.CURRENT_PROGRAM);   // once: the point program
    const n = Math.max(1, Math.min(MAXS, drawK.length)), P = progs[n] || (progs[n] = makeProg(n));
    gl.useProgram(P.prog);
    gl.uniformMatrix4fv(P.mvp, false, mvp);
    // light stage → stains absorb from the page (MIN); dark stage → stains emit (MAX)
    const light = stageLight();
    gl.uniform1f(P.light, light ? 1 : 0);
    // colours: one stain alone is greyscale in the page's neutral (light on dark, ink on paper);
    // several are each in their own colour
    const cols = new Float32Array(12), ints = new Float32Array(4);
    const grey = light ? [11 / 255, 11 / 255, 10 / 255] : [0.97, 0.975, 0.965];
    drawK.forEach((k, j) => {
      const h = hexRgb(colOf(k)), c = grey.map((g0, i) => g0 + (h[i] - g0) * cmix);
      cols.set(c, j * 3);
    });
    gl.uniform3fv(P.col, cols);
    const lo = new Float32Array(4), hi = new Float32Array(4).fill(1), ga = new Float32Array(4).fill(1);
    drawK.forEach((k, j) => { const L = levelsOf(k); lo[j] = L.lo; hi[j] = L.hi; ga[j] = L.g; });
    gl.uniform1fv(P.lo, lo); gl.uniform1fv(P.hi, hi); gl.uniform1fv(P.gam, ga);
    gl.bindVertexArray(vao);
    gl.depthMask(false);
    gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // premultiplied over
    const FULL = [0, 0, 1, 1];
    const quadDraw = (g, texs, a, uv) => {
      gl.uniform3fv(P.o, g.o); gl.uniform3fv(P.u, g.u); gl.uniform3fv(P.v, g.v);
      gl.uniform4fv(P.uv, uv || FULL);
      for (let j = 0; j < n; j++) {
        const tx = texs[j];
        const k = drawK[j];
        ints[j] = tx ? opacity * (1 + ((sInt[k] || 1) - 1) * cmix) * (sw[k] || 0) : 0;
        gl.activeTexture(gl.TEXTURE0 + j);
        gl.bindTexture(gl.TEXTURE_2D, tx ? tx.t : black);
      }
      gl.uniform1fv(P.int, ints);
      gl.uniform1f(P.fade, a);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };
    for (const it of items) {
      // overview, unless this section's base tiles have all arrived and faded in
      if (!(it.plan && it.plan.skipOverview)) quadDraw(it.g, it.texs, it.a);
      // then its tiles: the base level first, finer ones over it, each fading in as it lands
      if (it.plan) for (const d of it.plan.draws) quadDraw(d.g, d.texs, it.a * d.a, d.uv);
    }
    for (let j = n - 1; j >= 0; j--) { gl.activeTexture(gl.TEXTURE0 + j); gl.bindTexture(gl.TEXTURE_2D, null); }
    // back to what render() left: point program, no VAO (drawScene binds its own), default
    // texture unit, additive blend equation (the blend func was never changed), depth writes on
    gl.bindVertexArray(null);
    gl.useProgram(rprog);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // render()'s own
    gl.depthMask(true);
    return true;
  }
  // Something is still easing, or waiting on this frame loop to finish (a decoded image to
  // upload). Network waits are not counted: their arrival calls invalidate().
  function isAnimating() {
    if (upQ.length || tUpQ.length || view2d.animating()) return true;
    if (on && !stainsSettled()) return true;
    for (const e of tcache.values()) if (e.t && e.a < 1) return true;
    if (!on) return false;
    if (morphT < 1) return true;
    for (let ci = 0; ci < 2; ci++) if (fadeC[ci] !== (casteVisible(ci) ? 1 : 0)) return true;
    for (const q of secs) if (q.at !== undefined && q.a !== q.at) return true;
    return false;
  }
  function makeProg(n) {
    const sh = (type, src) => {
      const o = gl.createShader(type); gl.shaderSource(o, src); gl.compileShader(o);
      if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error("histo shader: " + gl.getShaderInfoLog(o));
      return o;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fsFor(n)));
    gl.linkProgram(prog);
    const u = (nm) => gl.getUniformLocation(prog, nm);
    const P = { prog, mvp: u("uMVP"), o: u("uO"), u: u("uU"), v: u("uV"), uv: u("uUV"), col: u("uCol[0]"), int: u("uInt[0]"), light: u("uLight"), fade: u("uFade"),
      lo: u("uLo[0]"), hi: u("uHi[0]"), gam: u("uGam[0]") };
    const cur = gl.getParameter(gl.CURRENT_PROGRAM);
    gl.useProgram(prog);
    for (let j = 0; j < n; j++) gl.uniform1i(u("uT" + j), j);
    gl.useProgram(cur);
    return P;
  }
  function setupGL(glx) {
    gl = glx;
    const prevVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
    vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(prevVao);
    // a 1×1 black texture for a stain slot whose image has not arrived
    black = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, black);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
  // for tests / the console
  function stats() {
    const all = secs.flatMap((q) => Object.entries(q.tx).map(([k, t]) => [k, t]));
    return { on, warming, mode2d, stains: active.slice(), sel: selC.slice(), lead, match: matchDepth,
      slab: (BIMP.state.s.slab || []).slice(), drawn: lastDrawn, sections: secs.length,
      small: all.filter(([, t]) => t.small).length, large: all.filter(([, t]) => t.large).length,
      queued: upQ.length, loading };
  }

  // ---- camera: face the picked section(s) ----------------------------------------------
  // Eases the camera round to look straight at the picked section of each brain on screen
  // (both, side by side at their split offsets, in Both), from the side the home view sees
  // them from, fitted to the visible stage. Only the rail's explicit "Face" button and
  // entering 2D call this; picking a section never moves the camera.
  function secOf(ci) {
    if (selC[ci] == null) return null;
    return order(CASTES[ci]).find((x) => x.i === selC[ci]) || null;
  }
  function face(ms) {
    const cam = BIMP.camera; if (!cam || !cam.flyTo) return;
    const s = BIMP.state.s;
    const picks = [0, 1].filter((ci) => casteVisible(ci) && secOf(ci));
    if (!picks.length) return;
    if (s.spin) { s.spin = false; const t = $("tg-spin"); if (t) t.setAttribute("aria-pressed", "false"); }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, z = 0;
    picks.forEach((ci) => {
      const g = quad(secOf(ci), s.z, splitOf(s.casteFilter, ci));
      [[0, 0], [1, 0], [0, 1], [1, 1]].forEach(([a, b]) => {
        const x = g.o[0] + a * g.u[0] + b * g.v[0], y = g.o[1] + a * g.u[1] + b * g.v[1];
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      });
      z += g.o[2] / picks.length;
    });
    // Fit to the visible stage: the canvas runs under both side panels and the section rail,
    // so size to what is left between them and shift the target so the section(s) sit in the
    // middle of that, not of the canvas. (Front-on, screen right is world +x.)
    const cv = $("gl"), cs = getComputedStyle(document.documentElement);
    const CW = cv ? cv.clientWidth : innerWidth, H = cv ? cv.clientHeight : innerHeight;
    const lw = parseFloat(cs.getPropertyValue("--lw")) || 0;
    let rw = parseFloat(cs.getPropertyValue("--rw")) || 0;
    const rl = $("histo-rail");
    if (rl && rl.classList.contains("on") && !rl.classList.contains("collapsed") && !PHONE) rw += rl.offsetWidth + 10;
    const W = Math.max(200, CW - lw - rw);
    const t = Math.tan(24 * Math.PI / 180);
    const r = Math.max(cam.cam.minR, 1.1 * Math.max((y1 - y0) / 2, ((x1 - x0) / 2) * (H / W)) / t);
    const dxPx = (lw + (CW - rw)) / 2 - CW / 2;            // visible-stage centre vs canvas centre
    const dx = dxPx * (2 * r * t / H);
    cam.flyTo({ theta: Math.PI / 2, phi: Math.PI / 2 - 0.001, radius: r,
      target: [(x0 + x1) / 2 - dx, (y0 + y1) / 2, z] }, ms || 900);
  }

  // Arrow keys mean depth, not list order: Up / Right go deeper, Down / Left toward 0 µm. The
  // desktop column lists deepest first (so deeper = back up the list), the phone strip 0 µm first.
  function domDir(key) {
    const deeper = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[key];
    if (!deeper) return 0;
    return PHONE ? deeper : -deeper;
  }

  // ---- section rail ----------------------------------------------------------------------
  // One small thumbnail per section, front to back, along the right edge of the stage (a strip
  // above the bottom sheet on phones), labelled by depth. With both brains on screen it is two
  // labelled columns, Queen | Worker, each with its own pick; arrow keys act on whichever
  // column has focus, and "Match depth" (off by default) ties them. Shown while the layer is
  // on, and always in 2D, where it is the navigation. Picking switches the section in place;
  // the camera does not move. Thumbnails are the small images of the first stain on, fetched
  // only once the rail is shown and the item scrolls into view. Sections without an image stay
  // listed, dimmed; in 3D they are not selectable, in 2D they are (their cells are real).
  const rail = (function () {
    let el = null, list = null, io = null, builtFor = null, setCollapsed = null, phoneCi = 0;
    function ensure() {
      if (el) return el;
      const app = $("app") || document.body;
      el = document.createElement("nav");
      el.id = "histo-rail";
      el.setAttribute("aria-label", "Sections");
      el.innerHTML = `<button type="button" class="hr-head" aria-expanded="true" title="Hide the section list">` +
        `<span class="hr-title">Sections</span><i class="hr-chev" aria-hidden="true"></i></button>` +
        `<button type="button" class="hr-match" aria-pressed="false" hidden title="Keep the two brains at the same depth">Match depth</button>` +
        // phones, both brains: one brain's sections at a time, this picks which
        `<div class="hr-cswitch seg" role="group" aria-label="Brain" hidden><button type="button" data-ci="0" aria-pressed="true">Queen</button>` +
        `<button type="button" data-ci="1" aria-pressed="false">Worker</button></div>` +
        `<button type="button" class="hr-face" hidden title="Turn the view to face the selected section">Face</button>` +
        `<div class="hr-list"></div><div class="hr-tip" role="tooltip" hidden></div>` +
        `<div class="hr-note" role="status"></div>`;
      app.appendChild(el);
      list = el.querySelector(".hr-list");
      let collapsed = false;
      try { collapsed = localStorage.getItem("bimp.histoRail") === "collapsed"; } catch (e) { /* storage off */ }
      const Pr = new URLSearchParams(location.search);
      if (Pr.has("srail")) collapsed = Pr.get("srail") === "0";      // a shared link wins
      const head = el.querySelector(".hr-head");
      setCollapsed = (v) => {
        collapsed = v; el.classList.toggle("collapsed", v);
        head.setAttribute("aria-expanded", String(!v));
        head.title = v ? "Show the section list" : "Hide the section list";
        try { localStorage.setItem("bimp.histoRail", v ? "collapsed" : "open"); } catch (e) { /* storage off */ }
        urlSet("srail", v ? "0" : null);
      };
      setCollapsed(collapsed);
      head.addEventListener("click", () => setCollapsed(!collapsed));
      el.querySelector(".hr-match").addEventListener("click", () => setMatch(!matchDepth));
      el.querySelector(".hr-cswitch").addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        phoneCi = +b.dataset.ci; lead = phoneCi; builtFor = null; build(); syncSlider();
      });
      // the only thing in the rail that moves the camera, and only when asked
      el.querySelector(".hr-face").addEventListener("click", () => face());
      // one shared label for the full name, outside the scroller so it is never clipped:
      // left of the rail on desktop, above the strip on phones
      const tip = el.querySelector(".hr-tip");
      const tipFor = (b) => {
        if (!b || !b.getAttribute("aria-label")) { tip.hidden = true; return; }
        tip.textContent = b.getAttribute("aria-label"); tip.hidden = false;
        const r = b.getBoundingClientRect(), R = el.getBoundingClientRect();
        if (PHONE) { tip.style.left = (r.left + r.width / 2 - R.left) + "px"; tip.style.top = ""; tip.dataset.side = "up"; }
        else { tip.style.top = (r.top + Math.min(r.height, 50) / 2 - R.top) + "px"; tip.style.left = ""; tip.dataset.side = "left"; }
      };
      list.addEventListener("pointerover", (e) => tipFor(e.target.closest("button")));
      list.addEventListener("pointerleave", () => { if (!list.contains(document.activeElement)) tip.hidden = true; });
      list.addEventListener("focusin", (e) => tipFor(e.target.closest("button")));
      list.addEventListener("focusout", () => { tip.hidden = true; });
      list.addEventListener("scroll", () => { tip.hidden = true; }, { passive: true });
      // the rail scrolls itself and nothing else
      el.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
      // A tap acts on the way up, unless the finger travelled (that was a scroll of the strip),
      // and the thumbnail shows it is pressed the moment it is touched. Mouse and keyboard go
      // through click as before; picking twice is harmless.
      let press = null;
      const unpress = () => { if (press) { press.b.classList.remove("hr-press"); clearTimeout(press.hold); } press = null; };
      list.addEventListener("pointerdown", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        unpress();
        // a long press shows the full name (and, for a dimmed section, why it is dimmed)
        const hold = setTimeout(() => tipFor(b), 450);
        if (b.getAttribute("aria-disabled") === "true") { press = { b, x: e.clientX, y: e.clientY, touch: false, hold }; return; }
        press = { b, x: e.clientX, y: e.clientY, touch: e.pointerType !== "mouse", hold };
        b.classList.add("hr-press");
      });
      list.addEventListener("pointermove", (e) => { if (press && Math.hypot(e.clientX - press.x, e.clientY - press.y) > 8) unpress(); });
      list.addEventListener("pointercancel", unpress);
      list.addEventListener("pointerup", () => { if (press && press.touch) pick(press.b); unpress(); });
      list.addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b || b.getAttribute("aria-disabled") === "true") return;
        pick(b);
      });
      // arrow keys walk the focused brain's column
      el.addEventListener("keydown", (e) => {
        const dir = domDir(e.key);
        const b = e.target.closest && e.target.closest(".hr-list button");
        if (!dir || !b) return;
        e.preventDefault(); e.stopPropagation();
        const items = [...list.querySelectorAll(`button[data-ci="${b.dataset.ci}"]:not([aria-disabled=true])`)];
        const i = items.indexOf(b);
        const nx = items[Math.max(0, Math.min(items.length - 1, (i < 0 ? 0 : i + dir)))];
        if (nx) { nx.focus(); pick(nx); }
      });
      if (typeof IntersectionObserver !== "undefined") {
        io = new IntersectionObserver((ents) => ents.forEach((en) => {
          if (!en.isIntersecting) return;
          const im = en.target; if (im.dataset.src) { im.src = im.dataset.src; delete im.dataset.src; }
          io.unobserve(im);
        }), { root: list, rootMargin: "120px" });
      }
      return el;
    }
    // which castes the rail lists: the one on screen, or both (two columns) in Both
    function castesNow() { const f = BIMP.state.s.casteFilter; return f < 0 ? [0, 1] : [f]; }
    // on phones with both brains, one column at a time (the caste switch picks it)
    const splitPhone = () => PHONE && castesNow().length > 1;
    function listed() { return splitPhone() ? [phoneCi] : castesNow(); }
    function keyNow() { return listed().join() + "|" + castesNow().length + "|" + active.join() + "|" + (mode2d ? "2d" : "3d"); }
    function build() {
      ensure();
      const key = keyNow();
      if (builtFor === key) return;
      builtFor = key;
      if (io) io.disconnect();
      const both = castesNow().length > 1, cols = listed();
      const cs = el.querySelector(".hr-cswitch");
      cs.hidden = !splitPhone();
      cs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.ci === phoneCi)));
      let h = "";
      cols.forEach((ci) => {
        const c = CASTES[ci], L = order(c);
        if (!L.length) return;
        h += `<div class="hr-col" data-ci="${ci}">`;
        if (cols.length > 1) h += `<div class="hr-caste">${c}</div>`;
        // "All" belongs to the 3D stack; in 2D there is always exactly one section per brain
        if (!mode2d) h += `<button class="hr-all" type="button" data-all="1" data-ci="${ci}" aria-label="All ${c} sections">` +
          `<span class="hr-cap">All</span><i class="hr-dot"></i></button>`;
        // The desktop rail reads like the physical stack: deepest section at the top, 0 µm (the
        // first cut) at the bottom. The phone strip reads left to right from 0 µm.
        (PHONE ? L.map((x, j) => [x, j]) : L.map((x, j) => [x, j]).reverse()).forEach(([sec, j]) => {
          const why = offReason(c, sec), d = depthUm(c, sec);
          const dis = why && !mode2d;
          const full = `${both ? c[0].toUpperCase() + c.slice(1) + " · " : ""}Section ${j + 1} · ${d} µm` +
            (why ? ` · ${mode2d ? "histology " : ""}${why}` : "");
          const k = active.find((x) => hasStain(sec, x)) || active[0];
          h += `<button type="button" class="hr-item${why ? " hr-off" : ""}" data-ci="${ci}" data-i="${sec.i}" ` +
            `aria-label="${full}"${dis ? ` aria-disabled="true"` : ""}>` +
            `<span class="hr-th"><img alt="" decoding="async" data-src="${base + fileOf(k, c, sec, true)}"></span>` +
            `<span class="hr-cap">${d}<span class="hr-u"> µm</span></span><i class="hr-dot"></i></button>`;
        });
        h += `</div>`;
      });
      list.innerHTML = h;
      list.classList.toggle("both", cols.length > 1);
      el.classList.toggle("both", cols.length > 1);
      list.querySelectorAll("img[data-src]").forEach((im) => {
        if (io) io.observe(im); else { im.src = im.dataset.src; delete im.dataset.src; }
        im.addEventListener("error", () => { im.style.visibility = "hidden"; }, { once: true });
        im.addEventListener("load", () => im.classList.add("ld"), { once: true });   // fades in (CSS), never pops
      });
      mark();
    }
    function rebuild() { builtFor = null; if (el && el.classList.contains("on")) build(); }
    // the first time the rail comes up in a visit, one line on it says what it is for; it goes
    // after the first pick or a few seconds
    let hintT = 0;
    function showHint() {
      let done = false; try { done = sessionStorage.getItem("bimp.railHint") === "1"; } catch (e) { /* storage off */ }
      if (done) return;
      const n = el.querySelector(".hr-note");
      const coarse = PHONE || (typeof matchMedia !== "undefined" && matchMedia("(pointer:coarse)").matches);
      n.textContent = coarse ? "Tap a section to show it" : "Click a section to show it · ← → to step";
      requestAnimationFrame(() => n.classList.add("show"));
      clearTimeout(hintT); hintT = setTimeout(hideHint, 4000);
    }
    function hideHint() {
      if (!el) return;
      el.querySelector(".hr-note").classList.remove("show");
      try { sessionStorage.setItem("bimp.railHint", "1"); } catch (e) { /* storage off */ }
    }
    function pick(b) {
      hideHint();
      const ci = +b.dataset.ci;
      if (b.dataset.all) { clearCaste(ci); return; }
      // switches the section in place: the camera stays wherever the reader left it
      select(ci, +b.dataset.i);
    }
    function mark() {
      if (!el) return;
      if (splitPhone() && lead !== phoneCi && selC[lead] != null && builtFor) { phoneCi = lead; builtFor = null; build(); return; }
      list.querySelectorAll("button[aria-current]").forEach((b) => b.removeAttribute("aria-current"));
      const acts = [];
      castesNow().forEach((ci) => {
        const a = selC[ci] != null ? list.querySelector(`button.hr-item[data-ci="${ci}"][data-i="${selC[ci]}"]`)
          : (!slotK ? list.querySelector(`button[data-all][data-ci="${ci}"]`) : null);
        if (a) { a.setAttribute("aria-current", "true"); acts.push(a); }
      });
      const mt = el.querySelector(".hr-match");
      mt.hidden = castesNow().length < 2 || PHONE;          // phones: it lives in the panel instead
      mt.setAttribute("aria-pressed", String(matchDepth));
      const pm = $("tg-hmatch");
      if (pm) { pm.hidden = castesNow().length < 2; pm.setAttribute("aria-pressed", String(matchDepth)); }
      // Face is a 3D action; in 2D the view already faces the section
      el.querySelector(".hr-face").hidden = mode2d || !castesNow().some((ci) => selC[ci] != null);
      if (el.classList.contains("collapsed") || !el.classList.contains("on")) return;
      // keep the active section centred in the strip (always on phones, where the strip is
      // narrow; on desktop only once it has left view). Scrolls the list itself, never the page.
      const R = list.getBoundingClientRect(), row = getComputedStyle(list).flexDirection === "row" && !list.classList.contains("both") || PHONE;
      const a = acts.find((x) => +x.dataset.ci === leadCaste()) || acts[0];
      if (!a) return;
      const r = a.getBoundingClientRect();
      const out = r.top < R.top || r.bottom > R.bottom || r.left < R.left || r.right > R.right;
      if (!(out || PHONE)) return;
      if (row) list.scrollBy({ left: (r.left + r.width / 2) - (R.left + R.width / 2), behavior: REDUCED ? "auto" : "smooth" });
      else list.scrollBy({ top: (r.top + r.height / 2) - (R.top + R.height / 2), behavior: REDUCED ? "auto" : "smooth" });
    }
    // shown while the layer is on, and always in 2D; fades (CSS on .on)
    let mode = null;
    function update() {
      const v = on || mode2d;
      if (v) {
        const was = el && el.classList.contains("on");
        build();
        // phones: in 3D the open strip would cover the brain's lower edge, so it comes up as
        // its slim "Sections" tab; in 2D, where it is the navigation, it comes up open
        if (PHONE && setCollapsed && (mode !== (mode2d ? "2d" : "3d"))) { mode = mode2d ? "2d" : "3d"; setCollapsed(!mode2d); }
        requestAnimationFrame(() => { el.classList.add("on"); mark(); if (!was && !el.classList.contains("collapsed")) showHint(); });
      } else if (el) el.classList.remove("on");
    }
    // a caste switch rebuilds the list (cheap check, run every frame from draw)
    function sync() { if (el && el.classList.contains("on") && builtFor !== keyNow()) build(); }
    // step the lead brain to its next / previous selectable section (2D arrow keys)
    function step(dir) {
      ensure();
      const ci = splitPhone() ? phoneCi : leadCaste();
      const items = [...list.querySelectorAll(`button.hr-item[data-ci="${ci}"]:not([aria-disabled=true])`)];
      const cur = selC[ci] != null ? items.findIndex((b) => +b.dataset.i === selC[ci]) : -1;
      const nx = items[Math.max(0, Math.min(items.length - 1, cur < 0 ? 0 : cur + dir))];
      if (nx) pick(nx);
    }
    return { update, sync, mark, rebuild, step };
  })();

  // ---- 2D section view --------------------------------------------------------------------
  // One section per brain, flat: only that section's cells (the cell shader drops everything
  // outside a thin slab around its z, per brain, via state.s.slab → uSlab), the camera eased
  // square-on with orbit off (pan and zoom kept), the same section's histology under it when
  // the layer is on (wanted(): the very manifest entry the slab is cut from), and the rail as
  // the way through the stack. With both brains on screen each shows its own pick side by side
  // (a brain with no pick starts at the section nearest the other's depth). Entering keeps the
  // picks, else the middle section; leaving restores the 3D camera, orbit and picks. This is
  // the one place the camera moves on its own: the reader asked for the mode.
  const view2d = (function () {
    let btn = null, origOrbit = null;
    const MIN_R_2D = 0.12;
    function init() {
      // orbit is off in 2D: the drag and inertia handlers in boot.js call camera.orbit
      const cam = BIMP.camera;
      if (cam && !origOrbit) { origOrbit = cam.orbit; cam.orbit = function () { if (!mode2d) return origOrbit.apply(cam, arguments); }; }
      const app = $("app") || document.body;
      btn = document.createElement("div");
      btn.id = "view-mode"; btn.className = "seg"; btn.setAttribute("role", "group"); btn.setAttribute("aria-label", "View");
      btn.innerHTML = `<button type="button" data-v="2d" aria-pressed="false" title="One section, flat (2)">2D</button>` +
        `<button type="button" data-v="3d" aria-pressed="true" title="The whole brain in 3D (3)">3D</button>`;
      app.appendChild(btn);
      btn.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) (b.dataset.v === "2d" ? enter : exit)(); });
      window.addEventListener("keydown", (e) => {
        const t = e.target || {};
        if (/INPUT|TEXTAREA|SELECT/.test(t.tagName || "") || t.isContentEditable || e.metaKey || e.ctrlKey || e.altKey) return;
        if (document.documentElement.dataset.view && document.documentElement.dataset.view !== "atlas") return;
        if (e.key === "2") { enter(); e.preventDefault(); }
        else if (e.key === "3") { exit(); e.preventDefault(); }
        else if (mode2d && !(t.closest && t.closest("#histo-rail"))) {
          const dir = domDir(e.key);
          if (dir) { rail.step(dir); e.preventDefault(); }
        }
        // R / Home would send the camera to the 3D home pose; in 2D re-face the section instead
        if (mode2d && (e.key === "r" || e.key === "R" || e.key === "Home")) {
          e.stopImmediatePropagation(); e.preventDefault(); face(500);
        }
      }, true);
      const rb = $("btn-reset");
      if (rb) rb.addEventListener("click", (e) => { if (mode2d) { e.stopImmediatePropagation(); face(500); } }, true);
    }
    function pressed() {
      if (!btn) return;
      btn.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.v === "2d") === mode2d)));
      document.documentElement.classList.toggle("view-2d", mode2d);
    }
    // with no pick, 2D starts where the rail starts: the top of the stack (the deepest section)
    function middle(ci) { const L = order(CASTES[ci]); return L[L.length - 1]; }
    // every brain on screen gets a section: its own pick, else (with the other picked) the
    // section nearest the other's depth, else its middle
    function fillPicks() {
      const f = BIMP.state.s.casteFilter, cis = f < 0 ? [lead, 1 - lead] : [f];
      cis.forEach((ci) => {
        if (secOf(ci)) return;
        const o = 1 - ci, os = cis.includes(o) ? secOf(o) : null;
        const sec = os ? nearestByDepth(ci, depthUm(CASTES[o], os)) : middle(ci);
        selC[ci] = sec.i;
      });
      lead = cis[0];
      syncSlider(); rail.mark(); slab(); writeUrl(); pump();
    }
    function enter() {
      if (mode2d || !manifest) return;
      const s = BIMP.state.s, c = BIMP.camera.cam;
      saved3d = { pose: { theta: c.theta, phi: c.phi, radius: c.radius, target: c.target.slice() },
        spin: s.spin, selC: selC.slice(), slotK, lead };
      if (s.spin) { s.spin = false; const t = $("tg-spin"); if (t) t.setAttribute("aria-pressed", "false"); }
      mode2d = true;
      // a flat section can be looked at much closer than the 3D brain: down to ~17× the fit,
      // past one level-0 texel per screen pixel
      saved3d.minR = BIMP.camera.cam.minR; BIMP.camera.cam.minR = MIN_R_2D;
      // and at the screen's own resolution: the 3D view's resolution may have been lowered to
      // keep a million cells moving (renderer adaptDpr, and a 2x cap), which leaves a stained
      // section soft; one section's cells are cheap, so 2D asks for the full density (up to 3x)
      // and gives the 3D limit back on the way out. If 2D itself runs slow, adaptDpr still steps down.
      const R = BIMP.renderer;
      if (R && R.setDpr && R.getDpr) { saved3d.dpr = R.getDpr(); R.setDpr(Math.min(window.devicePixelRatio || 1, 3)); }
      pressed();
      fillPicks();
      rail.update();
      face(900);
    }
    function exit() {
      if (!mode2d) return;
      mode2d = false;
      openSlab();                 // the rest of the brain grows back out from the section
      pressed();
      const v = saved3d; saved3d = null;
      if (v) {
        if (v.minR != null) BIMP.camera.cam.minR = v.minR;
        if (v.dpr && BIMP.renderer && BIMP.renderer.setDpr) BIMP.renderer.setDpr(v.dpr);
        selC = v.selC; slotK = v.slotK; lead = v.lead;
        BIMP.camera.flyTo(v.pose, 900);
        if (v.spin) { BIMP.state.s.spin = true; const t = $("tg-spin"); if (t) t.setAttribute("aria-pressed", "true"); }
      }
      if (!on) { selC = [null, null]; slotK = 0; }
      syncSlider(); writeUrl();
      rail.update(); rail.mark();
    }
    // the slab around each brain's section: [z queen, z worker, half thickness, 0], half the
    // gap to the nearest neighbour section, so exactly one section's cells pass. z comes from
    // the same manifest entry wanted() draws, in the same normalised frame as the cells.
    // The slab's thickness eases between one section and the whole brain (SLAB_OPEN passes every
    // cell), log-spaced over SLAB_SECS, so entering 2D peels the brain down to its section and
    // leaving grows it back, instead of every other cell vanishing or reappearing in one frame.
    const SLAB_OPEN = 0.4, SLAB_SECS = 0.6;
    let anim = null;              // {from, to, t, done}
    function animTo(to, done) {
      const cur = BIMP.state.s.slab && BIMP.state.s.slab[2] > 0 ? BIMP.state.s.slab[2] : SLAB_OPEN;
      anim = REDUCED ? null : { from: cur, to, t: 0, done };
      if (REDUCED) { BIMP.state.s.slab[2] = to; if (done) done(); }
      inv();
    }
    function openSlab() {
      if (!BIMP.state.s.slab || !(BIMP.state.s.slab[2] > 0)) { BIMP.state.s.slab = [0, 0, 0, 0]; return; }
      animTo(SLAB_OPEN, () => { if (!mode2d) BIMP.state.s.slab = [0, 0, 0, 0]; });
    }
    function tick(dt) {
      if (!anim) return false;
      anim.t = Math.min(1, anim.t + dt / SLAB_SECS);
      const e = ease(anim.t);
      BIMP.state.s.slab[2] = Math.exp(Math.log(anim.from) + (Math.log(anim.to) - Math.log(anim.from)) * e);
      if (anim.t >= 1) { const d = anim.done; anim = null; if (d) d(); }
      return true;
    }
    const animating = () => !!anim;
    function slab() {
      if (!mode2d) return;
      const z = [9, 9];           // a brain with no pick: nothing passes (it is hidden anyway)
      let half = Infinity;
      [0, 1].forEach((ci) => {
        const sec = secOf(ci); if (!sec) return;
        const L = order(CASTES[ci]), j = L.indexOf(sec);
        z[ci] = sec.z;
        if (j > 0) half = Math.min(half, 0.45 * (L[j].z - L[j - 1].z));
        if (j < L.length - 1) half = Math.min(half, 0.45 * (L[j + 1].z - L[j].z));
      });
      if (!isFinite(half)) half = 0.002;
      const was = BIMP.state.s.slab && BIMP.state.s.slab.length === 4 && BIMP.state.s.slab[2] > 0 ? BIMP.state.s.slab[2] : SLAB_OPEN;
      BIMP.state.s.slab = [z[0], z[1], anim || was > half * 1.01 ? was : half, 0];
      if (was > half * 1.01 && (!anim || anim.to !== half)) animTo(half);   // closing in from the whole brain
      else if (!anim) BIMP.state.s.slab[2] = half;                          // stepping: same thickness, new z
    }
    // a caste switch made elsewhere: every brain now on screen needs a section
    let lastF = null;
    function sync() {
      const f = BIMP.state.s.casteFilter;
      if (f === lastF) return;
      lastF = f;
      if (!mode2d) return;
      if (f >= 0) lead = f;
      fillPicks();
    }
    return { init, enter, exit, slab, sync, tick, animating };
  })();

  const api = { init, draw, isAnimating, warmHome, homeSection, setOn, isOn: () => on, onChange, setOpacity, getOpacity: () => opacity,
    setSection, setStains, getStains: () => active.slice(), select, setMatch, stats, warm,
    // display levels per stain, for tuning from the console: setLevels("dapi", {lo, hi, g})
    // tests: graft {tiles, sections: {"queen/14": {stains, levels}}} onto the loaded manifest
    __tiles: (o) => { manifest.tiles = o.tiles; Object.entries(o.sections || {}).forEach(([key, v]) => { const [c, i] = key.split("/"); const sec = manifest.castes[c].sections.find((x) => x.i === +i); if (sec) sec.tiles = v; }); inv(); },
    tileStats: () => ({ cached: tcache.size, loaded: [...tcache.values()].filter((e) => e.t).length, loading: tLoading, plan: lastPlan }),
    setLevels: (k, L) => { LEVELS[k] = Object.assign({}, levelsOf(k), L); inv(); }, getLevels: (k) => Object.assign({}, levelsOf(k)),
    enter2D: () => view2d.enter(), exit2D: () => view2d.exit(), is2D: () => mode2d,
    get on() { return on; }, get available() { return !!manifest; } };
  return api;
})();
// short alias, for layer toggles written against BIMP.histo
BIMP.histo = BIMP.histology;
