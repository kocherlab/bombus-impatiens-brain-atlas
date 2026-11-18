/* Guided "show me around" tour. Step-through: a caption card with Back/Next, and each
   stop drives the real controls (color mode, caste, region mask, gene pick) AND flies the
   camera to frame what it's talking about. Reuses the existing wired buttons so behaviour
   stays identical to manual use. */
window.BIMP = window.BIMP || {};
BIMP.tour = (function () {
  const { state, camera } = BIMP;
  const s = state.s;
  const $ = (id) => document.getElementById(id);
  let idx = 0, stops = [], active = false;

  // drive the existing controls (their click handlers own all the side-effects)
  const click = (sel) => { const b = document.querySelector(sel); if (b) b.click(); };
  const setMode = (m) => click(`#mode-seg [data-mode="${m}"]`);
  const setCaste = (c) => click(`#caste-seg [data-v="${c}"]`);
  const setLabeling = (l) => click(`#label-seg [data-v="${l}"]`);
  function setRegion(mask) {
    s.regionMask = mask;
    BIMP.ui.syncRegionUI && BIMP.ui.syncRegionUI();
    BIMP.ui.updateStat && BIMP.ui.updateStat();
  }
  function pickGene(sym) {
    const i = BIMP.ui.geneIndexBySymbol ? BIMP.ui.geneIndexBySymbol(sym) : -1;
    if (i >= 0) BIMP.ui.selectGene(i);
  }
  const bitsFor = (keys) => (BIMP.manifest.regionDefs || [])
    .reduce((m, d) => (keys.indexOf(d.key) >= 0 ? m | (1 << d.bit) : m), 0);

  // centroid (normalized world space, matching the renderer's y,-x,z*Z mapping) of the
  // cells in a region mask → a camera target so region stops actually zoom onto them.
  function regionCentroid(mask) {
    const D = BIMP.data;
    if (!D || !mask) return [0, 0, 0];
    if (_centroid[mask]) return _centroid[mask];   // cache: the 1.3M scan hitches the fly
    let X = 0, Y = 0, Z = 0, n = 0;
    for (let i = 0; i < D.n; i++) {
      if (D.region[i] & mask) { X += D.y[i]; Y += -D.x[i]; Z += D.z[i]; n++; }
    }
    if (!n) return [0, 0, 0];
    return (_centroid[mask] = [X / n / 32767, Y / n / 32767, (Z / n / 32767) * s.z]);
  }
  const _centroid = {};

  function defineStops() {
    const MB = bitsFor(["in_kc_ic", "in_kc_nc", "in_kc_oc"]);
    const OL = bitsFor(["in_lamina", "in_medulla", "in_lobula"]);
    stops = [
      { title: "The whole brain",
        body: "1.3&nbsp;million cells from a queen <i>Bombus&nbsp;impatiens</i> brain, each mapped to a cell type and placed back into the reconstructed 3D volume. You can drag to orbit at any time.",
        run() { setMode("celltype"); setLabeling("canonical"); setCaste("queen"); setRegion(0); },
        cam: { theta: 0.9, phi: 1.12, radius: 4.2, target: [0, 0, 0] } },
      { title: "Painting by gene — tachykinins",
        body: "Switch to <b>Gene</b> view and the expressing cells light up while the rest fade back. Tachykinins, a neuropeptide, concentrate in the optic lobes.",
        run() { setCaste("queen"); setMode("gene"); pickGene("tachykinins"); setRegion(0); },
        gene: "tachykinins",
        cam: { theta: 1.2, phi: 1.08, radius: 3.9, target: [0, 0, 0] } },
      { title: "sNPF",
        body: "Short neuropeptide&nbsp;F — the atlas's headline caste signal. Its pattern differs from tachykinins, favouring the mushroom-body Kenyon cells.",
        run() { setMode("gene"); pickGene("sNPF"); setRegion(0); },
        gene: "sNPF",
        cam: { theta: 0.9, phi: 1.12, radius: 3.8, target: [0, 0, 0] } },
      { title: "Queen vs worker",
        body: "Split the two brains side by side (<b>Caste&nbsp;→&nbsp;Both</b>) with tachykinins still painted on — the caste difference is obvious at a glance.",
        run() { setMode("gene"); pickGene("tachykinins"); setRegion(0); },
        late() { setCaste("both"); },
        gene: "tachykinins",
        cam: { theta: 1.05, phi: 1.08, radius: 5.2, target: [0, 0, 0] } },
      { title: "The mushroom body — Kenyon cells",
        body: "Isolate the hand-labeled Kenyon cell compartments, each named for the population that dominates it — <b>Large-type&nbsp;·&nbsp;Small-type&nbsp;·&nbsp;Class&nbsp;II</b> (non-compact, inner compact, outer compact). Everything else is hidden, and the cells are coloured by compartment.",
        // Region view: Type view no longer repaints the calyx by compartment (the tree's
        // Kenyon cells row is one colour until opened), so the compartments need this one
        run() { setCaste("queen"); setMode("region"); setRegion(MB); },
        camAt: () => ({ theta: 1.0, phi: 1.06, radius: 2.7, target: regionCentroid(MB) }) },
      { title: "The optic lobe",
        body: "The visual neuropils — <b>lamina · medulla · lobula</b> — arranged laterally, again from hand-drawn masks.",
        run() { setCaste("queen"); setMode("celltype"); setRegion(OL); },
        camAt: () => ({ theta: 1.12, phi: 1.1, radius: 2.9, target: regionCentroid(OL) }) },
      { title: "Queen vs worker, side by side",
        body: "The queen and worker brains laid out together, colored by cell type — same architecture, mapped independently. That's the tour: from here, pick any gene, isolate a region, spin the brain, or click a cell to inspect it.",
        run() { setMode("celltype"); setLabeling("canonical"); setRegion(0); },
        late() { setCaste("both"); },
        cam: { theta: 0.92, phi: 1.1, radius: 5.2, target: [0, 0, 0] } },
    ];
  }

  // Which control(s) each stop is changing — highlighted (pulsing glow) as the camera
  // flies, so you can see WHERE in the panel the tour is "clicking". Indexed by stop.
  const HI = [
    ["#mode-seg [data-mode='celltype']", "#caste-seg [data-v='queen']"],
    ["#mode-seg [data-mode='gene']", "#gene-list [aria-selected='true']"],
    ["#mode-seg [data-mode='gene']", "#gene-list [aria-selected='true']"],
    ["#caste-seg [data-v='both']", "#mode-seg [data-mode='gene']"],
    ["#mode-seg [data-mode='region']", "#region-list .reg-item[data-bit='0']", "#region-list .reg-item[data-bit='1']", "#region-list .reg-item[data-bit='2']"],
    ["#region-list .reg-item[data-bit='3']", "#region-list .reg-item[data-bit='4']", "#region-list .reg-item[data-bit='5']"],
    ["#caste-seg [data-v='both']", "#mode-seg [data-mode='celltype']"],
  ];
  function clearHighlight() { document.querySelectorAll(".tour-hl").forEach((e) => e.classList.remove("tour-hl")); }
  function applyHighlight() {
    clearHighlight();
    (HI[idx] || []).forEach((sel) => document.querySelectorAll(sel).forEach((e) => e.classList.add("tour-hl")));
  }

  // The card changes height from stop to stop (the captions differ by a line or two) and it
  // is anchored at the bottom, so swapping the text outright made its top edge jump 20-40 px
  // and the words cut. Instead: the old words fade out, the new ones fade in, and the card's
  // height glides between the two (FLIP: measure, swap, measure, animate). The dots are made
  // once and only toggled, so their own colour transition runs instead of being rebuilt.
  const FADE_OUT = 120, FADE_IN = 200, GROW = 320;
  let cardTok = 0, grow = null;
  function fill(st) {
    $("tour-step").textContent = "Stop " + (idx + 1) + " / " + stops.length;
    $("tour-title").innerHTML = st.title;
    $("tour-body").innerHTML = st.body;
    $("tour-back").disabled = idx === 0;
    $("tour-next").textContent = idx === stops.length - 1 ? "Done" : "Next →";
    const dots = $("tour-dots");
    if (!dots.children || dots.children.length !== stops.length)
      dots.innerHTML = stops.map(() => '<span class="tdot"></span>').join("");
    for (let i = 0; i < (dots.children ? dots.children.length : 0); i++) dots.children[i].classList.toggle("on", i === idx);
  }
  function render(instant) {
    const st = stops[idx], card = $("tour"), tok = ++cardTok;
    const parts = ["tour-step", "tour-title", "tour-body"].map($);
    const canAnim = card && card.animate && card.getBoundingClientRect && !REDUCED;
    // a quick second click: drop the previous stop's fades (their held opacity 0 would
    // otherwise outlive this stop's fade-in) and let the height carry on from where it is
    const o = canAnim && parts[0] ? +getComputedStyle(parts[0]).opacity : 1;   // read before cancelling
    parts.forEach((e) => e && e.getAnimations && e.getAnimations().forEach((a) => a.cancel()));
    if (instant || !canAnim || !card.classList.contains("show")) {
      if (grow) { grow.cancel(); grow = null; card.style.overflow = ""; }
      fill(st); return;
    }
    const fade = (o0, o1, ms) => parts.map((e) => e.animate([{ opacity: o0 }, { opacity: o1 }], { duration: ms, easing: "ease-out", fill: "forwards" }));
    const out = fade(o, 0, FADE_OUT * o);
    out[0].finished.then(() => {
      if (tok !== cardTok) return;              // a newer stop took over mid-fade
      const h0 = card.getBoundingClientRect().height;
      if (grow) { grow.cancel(); grow = null; }
      fill(st);
      const h1 = card.getBoundingClientRect().height;
      out.forEach((a) => a.cancel());
      fade(0, 1, FADE_IN).forEach((a) => a.finished.then(() => a.cancel(), () => {}));
      if (Math.abs(h1 - h0) > 0.5) {
        card.style.overflow = "hidden";
        const g = grow = card.animate([{ height: h0 + "px" }, { height: h1 + "px" }], { duration: GROW, easing: "cubic-bezier(.4,0,.2,1)" });
        g.finished.then(() => { if (grow === g) { grow = null; card.style.overflow = ""; } refit(); }, () => {});
      }
    }, () => {});
  }
  // on a phone the brain is centred above the card, so a taller or shorter card re-centres it
  // (the renderer glides the lens shift)
  const refit = () => { if (BIMP.ui && BIMP.ui.refit) BIMP.ui.refit(); };
  const REDUCED = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  function go(i, instant) {
    idx = Math.max(0, Math.min(stops.length - 1, i));
    const st = stops[idx];
    s.spin = false; const sp = $("tg-spin"); if (sp) sp.setAttribute("aria-pressed", "false");
    // The camera leaves immediately, from wherever it is now. Duration scales with how far
    // it travels, so a big move (pulling back for "both", then diving into the calyx) glides
    // at a consistent speed instead of snapping. Region goals are state-independent
    // (centroid of a mask), so the goal can be computed before the state swap.
    const c = camera.cam;
    const goal0 = st.camAt ? st.camAt() : st.cam;
    // the stops are framed for a desktop stage; a portrait phone pulls back by the same factor
    // the viewer's own distances use (ui.fitScale)
    const k = BIMP.ui && BIMP.ui.fitScale ? BIMP.ui.fitScale() : 1;
    const goal = goal0.radius != null && k !== 1 ? Object.assign({}, goal0, { radius: goal0.radius * k }) : goal0;
    const dr = Math.abs((goal.radius != null ? goal.radius : c.radius) - c.radius);
    const gt = goal.target || c.target;
    const dtar = Math.hypot(gt[0] - c.target[0], gt[1] - c.target[1], gt[2] - c.target[2]);
    const dur = Math.min(2600, 1200 + dr * 260 + dtar * 700);
    camera.flyTo(goal, dur);   // one smooth eased glide to the goal
    render(instant);
    applyHighlight();   // pulse the control(s) this stop is driving
    // …and the view state changes right away. The renderer sees the change and dissolves
    // into it over ~0.55 s (colors interpolate, filtered cells fade), so the cells stay on
    // screen throughout — no blink to empty. Two exceptions, both sequenced off the flight:
    //  - a gene stop waits (briefly) for its gene file, so the colours never dissolve into
    //    a gene that is not there yet and then pop when it lands;
    //  - leaving a region close-up for a wide view (a masked region, camera pulling well
    //    back) unmasks only once the flight is under way: unmasked at the start, the whole
    //    brain bloomed in at close-up scale and overflowed a phone's width before shrinking;
    //  - the caste split (st.late) waits until the camera has pulled back far enough. Split
    //    at the start, the two brains slid apart in 0.55 s while the eased flight had barely
    //    left a close-up, so they swelled off the edge of the frame and then shrank back.
    //    It now fires LATE_LEAD ms before landing, so the split ends just as the camera
    //    settles. The caste handler's own zoom is skipped while the flight owns the camera.
    if (!geneReady(st.gene)) prefetch(st.gene);   // normally already warm (start)
    const widen = s.regionMask !== 0 && goal.radius != null && goal.radius > c.radius * 1.25;
    const runAt = widen ? dur * WIDEN_AT : 0;
    const t0 = performance.now(), my = ++goTok;
    let ran = false, lateDone = !st.late;
    pending = () => { if (!ran) { ran = true; if (st.run) st.run(); } if (!lateDone) { lateDone = true; st.late(); } };
    const loop = () => {
      if (my !== goTok) return;
      const t = performance.now() - t0;
      if (!ran && t >= runAt && (geneReady(st.gene) || t > runAt + GENE_WAIT)) { ran = true; if (st.run) st.run(); applyHighlight(); }
      if (ran && !lateDone && t >= Math.max(0, dur - LATE_LEAD)) { lateDone = true; st.late(); applyHighlight(); }
      if (!(ran && lateDone)) requestAnimationFrame(loop); else pending = null;
    };
    loop();
  }
  const GENE_WAIT = 1500, LATE_LEAD = 800, WIDEN_AT = 0.4;
  let goTok = 0, pending = null;
  // is this stop's gene already in memory? (fetch mode only: an embedded build has them all)
  function geneReady(sym) {
    if (!sym || !BIMP.manifest || BIMP.manifest.mode !== "fetch") return true;
    if (!BIMP.fullReady) return false;  // genes come with the full tier: wait (up to GENE_WAIT) for it
    const gi = BIMP.ui.geneIndexBySymbol ? BIMP.ui.geneIndexBySymbol(sym) : -1;
    const G = BIMP.data && BIMP.data.genes;
    return gi < 0 || !!(G && G[gi]);
  }
  function prefetch(sym) {
    const gi = BIMP.ui.geneIndexBySymbol ? BIMP.ui.geneIndexBySymbol(sym) : -1;
    if (gi >= 0 && BIMP.ui.prefetchGene) BIMP.ui.prefetchGene(gi);
  }
  // apply whatever the current stop still owes (the tour is ending mid-flight)
  function flush() { goTok++; const p = pending; pending = null; if (p) p(); }
  function start() {
    if (active) return;
    if (!stops.length) defineStops();
    active = true; idx = 0;
    // warm the genes the tour paints, so a stop never waits mid-flight for a fetch and
    // then pops in (the cache is shared with the normal hover-prefetch path)
    // From the home page the tour starts before the full tier has arrived, when prefetch is
    // a no-op, so keep trying until it is in.
    const warm = () => {
      if (!active) return;
      if (!BIMP.fullReady) { requestAnimationFrame(warm); return; }
      ["tachykinins", "sNPF"].forEach(prefetch);
    };
    warm();
    document.body.classList.add("touring");
    $("tour").classList.add("show");
    const b = $("tour-btn"); if (b) b.classList.add("hide");
    go(0, true);   // the card fades in as a whole; nothing to cross-fade yet
  }
  function stop() {
    active = false;
    camera.cancelFly();
    flush();      // the stop being read finishes what it was doing (e.g. a pending split)
    document.body.classList.remove("touring");
    $("tour").classList.remove("show");
    clearHighlight();
    s.spin = true; const sp = $("tg-spin"); if (sp) sp.setAttribute("aria-pressed", "true");  // resume the gentle idle orbit
    const b = $("tour-btn"); if (b) b.classList.remove("hide");
  }
  const next = () => (idx >= stops.length - 1 ? stop() : go(idx + 1));
  const back = () => go(idx - 1);

  function init() {
    const on = (id, fn) => { const e = $(id); if (e) e.addEventListener("click", fn); };
    on("tour-next", next); on("tour-back", back); on("tour-skip", stop); on("tour-btn", start);
  }
  return { init, start, stop, next, back, isActive: () => active };
})();
