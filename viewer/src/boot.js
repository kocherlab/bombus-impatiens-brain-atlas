/* Bootstrap with progressive loading:
   1. decode the LITE tier (small, no genes) → paint + interact in well under a
      second,
   2. stream the FULL tier (all cells + 100 genes) in the background — from an
      IndexedDB cache when available, else decode once and cache it — then hot-swap
      it in without interrupting the view.
   Exposes BIMP.ready (resolves when the full tier is live) and BIMP.selfTest(). */
window.BIMP = window.BIMP || {};
(function () {
  const { renderer: R, camera, decode, ui, state } = BIMP;
  const s = state.s;
  const canvas = document.getElementById("gl");
  let DATA = null, fpsAcc = 0, fpsCount = 0;

  const setLoad = (m) => { const el = document.getElementById("loadmsg"); if (el) el.textContent = m; };
  const setStatus = (m) => { const el = document.getElementById("status"); if (el) el.textContent = m; };

  // ---- tiny IndexedDB cache for the decoded full buffer (keyed by data version).
  // Every op is time-boxed and fail-safe: on an opaque (file://) origin, private
  // mode, or a hang, it resolves null and we just decode from the inline blob. ----
  const withTimeout = (p, ms) => Promise.race([p, new Promise((r) => setTimeout(() => r(null), ms))]);
  const IDB = {
    open() {
      return withTimeout(new Promise((res) => {
        try {
          if (typeof indexedDB === "undefined") return res(null);
          const r = indexedDB.open("bimp-atlas", 1);
          r.onupgradeneeded = () => r.result.createObjectStore("tiers");
          r.onsuccess = () => res(r.result);
          r.onerror = () => res(null);
          r.onblocked = () => res(null);
        } catch (e) { res(null); }
      }), 1500);
    },
    async get(key) {
      const db = await this.open(); if (!db) return null;
      return withTimeout(new Promise((res) => {
        try {
          const t = db.transaction("tiers", "readonly").objectStore("tiers").get(key);
          t.onsuccess = () => res(t.result || null); t.onerror = () => res(null);
        } catch (e) { res(null); }
      }), 1500);
    },
    async put(key, buf) {
      const db = await this.open(); if (!db) return;
      try { db.transaction("tiers", "readwrite").objectStore("tiers").put(buf, key); } catch (e) { /* best-effort */ }
    },
  };

  // Resolves when the FULL tier is live. The full blob is the last <script> in the
  // file, so it may not exist yet when boot runs — we register a callback it fires.
  let resolveFull, rejectFull;
  const fullReady = new Promise((res, rej) => { resolveFull = res; rejectFull = rej; });

  // shared setup once DATA is uploaded and the UI is built.
  // SPATIAL build-up: each cell has a per-vertex birth time (aBirth, radial-from-centroid
  // + jitter) and frame() advances a single `progress` clock 0→1. A coherent reveal front
  // sweeps outward from the brain core, each cell fading+popping in as the front reaches
  // it. This reads as the brain ASSEMBLING — unlike a uniform density ramp (draw-count or
  // random birth), which saturates in the first ~15% and looks like a snap because
  // perceived overplotting is inverse-logarithmic. (Earlier attempts — a gl_VertexID hash
  // cull, then a shuffled draw-count ramp — both failed for this reason.)
  const BUILD_SECS = 6.5;  // wall-clock for the full core→shell sweep
  let building = false, buildProgress = 0;
  function revealFull() {
    if (!R.setProgress) return;
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion:reduce)").matches) { R.setProgress(1); building = false; return; }
    buildProgress = 0; R.setProgress(0);   // start empty, then sweep the front outward
    building = true;   // frame() eases progress 0→1 over BUILD_SECS
  }
  // Enable the "Explore" button only when the FULL tier is decoded and live, so the
  // 7.5 MB download + gzip-decode happens WHILE the user reads the intro — clicking
  // Explore then builds from 0 immediately, with no blank "loading…" wait on-screen.
  function enableExplore() {
    const e = document.getElementById("enter");
    if (e) { e.disabled = false; e.textContent = "Take the guided tour"; }
    const et = document.getElementById("enter-tour");
    if (et) { et.classList.add("ready"); et.textContent = "Open the atlas"; }
    showTourBtn();
    // the histology layer's control appears only if data/histo/ was deployed (Pages build)
    if (BIMP.histology) BIMP.histology.init(window.__BIMP_SRC__);
  }
  // The persistent "Show me around" launch pill appears once the intro is gone and the
  // full atlas is live (so gene stops have data to paint).
  function showTourBtn() {
    const b = document.getElementById("tour-btn");
    if (b && introGone && BIMP.fullReady && !(BIMP.tour && BIMP.tour.isActive())) b.classList.add("ready");
  }
  // Gate: the build-up must play ON-SCREEN, so start it only once BOTH the full data
  // is uploaded AND the intro overlay is dismissed (otherwise it builds up hidden
  // behind the intro and you just see a finished brain when you click Explore).
  // ?atlas (or the older ?nointro) is a link straight into the viewer
  const ATLAS_LINK = /[?&](nointro|atlas)(=|&|$)/.test(location.search);
  // never hold the controls for long, whatever the data does
  if (ATLAS_LINK) setTimeout(() => document.documentElement.classList.remove("atlas-boot"), 6000);
  let dataReady = false, introGone = ATLAS_LINK;
  function maybeReveal() {
    if (!(dataReady && introGone && R.getProgress && R.getProgress() < 1 && !building)) return;
    // arriving from the home page's hand-off, the stipple's own dots have just landed on these
    // cells, so the brain is shown whole rather than assembled a second time
    if (BIMP.instantReveal) { R.setProgress(1); return; }
    revealFull();
  }
  function dismissIntro() {
    const el = document.getElementById("intro"); if (el) el.classList.add("gone");
    introGone = true; maybeReveal(); showTourBtn();
  }

  // ---- the home page's brain, and the way between it and the atlas --------------------
  // One renderer, one canvas, one set of cells. On the home page the renderer draws the HERO
  // tier (~20k real queen cells, build-pages.mjs) in ink on paper, inside the #hero-gl box,
  // rocking slowly. Entering the atlas is a single eased clock (tp 0 → 1) that moves every
  // channel together: the box grows to the full canvas, the camera flies from the hero pose to
  // the viewer's, the box behind the brain cuts to the dark stage and grows with it (a crisp
  // rectangle, never a fade through grey), light ink turns into the cell colours, the rest of the 1.3M
  // cells grow in around the hero cells, which never move or blink: the full atlas is
  // uploaded with those same cells born at -1, already present, so the slot swap is invisible.
  // Going home runs the same clock backwards. Both are reversible mid-flight.
  const heroSmall = Math.min(innerWidth, innerHeight) < 700 || (navigator.hardwareConcurrency || 8) <= 4;
  const HERO = window.__BIMP_HERO__, heroBox = document.getElementById("hero-gl");
  const heroOn = !!(HERO && heroBox && !introGone && window.__BIMP_SRC__);
  const DARK = R.look.bg.slice();
  // where the dark first screen ends, in the page's scroll coordinates: #intro paints its paper
  // from there down, so in flight the page can fade while the stage above it stays put
  {
    const intro = document.getElementById("intro"), hs = intro && intro.querySelector(".hero");
    if (hs) {
      const setB = () => intro.style.setProperty("--hero-b", (hs.offsetTop + hs.offsetHeight) + "px");
      setB(); new ResizeObserver(setB).observe(hs);
    }
  }
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  const sstep = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const T = { tp: introGone ? 1 : 0, dir: 0, slot: "none", heroKeys: null, pendingFull: null, atlasPose: null, heroData: null,
    secs: 2.2, secsBack: 3.0, onDone: null, ptrK: 0, ptr: [0, 0], visible: true, bu: 0, ba: 0, bb: 0, bLast: 0, homeP: 0 };
  // the home build: all cells over BUILD_S on a smoothstep; on a slow link the sample first, in the
  // reveal's first BUILD_H
  const BUILD_S = 5.5, BUILD_H = 0.4, BUILD_WAIT = 2500;
  const root = document.documentElement, appEl = document.getElementById("app");
  const setView = (v) => { root.dataset.view = v; };
  setView(introGone ? "atlas" : "home");
  // The brain is drawn into a margin around its box (the bleed) and the copy on the page fades out
  // over that margin, so when the story zooms in the cells drift out of view rather than being cut
  // by the box's edge. heroRect is the box plus its bleed; the camera fits the brain to the box.
  const narrowHero = matchMedia("(max-width: 1000px)");
  const bleed = () => (narrowHero.matches ? [0.06, 0.12] : [0.12, 0.16]);
  function heroRect() {
    const r = heroBox.getBoundingClientRect(), [bx, by] = bleed();
    return { x: r.left - r.width * bx, y: r.top - r.height * by, w: r.width * (1 + 2 * bx), h: r.height * (1 + 2 * by) };
  }
  function heroPose(now, rect) {
    // fit the brain to the box inside the bleed: the same size on the page as before the bleed
    const b = HERO.box, r = rect || heroRect(), [bx, by] = bleed(), t = Math.tan(24 * Math.PI / 180);
    const A = (r.w / (1 + 2 * bx)) / Math.max(1, r.h / (1 + 2 * by));
    const bw = (b[2] - b[0]) / 2, bh = (b[3] - b[1]) / 2;
    const radius = (Math.max(bw / (t * A), bh / t) * 1.18 + 0.15) * (1 + 2 * by);   // fit, plus the near half's perspective
    // front-on from +z, the same side the viewer opens on, so the flight in turns the brain only
    // a little instead of swinging it edge-on
    // a little from above (phi < 90°), so the anterior axis never lines up with the lateral one
    const c = story.cur, cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
    // on a phone the brain fills its width: a little closer, its turning edges may graze the sides
    const near = narrowHero.matches ? 0.86 : 1;
    return { theta: Math.PI / 2 + c.amp * 0.52 * Math.sin(now / 1000 * 0.32), phi: Math.PI / 2 - 0.32 * (1 - c.front),
      radius: radius * c.zoom * near, target: [cx + c.off[0], cy + c.off[1], c.off[2]] };
  }
  // The scroll story (body.html .story): as each step's text arrives the pinned brain glides to it,
  // coloured as in the figures: the Kenyon cell compartments, the Large-type subclusters
  // (fig:kc_subtypes), the optic lobe neuropils, then front-on and cut down to a section. Every change is
  // an eased approach or the renderer's own dissolve, never a cut; the section's cut waits for the
  // other cells to fade back first, then closes in as a soft band (uSlab.w) so cells leave one by
  // one instead of a hard plane sweeping through. Entering the atlas eases it all back to the
  // whole brain as the move begins, and it never touches the viewer's state once there.
  const story = (() => {
    const steps = [...document.querySelectorAll("#intro .story .step")];
    const fig = document.querySelector("#intro .hero-fig"), narrow = matchMedia("(max-width: 1000px)");
    // phones: each step is the screen left under the pinned brain, so the scroll can settle with the
    // brain above and the step's words below (styles.css reads --fig-h)
    if (fig && "ResizeObserver" in window) new ResizeObserver(() => document.getElementById("intro").style.setProperty("--fig-h", fig.offsetHeight + "px")).observe(fig);
    const still = matchMedia("(prefers-reduced-motion: reduce)");
    const cur = { amp: 1, zoom: 1, front: 0, off: [0, 0, 0], thin: 0 };
    const S = { cur, want: 0, goals: null, slab: null, last: 0, maskAt: 0, grow: 0, secP: 1, secDrawn: 1 };
    const bits = (keys) => keys.reduce((m, k) => (HERO && HERO.regions && k in HERO.regions ? m | (1 << HERO.regions[k]) : m), 0);
    function prepare() {
      // region centroids and the section planes, from the hero cells themselves
      const hd = T.heroData; if (!hd || S.goals) return;
      const MB = bits(["in_kc_ic", "in_kc_nc", "in_kc_oc"]), OL = bits(["in_lamina", "in_medulla", "in_lobula"]);
      const cent = (mask) => {
        let X = 0, Y = 0, n = 0;
        for (let i = 0; i < hd.n; i++) if (hd.region[i] & mask) { X += hd.y[i]; Y += -hd.x[i]; n++; }
        return n ? [X / n / 32767, Y / n / 32767] : [0, 0];
      };
      const b = HERO.box, cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2, mb = cent(MB);
      const zs = [...new Set(hd.z)].sort((a, c) => a - c), mid = zs[zs.length >> 1];
      const gap = zs.length > 1 ? (zs[zs.length - 1] - zs[0]) / (zs.length - 1) : 300;
      // exactly one section: the story draws the full cells, so one plane holds ~23k of them, and
      // only that section's own cells may sit over its DAPI image (two sections' depth put the
      // neighbour's cells over it, where they could not line up)
      S.slab = { z: mid / 32767, thin: 0.45 * gap / 32767, open: 2 };
      const mbOff = [(mb[0] - cx) * 0.8, (mb[1] - cy) * 0.8, 0];
      S.goals = [   // mode: 0 cell type · 4 region (figure colours) · 5 Large-type subclusters
        { amp: 1, zoom: 1, front: 0, off: [0, 0, 0], thin: 0, mask: 0, mode: 0 },
        { amp: 0.7, zoom: 0.6, front: 0.2, off: mbOff, thin: 0, mask: MB, mode: 4 },
        { amp: 0.7, zoom: 0.6, front: 0.2, off: mbOff, thin: 0, mask: MB, mode: 5 },
        { amp: 0.75, zoom: 0.95, front: 0.3, off: [0, 0, 0], thin: 0, mask: OL, mode: 4 },
        { amp: 0.4, zoom: 1, front: 0.45, off: [0, 0, 0], thin: 1, mask: 0, mode: 0 },   // a slice in space: tilted and gently turning with its DAPI, the cells fixed on their nuclei (no drift)
      ];
    }
    // a step is on once its text reaches the reading line: 60% up the screen beside the brain on wide
    // screens; on a phone, the pinned brain's lower edge, so the words and the picture always match.
    // Measured on scroll, not per frame.
    function measure() {
      // Scrollama's rule: a step is on once its card (on wide screens its text) crosses the line
      const line = innerHeight * (narrow.matches ? 0.82 : 0.6);   // phones: as soon as a card is properly in view, not once it is half way up
      let k = 0; steps.forEach((el, i) => { if ((narrow.matches ? el.querySelector(".card") || el : el).getBoundingClientRect().top < line) k = i + 1; });
      if (k !== S.want) { S.want = k; R.invalidate(); if (intro) intro.classList.toggle("in-story", k > 0); }
      // reaching the story is intent to see more: the full atlas starts loading (and is swapped in
      // under the story once staged) rather than waiting for a hover on the way into the atlas
      if (k > 0 && window.__BIMP_GO__) window.__BIMP_GO__();
    }
    const intro = document.getElementById("intro");
    if (steps.length && intro) {
      // scrolling stays entirely the reader's, as in a newspaper's scrolling graphics
      intro.addEventListener("scroll", measure, { passive: true });
      addEventListener("resize", measure); measure();
    }
    S.tick = (now) => {
      prepare(); if (!S.goals) return;
      // the story runs at home and while leaving for the atlas (where it eases back to step 0);
      // in the atlas and on the way home the viewer owns the state
      const home = T.tp <= 0 && !T.dir, leaving = T.going === "atlas" && T.tp > 0 && T.tp < 1;
      const H = BIMP.histology;
      if (!home && !leaving) { S.last = now; return; }
      const g = S.goals[leaving ? 0 : Math.min(S.want, S.goals.length - 1)];
      // the real section under the cut: fetched on intent (the optic-lobe step, the one before
      // it), never on first load; the cut then centres on that section's own depth
      // (preloaded once the home brain has built, so the section's image is there before the
      // reader is; it waited for the step before, and on a phone often arrived after the cells)
      if (home && (S.want >= 3 || T.bu >= 1) && !S.histoAsked && H && H.warmHome) {
        S.histoAsked = true;
        H.warmHome(S.slab.z).then((z) => { if (z != null) S.slab.z = z; });
      }
      const dt = S.last ? Math.min(0.1, (now - S.last) / 1000) : 0; S.last = now;
      const k = still.matches ? 1 : 1 - Math.exp(-dt / (leaving ? 0.18 : 0.45));
      cur.amp += (g.amp - cur.amp) * k; cur.zoom += (g.zoom - cur.zoom) * k; cur.front += (g.front - cur.front) * k;
      for (let i = 0; i < 3; i++) cur.off[i] += (g.off[i] - cur.off[i]) * k;
      // Into the section, in three beats: the view on screen dissolves away as the brain turns
      // face-on, the section's own DAPI image fades in alone, and only then do that section's cells
      // settle in over it (cut, tissue, cells; before, the section's full colours landed on top of
      // the optic lobes in one frame and buried the image). Back out, the region view returns
      // first and the cut opens once the section's cells have gone.
      const cutting = g.thin > 0;
      if (!cutting) { S.lastMask = g.mask; S.lastMode = g.mode; }
      const NONE = 1 << 30;   // a region bit no cell has: every cell dissolves out
      // the cells come in once the tissue is fully up (DAPI is at full strength from thin 0.75):
      // waiting for 0.93 on an exponential approach held them back ~2.4 s, the slow part of a
      // curve that was nearly there
      const CELLS_AT = 0.8;
      const mask = cutting ? (cur.thin < CELLS_AT ? NONE : g.mask) : g.mask;
      const mode = cutting ? (cur.thin < CELLS_AT && S.lastMode != null ? S.lastMode : g.mode) : g.mode;
      if (s.regionMask !== mask) { s.regionMask = mask; S.maskAt = now; }   // against the live state: the atlas may have changed it
      if (s.mode !== mode) s.mode = mode;
      // Phones keep the home sample (no full tier), which holds only ~2k of the section's ~24k
      // cells; the sample also carries the rest of that one section, hidden (renderer upload
      // data.hx), and they sprinkle in with the section's cells and out again on the way back.
      if (home && T.slot === "hero" && T.bu >= 1 && HERO.nSection) {
        const want = cutting && cur.thin >= CELLS_AT ? 2 : 1;
        S.secP += (want - S.secP) * (still.matches ? 1 : 1 - Math.exp(-dt / 0.35));
        if (Math.abs(want - S.secP) < 0.002) S.secP = want;
        if (S.secP !== S.secDrawn) { S.secDrawn = S.secP; R.setProgress(S.secP); }
      }
      const thinGoal = cutting ? g.thin : (now - (S.maskAt || 0) > 550 ? 0 : cur.thin);
      // the section: the slab's half-thickness eases between the whole depth and one plane, on a
      // log scale so the last few sections fall away as smoothly as the first
      // the cut runs on a slower clock than the camera, so the three beats have time to read
      cur.thin += (thinGoal - cur.thin) * (still.matches ? 1 : 1 - Math.exp(-dt / (leaving ? 0.18 : 0.6)));
      // Eased values land exactly once close: an approach that never arrives kept moving the
      // section's band by hairs every frame, and cells at its edge blinked between the renderer's
      // two passes. A settled view is now truly still.
      const land = (a, b, e) => (Math.abs(a - b) < e ? b : a);
      cur.thin = land(cur.thin, thinGoal, 0.004); cur.amp = land(cur.amp, g.amp, 0.002);
      cur.zoom = land(cur.zoom, g.zoom, 0.0005); cur.front = land(cur.front, g.front, 0.001);
      for (let i = 0; i < 3; i++) cur.off[i] = land(cur.off[i], g.off[i], 1e-5);
      // Leaving for the atlas from the section, the rest of the brain grows back around it in step
      // with the move (on the move's own clock), instead of snapping whole in a fifth of a second
      // and then thinning mid-flight.
      if (leaving) { if (S.leaveThin == null) S.leaveThin = cur.thin; cur.thin = S.leaveThin * (1 - sstep(0.1, 0.8, T.tp)); }
      else S.leaveThin = null;
      // past the first screen, on devices that run the full atlas, the story draws all of the
      // queen's cells, not the home sample: the full tier is already staged on the GPU, and its extra
      // cells grow in around the ones on screen (the same swap the move into the atlas makes)
      // the sample's cells keep their birth times in the full tier, so the swap is invisible at any
      // moment of the first build: make it as soon as the full tier is staged
      // (not while the sample is still opening on a slow link: its own curve finishes first)
      if (home && BIMP.tier === "full" && T.slot === "hero" && T.staged && (T.ba === 0 || T.ba >= 1)) swapToFull();
      // one way: once the full cells are in they stay, so scrolling back up shows the same brain
      // instead of hundreds of thousands of cells dissolving away and growing back
      // An even ramp, not an exponential approach: that put 40% of 600k new cells on screen in
      // the first 0.3 s, which read as a snap. Eased at both ends over 2.4 s, the full brain
      // fills in at a steady rate around the sample.
      const sl = S.slab;
      if (cur.thin > 0.002) {
        const w = Math.exp(Math.log(sl.open) + (Math.log(sl.thin * 0.12) - Math.log(sl.open)) * cur.thin);   // a hairline edge once cut
        if (!s.slab || s.slab[0] !== sl.z || s.slab[3] !== w) s.slab = [sl.z, sl.z, sl.thin, w];
      }
      else if (s.slab && s.slab[2] > 0) s.slab = [];
      // the section's DAPI fades in on its own once the last view has gone, before the section's
      // cells arrive over it; out ahead of the move into the atlas
      if (H && H.homeSection) H.homeSection(S.slab.z, (home || leaving) && cur.thin > 0.002 ? 0.85 * Math.max(0, Math.min(1, (cur.thin - 0.3) / 0.45)) : 0);   // follows the cut both ways (scrolling back up it vanished in one frame)   // under the cells, not over them
      if (Math.abs(thinGoal - cur.thin) > 1e-3 || Math.abs(g.zoom - cur.zoom) > 1e-3) R.invalidate();
    };
    return S;
  })();
  async function startHero() {
    try {
      const hd = await decode.fetchBase(window.__BIMP_SRC__ + HERO.file, { n: HERO.n, offsets: HERO.offsets }, HERO.gzip);
      T.heroData = hd;
      // packed (x, y) → z of every hero cell, to find the same cells in the full atlas
      const K = new Map();
      for (let i = 0; i < hd.n; i++) K.set(((hd.x[i] + 32768) * 65536) + (hd.y[i] + 32768), hd.z[i]);
      T.heroKeys = K;
      if (T.slot === "none") {
        R.upload(hd, { half: BUILD_H }); R.setProgress(0); T.slot = "hero"; T.born = performance.now();
        // the atlas colours each cell by its group (ui.js; the manifest's canonColors), not by the renderer's
        // default subtype palette: use the same colours here, so the brain looks the same before and after
        if (HERO.canonColors) R.setPalette(HERO.canonColors);
      }
      // every cell is the goal on the first screen, so the full atlas downloads while the sample
      // opens rather than after it: it is usually in by the time the sample has settled
      if (window.__BIMP_GO__) window.__BIMP_GO__();
      // pause the GL work when the hero is off screen
      if ("IntersectionObserver" in window) new IntersectionObserver((es) => { T.visible = es[0].isIntersecting; }).observe(heroBox);
      const intro = document.getElementById("intro");
      const point = (e) => {
        const r = heroRect();
        T.ptr = [((e.clientX - r.x) / r.w) * 2 - 1, 1 - ((e.clientY - r.y) / r.h) * 2];
        T.ptrOn = e.clientX > r.x - 60 && e.clientX < r.x + r.w + 60 && e.clientY > r.y - 60 && e.clientY < r.y + r.h + 60;
      };
      intro.addEventListener("pointermove", point, { passive: true });
      intro.addEventListener("pointerdown", point, { passive: true });
      intro.addEventListener("pointerleave", () => { T.ptrOn = false; });
      requestAnimationFrame(frame);
    } catch (e) { console.warn("hero:", e); }
  }
  // the home look at tp = 0, the atlas look at tp = 1, and everything in between
  // the home look at tp = 0, the atlas look at tp = 1. The order is what keeps it clean:
  //   1. the ink becomes the cell colours, still on paper (colours read on light and dark alike);
  //   2. the lights go down, quickly, over the whole page (never a dark shape on the light page);
  //   3. the camera glides, the box grows to the stage and the other cells grow in;
  //   4. only then do the panels and their hairlines come in.
  function applyLook(now) {
    // Going in: colour, lights down, then the glide and the growth. Coming home is not the same
    // clock run backwards (that rushed the glide and dropped the cells all at once): the camera
    // settles back first, the extra cells then dissolve inward, and the page lightens last.
    const back = T.dir < 0 || T.going === "home";
    const L = R.look;
    const kCol = back ? sstep(0, 0.35, T.tp) : sstep(0, 0.3, T.tp);
    const kDark = back ? sstep(0.05, 0.4, T.tp) : sstep(0.06, 0.48, T.tp);
    const e = back ? ease(sstep(0.45, 1, T.tp)) : ease(sstep(0.18, 1, T.tp));
    // the box is measured once per move (T.hr), not every frame: a per-frame getBoundingClientRect
    // after the frame's style writes forced a synchronous layout of the whole page each frame
    const full = T.full || (T.full = { x: 0, y: 0, w: canvas.clientWidth, h: canvas.clientHeight });
    const hr = (T.dir || T.tp > 0) && T.hr ? T.hr : heroRect();
    // At rest on the home page the brain is drawn at the canvas's top-left and copied into a
    // small canvas inside the page (mirrorHero), which scrolls natively with the text. Drawing it
    // on the fixed full-screen canvas at the box's on-screen position lagged a frame behind the
    // scroll on phones, so the brain shook. The full-screen canvas takes over for the move.
    const atRest = T.tp <= 0 && !T.dir;
    L.rect = atRest ? { x: 0, y: 0, w: hr.w, h: hr.h }
      : (T.tp >= 1 ? null : { x: lerp(hr.x, 0, e), y: lerp(hr.y, 0, e), w: lerp(hr.w, full.w, e), h: lerp(hr.h, full.h, e) });
    L.scissor = T.tp <= 0;
    // the first screen is the dark stage itself, so the brain never changes ground: only the
    // paper below it fades (styles.css, #intro's background under html.stage-live)
    for (let k = 0; k < 3; k++) L.bg[k] = DARK[k];
    L.ink = 0;
    L.scaffold = 1 - kCol;
    L.birthHash = 0;   // the even sprinkle is in the full tier's birth times now (renderer upload opts.half)   // the story's swap to the full cells: an even sprinkle, not a second radial entrance   // the home story highlights regions over a grey scaffold; the atlas filters as before
    L.wob = 0.011 * (1 - kCol) * (1 - story.cur.thin);   // the home brain's drift, stilled for the section
    T.ptrK += ((T.ptrOn && T.tp === 0 ? 1 : 0) - T.ptrK) * 0.08;
    L.ptr[0] = T.ptr[0]; L.ptr[1] = T.ptr[1]; L.ptr[2] = 0.2; L.ptr[3] = 0.5 * T.ptrK;
    L.sizeMul = lerp(0.42 * (1 - 0.35 * story.cur.thin), 1, e);   // over the DAPI the section's cells are drawn finer, so the tissue shows between them
    // load in order: the page's text first, then the brain fades up once its cells are in
    // the brain fades up just after the page's entrance starts (intro-stipple.js sets READY_AT),
    // so a cold first load plays as one sequence instead of pieces arriving as they land
    const readyAt = window.__BIMP_READY_AT__;
    L.alpha = T.tp > 0 ? (T.hrOff && T.going === "atlas" ? sstep(0, 0.3, T.tp) : 1) : (readyAt ? sstep(0, 500, now - readyAt - 250) : 0);
    // and it opens the way the atlas does, its cells growing out from the centre (renderer birth
    // order) rather than the whole brain fading up; set only while it changes, so a page at rest idles
    // One build, one clock (BUILD_S). The sample's cells grow out from the centre over the first
    // BUILD_H of the reveal; the full atlas, downloading meanwhile, is swapped in under them
    // (invisibly: its copies of those cells keep their birth times) and its other cells sprinkle in
    // evenly over the rest, at the speed the build was already going. A sample left waiting for a
    // slow download holds there, and carries on when it lands; on devices that keep the sample
    // (the lite tier) it simply stops, its last cells in.
    if ((T.slot === "hero" || T.slot === "full") && T.tp <= 0 && !T.dir && T.bu < 1) {
      const d = T.bLast ? Math.min(0.05, (now - T.bLast) / 1000) : 0; T.bLast = now;
      const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
      // Every cell first, then one build: downloading, decoding and uploading 1.3M cells while the
      // brain grew stalled its frames (choppy, then a snap). So the build waits, its box empty,
      // until the full atlas is on the GPU, and then grows all of the queen's cells out from the
      // centre in one go over BUILD_S, as the atlas opens. Only a slow link (past BUILD_WAIT) opens
      // the sample first, easing to rest at BUILD_H of the reveal; the rest then eases in from rest
      // when it lands (each phase its own curve, so neither stops or starts at speed).
      const waiting = T.slot === "hero" && T.ba === 0 && BIMP.tier !== "lite" && readyAt && now - readyAt < 250 + BUILD_WAIT;
      let p = T.homeP;
      if (readyAt && now - readyAt > 250 && !waiting) {
        if (T.slot === "hero") {
          T.ba = still ? 1 : Math.min(1, T.ba + d / 3);
          p = BUILD_H * (1 - Math.pow(1 - T.ba, 3));
          if (T.ba >= 1) { T.heroRev = 1; if (BIMP.tier === "lite") { T.bu = 1; p = 1; } }
        } else {
          T.bb = still ? 1 : Math.min(1, T.bb + d / (T.ba > 0 ? 3.5 : BUILD_S));
          const e = T.bb * T.bb * (3 - 2 * T.bb);
          p = T.ba > 0 ? BUILD_H + (1 - BUILD_H) * e : e;
          T.heroRev = 1;
          if (T.bb >= 1) T.bu = 1;
        }
      }
      if (p !== T.homeP) { T.homeP = p; R.setProgress(p); }
    } else T.bLast = 0;
    // every cell at the two ends (the slot swap must match frame for frame); in the middle, the
    // viewer's usual motion thinning keeps the frame rate while the camera moves
    L.lockLod = T.tp < 0.25 || (T.going === "atlas" && story.leaveThin > 0.01);   // and every cell through a move that starts from the section, as it grows back
    // no newborn flash while cells grow in or dissolve during the move (it lit the core white),
    // and no depth fog while the ground passes through grey (it faded the cells into it)
    L.spark = T.tp >= 1 && !T.dir ? 1 : 0;
    L.fogK = 1 - 3 * kDark * (1 - kDark);   // eases down to 0.25 at mid-dim and back, no pop
    if (T.tp < 1) {
      story.tick(now);
      const h = heroPose(now, hr), a = T.atlasPose || camera.HOME, c = camera.cam;
      let dth = a.theta - h.theta; while (dth > Math.PI) dth -= 2 * Math.PI; while (dth < -Math.PI) dth += 2 * Math.PI;
      c.theta = h.theta + dth * e; c.phi = lerp(h.phi, a.phi, e); c.radius = lerp(h.radius, a.radius, e);
      for (let k = 0; k < 3; k++) c.target[k] = lerp(h.target[k], a.target[k], e);
      // on devices that run the full atlas the home page draws every cell too (the story swaps it
      // in), so the way home keeps them all: dissolving to the hero sample and then growing the
      // full brain back from nothing left the home page near-empty for a second after landing
      // Where the home page keeps only the sample (phones), the way home dissolves the other cells
      // down to it, never past: running the build to 0 emptied the brain, and the sample then
      // snapped back in at the swap. Its cells are born by BUILD_H, fully in once the front
      // (FADE wide) has passed that.
      const floor = T.stagedHalf ? (BUILD_H + (R.FADE || 0)) / (1 + (R.FADE || 0)) : 0;
      if (T.slot === "full" && T.tp > 0) R.setProgress(back && keepFull() ? 1 : Math.max(back ? floor : T.homeP, back ? sstep(0.2, 0.8, T.tp) : sstep(0.35, 1, T.tp)));
    }
    // only the atlas shell reads these, so they live on #app: set on <html> they restyled the whole
    // home page every frame. And only when they change.
    const tpS = T.tp.toFixed(3), dkS = kDark.toFixed(3);
    if (tpS !== T.tpS) { appEl.style.setProperty("--tp", tpS); T.tpS = tpS; }
    if (dkS !== T.dkS) { appEl.style.setProperty("--dk", dkS); T.dkS = dkS; }
  }
  // Home at rest: copy the brain from the full-screen canvas into the in-page canvas, in the same
  // frame it was drawn (the WebGL buffer is only readable until the frame is presented). In
  // flight or in the atlas, the page goes see-through (html.stage-live) and the full-screen
  // canvas is shown instead; the switch happens on one frame, so it is never visible.
  let mirror = null, mctx = null;
  function mirrorHero() {
    const atRest = T.tp <= 0 && !T.dir;
    if (atRest) {
      if (!mirror) {
        mirror = document.createElement("canvas"); mirror.className = "hero-mirror"; mirror.setAttribute("aria-hidden", "true");
        heroBox.insertBefore(mirror, heroBox.firstChild); mctx = mirror.getContext("2d");
      }
      // the renderer's own pixel ratio, which it lowers when frames run slow: assuming the screen's
      // copied the wrong patch of the canvas (the brain magnified and off to one side)
      const [bx, by] = bleed(), d = R.getDpr ? R.getDpr() : Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(heroBox.clientWidth * (1 + 2 * bx) * d), h = Math.round(heroBox.clientHeight * (1 + 2 * by) * d);
      if (mirror.width !== w || mirror.height !== h) {
        mirror.width = w; mirror.height = h;
        // explicit size: a canvas does not stretch between its offsets as a div does (with width
        // auto it showed at its pixel size, twice too large and half off the page)
        Object.assign(mirror.style, { left: `${-bx * 100}%`, top: `${-by * 100}%`, right: "auto", bottom: "auto",
          width: `${(1 + 2 * bx) * 100}%`, height: `${(1 + 2 * by) * 100}%` });
      }
      mctx.clearRect(0, 0, w, h);
      mctx.drawImage(canvas, 0, 0, w, h, 0, 0, w, h);
      mirror.style.visibility = "";
    } else if (mirror) mirror.style.visibility = "hidden";
    root.classList.toggle("stage-live", !atRest);
  }
  // dorsal / lateral / anterior, turned by the same view matrix the cells are drawn with, so the
  // rocking reads unmistakably as rotation in depth (world +y is dorsal, x lateral, z anterior)
  const triadG = heroBox && heroBox.querySelector(".triad g");
  function triad() {
    if (!triadG) return;
    const c = camera.cam, V = BIMP.mat4.lookAt(camera.eye(), c.target, [0, 1, 0]), L = 26;
    const ax = [[0, 1, 0, "dorsal"], [1, 0, 0, "lateral"], [0, 0, 1, "anterior"]];
    let out = "";
    for (const [x, y, z, t] of ax) {
      const dx = (V[0] * x + V[4] * y + V[8] * z) * L, dy = -(V[1] * x + V[5] * y + V[9] * z) * L;
      const len = Math.hypot(dx, dy), ux = len ? dx / len : 0, uy = len ? dy / len : -1;
      // the label is centred beyond the tip, pushed out along the axis by an ellipse fitted to its
      // own width, so it glides round the end as the axis turns instead of snapping sides
      const hw = t.length * 3.1 + 2, r = Math.hypot(hw * ux, 6 * uy) + 3;
      const lx = dx + ux * r, ly = dy + uy * r + 4;
      // an axis turning toward the viewer shortens to a point; fade it rather than let it spin
      const op = Math.max(0, Math.min(1, (len - 5) / 8));
      out += `<g opacity="${op.toFixed(2)}"><line x1="0" y1="0" x2="${dx.toFixed(1)}" y2="${dy.toFixed(1)}"/>` +
        `<circle cx="${dx.toFixed(1)}" cy="${dy.toFixed(1)}" r="1.6"/>` +
        `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle">${t}</text></g>`;
    }
    triadG.innerHTML = out;
    triadG.style.opacity = R.look.alpha;      // it arrives with the brain, not before it
  }
  // Stage the full atlas on the GPU while the reader is still on the home page (idle time), so
  // the click itself only flips which slot is drawn. Uploading 1.3M cells and matching them to
  // the hero cells at the click was the long first frame of the move.
  function stage() {
    if (!T.pendingFull || T.slot !== "hero") return;
    const full = T.pendingFull; T.pendingFull = null;
    // nothing shown yet: the full tier's own centre-out order, the atlas's opening; else carry on from the sample
    R.upload(full, T.ba > 0 ? { keep: true, heroKeys: T.heroKeys, heroData: T.heroData, half: BUILD_H } : { keep: true });   // full becomes current, the hero is held aside…
    R.swapSlots();                                          // …and swapped straight back: hero drawn, full staged
    DATA = full; BIMP.data = full; ui.setData(full);
    T.staged = true; T.stagedHalf = T.ba > 0;               // the sample's cells are all in by BUILD_H in it
  }
  function prestage() {
    const idle = window.requestIdleCallback || ((f) => setTimeout(f, 200));
    // not while the sample is still opening: the 1.3M upload is one long frame, and it hitched the reveal
    // A device that keeps the sample (lite) stages only once the sample has opened, so the atlas
    // tier inherits its cells' birth times (stage, opts.half) and the way home can dissolve down
    // to them; staged earlier it emptied and the sample snapped back in. A click before then
    // stages at once (toAtlas), as before.
    const go = () => {
      if (T.dir !== 0 || T.tp !== 0) return;
      if (BIMP.tier === "lite" && T.ba < 1) return void setTimeout(() => idle(go, { timeout: 2000 }), 250);
      stage();
    };
    idle(go, { timeout: 2000 });
  }
  const keepFull = () => BIMP.tier === "full" && T.heroRev === 1 && T.staged;
  function swapToFull() {
    if (T.slot === "full") return;
    if (T.pendingFull) stage();
    if (!T.staged || !R.swapSlots()) return;
    R.setProgress(0); T.slot = "full";
  }
  function step(dt) {
    if (!T.dir) return;
    T.tp = Math.max(0, Math.min(1, T.tp + T.dir * Math.min(T.rawDt || dt, 0.1) / (T.dir > 0 ? T.secs : T.secsBack)));
    const intro = document.getElementById("intro");
    if (T.dir > 0 && T.tp > 0.62 && !introGone) { intro.classList.add("gone"); introGone = true; }   // panels start to settle in
    // coming home, the page's text starts back while the brain is still settling, not after it
    if (T.dir < 0 && T.tp < 0.42 && intro.classList.contains("leaving")) intro.classList.remove("leaving");   // early enough to ease in over the last ~1.2 s
    if (T.tp >= 1 && T.dir > 0) {
      T.dir = 0; R.setProgress(1); setView("atlas"); maybeReveal(); showTourBtn();
      const f = T.onDone; T.onDone = null; if (f) f();
    } else if (T.tp <= 0 && T.dir < 0) {
      T.dir = 0;
      if (T.slot === "full" && keepFull()) { T.bu = 1; T.homeP = 1; }   // stays full: the story would only swap it back in
      else if (T.slot === "full" && R.swapSlots()) T.slot = "hero";   // at the floor: only the hero cells were showing
      R.setProgress(1); story.secP = 1; story.secDrawn = 1; setView("home"); setLoad("");
      intro.classList.remove("leaving");                          // the page comes back
    }
  }
  function toAtlas(opts) {
    opts = opts || {};
    if (!opts.fromHistory && heroOn) pushView(true);
    if (!heroOn) { dismissIntro(); if (opts.tour && BIMP.tour) BIMP.tour.start(); if (opts.then) opts.then(); return; }
    const go = () => {
      swapToFull();
      T.atlasPose = { theta: camera.HOME.theta, phi: camera.HOME.phi, radius: camera.HOME.radius, target: camera.HOME.target.slice() };
      T.onDone = () => {
        if (T.histoWas && BIMP.histology && BIMP.histology.setOn) { T.histoWas = false; BIMP.histology.setOn(true); }
        if (opts.tour && BIMP.tour) BIMP.tour.start();
        // the home story changes the region filter and colour mode without the atlas's panels, which
        // kept a stale snapshot (regions listed as cell types, compartments ticked): on landing, run
        // the panels' own 'all regions' and 'colour by type' so they rebuild from the real state
        if (!opts.then) {
          const all = document.getElementById("region-all"); if (all) all.click();
          const ty = document.querySelector('#mode-seg [data-mode="celltype"]'); if (ty) ty.click();
        }
        // a way in that opens on a particular view (both brains, gene mode) sets it once the move
        // has landed, so its own animation (the brains sliding apart) runs on a settled camera
        if (opts.then) setTimeout(opts.then, 120);
      };
      document.getElementById("intro").classList.add("leaving");
      if (camera.cancelFly) camera.cancelFly();
      T.hr = heroRect(); T.full = null; T.going = "atlas";
      // Entered from further down the page, the hero brain is scrolled out of view: start the
      // move from a box of the same size in the middle of the screen and fade the brain in,
      // rather than sliding it down from above the top edge.
      const vh = innerHeight, hd = parseFloat(getComputedStyle(root).getPropertyValue("--hd")) || 56;
      T.hrOff = T.hr.y + T.hr.h < hd + 40 || T.hr.y > vh - 40;
      if (T.hrOff) T.hr = { x: (innerWidth - T.hr.w) / 2, y: hd + (vh - hd - T.hr.h) / 2, w: T.hr.w, h: T.hr.h };
      setView("to-atlas"); T.dir = 1;
    };
    if (BIMP.fullReady && (T.pendingFull || T.staged || T.slot === "full")) return go();
    if (window.__BIMP_GO__) window.__BIMP_GO__();
    const b = document.getElementById(opts.tour ? "enter" : "enter-tour");
    const label = b ? b.textContent : "";
    if (b) { b.textContent = "Loading the atlas…"; b.setAttribute("aria-busy", "true"); }
    BIMP.ready.then(() => { if (b) { b.textContent = label; b.removeAttribute("aria-busy"); } go(); });
  }
  function toHome(opts) {
    if (!(opts && opts.fromHistory) && heroOn && T.tp > 0) pushView(false);
    if (!heroOn || T.slot === "none") { location.href = "./"; return; }
    if (BIMP.tour && BIMP.tour.isActive && BIMP.tour.isActive() && BIMP.tour.stop) BIMP.tour.stop();
    s.selected = -1; s.hover = -1; if (ui.hideTip) ui.hideTip();
    // The home page holds only the queen's cells, in cell-type order, unfiltered. Ease the viewer
    // back to that view through its own controls (so the panel stays in step), and its morph
    // runs the change while the camera flies: two brains slide back into one, filtered cells
    // return, colours dissolve. Otherwise the hand-back to the home cells jumped.
    const press = (sel) => { const b = document.querySelector(sel); if (b && b.getAttribute("aria-pressed") !== "true") b.click(); };
    press('#caste-seg button[data-v="queen"]');
    press('#mode-seg button[data-mode="celltype"]');
    if (s.regionMask) { const b = document.getElementById("region-all"); if (b) b.click(); }
    if (s.visible.some((v) => !v)) { const b = document.getElementById("leg-all"); if (b) b.click(); }
    s.iso = -1;
    if (camera.cancelFly) camera.cancelFly();   // the caste button queues its own camera flight; this move owns the camera
    // The home page draws only the hero cells: leave 2D and switch histology off for the way
    // home (the slab and the sections otherwise carried into the hero), and bring histology
    // back on the next visit to the atlas.
    const H = BIMP.histology;
    if (H) {
      try {
        if (H.is2D && H.is2D()) H.exit2D();
        s.slab = [];
        T.histoWas = !!(H.isOn && H.isOn());
        if (T.histoWas && H.setOn) H.setOn(false);
      } catch (e) { /* histology is optional */ }
    }
    const intro = document.getElementById("intro");
    if (T.tp >= 1) {
      // fly back from wherever the reader left the camera
      const c = camera.cam; T.atlasPose = { theta: c.theta, phi: c.phi, radius: c.radius, target: c.target.slice() };
      intro.scrollTop = 0;
    }
    intro.classList.add("leaving"); intro.classList.remove("gone"); introGone = false;
    T.hr = heroRect(); T.full = null; T.going = "home"; T.hrOff = false;
    setView("to-home"); T.dir = -1;
  }
  BIMP.transition = { toAtlas, toHome, get tp() { return T.tp; } };
  // "Load all cells": the lite tier (phones, weak devices; see deviceTier) upgrades to the full
  // one in place. The cells already on screen stay where they are and the other 1.2M grow in
  // around them (renderer upload opts.born); the choice is remembered on this device.
  // Resolves true once the full tier is showing. UI: atlas-ui's panel calls it.
  let upgrading = null;
  BIMP.loadFullDetail = function () {
    if (BIMP.tier !== "lite") return Promise.resolve(true);
    if (upgrading) return upgrading;
    try { localStorage.setItem("bimp-detail", "full"); } catch (e) { /* private mode: this visit only */ }
    const M = BIMP.manifest, SRC = window.__BIMP_SRC__;
    setStatus("loading all " + M.n.toLocaleString() + " cells…");
    return (upgrading = (async () => {
      const full = await decode.fetchBase(SRC + M.base, { n: M.n, offsets: M.offsets }, M.gzip, BASE_RETRY("the cells"));
      // never mid-build or mid-hand-off: wait for the view to settle
      await new Promise((r) => { const w = () => (building || T.dir ? setTimeout(w, 100) : r()); w(); });
      const lite = DATA, idx = lite.fullIndex, sel = s.selected;
      decode.setGather(null);
      DATA = full; BIMP.data = full; BIMP.tier = "full"; BIMP.tierWhy = "loaded on request";
      s.selected = -1;
      R.upload(full, { born: idx }); R.setProgress(0); ui.setData(full); T.slot = "full";
      if (sel >= 0 && idx) s.selected = idx[sel];
      revealFull();
      setStatus(""); setLoad("ready — " + M.n.toLocaleString() + " cells");
      url.set("detail", null);   // a link copied from here should open the full atlas
      dispatchEvent(new CustomEvent("bimp:tier", { detail: { tier: "full" } }));
      return true;
    })().catch((e) => { upgrading = null; setStatus("couldn't load all cells (" + e.message + ") — try again"); throw e; }));
  };
  // for tests and the perf harness: the loop's own state
  BIMP._loop = () => ({ building, buildProgress, dataReady, introGone, asleep: !!frame._asleep, idle: frame._idle || 0, last: frame._last || 0, slot: T.slot });

  // ---- linkable pages ----------------------------------------------------------------
  // The home page is the bare address (section anchors like #s-anatomy scroll the page);
  // the atlas is ?atlas plus its view state. Entering and leaving push history entries, so
  // the back button runs the same transition in reverse, and a copied atlas address reopens
  // the same view. Every part of the viewer records its own state through BIMP.url.set
  // (camera below; types, genes, histology and sections from ui.js / histology.js); the
  // existing readers (applyUrlParams, histology.js) restore it on load.
  const url = (function () {
    let pending = null, timer = 0;
    const flush = () => {
      timer = 0; if (!pending) return;
      const P = new URLSearchParams(location.search);
      for (const [k, v] of Object.entries(pending)) { if (v === null || v === undefined || v === false) P.delete(k); else P.set(k, v === true ? "" : String(v)); }
      pending = null;
      if (root.dataset.view !== "atlas") return;        // the home page's address stays clean
      const q = P.toString().replace(/=(&|$)/g, "$1");  // bare flags read as ?atlas, not ?atlas=
      history.replaceState(history.state, "", location.pathname + (q ? "?" + q : "") + location.hash);
    };
    return {
      get: (k) => new URLSearchParams(location.search).get(k),
      // batch writes and apply them at most every 250 ms: replaceState on every slider tick is costly
      set(k, v) { (pending || (pending = {}))[k] = v; if (!timer) timer = setTimeout(flush, 250); },
    };
  })();
  BIMP.url = url;
  function pushView(atlas) {
    if (atlas) history.pushState({ bimp: "atlas" }, "", location.pathname + "?atlas");
    else history.pushState({ bimp: "home" }, "", location.pathname);
  }
  addEventListener("popstate", () => {
    const wantAtlas = /[?&](nointro|atlas)(=|&|$)/.test(location.search);
    if (!heroOn) { if (!wantAtlas !== !introGone) location.reload(); return; }
    if (wantAtlas && T.tp < 1 && T.dir <= 0) toAtlas({ fromHistory: true });
    else if (!wantAtlas && T.tp > 0 && T.dir >= 0) toHome({ fromHistory: true });
  });
  // the camera, written once it settles
  let camKey = "";
  setInterval(() => {
    if (root.dataset.view !== "atlas" || (camera.isFlying && camera.isFlying())) return;
    const c = camera.cam, k = [c.theta, c.phi, c.radius].map((x) => x.toFixed(2)).join(",");
    if (k === camKey) return;
    camKey = k; url.set("theta", c.theta.toFixed(2)); url.set("phi", c.phi.toFixed(2)); url.set("radius", c.radius.toFixed(2));
  }, 700);

  function activate() {
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion:reduce)").matches) {
      s.spin = false; document.getElementById("tg-spin").setAttribute("aria-pressed", "false");
    }
    window.addEventListener("resize", () => R.resize(canvas));
    wireInput(); applyUrlParams();
    requestAnimationFrame(frame);
    // Don't enable Explore yet — wait until the full tier is live (enableExplore),
    // so clicking it never lands on an empty, still-downloading canvas.
    if (BIMP.fullReady) enableExplore();
  }

  // ---- device tier: decided once, before anything heavy loads ----------------------------
  // Phones, low-memory or few-core machines, software GL and saved-data or slow links draw
  // the 100k lite tier, never the 1.3M upload (11.5 MB, ~70 MB of GPU buffers). Genes still
  // work there: base-lite.idx (build-pages.mjs) maps each lite cell into the full tier's gene
  // files. BIMP.loadFullDetail() upgrades in place and the choice is remembered on this
  // device; ?detail=full|lite overrides both.
  function deviceTier() {
    const want = new URLSearchParams(location.search).get("detail");
    if (want === "full" || want === "lite") return { tier: want, why: "?detail" };
    let saved = null; try { saved = localStorage.getItem("bimp-detail"); } catch (e) { /* private mode */ }
    if (saved === "full") return { tier: "full", why: "chosen on this device" };
    if (!window.__BIMP_LITE_IDX__) return { tier: "full", why: "no lite index shipped" };
    const nav = navigator, c = nav.connection || {}, why = [];
    if (c.saveData) why.push("data saver");
    if (/2g|3g/.test(c.effectiveType || "")) why.push(c.effectiveType + " link");
    if (nav.deviceMemory && nav.deviceMemory <= 4) why.push(nav.deviceMemory + " GB memory");
    if (nav.hardwareConcurrency && nav.hardwareConcurrency <= 4) why.push(nav.hardwareConcurrency + " cores");
    const coarse = window.matchMedia && matchMedia("(pointer: coarse)").matches;
    if (coarse && Math.min(screen.width, screen.height) < 768) why.push("phone");
    const gl = R.gl;
    if (gl) {
      if (gl.getParameter(gl.MAX_TEXTURE_SIZE) < 8192) why.push("small GPU");
      const d = gl.getExtension("WEBGL_debug_renderer_info");
      if (d && /swiftshader|llvmpipe|softpipe|software|basic render/i.test(String(gl.getParameter(d.UNMASKED_RENDERER_WEBGL)))) why.push("software GL");
    }
    return why.length ? { tier: "lite", why: why.join(", ") } : { tier: "full", why: "capable device" };
  }
  // A failure the reader can read, in the stage, instead of an empty or frozen canvas. The
  // home page's text and figures never depend on the atlas, so they keep working.
  function showStageError(msg) {
    root.classList.add("atlas-failed");
    let el = document.getElementById("stage-error");
    if (!el) {
      el = document.createElement("div"); el.id = "stage-error"; el.setAttribute("role", "alert");
      (document.getElementById("app") || document.body).appendChild(el);
    }
    el.textContent = msg;
  }
  // WebGL contexts are lost when a phone backgrounds the tab or a GPU driver resets. Every
  // buffer and texture goes with them, so the cheapest correct recovery is a reload: the
  // address already carries the view (BIMP.url), so the reader lands where they were.
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    showStageError("The graphics context was lost. Restoring the atlas…");
  });
  canvas.addEventListener("webglcontextrestored", () => location.reload());
  const BASE_RETRY = (what) => ({ retries: 3, onRetry: (n) => setStatus("connection dropped, retrying " + what + " (" + n + "/3)…") });

  const liteReady = (async function boot() {
    try { R.init(canvas); } catch (e) {
      showStageError("This browser can't draw the 3D atlas (WebGL 2 is not available). The pages about the atlas work without it.");
      throw e;
    }
    R.resize(canvas);
    if (heroOn) startHero();
    // A device that keeps the sample (deviceTier: phones, weak machines) says so under the home
    // brain, decided here, before anything is drawn, so the caption never changes under the reader
    if (heroOn && HERO.nQueen && deviceTier().tier === "lite") {
      const cap = document.getElementById("hero-cap");
      if (cap && cap.parentNode) cap.parentNode.classList.add("sample");
      if (cap) cap.textContent = "A sample of the queen’s brain in 3D: " + (HERO.n - (HERO.nSection || 0)).toLocaleString() +
        " of its " + HERO.nQueen.toLocaleString() + " cells, coloured by cell type.";
    }
    // FETCH mode (GitHub Pages): no inline payload — fetch manifest, then a small
    // LITE base for a snappy paint, then stream the full 1.3M base and swap.
    if (!window.__BIMP__ && window.__BIMP_SRC__) {
      const SRC = window.__BIMP_SRC__;
      // Nothing heavy loads while the reader is on the home page: the atlas (about 11 MB) is
      // fetched on intent, when they hover, focus or touch a way in, or scroll toward the
      // atlas section. A link straight to the viewer (?nointro) starts at once.
      if (!introGone && window.__BIMP_INTENT__) { setLoad(""); await window.__BIMP_INTENT__; }
      setLoad("loading atlas…");
      const M = JSON.parse(new TextDecoder().decode(await decode.fetchBuf(SRC + "manifest.json", false, BASE_RETRY("the atlas"))));
      BIMP.manifest = M;
      const liteTier = M.lite || { file: M.base, n: M.n, offsets: M.offsets };
      // SINGLE-BUFFER load (no tier swap → no snap). Swapping a 100k preview for the
      // 1.3M set reshuffles every point in one frame (different cells/ids), which reads
      // as a snap no matter how the density is matched. So we fetch the lite base only
      // to build the UI (which needs the category lists), keep the canvas empty, then
      // load the full base, upload it ONCE, and reveal 0→1 — the build-up comes from
      // the reveal alone, and nothing is ever swapped out from under it.
      // Upload the small lite tier so the UI (which needs a live VAO for label/palette
      // swaps) initializes — but keep reveal at 0, so the CANVAS STAYS EMPTY. When the
      // full base arrives we swap the buffer while nothing is drawn (reveal 0 = empty),
      // so the swap is invisible, then reveal 0→1 builds the full 1.3M up from nothing.
      // No visible tier swap → no snap.
      const sub = document.getElementById("title-sub");
      const LI = window.__BIMP_LITE_IDX__;
      const TIER = deviceTier();
      BIMP.tier = TIER.tier; BIMP.tierWhy = TIER.why;
      // the lite tier's full-tier indices: what lets it show genes, and what lets the full tier
      // grow around it later without a blank (84 KB, fetched beside the lite base)
      const idxP = LI && M.lite && LI.n === M.lite.n && LI.of === M.version
        ? decode.fetchLiteIndex(SRC + LI.file, LI.n).catch(() => null) : Promise.resolve(null);
      const [lite, liteIdx] = await Promise.all([
        decode.fetchBase(SRC + liteTier.file, { n: liteTier.n, offsets: liteTier.offsets }, M.gzip, BASE_RETRY("the atlas")), idxP]);
      if (liteIdx) lite.fullIndex = liteIdx;
      if (BIMP.tier === "lite" && !liteIdx) { BIMP.tier = "full"; BIMP.tierWhy = "lite index unavailable"; }
      if (BIMP.tier === "lite") decode.setGather(liteIdx);
      DATA = lite; BIMP.data = lite;
      // with the hero up, the lite tier only feeds the UI's category lists; nothing replaces the hero
      if (!heroOn) { R.upload(lite); R.setProgress(0); T.slot = "lite"; }   // uploaded but not yet revealed (empty canvas)
      ui.build({ manifest: M, data: lite, renderer: R });
      document.getElementById("hud-n").textContent = M.n.toLocaleString();
      if (sub) sub.innerHTML = "Each point is one cell, mapped to a cell type and placed back into the reconstructed volume. Loading the full " + M.n.toLocaleString() + "-cell atlas…";
      activate();
      setStatus("loading all " + M.n.toLocaleString() + " cells…");
      if (location.search.indexOf("nofull") >= 0) {   // tests / preview-only: show the lite tier
        if (liteIdx) decode.setGather(liteIdx);       // genes still colour the lite cells
        // ?gate keeps the real reveal gating (progress 0 until the intro is dismissed) on
        // the small tier — the only way to test the intro doors headless, since software
        // GL cannot drive the full 1.3M tier at any usable speed.
        if (location.search.indexOf("gate") >= 0) { dataReady = true; maybeReveal(); }
        else R.setProgress(1);   // show it fully (no build-up in tests)
        BIMP.fullReady = true; enableExplore();
        setStatus(""); resolveFull(true); return true;
      }
      (async () => {
        let isLite = BIMP.tier === "lite", fullErr = null;
        // No blank stage while the full tier downloads: on a slow link (it is 11.5 MB) the lite
        // cells are revealed after a short grace period, and the full tier later grows around
        // them (renderer upload opts.born). A fast link never sees the lite tier.
        let liteShown = false;
        const grace = !isLite && !heroOn && liteIdx ? setTimeout(() => {
          if (BIMP.fullReady || building) return;
          liteShown = true; dataReady = true; maybeReveal();
          setStatus("showing " + lite.n.toLocaleString() + " cells · loading all " + M.n.toLocaleString() + "…");
        }, 900) : 0;
        let full = lite;
        // a full tier that fails for good (after retries) falls back to showing the lite one,
        // with the way to try again, rather than leaving the stage empty
        if (!isLite) {
          try { full = await decode.fetchBase(SRC + M.base, { n: M.n, offsets: M.offsets }, M.gzip, BASE_RETRY("the cells")); } catch (e) {
            if (!liteIdx) throw e;
            fullErr = e; full = lite; isLite = true;
            BIMP.tier = "lite"; BIMP.tierWhy = "the full atlas failed to load"; decode.setGather(liteIdx);
            if (!heroOn && T.slot !== "lite") { R.upload(lite); R.setProgress(0); T.slot = "lite"; }
          }
        }
        if (grace) clearTimeout(grace);
        if (liteShown) await new Promise((r) => { const w = () => (building ? setTimeout(w, 100) : r()); w(); });
        const sel = s.selected;
        s.selected = -1;
        if (heroOn && T.slot !== "full") { T.pendingFull = full; prestage(); }   // staged on the GPU now, shown at the hand-off
        else if (full === lite) { if (T.slot !== "lite") { R.upload(lite); R.setProgress(0); } T.slot = "full"; }   // lite IS the final tier here
        else if (liteShown) {             // grow the rest around the cells already on screen
          decode.setGather(null);
          DATA = full; BIMP.data = full; R.upload(full, { born: liteIdx }); R.setProgress(0); ui.setData(full); T.slot = "full";
          if (sel >= 0) s.selected = liteIdx[sel];
          building = false; R.setProgress(0); maybeReveal();   // the intro gate still applies
        } else { DATA = full; BIMP.data = full; R.upload(full); R.setProgress(0); ui.setData(full); T.slot = "full"; }   // swap while canvas is empty (progress 0)
        BIMP.fullReady = true; setStatus(""); setLoad("ready — " + (isLite ? lite.n : M.n).toLocaleString() + " cells");
        // a gene chosen before now (a ?color=gene link) was told to wait for the full atlas, and
        // nothing asked again: the brain stayed black under "gene expression shortly"
        if (s.mode === 3 && s.gene >= 0) ui.selectGene(s.gene);
        if (sub) sub.textContent = "Each point is one cell, mapped to a cell type and placed back into the reconstructed volume. " +
          (isLite ? "Showing " + lite.n.toLocaleString() + " of " + M.n.toLocaleString() + " cells." : "Showing all " + M.n.toLocaleString() + " cells.");
        if (fullErr) setStatus("couldn't load all " + M.n.toLocaleString() + " cells (" + fullErr.message.replace(/^fetch \S+ → /, "") + ") — showing " + lite.n.toLocaleString());
        if (liteShown) { enableExplore(); resolveFull(true); return; }
        if (location.search.indexOf("revealhalf") >= 0) { R.setProgress(0.5); return; }  // test: pin mid-build
        enableExplore();                   // full tier live → the intro's Explore button unlocks
        dataReady = true; maybeReveal();   // build up ON-SCREEN — only once the intro is dismissed too
        resolveFull(true);
      })().catch(rejectFull);
      return true;
    }
    // INLINE mode (single-file artifact): LITE tier paints, FULL blob streams last.
    const M = window.__BIMP__;
    if (!M) throw new Error("payload missing (window.__BIMP__ / __BIMP_SRC__)");
    BIMP.manifest = M;
    setLoad("decoding preview…");
    const lite = await decode.decodeBlob(M.b64lite, M.lite, 0, M.gzip);
    DATA = lite.data; BIMP.data = DATA;
    R.upload(DATA);
    ui.build({ manifest: M, data: DATA, renderer: R });
    document.getElementById("hud-n").textContent = M.full.n.toLocaleString();
    const sub2 = document.getElementById("title-sub");
    if (sub2) sub2.innerHTML = "Each point is one cell, mapped to a cell type and placed back into the reconstructed volume. This single-file view renders a " + M.full.n.toLocaleString() + "-cell representative sample of the full " + M.nTotal.toLocaleString() + "-cell atlas.";
    activate();
    setLoad("preview ready · streaming full atlas…");
    setStatus("streaming full atlas…");
    const startFull = () => loadFull(M).then(() => resolveFull(true)).catch(rejectFull);
    if (M.b64) startFull(); else window.__BIMP_needFull__ = startFull;
    return true;
  })().catch((err) => { setLoad("Failed: " + err.message); console.error(err); rejectFull(err); throw err; });
  BIMP.ready = liteReady.then(() => fullReady);
  BIMP.ready.catch(() => { /* reported in the stage (showStageError / status line) */ });

  async function loadFull(M) {
    let buf = await IDB.get(M.version);
    let data;
    if (buf) {
      data = decode.views(buf, M.full, (M.genes || []).length);
    } else {
      const r = await decode.decodeBlob(M.b64, M.full, (M.genes || []).length, M.gzip);
      buf = r.buf; data = r.data;
      IDB.put(M.version, buf);
    }
    DATA = data; BIMP.data = data;
    s.selected = -1;                    // lite indices ≠ full indices
    R.upload(data);
    ui.setData(data);
    setStatus("");
    setLoad("ready — " + data.n.toLocaleString() + " cells");
    BIMP.fullReady = true;
  }

  // ---- URL params (deep-links / headless shots) ----
  function applyUrlParams() {
    const P = new URLSearchParams(location.search);
    const M = window.__BIMP__;
    if (ATLAS_LINK) { const el = document.getElementById("intro"); el.classList.add("gone"); el.style.display = "none"; }
    if (P.has("spin")) { s.spin = P.get("spin") !== "0"; document.getElementById("tg-spin").setAttribute("aria-pressed", String(s.spin)); }
    if (P.has("progress") && R.setProgress) R.setProgress(parseFloat(P.get("progress")));  // debug: pin build front 0..1
    if (P.has("theta")) camera.cam.theta = parseFloat(P.get("theta"));
    if (P.has("phi")) camera.cam.phi = parseFloat(P.get("phi"));
    if (P.has("radius")) camera.cam.radius = parseFloat(P.get("radius"));
    if (P.has("caste")) { const btn = document.querySelector(`#caste-seg button[data-v="${P.get("caste")}"]`); if (btn) btn.click(); }
    if (P.has("region")) {
      const v = P.get("region");
      const grp = { mb: "Mushroom body", ol: "Optic lobe" }[v];
      if (grp) s.regionMask = M.regionDefs.filter((d) => d.group === grp).reduce((m, d) => m | (1 << d.bit), 0);
      ui.syncRegionUI();
    }
    const view = P.get("view");
    if (view) {
      const btn = document.querySelector(`#mode-seg button[data-mode="${view}"]`);
      if (btn) btn.click();
      if (view === "gene" && P.has("gene")) ui.selectGene(parseInt(P.get("gene"), 10) || 0);
    }
    if (P.has("label")) { const btn = document.querySelector(`#label-seg button[data-v="${P.get("label")}"]`); if (btn) btn.click(); }
    // legacy select= is a full-tier index; on the lite tier, map it (not drawn there: skip)
    if (P.has("select")) {
      let i = parseInt(P.get("select"), 10) || 0;
      if (DATA && DATA.fullIndex && BIMP.tier === "lite") i = DATA.fullIndex.indexOf(i);
      if (i >= 0) { s.spin = false; ui.inspect(i); }
    }
    // shareable deep-links: ?tour[=stopIndex] starts the guided tour (optionally at a stop);
    // ?gm=<symbol> opens a gene's details modal. Deferred so the full tier is live first.
    if (P.has("tour")) setTimeout(() => {
      if (!BIMP.tour) return;
      dismissIntro();          // otherwise the tour runs behind the intro, on an unrevealed cloud
      BIMP.tour.start();
      const n = parseInt(P.get("tour"), 10) || 0;
      for (let k = 0; k < n; k++) BIMP.tour.next();
    }, 500);
    if (P.has("gm")) setTimeout(() => {
      const i = ui.geneIndexBySymbol(P.get("gm")); if (i >= 0) ui.openGeneModal(i);
    }, 500);
    // headless regression for touch: synthesize a 2-finger pinch + a 1-finger orbit and
    // confirm the camera zoomed and orbited (can't do real gestures via --screenshot).
    if (P.has("touchtest")) setTimeout(() => {
      const cam = camera.cam, r0 = cam.radius, th0 = cam.theta;
      const pe = (t, id, x, y) => canvas.dispatchEvent(new PointerEvent(t, { pointerId: id, clientX: x, clientY: y, bubbles: true, button: 0 }));
      pe("pointerdown", 1, 300, 300); pe("pointerdown", 2, 340, 300);
      pe("pointermove", 1, 250, 300); pe("pointermove", 2, 390, 300);   // spread → zoom in
      pe("pointerup", 1, 250, 300); pe("pointerup", 2, 390, 300);
      const rP = cam.radius;
      pe("pointerdown", 3, 420, 300); pe("pointermove", 3, 500, 300); pe("pointerup", 3, 500, 300);  // drag → orbit
      const thO = cam.theta;
      const ok = Math.abs(rP - r0) > 1e-3 && Math.abs(thO - th0) > 1e-3;
      const el = document.createElement("div"); el.id = "touchtest-result";
      el.textContent = "TOUCH zoom " + r0.toFixed(3) + "→" + rP.toFixed(3) + " | orbit θ " + th0.toFixed(3) + "→" + thO.toFixed(3);
      document.body.appendChild(el);
      document.title = ok ? "TOUCHTEST PASS" : "TOUCHTEST FAIL";
    }, 700);
  }

  // ---- input ----
  let pending = null, lastPickT = 0, pickCost = 0, velTheta = 0, velPhi = 0, lastTipAt = null, lastTipKey = "";
  const hover = BIMP.state.makeHoverHold();
  let gesture = null;   // active pointer gesture (orbit/pan/pinch) — the frame loop reads it too
  function wireInput() {
    // Unified pointer model (mouse + touch): 1 pointer orbits (or pans for mouse
    // right/middle/shift); 2 pointers pinch-zoom + two-finger pan. Tracking every active
    // pointer in a map is what makes multi-touch behave — the old code only followed one.
    const pointers = new Map();   // pointerId → {x, y}
    let moved = false, pinchDist = 0, pinchCx = 0, pinchCy = 0;
    const stopSpin = () => { if (s.spin) { s.spin = false; document.getElementById("tg-spin").setAttribute("aria-pressed", "false"); } };
    const two = () => { const p = [...pointers.values()]; return [p[0], p[1]]; };
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());   // allow right-drag pan
    // Safari: stop the page itself pinch-zooming under our gesture (touch-action:none
    // covers modern browsers; these cover the rest).
    canvas.addEventListener("touchmove", (e) => { if (e.touches.length > 1) e.preventDefault(); }, { passive: false });
    canvas.addEventListener("gesturestart", (e) => e.preventDefault());

    const flat = () => !!(BIMP.histology && BIMP.histology.is2D && BIMP.histology.is2D());
    canvas.addEventListener("pointerdown", (e) => {
      try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic/again */ }
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      camera.cancelFly && camera.cancelFly();   // user takes control from any tour flight
      stopSpin(); velTheta = 0; velPhi = 0;
      if (pointers.size >= 2) {                  // second finger down → pinch/pan
        const [a, b] = two();
        pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
        pinchCx = (a.x + b.x) / 2; pinchCy = (a.y + b.y) / 2;
        gesture = "pinch"; moved = true; ui.hideTip();
      } else {                                   // first pointer → orbit (mouse modifiers → pan)
        moved = false;
        // a flat section has nothing to orbit (histology turns orbit off in 2D): there a plain drag
        // slides the section, as on any map; before, it did nothing at all
        gesture = (e.button === 2 || e.button === 1 || e.shiftKey || flat()) ? "pan" : "orbit";
      }
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const p = pointers.get(e.pointerId);
      if (!p) { pending = { x: e.clientX - rect.left, y: e.clientY - rect.top }; return; }  // mouse hover
      const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
      if (gesture === "pinch" && pointers.size >= 2) {
        const [a, b] = two();
        const d = Math.hypot(a.x - b.x, a.y - b.y), cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
        if (pinchDist) camera.zoom((pinchDist - d) * 0.02);   // spread fingers → zoom in
        camera.pan(cx - pinchCx, cy - pinchCy);               // slide both → pan
        pinchDist = d; pinchCx = cx; pinchCy = cy; ui.hideTip();
      } else if (gesture === "pan") {
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        camera.pan(dx, dy); ui.hideTip();
      } else if (gesture === "orbit") {
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        camera.orbit(dx, dy); velTheta = dx * 0.006; velPhi = -dy * 0.006; ui.hideTip();   // inertia
      }
    });
    function endPointer(e) {
      if (!pointers.has(e.pointerId)) return;
      pointers.delete(e.pointerId);
      if (pointers.size === 0) {
        if ((gesture === "orbit" || (gesture === "pan" && flat())) && !moved) {     // a tap/click that didn't drag → pick the cell
          const rect = canvas.getBoundingClientRect();
          const idx = R.pick(e.clientX - rect.left, e.clientY - rect.top);
          ui.inspect(idx); if (idx >= 0) stopSpin();
        }
        gesture = null; canvas.classList.remove("dragging");
      } else if (pointers.size === 1) {          // lifted one of two fingers → resume orbit
        gesture = flat() ? "pan" : "orbit"; moved = true;         // (don't treat the lift as a tap)
      }
    }
    canvas.addEventListener("pointerup", endPointer);
    canvas.addEventListener("pointercancel", endPointer);
    // the pointer really is gone — drop the held hover at once rather than letting it expire
    canvas.addEventListener("pointerleave", () => {
      if (!pointers.size) { ui.hideTip(); pending = null; hover.clear(); s.hover = -1; lastTipAt = null; }
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      // Mac trackpad: pinch fires wheel with ctrlKey → ZOOM; a plain two-finger swipe
      // (has a horizontal component, or fractional pixel deltas) → ORBIT. A classic
      // mouse wheel (line-mode, or pure integer vertical with no deltaX) → ZOOM.
      const isMouseWheel = e.deltaMode !== 0 || (e.deltaX === 0 && Math.abs(e.deltaY) >= 40 && Number.isInteger(e.deltaY));
      if (e.ctrlKey || isMouseWheel) {
        camera.zoom(Math.sign(e.deltaY || 1));
      } else {
        if (s.spin) { s.spin = false; document.getElementById("tg-spin").setAttribute("aria-pressed", "false"); }
        if (flat()) camera.pan(-e.deltaX, -e.deltaY);   // in 2D it slides the section
        else camera.orbit(-e.deltaX, -e.deltaY);   // two-finger swipe orbits
      }
    }, { passive: false });
    // reset view: button + "R" / Home key
    const resetView = () => { camera.reset(); ui.hideTip(); };
    const rb = document.getElementById("btn-reset");
    if (rb) rb.addEventListener("click", resetView);
    window.addEventListener("keydown", (e) => {
      if ((e.key === "r" || e.key === "R" || e.key === "Home") && !/INPUT|TEXTAREA/.test((e.target || {}).tagName || "")) resetView();
    });
    // Jump into the tour with the brain fully shown — the camera flights shouldn't fight the
    // 0→1 assembly build-up, so finish the reveal instantly instead of playing it.
    // Taking the tour goes through the SAME door as skipping it: dismissIntro() is what
    // starts the build-up (via maybeReveal — the cloud sits at progress 0 until then).
    // Open-coding the intro teardown here once left progress at 0 and the canvas empty.
    // The build then plays on under the tour: stop 1 barely moves the camera, so you
    // watch the brain assemble while it's introduced.
    const startTour = () => {
      dismissIntro();
      if (BIMP.tour) BIMP.tour.start();
    };
    const enter = document.getElementById("enter");                 // big primary → guided tour
    if (enter) enter.addEventListener("click", () => (heroOn ? toAtlas({ tour: true }) : startTour()));
    const enterTour = document.getElementById("enter-tour");        // "skip to the atlas →"
    if (enterTour) enterTour.addEventListener("click", () => (heroOn ? toAtlas() : dismissIntro()));
    if (BIMP.tour) BIMP.tour.init();
    // mobile bottom-sheet toggle
    const sheetBtn = document.getElementById("sheet-toggle"), controls = document.getElementById("controls");
    const setSheet = (open) => {
      controls.classList.toggle("sheet-open", open);
      document.body.classList.toggle("sheet-open", open);   // CSS hides the pill + the open button
      sheetBtn.textContent = "Controls";                    // it only ever means "open" now
    };
    if (sheetBtn && controls) sheetBtn.addEventListener("click", () => setSheet(!controls.classList.contains("sheet-open")));
    const sheetClose = document.getElementById("sheet-close");
    if (sheetClose) sheetClose.addEventListener("click", () => setSheet(false));
  }

  function frame(now) {
    // asleep under the home page: no GL work until it is dismissed (or the hand-off begins)
    if (!introGone && !heroOn) { frame._last = now; requestAnimationFrame(frame); return; }
    if (frame._queued === now) return;                            // one loop, however many starters
    frame._queued = now;
    // waking from sleep (render on demand), the gap is idle time, not a slow frame: count one
    // ordinary frame, so a flight or the spin starting from rest doesn't open with a 50 ms jump
    const dt = frame._fromSleep ? 1 / 60 : Math.min((now - (frame._last || now)) / 1000, 0.05);
    frame._fromSleep = false;
    if (heroOn && (T.tp < 1 || T.dir)) {
      // home page or mid-flight: the hero look, no picking, no spin
      if (T.slot === "none") { requestAnimationFrame(frame); return; }
      const home = T.tp === 0 && !T.dir;
      // off screen or hidden: no animation, but still repaint the page a few times a second — the
      // canvas IS the page's ground, and a resize clears it to black
      if (home && (!T.visible || document.hidden)) {
        frame._last = now;
        if (!document.hidden) { applyLook(now); R.render(); }
        setTimeout(() => requestAnimationFrame(frame), 250); return;
      }
      if (home && heroSmall && now - (frame._last || 0) < 32) { requestAnimationFrame(frame); return; }   // ~30 fps on small devices
      T.rawDt = (now - (frame._last || now)) / 1000;
      frame._last = now;
      step(dt); applyLook(now); R.render();
      mirrorHero();
      if (T.tp === 0) triad();
      // the page goes transparent onto the canvas only once the canvas has drawn its paper: a
      // fresh WebGL canvas is black, and that is what flashed before the home page
      if (!T.shown) { T.shown = true; requestAnimationFrame(() => { root.classList.add("hero-gl"); dispatchEvent(new Event("bimp:hero")); }); }
      requestAnimationFrame(frame); return;
    }
    frame._last = now;
    // an atlas link: the controls come in together once the first cells are on screen
    if (ATLAS_LINK && root.classList.contains("atlas-boot") && R.N > 0 && R.getProgress() > 0.12) {
      requestAnimationFrame(() => root.classList.remove("atlas-boot"));
    }
    // Advance the reveal-front clock 0→1 over BUILD_SECS with an ease-out, so the front
    // sweeps quickly through the dense core and decelerates as it reaches the sparse shell
    // — it "settles" into place rather than stopping abruptly. The spatial ordering of
    // aBirth (in the renderer) is what makes this read as an assembling brain; here we
    // just move the clock.
    if (building && R.setProgress) {
      buildProgress = Math.min(1, buildProgress + dt / BUILD_SECS);
      const eased = 1 - (1 - buildProgress) * (1 - buildProgress);   // ease-out quad
      R.setProgress(eased);
      if (buildProgress >= 1) building = false;
    }
    // scripted camera flight (guided tour) takes precedence over spin/momentum
    const flying = camera.tick ? camera.tick(dt) : false;
    if (!flying) {
      if (s.spin) camera.spin(dt);
      // momentum: keep spinning briefly after a flick-release, then decay
      else if (Math.abs(velTheta) > 1e-4 || Math.abs(velPhi) > 1e-4) {
        camera.orbit(velTheta / 0.006, velPhi / -0.006);
        velTheta *= 0.92; velPhi *= 0.92;
      }
    }
    // Hover picking costs a WHOLE extra scene pass (1.3M points into the pick FBO), so
    // don't spend it while the view is already moving: during a scripted flight or a drag
    // the tooltip is unreadable anyway, and that pass is exactly the frame budget the
    // motion needs. Idle hover keeps working, just at a calmer cadence.
    //
    // A morph counts as moving. Those are the most expensive frames the viewer ever draws —
    // the shader evaluates BOTH view states per point (see shaders.js) for 0.55 s — and
    // landing a second full-scene pass in the middle of one is what made switching colour
    // mode or splitting the castes hitch. Unlike a drag, nothing here told the picker the
    // view was busy, because the camera never moved.
    const morphing = R.isMorphing ? R.isMorphing() : false;
    // The cadence follows the pick's own measured cost: the pass blocks the main thread
    // (readPixels waits for the GPU), so a machine where it takes 300 ms would otherwise be
    // frozen solid while the pointer moves. At least 90 ms apart, and never more than a
    // fifth of the time; on a fast GPU (a few ms per pick) that is the old 90 ms exactly.
    if (pending && now - lastPickT > Math.max(90, pickCost * 4) && !flying && !gesture && !morphing) {
      lastPickT = now;
      const t0 = performance.now();
      const idx = R.pick(pending.x, pending.y, 4);
      pickCost = pickCost * 0.5 + (performance.now() - t0) * 0.5;
      // A single missed pick is not "the pointer left the cloud" — hold the last hit for a
      // moment so the highlight and the tooltip stop strobing. See makeHoverHold.
      lastTipAt = { x: pending.x, y: pending.y };
      s.hover = hover.report(idx);
      pending = null;
    } else if (pending && (flying || gesture)) {
      pending = null; hover.clear(); s.hover = -1;
    }
    // Only touch the tooltip when something about it actually changed — this runs every
    // frame now that the hover can expire on its own, and showTip writes to the DOM.
    const tipKey = s.hover >= 0 && lastTipAt ? s.hover + ":" + lastTipAt.x + ":" + lastTipAt.y : "";
    if (tipKey !== lastTipKey) {
      lastTipKey = tipKey;
      if (tipKey) ui.showTip(s.hover, lastTipAt.x, lastTipAt.y); else ui.hideTip();
    }
    // Render on demand (renderer needsFrame): nothing is drawn while nothing changes.
    // The idle auto-orbit is the one motion that is slow enough to draw at half rate
    // (0.12 rad/s is under a pixel per frame at 30 fps), so when it is the ONLY thing
    // moving, every other tick is skipped: half the GPU for the default idle view.
    const spinOnly = s.spin && !flying && !gesture && !building && !morphing && !pending;
    let drew = false, want = true;
    // a fault in the gate must not stop the loop for good: draw, and say so once
    try { want = R.needsFrame(now); } catch (e) { if (!frame._gateErr) { frame._gateErr = true; console.error("needsFrame:", e); } }
    if (!(spinOnly && now - (frame._drawn || 0) < 30) && want) {
      R.render(); drew = true;
      // adaptive resolution: feed the interval between back-to-back rendered frames
      // (not during the build-up, a one-off burst that would soften the atlas for good)
      if (frame._drewLast && R.adaptDpr && !building) R.adaptDpr(now - frame._drawn);
      frame._drawn = now; fpsCount++;
    }
    frame._drewLast = drew || (spinOnly && now - (frame._drawn || 0) < 30);
    if (ui.tickUmap) ui.tickUmap(dt);   // live UMAP marker eases toward the hovered cell
    fpsAcc += dt;
    if (fpsAcc >= 0.5) { document.getElementById("hud-fps").textContent = fpsCount ? Math.round(fpsCount / fpsAcc) : "idle"; fpsAcc = 0; fpsCount = 0; }
    // Asleep: after a second with nothing drawn, stop asking for animation frames at all and
    // look in four times a second (a safety net for changes that arrive with no event).
    // Input, a renderer invalidate() and visibility changes wake it at once (wakeLoop).
    if (drew) frame._idle = 0; else frame._idle = (frame._idle || 0) + 1;
    if (frame._idle > 60) { frame._asleep = true; frame._fromSleep = true; setTimeout(() => { if (frame._asleep) requestAnimationFrame(frame); }, 250); return; }
    frame._asleep = false;
    requestAnimationFrame(frame);
  }
  // Wake the frame loop from sleep (see the end of frame). Cheap when it is already awake.
  // Also a watchdog: a loop that has not ticked for a second while it should be awake (an
  // exception escaped a frame) is started again by the next input or invalidate.
  function wakeLoop() {
    const stalled = !frame._asleep && frame._last && performance.now() - frame._last > 1000 && !document.hidden;
    if (frame._asleep || stalled) { frame._asleep = false; frame._idle = 0; requestAnimationFrame(frame); }
  }
  R.setOnWake && R.setOnWake(wakeLoop);
  setInterval(wakeLoop, 1000);
  // a scripted flight can start with no input event (tour steps on a timer, histology's Face)
  if (camera.flyTo) { const fly0 = camera.flyTo; camera.flyTo = function () { const r = fly0.apply(this, arguments); wakeLoop(); return r; }; }
  ["pointermove", "pointerover", "pointerout", "wheel", "focusin"]
    .forEach((t) => document.addEventListener(t, wakeLoop, { capture: true, passive: true }));
  // A discrete action may change state the frame signature does not list (a new control,
  // a new uniform): those force one frame, not just a look, so nothing new can go stale.
  // Deferred a tick so the handlers that run after this capture listener have applied it.
  const poke = () => setTimeout(() => R.invalidate ? R.invalidate() : wakeLoop(), 0);
  ["pointerdown", "pointerup", "click", "input", "change", "keydown"]
    .forEach((t) => document.addEventListener(t, poke, { capture: true, passive: true }));
  window.addEventListener("resize", wakeLoop);
  document.addEventListener("visibilitychange", wakeLoop);

  // ---- headless self-test ----
  BIMP.selfTest = async function () {
    await BIMP.ready;
    const gl = R.gl, out = { ok: true, checks: [] };
    const rec = (name, pass, info) => { out.checks.push({ name, pass, info: info || "" }); if (!pass) out.ok = false; };
    const M = BIMP.manifest;
    const wantN = M.full ? M.full.n : M.n;
    rec("full-loaded", DATA && DATA.n === wantN, "n=" + (DATA && DATA.n));
    rec("points-uploaded", R.N === wantN, "N=" + R.N);
    const w = gl.canvas.width, h = gl.canvas.height;
    // Each assertion samples ONE frame, so any pending view dissolve is applied instantly
    // first — otherwise we'd photograph the transition rather than the result. (The
    // dissolve itself is covered by test/morph.mjs, against the real shader.)
    const settle = () => { if (R.settleMorph) R.settleMorph(); };
    const litCount = () => { settle(); R.render(); const p = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, p); let c = 0; for (let i = 0; i < p.length; i += 4) if (p[i] > 20 || p[i + 1] > 24 || p[i + 2] > 30) c++; return c; };
    // Finish the intro reveal before measuring anything. The reveal is an animation driven
    // by the frame loop, and BIMP.ready only promises the DATA has arrived — in the inline
    // artifact the two coincide (the buffer is embedded, so progress is already 1), but the
    // fetch build streams in tiers and is still at progress 0 here. Every pixel assertion
    // below would then photograph an empty canvas and read 0 — not because the atlas is
    // broken, but because it has not been shown yet.
    if (R.setProgress) R.setProgress(1);
    const base = litCount();
    rec("pixels-lit", base > w * h * 0.01, "lit=" + base + "/" + (w * h));
    const cx = Math.floor(w / (2 * Math.min(window.devicePixelRatio || 1, 2)));
    const cy = Math.floor(h / (2 * Math.min(window.devicePixelRatio || 1, 2)));
    rec("pick-center", R.pick(cx, cy) >= 0, "idx=" + R.pick(cx, cy));
    if ((M.genes || []).length) {
      const p0 = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, p0);
      s.mode = 3; ui.selectGene(0); settle(); R.render();
      const p1 = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, p1);
      let diff = 0; for (let i = 0; i < p0.length; i += 4) if (Math.abs(p0[i] - p1[i]) > 8) diff++;
      rec("gene-recolor", diff > 100, "changed=" + diff); s.mode = 0; settle();
    }
    s.visible.fill(0); const off = litCount(); s.visible.fill(1); const on = litCount();
    rec("legend-toggle", off < base * 0.2 && on > off * 4, "off=" + off + " on=" + on);
    if ((M.regionDefs || []).length) {
      s.regionMask = 1; const one = litCount(); s.regionMask = 0;
      rec("region-mask", one > 0 && one < base, "masked=" + one + " all=" + base);
    }
    // Spatial build-up: an early front (progress 0.3, core only) must light strictly
    // fewer pixels than the finished brain. Proves cells reveal incrementally by position.
    if (R.setProgress && R.getProgress) {
      const saved = R.getProgress();
      R.setProgress(0.3); const few = litCount();
      R.setProgress(1); const all2 = litCount();
      R.setProgress(saved);
      rec("stream-buildup", few > 0 && few < all2 * 0.7, "30%=" + few + " 100%=" + all2);
    }
    return out;
  };

  if (typeof location !== "undefined" && location.search.indexOf("selftest") >= 0) {
    BIMP.ready.then(() => BIMP.selfTest()).then((r) => {
      const el = document.createElement("div"); el.id = "selftest-result"; el.style.display = "none";
      el.textContent = JSON.stringify(r); document.body.appendChild(el);
      document.title = "SELFTEST " + (r.ok ? "PASS" : "FAIL");
    }).catch((e) => {
      const el = document.createElement("div"); el.id = "selftest-result";
      el.textContent = JSON.stringify({ ok: false, error: String(e && e.message || e) }); document.body.appendChild(el);
      document.title = "SELFTEST FAIL";
    });
  }
})();
