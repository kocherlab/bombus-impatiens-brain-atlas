/* Home page scripts. The hero brain itself is NOT drawn here: it is the atlas's own WebGL
 * renderer (boot.js, hero mode), drawing ~20k real queen cells from the hero tier into the
 * #hero-gl box, so entering the atlas is one continuous camera move on the same cells. */
/* Load on intent, McMaster-Carr style. The home page loads only itself; the atlas data waits
   until the reader shows they are heading there: hover, focus or touch on a way in, or a
   scroll toward the atlas section. While they read, an idle prefetch warms the HTTP cache with
   the big file without decoding it (no memory cost); intent then decodes from cache. */
(function () {
  if (/[?&](nointro|atlas)(=|&|$)/.test(location.search)) return;   // a link straight to the viewer: load now
  let go;
  window.__BIMP_INTENT__ = new Promise((r) => { go = r; });
  const fire = () => { if (go) { go(); go = null; } };
  window.__BIMP_GO__ = fire;
  const ways = () => document.querySelectorAll('#enter, #enter-tour, #intro .ways .go-link, #masthead .nav a[href="#s-atlas"]');
  const arm = () => {
    for (const el of ways()) for (const ev of ["pointerenter", "focusin", "touchstart"]) el.addEventListener(ev, fire, { once: true, passive: true });
    // the page scrolls inside #intro, not the window, so that is the observer's root (with the
    // viewport as root a rootMargin cannot reach past #intro's clip)
    const atlas = document.getElementById("s-atlas");
    if (atlas && "IntersectionObserver" in window)
      new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { fire(); o.disconnect(); } },
        { root: document.getElementById("intro"), rootMargin: "400px" }).observe(atlas);
    // prefetch into the HTTP cache, only on a connection that can afford it, and only once the
    // first screen has finished arriving (so it never competes with the page itself): the small
    // files when the page is idle, the 11 MB atlas after the reader has stayed a few seconds
    const c = navigator.connection || {};
    if (!c.saveData && !/2g|3g/.test(c.effectiveType || "") && innerWidth >= 900) {   // not on phones or slow links
      const idle = window.requestIdleCallback || ((f) => setTimeout(f, 1000));
      const pre = (f) => { const l = document.createElement("link"); l.rel = "prefetch"; l.href = (window.__BIMP_SRC__ || "./data/") + f; l.as = "fetch"; document.head.appendChild(l); };
      const later = () => idle(() => { pre("manifest.json"); pre("base-lite.bin"); setTimeout(() => idle(() => go && pre("base.bin"), { timeout: 4000 }), 4000); }, { timeout: 4000 });
      if (document.readyState === "complete") later(); else addEventListener("load", later, { once: true });
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", arm); else arm();
})();

/* Load in order, so the page assembles rather than pops: paper and header, then the text rises
   line by line as soon as the webfonts are in (at most 1.5 s), then the brain fades up once it
   has drawn its first frame. The atlas code (renderer, WebGL setup: the heaviest work on the
   page) only starts after the text's entrance has been painted, so it can never hold the
   first screen back. boot.js fades the brain in from READY_AT, which is set here when both the
   text has started and the brain has drawn ("bimp:hero"), so it never pops in at full strength.
   Each drawing further down fades in once decoded. */
(function () {
  const root = document.documentElement;
  let textAt = 0, heroAt = 0;
  const brain = () => { if (textAt && heroAt && !window.__BIMP_READY_AT__) window.__BIMP_READY_AT__ = performance.now(); };
  const ready = () => {
    if (root.classList.contains("ready")) return;
    root.classList.add("ready"); textAt = performance.now(); brain();
    dispatchEvent(new Event("bimp:ready"));
  };
  addEventListener("bimp:hero", () => { heroAt = performance.now(); brain(); }, { once: true });
  // every face the first screen uses, loaded explicitly (document.fonts.ready alone can resolve
  // before a face that is only needed later has been requested); all are preloaded in <head>
  const faces = ['400 1em "Alegreya Sans"', '500 1em "Alegreya Sans"', '800 1em "Alegreya Sans"',
    '500 1em "Alegreya Sans SC"', 'italic 400 1em "Alegreya"'];
  const fonts = document.fonts && document.fonts.load
    ? Promise.all(faces.map((f) => document.fonts.load(f).catch(() => null))) : Promise.resolve();
  Promise.race([fonts, new Promise((r) => setTimeout(r, 1500))]).then(() => {
    root.classList.add("fonts"); requestAnimationFrame(ready); setTimeout(ready, 250);   // the timer: a hidden tab has no frames
  });
  const fade = () => document.querySelectorAll("#intro img").forEach((im) => {
    if (im.complete && im.naturalWidth) return;
    im.classList.add("ld");
    const done = () => im.classList.remove("ld");
    im.addEventListener("load", done, { once: true }); im.addEventListener("error", done, { once: true });
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fade); else fade();
})();

/* The atlas code ships as one cached file (build-pages.mjs writes assets/atlas.<hash>.js and
   names it in __BIMP_ATLAS__; it is preloaded in <head>, so it is already here when it runs).
   It runs one frame after the text's entrance starts, or at once for a link straight into the
   viewer.
   A click on a way in is held until the viewer can take it, then replayed: boot.js wires those
   buttons only once the small atlas tier is in (it is fetched on intent, i.e. on the same hover
   or touch that precedes the click), so without this a quick first tap did nothing at all. */
(function () {
  const url = window.__BIMP_ATLAS__;
  const root = document.documentElement;
  const WAYS = '#enter, #enter-tour, #intro .ways .go-link, #masthead .nav .open-atlas';
  const wired = () => !!(window.BIMP && BIMP.data);             // boot.js sets it, then wires the buttons
  const held = [];
  const hold = (e) => {
    if (wired()) { document.removeEventListener("click", hold, true); return; }
    const t = e.target.closest && e.target.closest(WAYS);
    if (!t) return;
    e.preventDefault(); e.stopImmediatePropagation();
    if (window.__BIMP_GO__) window.__BIMP_GO__();               // a click is intent too (keyboard, or no hover)
    if (held.indexOf(t) < 0) { held.push(t); t.setAttribute("aria-busy", "true"); }
    if (held.length === 1) (function wait() {
      if (!wired()) return void setTimeout(wait, 50);
      document.removeEventListener("click", hold, true);
      held.splice(0).forEach((h) => { h.removeAttribute("aria-busy"); h.click(); });
    })();
  };
  if (!url) return;                                              // inline build: the scripts follow
  if (!/[?&](nointro|atlas)(=|&|$)/.test(location.search)) document.addEventListener("click", hold, true);
  const load = () => {
    const s = document.createElement("script");
    s.src = url;
    s.onerror = () => document.removeEventListener("click", hold, true);
    document.body.appendChild(s);
  };
  let started = false;
  const start = () => { if (!started) { started = true; load(); } };
  // after the next paint; the timer is the fallback for a hidden tab, where frames never come
  const soon = () => { requestAnimationFrame(() => setTimeout(start, 0)); setTimeout(start, 250); };
  if (root.dataset.view === "atlas") start();                    // a link straight into the viewer: at once
  else if (root.classList.contains("ready")) soon();
  else addEventListener("bimp:ready", soon, { once: true });
})();

/* Keyboard focus across the hand-off. The control you pressed belongs to the page you are
   leaving, which is then hidden, so focus would fall to <body> and the next Tab start from the
   top. Arriving in the viewer it moves to the first control there; back home, to "Open the
   atlas". Only for keyboard use: a mouse or touch user's focus is left alone. */
(function () {
  const root = document.documentElement;
  let keys = false, last = root.dataset.view;
  addEventListener("keydown", (e) => { if (!e.metaKey && !e.ctrlKey) keys = true; }, true);
  addEventListener("pointerdown", () => { keys = false; }, true);
  const lost = () => {
    const a = document.activeElement;
    return !a || a === document.body || !a.offsetParent || !!a.closest("#intro.gone, #intro.leaving") || getComputedStyle(a).visibility === "hidden";
  };
  new MutationObserver(() => {
    const v = root.dataset.view;
    if (v === last) return; last = v;
    if (!keys || !lost()) return;
    const shown = (b) => b.offsetParent && getComputedStyle(b).visibility !== "hidden";
    const target = () => v === "atlas" ? [...document.querySelectorAll("#controls button:not([disabled])")].find(shown)
      : v === "home" ? document.getElementById("enter-tour") : null;
    // the panels can still be fading in (hidden, so not focusable) for a moment: try for a second
    let n = 0;
    (function put() {
      if (root.dataset.view !== v || !lost()) return;
      const to = target();
      if (to) to.focus({ preventScroll: true });
      if ((!to || document.activeElement !== to) && ++n < 10) setTimeout(put, 100);
    })();
  }).observe(root, { attributes: true, attributeFilter: ["data-view"] });
})();

/* Ian Traniello's labelled mushroom-body close-up (93 KB of paths) is fetched as the Anatomy
   section nears, then inlined rather than shown as an <img>, so its labels use the page's webfonts. */
(function () {
  const z = document.getElementById("z3"), url = z && z.dataset.svg;
  if (!url) return;
  let done = false;
  const inject = () => {
    if (done) return; done = true;
    fetch(url).then((r) => (r.ok ? r.text() : Promise.reject(r.status)))
      .then((t) => z.insertAdjacentHTML("afterbegin", t)).catch(() => { done = false; });
  };
  const sec = z.closest("section") || z;
  if ("IntersectionObserver" in window)
    new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { o.disconnect(); inject(); } },
      { root: document.getElementById("intro"), rootMargin: "1200px 0px" }).observe(sec);
  else inject();
})();

/* zoom leader lines on the home page: from the corners of each frame's dashed box to the
   corners of the next frame, and the to-scale wedge; redrawn whenever the layout changes */
(function () {
  function draw() {
    const z = document.querySelector("#intro .zoom"), svg = document.getElementById("zl");
    if (z && svg) {
      const o = z.getBoundingClientRect(); let out = "";
      [["b1", "z2"], ["b2", "z3"]].forEach(([b, f]) => {
        const a = document.getElementById(b).getBoundingClientRect(), r = document.getElementById(f).getBoundingClientRect();
        out += `<line x1="${a.right - o.left}" y1="${a.top - o.top}" x2="${r.left - o.left}" y2="${r.top - o.top}"/>`;
        out += `<line x1="${a.right - o.left}" y1="${a.bottom - o.top}" x2="${r.left - o.left}" y2="${r.bottom - o.top}"/>`;
      });
      svg.innerHTML = out;
    }
    const f = document.getElementById("tosc"), w = document.getElementById("tw"),
          big = document.getElementById("tb"), tiny = document.getElementById("tt");
    if (f && w && big && tiny && f.offsetWidth) {
      const fo = f.getBoundingClientRect(), r = big.getBoundingClientRect(), t = tiny.getBoundingClientRect();
      // each edge of the wedge is a tangent from the actual-size brain to the drawing's outline,
      // so it just grazes the silhouette and the rest of the wedge hides behind it
      const tx = t.left + t.width / 2 - fo.left, ty = t.top + t.height / 2 - fo.top;
      if (!draw.pts) {
        const pa = document.querySelector("#brain-sil path"), n = 400, L = pa.getTotalLength();
        draw.pts = []; for (let k = 0; k < n; k++) { const q = pa.getPointAtLength(L * k / n); draw.pts.push([q.x, q.y]); }
      }
      const kx = r.width / 215.7, ky = r.height / 111.4;
      let up = null, lo = null, au = Infinity, al = -Infinity;
      for (const [px, py] of draw.pts) {
        const sx = r.left - fo.left + (px - 71.7) * kx, sy = r.top - fo.top + (py - 72) * ky;
        const ang = Math.atan2(sy - ty, tx - sx);
        if (ang < au) { au = ang; up = [sx, sy]; }
        if (ang > al) { al = ang; lo = [sx, sy]; }
      }
      const cx = r.left - fo.left + r.width * 0.5, cy = r.top - fo.top + r.height * 0.5;
      w.innerHTML = `<polygon points="${tx},${ty} ${up} ${cx},${cy} ${lo}"/>`;
    }
  }
  // only while the home page is what's showing: measuring these boxes forces a layout of the
  // whole home page, which on an atlas link cost ~340 ms on a throttled phone for nothing
  const onHome = () => document.documentElement.dataset.view === "home";
  const drawIfHome = () => { if (onHome()) draw(); };
  if (document.readyState === "complete") requestAnimationFrame(drawIfHome); else addEventListener("load", drawIfHome);
  addEventListener("resize", drawIfHome);
  const i = document.getElementById("intro"); if (i) i.addEventListener("scroll", drawIfHome, { passive: true });
  // back on the home page: redraw once the page is laid out again
  new MutationObserver(() => { if (onHome()) requestAnimationFrame(draw); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-view"] });
})();
