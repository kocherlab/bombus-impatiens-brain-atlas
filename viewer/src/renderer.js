/* WebGL2 point-cloud renderer: program, buffers, per-frame draw, and a parallel
   pick pass. Depends on BIMP.shaders, BIMP.mat4, BIMP.camera, BIMP.palette, BIMP.state. */
window.BIMP = window.BIMP || {};
BIMP.renderer = (function () {
  const { mat4, camera, palette, shaders, state } = BIMP;
  const BG = [7 / 255, 9 / 255, 14 / 255];
  // ---- the home page shares this renderer ----------------------------------------
  // On the home page the same program draws ~20k real queen cells (the hero tier) in ink on
  // paper, into the box the page leaves for it (rect, CSS px). Entering the atlas eases every
  // field here toward the viewer's look on one clock (boot.js), so nothing is ever swapped
  // visibly: ink → cell colours, paper → dark, hero box → full canvas.
  //   ink 0..1, inkCol rgb, bg rgb (also the fog colour), sizeMul, wob (NDC wander),
  //   ptr [ndcX, ndcY, radius, strength] (cells bubble away from the pointer), lockLod.
  const look = { birthHash: 0, scaffold: 0, alpha: 1, spark: 1, fogK: 1, ink: 0, inkCol: [18 / 255, 17 / 255, 14 / 255], bg: BG.slice(), sizeMul: 1, wob: 0,
    ptr: [0, 0, 0.18, 0], lockLod: false, rect: null, scissor: false };
  let gl, prog, U, vao, N = 0, geneBuf = null, genePrevBuf = null, prevIsCur = true, geneF32 = null, ctBuf = null, ownedBufs = [];
  let ctSub = null, ctCanon = null;
  // µm per world unit: export_data.py divides x, y and z (µm) by one scale; the export
  // writes it as manifest.scaleUm (older exports: the histology frame's scale_um, 1949.04)
  const scaleUm = () => (BIMP.manifest && BIMP.manifest.scaleUm) || SCALE_UM_FALLBACK;
  const SCALE_UM_FALLBACK = 1949.0432870581344;
  // a cell's diameter: the median segment of a labelled cell is 34.0 µm² (Supplementary
  // Methods, "What the Other category contains") → 2·√(34/π) ≈ 6.6 µm
  const CELL_UM = 6.6;
  let viewH = 1;
  let DPR = 1, projM = null, viewShift = 0, viewShiftCur = 0, viewShiftT = 0;
  // "Both": queen at −mag·dir, worker at +mag·dir in world x/y. Desktop is side by side; ui.js
  // stacks them (queen above) on a portrait phone. histology.js reads getSplit() to follow.
  let split = { mag: 1.15, dir: [1, 0] };
  // Build-up clock: progress in [0,1] is the global reveal time. Each cell's aBirth (in
  // [0,1], spatially ordered) fades it in when progress passes it. uProgress is driven to
  // 1+FADE at progress==1 so even the last (shell) cells fully resolve. FADE is the width
  // of the moving front — larger = softer, more cells mid-fade at once.
  let progress = 1;
  // ---- view morphing ----------------------------------------------------------
  // Changing the view is a MOVE, not a cut: the same cells stay on screen, so the
  // renderer keeps the state it is coming FROM and hands the shader both, plus a 0→1
  // clock. Colors dissolve, filtered cells fade, the caste split slides. Detected
  // automatically from state, so manual clicks morph exactly like the guided tour.
  const MORPH_SECS = 0.55;
  // honour the OS "reduce motion" setting: changes apply instantly, no dissolve
  const REDUCED = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  let morph = 1;                       // 1 = settled on the current state
  // The clock stays LINEAR — settling, isMorphing and the tests all read it — but what the
  // shader gets is eased. uMorph drives the caste split's position mix, so a linear clock
  // slid every point at constant velocity and stopped it dead, while the camera pulled back
  // through easeInOutCubic over the same 550 ms. Two motions with different velocity
  // profiles, one of them starting and stopping instantly: that is the jerk. Same curve as
  // camera.js flyTo, so the brains and the camera now accelerate and arrive together.
  // Monotone with e(0)=0 and e(1)=1, so `uMorph < 1.0` still means exactly "morphing".
  const easeMorph = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  let curSig = null, curState = null, prevState = null, lastT = 0;
  let heroBirth = null;   // the home sample's birth times (upload opts.half), reused for the same cells in the full tier
  // ---- motion LOD --------------------------------------------------------------
  // The heavy cost at 1.3M points is fill, and it only hurts while things are moving.
  // So thin the cloud during camera motion and restore it the moment things settle:
  // drop instantly (protect the frame we are about to miss), ease back over ~0.35 s.
  // Region bits of the three Kenyon-cell compartments, read from the manifest (never
  // hardcoded — bit order comes from export_data.py). The shader shades them apart.
  let kcBits = null, regUploaded = false;
  function kcBitsFor() {
    if (kcBits) return kcBits;
    const defs = (BIMP.manifest || window.__BIMP_HERO__ || {}).regionDefs;   // the hero carries them on the home page
    if (!defs) return [0, 0, 0];
    const bit = (k) => { const d = defs.find((x) => x.key === k); return d ? (1 << d.bit) : 0; };
    return (kcBits = [bit("in_kc_ic"), bit("in_kc_nc"), bit("in_kc_oc")]);
  }
  // The six hand-drawn masks and their figure colours, for the "Region" colour mode.
  // Built from the manifest so bit order always matches export_data.py.
  function uploadRegions() {
    if (regUploaded) return;
    const defs = (BIMP.manifest || window.__BIMP_HERO__ || {}).regionDefs;   // the hero carries them on the home page
    if (!defs) return;
    const bits = new Int32Array(6), cols = new Float32Array(18);
    defs.slice(0, 6).forEach((d, i) => {
      bits[i] = 1 << d.bit;
      const c = palette.hexToRgb(palette.KC_REGION[d.key] || "#8A93A6");
      cols[i * 3] = c[0]; cols[i * 3 + 1] = c[1]; cols[i * 3 + 2] = c[2];
    });
    gl.uniform1iv(U.regBits, bits);
    gl.uniform3fv(U.regCol, cols);
    gl.uniform3fv(U.regNone, new Float32Array(palette.hexToRgb(palette.UNREGIONED)));
    const lk = (BIMP.manifest || window.__BIMP_HERO__ || {}).ltkc;
    if (lk) gl.uniform3fv(U.ltCol, new Float32Array(lk.colors.flatMap((h) => palette.hexToRgb(h))));
    regUploaded = true;
  }
  const LOD_MIN = 0.6;          // never thin below this
  let lod = 1, lastPose = null, speedEma = 0, sectionHold = false;
  const FADE = 0.14;
  let pickFbo = null, pickTex = null, pickDepth = null, pickW = 0, pickH = 0;

  function compile(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(sh));
    return sh;
  }
  function init(canvas) {
    // No MSAA. The point sprites antialias their own edges in the fragment shader (discard +
    // smoothstep on r), so 4x multisampling bought nothing visible on them, but it multiplied
    // the fill and framebuffer memory of every frame: 2-3x the GPU time per render, measured.
    gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 unavailable");
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, shaders.VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, shaders.FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    const u = (n) => gl.getUniformLocation(prog, n);
    U = {
      MVP: u("uMVP"), MV: u("uMV"), pal: u("uPalette"), queen: u("uQueen"), worker: u("uWorker"),
      mode: u("uMode"), casteF: u("uCasteFilter"), split: u("uSplit"), splitDir: u("uSplitDir"), regionMask: u("uRegionMask"), iso: u("uIso"),
      vis: u("uVisible"), z: u("uZ"), size: u("uSize"), cellW: u("uCellW"), pxK: u("uPxK"), cellTrue: u("uCellTrue"),
      dpr: u("uDpr"), hover: u("uHover"), selected: u("uSelected"), pick: u("uPick"),
      opacity: u("uOpacity"), progress: u("uProgress"), fade: u("uFade"), geneLo: u("uGeneLo"), geneHi: u("uGeneHi"), bg: u("uBg"), fogRef: u("uFogRef"),
      lod: u("uLOD"), slab: u("uSlab"), pass: u("uPass"), kcBits: u("uKcBits[0]"), regBits: u("uRegBits[0]"), regCol: u("uRegCol[0]"), regNone: u("uRegNone"), ltCol: u("uLtCol[0]"), morph: u("uMorph"), modeP: u("uModePrev"), casteFP: u("uCasteFilterPrev"),
      regionMaskP: u("uRegionMaskPrev"), geneLoP: u("uGeneLoPrev"), geneHiP: u("uGeneHiPrev"),
      birthHash: u("uBirthHash"), scaffold: u("uScaffold"), ink: u("uInk"), inkCol: u("uInkCol"), spark: u("uSpark"), fogK: u("uFogK"), wob: u("uWob"), time: u("uTime"), ptr: u("uPtr"), aspect: u("uAspect"),
    };
    setPalette(Array.from({ length: 29 }, (_, i) => palette.HEX[i]));
    gl.uniform3fv(U.queen, palette.QUEEN);
    gl.uniform3fv(U.worker, palette.WORKER);
    gl.uniform3f(U.bg, BG[0], BG[1], BG[2]);
    return gl;
  }
  // Upload point positions + per-cell attributes. Safe to call again (tier swap):
  // it releases the previous tier's GPU resources first.
  // Two slots: the one drawn, and one held aside (the hero tier while the atlas is up, so going
  // home is free; or the full atlas while the home page is up, so coming back is too).
  let alt = null;
  const snap = () => ({ vao, N, geneBuf, genePrevBuf, prevIsCur, geneF32, ctBuf, ctSub, ctCanon, ownedBufs, morphSig: curSig });
  function restore(o) { ({ vao, N, geneBuf, genePrevBuf, prevIsCur, geneF32, ctBuf, ctSub, ctCanon, ownedBufs } = o); }
  function free(o) { if (o && o.vao) { gl.deleteVertexArray(o.vao); o.ownedBufs.forEach((b) => gl.deleteBuffer(b)); } }
  function swapSlots() { if (!alt) return false; invalidate(); const cur = snap(); restore(alt); alt = cur; return true; }
  function dropAlt() { free(alt); alt = null; }
  function hasAlt() { return !!alt; }
  // opts.keep: hold the current slot aside instead of freeing it.
  // opts.heroKeys: a Map of the hero cells' packed positions; cells found there are born at -1,
  // i.e. already fully present at progress 0, so the swap from the hero slot is invisible and
  // the reveal grows every OTHER cell around them.
  function upload(data, opts) {
    opts = opts || {}; invalidate();
    if (opts.keep) { free(alt); alt = vao ? snap() : null; vao = null; ownedBufs = []; }
    else if (vao) { gl.deleteVertexArray(vao); ownedBufs.forEach((b) => gl.deleteBuffer(b)); ownedBufs = []; }
    N = data.n;
    // Rotate into natural insect-brain orientation. The stored spatial_aligned x/y
    // put the brain upright (long axis vertical); a −90° in-plane rotation
    // (x,y)→(y,−x) lays it horizontal with the optic lobes lateral, as a brain is
    // normally figured. Kept in int16 range — the shader divides by 32767 and applies
    // Z-exaggeration to .z, so only swap/negate the axes here.
    const pos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      pos[i * 3] = data.y[i];
      pos[i * 3 + 1] = -data.x[i];
      pos[i * 3 + 2] = data.z[i];
    }
    ctSub = data.ct; ctCanon = data.ctCanon || data.ct;
    vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    fbuf(0, pos, 3);
    // Collapsed tree (the coarse groups) until ui.js composes the legend's codes and calls
    // setCodes — the hero tier never reaches ui.js, and it is shown collapsed anyway.
    ctBuf = fbuf(1, Float32Array.from(ctCanon), 1);
    fbuf(2, Float32Array.from(data.caste), 1);
    const conf = new Float32Array(N); for (let i = 0; i < N; i++) conf[i] = data.conf[i] / 255;
    fbuf(3, conf, 1);
    fbuf(5, Float32Array.from(data.region), 1);
    // location 8: fig:kc_subtypes' Large-type subcluster projection (S1-S3 probabilities), or zeros
    const lt = new Float32Array(N * 3);
    if (data.ltkc1) for (let i = 0; i < N; i++) { lt[i * 3] = data.ltkc1[i] / 255; lt[i * 3 + 1] = data.ltkc2[i] / 255; lt[i * 3 + 2] = data.ltkc3[i] / 255; }
    fbuf(8, lt, 3);
    // gene attribute (location 4) — filled on demand, zero to start
    geneBuf = gl.createBuffer(); ownedBufs.push(geneBuf); gl.bindBuffer(gl.ARRAY_BUFFER, geneBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(N), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 1, gl.FLOAT, false, 0, 0);
    // location 7 holds the PREVIOUS gene's values so switching genes cross-dissolves
    // rather than cutting; setGene rolls the outgoing values into it.
    genePrevBuf = gl.createBuffer(); ownedBufs.push(genePrevBuf); gl.bindBuffer(gl.ARRAY_BUFFER, genePrevBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(N), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(7); gl.vertexAttribPointer(7, 1, gl.FLOAT, false, 0, 0);
    geneF32 = null; prevIsCur = true;
    // Reveal-order attribute (location 6): a per-cell birth time in [0,1], ordered
    // SPATIALLY so the build-up reads as a coherent front sweeping outward from the brain
    // core rather than a uniform fade (which saturates instantly → looks like a snap).
    // birth = 0.8·(normalized distance from centroid) + 0.2·(hashed jitter): the radial
    // term is the travelling front; the jitter frays the front into an organic edge.
    const birth = buildBirth(pos, N);
    // The hero tier's cells are an ordered subsequence of this tier (hero ⊂ lite ⊂ full, rows
    // kept in order: build-pages.mjs, export_data.py), so one merge scan finds them. The Map
    // lookup below cost 35-90 ms for 1.3M cells, in the first frames of the home → atlas
    // hand-off (the hitch at its start); the scan is a few ms. The Map stays as the fallback.
    let heroDone = false;
    if (opts.heroData) {
      const h = opts.heroData, n = h.n, marks = [];
      let i = 0;
      for (let j = 0; j < n; j++) {
        const hx = h.x[j], hy = h.y[j], hz = h.z[j], hc = h.caste[j];
        while (i < N && !(data.x[i] === hx && data.y[i] === hy && data.z[i] === hz && data.caste[i] === hc)) i++;
        if (i >= N) break;
        marks.push(i++);
      }
      if (marks.length === n) {
        if (opts.half && heroBirth && heroBirth.length === n) {
          // One build in two halves (see opts.half below): the sample's cells keep exactly the
          // birth times they had in the sample, so the swap is invisible at any point of it, and
          // every other cell is born in the second half, in an even per-cell order (by position).
          const H = opts.half;
          for (let i = 0; i < N; i++) birth[i] = H + (1 - H) * hashPos(data, i);
          // (the home section's extras, born past 1 in the sample, are ordinary cells here)
          for (let k = 0; k < n; k++) if (heroBirth[k] <= 1) birth[marks[k]] = heroBirth[k];
        } else for (let k = 0; k < marks.length; k++) birth[marks[k]] = -1;
        heroDone = true;
      }
    }
    if (opts.heroKeys && !heroDone) {
      const K = opts.heroKeys;
      for (let i = 0; i < N; i++) {
        const z = K.get(((data.x[i] + 32768) * 65536) + (data.y[i] + 32768));
        if (z !== undefined && z === data.z[i] && data.caste[i] === 0) birth[i] = -1;
      }
    }
    // opts.born: full-tier indices of cells already on screen (the lite tier, when the full
    // one replaces it). Born at -1 like the hero cells, so they stay put and the reveal grows
    // every other cell around them: an upgrade with no blank and no snap.
    // opts.half (the home sample): its own build fills only the first part of the reveal, so the
    // full tier can carry on from there (above) as one continuous build rather than two
    if (opts.half && !opts.heroData) for (let i = 0; i < N; i++) birth[i] *= opts.half;
    // the home section's extra cells (data.hx) are born past the end of the ordinary reveal, in an
    // even order: hidden at progress 1, sprinkled in as the story's section takes progress to 2
    if (data.hx) for (let i = 0; i < N; i++) if (data.hx[i]) birth[i] = 1.02 + 0.9 * hashPos(data, i);
    if (opts.half && !opts.heroData) heroBirth = birth.slice();
    if (opts.born) { const B = opts.born; for (let j = 0; j < B.length; j++) if (B[j] < N) birth[B[j]] = -1; }
    fbuf(6, birth, 1);
    gl.bindVertexArray(null);
  }
  // Compute the spatially-ordered per-cell birth times (see upload). Deterministic:
  // a per-index integer hash supplies the jitter (no Math.random → stable across reloads).
  function buildBirth(pos, N) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < N; i++) { cx += pos[i * 3]; cy += pos[i * 3 + 1]; cz += pos[i * 3 + 2]; }
    cx /= N; cy /= N; cz /= N;
    const rad = new Float32Array(N); let rMax = 1e-6;
    for (let i = 0; i < N; i++) {
      const dx = pos[i * 3] - cx, dy = pos[i * 3 + 1] - cy, dz = pos[i * 3 + 2] - cz;
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz); rad[i] = r; if (r > rMax) rMax = r;
    }
    const birth = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
      const j = (h >>> 0) / 4294967296;               // deterministic jitter in [0,1)
      birth[i] = 0.8 * (rad[i] / rMax) + 0.2 * j;      // in [0,1)
    }
    return birth;
  }
  // a cell's even-order hash in [0,1), from its position: the same cell gets the same value in every tier
  function hashPos(data, i) {
    let h = Math.imul((data.x[i] * 73856093) ^ (data.y[i] * 19349663) ^ (data.z[i] * 83492791), 0x85ebca6b);
    h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }
  function fbuf(loc, arr, size) {
    const b = gl.createBuffer(); ownedBufs.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    return b;
  }
  // Swap the cell-type coloring between canonical (cluster vote) and raw subtypes.
  function setLabeling(which) { setCodes(which === "canonical" ? ctCanon : ctSub); }
  // The per-cell colour/visibility code (location 1). ui.js composes it from the tree
  // legend's expand state: a group index while its group is collapsed, a fine-type or
  // restricted-Kenyon code once expanded, all indexing the same uPalette / uVisible.
  function setCodes(codes) {
    invalidate();
    // Only for the slot on screen: on the home page the hero slot (20k cells) is bound while
    // ui.js composes codes for the lite or full tier, which would land in the wrong buffer.
    // That tier's own upload() is followed by ui.setData, which composes again.
    if (!vao || !codes || codes.length !== N) return;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, ctBuf);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from(codes), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }
  // Set the categorical palette (list of hex colors, index = cell-type code).
  function setPalette(hexList) {
    invalidate();
    const palArr = new Float32Array(32 * 3);
    for (let i = 0; i < hexList.length && i < 32; i++) {
      const c = palette.hexToRgb(hexList[i]);
      palArr[i * 3] = c[0]; palArr[i * 3 + 1] = c[1]; palArr[i * 3 + 2] = c[2];
    }
    gl.useProgram(prog);
    gl.uniform3fv(U.pal, palArr);
  }
  let geneLo = 0, geneHi = 1;
  // Swap the active gene expression (Uint8 → normalized float attribute). Returns an
  // auto color window {lo, hi} in [0,1] computed from the value distribution: hi is a
  // high percentile of NONZERO cells so a handful of very-high-prediction cells don't
  // compress everyone else into the dark end (common for Tangram-inferred genes).
  function setGene(u8) {
    invalidate();
    // a gene array for the other tier (fetched just before a tier change) must not reach the
    // GPU: it would be read past its end
    if (!u8 || u8.length !== N) return { lo: geneLo, hi: geneHi };
    const f = new Float32Array(u8.length);
    for (let i = 0; i < u8.length; i++) f[i] = u8[i] / 255;
    gl.bindVertexArray(vao);
    // roll the outgoing gene into the prev slot first — that's what the dissolve reads
    gl.bindBuffer(gl.ARRAY_BUFFER, genePrevBuf);
    gl.bufferData(gl.ARRAY_BUFFER, geneF32 || new Float32Array(u8.length), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, geneBuf);
    gl.bufferData(gl.ARRAY_BUFFER, f, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    geneF32 = f; prevIsCur = false;
    // histogram of nonzero values (256 bins) → robust hi at the 98th percentile
    const hist = new Uint32Array(256); let nz = 0;
    for (let i = 0; i < u8.length; i++) { const v = u8[i]; if (v > 0) { hist[v]++; nz++; } }
    let hi = 255;
    if (nz > 0) { let c = 0; const target = nz * 0.98; for (let b = 1; b < 256; b++) { c += hist[b]; if (c >= target) { hi = b; break; } } }
    geneLo = 0; geneHi = Math.max(hi / 255, 0.02);
    return { lo: geneLo, hi: geneHi };
  }
  function setGeneRange(lo, hi) { invalidate(); geneLo = lo; geneHi = Math.max(hi, lo + 1e-3); }
  // ---- device pixel ratio, adaptive ---------------------------------------------------
  // The backing store starts at the screen's own ratio (capped at 2) and steps down when
  // continuous rendering stays slow: frame cost is fill plus 1.3M vertices, and on a weak
  // GPU the fill half is the part we can give back. Points are sized by uDpr, so the cloud
  // keeps its on-screen size and only gets softer. Never steps back up within a session:
  // a ratio that oscillated would re-allocate the canvas and visibly change sharpness.
  // ?dpr=<x> pins it (and turns adaptation off).
  const DPR_STEPS = [2, 1.5, 1.25, 1];
  const dprPinned = (function () { try { const v = parseFloat(new URLSearchParams(location.search).get("dpr")); return v > 0 ? v : 0; } catch (e) { return 0; } })();
  let dprCap = 2, slowMs = 0, lastCanvas = null;
  function setDpr(cap) { dprCap = cap; if (lastCanvas) resize(lastCanvas); }
  // Called by the frame loop with the interval between two consecutive RENDERED frames.
  // 40 ms sustained over ~1.2 s of real motion steps the ratio down one notch.
  function adaptDpr(frameMs) {
    if (dprPinned || !(frameMs > 0) || frameMs > 1000) return;
    if (frameMs > 40) slowMs += frameMs; else slowMs = Math.max(0, slowMs - frameMs * 0.5);
    if (slowMs < 1200) return;
    slowMs = 0;
    const cur = Math.min(window.devicePixelRatio || 1, dprCap);
    const next = DPR_STEPS.find((d) => d < cur - 1e-3);
    if (next) setDpr(next);
  }
  function getDpr() { return DPR; }
  function resize(canvas) {
    lastCanvas = canvas; invalidate();
    DPR = dprPinned || Math.min(window.devicePixelRatio || 1, dprCap);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(w * DPR));
    canvas.height = Math.max(1, Math.floor(h * DPR));
    gl.viewport(0, 0, canvas.width, canvas.height);
    projM = mat4.persp(48 * Math.PI / 180, canvas.width / canvas.height, 0.05, 60);
    ensurePick(Math.max(1, Math.floor(w)), Math.max(1, Math.floor(h)));
  }
  function ensurePick(w, h) {
    if (pickFbo && pickW === w && pickH === h) return;
    pickW = w; pickH = h;
    if (pickFbo) { gl.deleteFramebuffer(pickFbo); gl.deleteTexture(pickTex); gl.deleteRenderbuffer(pickDepth); }
    pickTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, pickTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    pickDepth = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, pickDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    pickFbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pickDepth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  function setCommon(pick) {
    const s = state.s;
    gl.uniform1i(U.mode, s.mode);
    gl.uniform1i(U.casteF, s.casteFilter);
    gl.uniform1f(U.split, split.mag);   // the "both" offset: side by side, or stacked on a portrait phone
    gl.uniform2f(U.splitDir, split.dir[0], split.dir[1]);
    gl.uniform1i(U.regionMask, s.regionMask);
    gl.uniform1i(U.iso, pick ? -1 : s.iso);
    gl.uniform1fv(U.vis, s.visible);
    gl.uniform1f(U.z, s.z);
    gl.uniform1f(U.size, s.size * look.sizeMul);
    // cell-true dots: diameter in world units, and pixels per world unit at distance 1 for
    // the viewport being drawn (the pick buffer is CSS pixels, the screen device pixels)
    gl.uniform1f(U.cellW, CELL_UM / scaleUm());
    gl.uniform1f(U.pxK, (pick ? pickH : viewH) / (2 * Math.tan(24 * Math.PI / 180)));
    // the home page keeps its own look; the atlas gets cell-true dots, eased in by the hand-off
    gl.uniform1f(U.cellTrue, BIMP.transition ? BIMP.transition.tp : 1);
    gl.uniform1f(U.opacity, s.opacity * look.alpha);
    gl.uniform1i(U.hover, pick ? -1 : s.hover);
    gl.uniform1i(U.selected, pick ? -1 : s.selected);
    // fog reference = how far the camera is orbiting out; keeps the haze a depth cue
    // within the cloud instead of an overall dimmer that grows as you zoom out
    gl.uniform1f(U.fogRef, camera.cam.radius);
    gl.uniform1f(U.geneLo, geneLo);
    gl.uniform1f(U.geneHi, geneHi);
    // The state being morphed away from. The pick pass forces morph=1 so hit-testing
    // always uses the CURRENT view — a cell half-way through fading in is still clickable
    // where it is now, and one fading out is not clickable where it used to be.
    const p = prevState || { mode: s.mode, caste: s.casteFilter, region: s.regionMask, lo: geneLo, hi: geneHi };
    gl.uniform1f(U.morph, pick ? 1.0 : easeMorph(morph));
    gl.uniform1f(U.lod, pick ? 1.0 : lod);   // picking always sees every cell
    gl.uniform4fv(U.slab, s.slab && s.slab.length === 4 ? s.slab : [0, 0, 0, 0]);  // 2D section view (histology.js); off unless set
    gl.uniform1iv(U.kcBits, kcBitsFor());
    uploadRegions();
    gl.uniform1i(U.modeP, p.mode);
    gl.uniform1i(U.casteFP, p.caste);
    gl.uniform1i(U.regionMaskP, p.region);
    gl.uniform1f(U.geneLoP, p.lo);
    gl.uniform1f(U.geneHiP, p.hi);
    gl.uniform1i(U.pick, pick ? 1 : 0);
  }
  // Start a morph whenever the view state actually changed since the last frame. Keyed on
  // the fields the shader interpolates; anything else (point size, opacity, labeling) is
  // applied instantly, as it always was.
  // The signature deliberately EXCLUDES the colour window: dragging the scale sliders
  // is a continuous adjustment and must track the handle, not restart a 0.55 s dissolve
  // on every input event (which read as jumpy). lo/hi still ride along in prevState, so
  // switching genes dissolves from the old gene's window correctly.
  function sigOf(s) { return [s.mode, s.casteFilter, s.regionMask, s.gene].join("|"); }
  function sigOf2(st) { return [st.mode, st.caste, st.region, st.gene].join("|"); }   // the same, for a stored state
  function syncMorph() {
    const s = state.s;
    const now = { mode: s.mode, caste: s.casteFilter, region: s.regionMask, gene: s.gene, lo: geneLo, hi: geneHi };
    const sig = sigOf(s);
    let started = false;
    if (curSig === null) {                       // first frame: nothing to morph from
      curSig = sig; curState = now; prevState = now; morph = 1;
    } else if (sig !== curSig) {                 // a change: morph from what we were showing
      // Interrupted mid-dissolve: going straight back reverses it from where it stands (no
      // jump at all); anything else continues from whichever end the view is closer to,
      // instead of always snapping to the half-reached target first
      // (not across a gene change: the outgoing gene's values live in their own buffer)
      const back = morph < 1 && prevState && sigOf2(prevState) === sig && prevState.gene === curState.gene;
      if (back) { const t = prevState; prevState = curState; curState = t; morph = 1 - morph; }
      else {
        if (!(morph < 1 && easeMorph(morph) < 0.5)) prevState = curState;
        curState = now; morph = 0;
        // The prev slot still holds the gene before the current one until another gene uploads,
        // so leaving gene view after A → B dissolved out of A's values through B's window (a cut
        // on the first frame). When the gene itself is not what changed, the outgoing colours
        // are the current gene's: copy them across on the GPU.
        if (prevState.gene === now.gene && !prevIsCur && geneF32) {
          gl.bindBuffer(gl.COPY_READ_BUFFER, geneBuf); gl.bindBuffer(gl.COPY_WRITE_BUFFER, genePrevBuf);
          gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, 0, 0, N * 4);
          gl.bindBuffer(gl.COPY_READ_BUFFER, null); gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
          prevIsCur = true;
        }
      }
      curSig = sig; started = true;
    }
    const t = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
    const dt = lastT ? Math.min(t - lastT, 0.05) : 0;
    lastT = t;
    // The dissolve starts at exactly 0 on the frame that sees the change. Frames are drawn on
    // demand, so the previous one can be seconds old, and that idle gap (clamped to 50 ms) used
    // to land in the first step: 9% of the dissolve in one frame, the snap at the start.
    if (morph < 1 && !started) morph = REDUCED ? 1 : Math.min(1, morph + dt / MORPH_SECS);
    syncLod(dt);
  }
  // Apply any pending view change instantly (no dissolve). Used by the self-test, which
  // samples a single frame per assertion and would otherwise photograph the transition.
  function settleMorph() { syncMorph(); morph = 1; }
  // Track camera motion → target LOD. Uses the pose itself, so it covers every source of
  // movement (drag, pinch, momentum, auto-orbit, scripted tour flights) with no wiring.
  function syncLod(dt) {
    const c = camera.cam;
    const p = lastPose;
    lastPose = { theta: c.theta, phi: c.phi, radius: c.radius, tx: c.target[0], ty: c.target[1], tz: c.target[2] };
    if (!p) return;
    const moved = Math.abs(c.theta - p.theta) + Math.abs(c.phi - p.phi) + Math.abs(c.radius - p.radius) * 0.4
      + Math.abs(c.target[0] - p.tx) + Math.abs(c.target[1] - p.ty) + Math.abs(c.target[2] - p.tz);
    // Thin in PROPORTION to how fast the view is moving, not on/off: the idle auto-orbit
    // is slow (0.12 rad/s) and stays at essentially full density, while a drag or a tour
    // flight — where detail is masked by the motion anyway — drops to LOD_MIN. A binary
    // switch would have left the atlas permanently thinned, since auto-orbit is on by default.
    // Both the measurement and the response are damped. A raw per-frame speed is very
    // noisy — pointer deltas arrive in bursts and rAF intervals jitter — and feeding that
    // straight into the threshold made the kept fraction jump around every frame, which
    // is visible as the cloud boiling however soft the cull band is. So: smooth the speed,
    // then ease the threshold in BOTH directions (fast down, slow back up).
    const step = Math.min(1, (dt || 0.016) / 0.09);
    speedEma += (moved / Math.max(dt || 0.016, 0.004) - speedEma) * step;
    const want = Math.max(LOD_MIN, Math.min(1, 1 - (speedEma - 0.2) * 0.45));
    // NOT thinned for a caste split, though every point is moving and the frames are the
    // most expensive the viewer draws. That was tried and reverted: pinning the kept
    // fraction to the floor for the length of the morph measured as a hard 40% density drop
    // over 120 ms at the click and an 8%-lit-pixel snap back at the moment the brains land
    // — two brightness lurches bracketing exactly the motion they were meant to smooth.
    // The travel does NOT hide it, because the eye is tracking the cloud rather than a
    // blurred background. Camera-driven thinning above stays: it rides the flight's own
    // ease in and out, which is the well-behaved case. (test/split-smooth.mjs)
    if (want < lod) lod = Math.max(want, lod - (dt || 0.016) / 0.12);   // down over ~0.12 s
    else lod = Math.min(want, lod + (dt || 0.016) / 0.35);              // back up over ~0.35 s
    // Not in a 2D section either (histology's hard-edged slab), nor on the way back out of one
    // until the camera has settled: the camera flies into the section, the thinning dropped
    // 40% of its cells, and they filled back in just after it landed, a late snap. One
    // section's cells are cheap; the peel down to it and back draws every cell, as at rest.
    const sl = state.s.slab;
    if (sl && sl.length === 4 && sl[2] > 0 && sl[3] === 0) sectionHold = true;
    else if (sectionHold && speedEma < 0.05) sectionHold = false;
    if (sectionHold) lod = 1;
  }
  function drawScene(pick) {
    const e = camera.eye();
    const view = mat4.lookAt(e, camera.cam.target, [0, 1, 0]);
    const mvp = mat4.mul(projM, view);
    if (!pick && BIMP.histology) BIMP.histology.draw(gl, mvp, view);   // DAPI layer under the cells; restores GL state
    // the Cells layer switched off (ui.js layer row, key C): the stain alone, nothing to pick
    if (state.s.cellsOn === false) return;
    gl.uniformMatrix4fv(U.MVP, false, mvp);
    gl.uniformMatrix4fv(U.MV, false, view);
    gl.uniform1f(U.dpr, pick ? 1.0 : DPR);
    setCommon(pick);
    // Reveal front: drive uProgress to progress·(1+FADE) so the last cells fully resolve
    // at progress==1. The pick pass forces it past the end (2.0) so EVERY cell is always
    // clickable, even mid-build. All N are drawn every frame; the shader offscreens the
    // unborn ones (gl_PointSize 0 + reveal<=0), so they cost no fragments early on.
    gl.uniform1f(U.progress, pick ? 2.0 : progress * (1.0 + FADE));
    gl.uniform1f(U.fade, FADE);
    gl.bindVertexArray(vao);
    // One pass or two? Deliberately-faded cells (the grey region wash, near-zero cells in
    // gene view, anything mid-dissolve or mid-build) must not write depth — see the pass
    // split in the vertex shader. Most of the time no such cells exist, and then the
    // second pass would only burn 1.3M vertex invocations for nothing, so ask first.
    if (pick || !hasFaintCells()) {
      gl.uniform1i(U.pass, 0);
      gl.drawArrays(gl.POINTS, 0, N);
      return;
    }
    gl.uniform1i(U.pass, 1); gl.depthMask(true);    // solid: fills the depth buffer
    gl.drawArrays(gl.POINTS, 0, N);
    gl.uniform1i(U.pass, 2); gl.depthMask(false);   // faint: tests against it, never hides it
    if (look.scaffold > 0.5) gl.blendEquation(gl.MAX);   // the home scaffold: no additive build-up (see FS)
    gl.drawArrays(gl.POINTS, 0, N);
    gl.blendEquation(gl.FUNC_ADD);
    gl.depthMask(true);
  }
  // Only two views deliberately push a whole class of cells into the background: gene view's
  // near-zero sea and region view's grey context wash. Those are the ones where a faint cell
  // was erasing a bright one, so those are the ones worth a second pass — everywhere else
  // (including the intro build, where every cell is briefly faint on its way in, and which
  // is the most frame-hungry moment there is) one pass costs exactly what it always did.
  function hasFaintCells() {
    const m = state.s.mode, p = prevState;
    return look.scaffold > 0 || m === 3 || m === 4 || (morph < 1 && p && (p.mode === 3 || p.mode === 4));
  }
  // ---- render on demand ----------------------------------------------------------------
  // A still atlas used to be redrawn 60 times a second for nothing: about 15 ms of GPU per
  // frame on an M3 Pro (a saturated GPU, forever), seven CPU cores under a software GPU.
  // The frame loop now asks needsFrame() first. It says yes while anything is moving
  // (morph, LOD easing back, reveal, the home look), for TAIL ms after the last change
  // (covers the histology layer's own fades, the UMAP marker glide, the LOD ease back),
  // and whenever the drawn inputs differ from the last frame drawn: camera pose, every
  // state.s field the shader reads, the look, the canvas size. invalidate() forces one,
  // for changes that live outside those (a buffer upload, a texture arriving); onWake
  // lets the frame loop resume when it has gone to sleep.
  const TAIL = 700;
  let dirty = true, busyUntil = 0, onWake = null, histoBusy = false;
  let lastWhy = "", lastChanged = [];   // why the last frame was drawn (tests, the perf harness)
  function whyFrame() { return { why: lastWhy, changed: lastChanged.slice(0, 12) }; }
  const sigA = new Float64Array(96), sigB = new Float64Array(96);
  let sigCur = sigA, sigPrev = sigB, sigLab = null;
  function invalidate() { dirty = true; if (onWake) onWake(); }
  let viewShiftSet = false;
  function setViewShift(ndcY) {
    if (!viewShiftSet) { viewShiftSet = true; viewShiftCur = ndcY; }   // the first value is where the view starts: no glide
    if (ndcY !== viewShift) { viewShift = ndcY; invalidate(); }
  }
  function setSplit(mag, dir) { split = { mag, dir: dir.slice() }; invalidate(); }
  function getSplit() { return { mag: split.mag, dir: split.dir.slice() }; }
  function setOnWake(fn) { onWake = fn; }
  function writeSig(o) {
    const s = state.s, c = camera.cam, L = look;
    let k = 0;
    o[k++] = c.theta; o[k++] = c.phi; o[k++] = c.radius; o[k++] = c.target[0]; o[k++] = c.target[1]; o[k++] = c.target[2];
    o[k++] = s.mode; o[k++] = s.casteFilter; o[k++] = s.regionMask; o[k++] = s.gene; o[k++] = s.size; o[k++] = s.z;
    o[k++] = s.opacity; o[k++] = s.iso; o[k++] = s.hover; o[k++] = s.selected;
    const sl = s.slab || []; for (let i = 0; i < 4; i++) o[k++] = sl[i] || 0;
    const v = s.visible; for (let i = 0; i < 32; i++) o[k++] = v[i];
    o[k++] = L.alpha; o[k++] = L.spark; o[k++] = L.fogK; o[k++] = L.ink; o[k++] = L.sizeMul; o[k++] = L.lockLod ? 1 : 0; o[k++] = L.scissor ? 1 : 0;
    for (let i = 0; i < 3; i++) { o[k++] = L.inkCol[i]; o[k++] = L.bg[i]; }
    const r = L.rect; o[k++] = r ? r.x : -1; o[k++] = r ? r.y : -1; o[k++] = r ? r.w : -1; o[k++] = r ? r.h : -1;
    o[k++] = gl ? gl.canvas.width : 0; o[k++] = gl ? gl.canvas.height : 0; o[k++] = progress; o[k++] = geneLo; o[k++] = geneHi; o[k++] = N;
    // the histology layer's textures arrive on their own (and are drained inside its draw),
    // so its load counts are part of what is drawn: each arrival is a change, with a tail
    // that covers its fade-in
    // isAnimating() (histology.js) covers all of this when present; stats() is the fallback
    const H = BIMP.histology;
    let hs = null;
    if (H && !H.isAnimating && H.isOn && H.isOn() && H.stats) { try { hs = H.stats(); } catch (e) { hs = null; } }
    // (coerced: a renamed field must read 0, not NaN, which never equals itself and would
    // redraw every frame)
    o[k++] = hs ? 1 : 0; o[k++] = hs ? +hs.small || 0 : 0; o[k++] = hs ? +hs.large || 0 : 0; o[k++] = hs ? +hs.sections || 0 : 0;
    o[k++] = hs ? +hs.queued || 0 : 0; o[k++] = hs ? +hs.loading || 0 : 0; o[k++] = hs && hs.warming ? 1 : 0;
    histoBusy = !!(hs && (hs.queued > 0 || hs.warming));
    return k;
  }
  function needsFrame(now) {
    const t = now || performance.now();
    const n = writeSig(sigCur);
    let changed = dirty || sigLab !== state.s.labeling;
    for (let i = 0; i < n && !changed; i++) if (sigCur[i] !== sigPrev[i]) changed = true;
    // things that animate on their own clock, with no input changing
    const H = BIMP.histology;
    let hAnim = false;
    if (H && H.isAnimating) { try { hAnim = !!H.isAnimating(); } catch (e) { hAnim = true; } }
    const anim = morph < 1 || lod < 1 || isMorphing() || look.wob > 0 || look.ptr[3] > 0 || viewShiftCur !== viewShift
      || histoBusy || hAnim;
    lastWhy = changed ? "changed" : morph < 1 ? "morph" : lod < 1 ? "lod" : isMorphing() ? "morphing" : (look.wob > 0 || look.ptr[3] > 0) ? "look" : viewShiftCur !== viewShift ? "viewShift" : histoBusy ? "histoBusy" : hAnim ? "histology" : t < busyUntil ? "tail" : "";
    if (changed) { busyUntil = t + TAIL; lastChanged = []; for (let i = 0; i < n; i++) if (sigCur[i] !== sigPrev[i]) lastChanged.push(i); if (dirty) lastChanged.push("dirty"); }
    if (!(changed || anim || t < busyUntil)) return false;
    const tmp = sigPrev; sigPrev = sigCur; sigCur = tmp;
    sigLab = state.s.labeling; dirty = false;
    return true;
  }
  function render() {
    syncMorph();          // detect view changes and advance the dissolve clock
    if (look.lockLod) lod = 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const W = gl.canvas.width, H = gl.canvas.height, r = look.rect;
    BG[0] = look.bg[0]; BG[1] = look.bg[1]; BG[2] = look.bg[2];
    // the whole canvas takes the ground colour: the page's paper at home, the dark stage in the
    // atlas, and in between the one quick "lights down" (boot.js)
    gl.disable(gl.SCISSOR_TEST); gl.viewport(0, 0, W, H);
    gl.clearColor(BG[0], BG[1], BG[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    let vw = W, vh = H;
    if (r) {
      const x = Math.round(r.x * DPR), y = Math.round(H - (r.y + r.h) * DPR);
      vw = Math.max(1, Math.round(r.w * DPR)); vh = Math.max(1, Math.round(r.h * DPR));
      gl.viewport(x, y, vw, vh);
      // at rest in the box, clip the sprites that would spill past its edge onto the page
      if (look.scissor) { gl.enable(gl.SCISSOR_TEST); gl.scissor(x, y, vw, vh); }
    }
    projM = mat4.persp(48 * Math.PI / 180, vw / vh, 0.05, 60); viewH = vh;
    // phones: a vertical lens shift (NDC) so the brain centres between the header and the
    // bottom sheet's peek, not behind the sheet (ui.js sets it; eased in with the hand-off)
    {
      // the shift glides to its target (≈200 ms) instead of jumping when the inspector or the
      // tour card changes what covers the bottom of a phone
      const t = performance.now(), dt = viewShiftT ? Math.min(0.1, (t - viewShiftT) / 1000) : 1;
      viewShiftT = t;
      viewShiftCur += (viewShift - viewShiftCur) * (1 - Math.exp(-dt / 0.06));
      if (Math.abs(viewShift - viewShiftCur) < 1e-4) viewShiftCur = viewShift;
      if (viewShiftCur) projM[9] -= viewShiftCur * (BIMP.transition ? BIMP.transition.tp : 1);
    }
    gl.uniform3f(U.bg, BG[0], BG[1], BG[2]);
    gl.uniform1f(U.ink, look.ink); gl.uniform3fv(U.inkCol, look.inkCol); gl.uniform1f(U.scaffold, look.scaffold); gl.uniform1f(U.birthHash, look.birthHash);
    gl.uniform1f(U.spark, look.spark); gl.uniform1f(U.fogK, look.fogK);
    gl.uniform1f(U.wob, look.wob); gl.uniform1f(U.time, performance.now() / 1000);
    gl.uniform4fv(U.ptr, look.ptr); gl.uniform1f(U.aspect, vw / vh);
    gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    drawScene(false);
  }
  // Pick the nearest cell within a small radius of (px,py). Reading a block instead
  // of a single pixel makes clicking a small point on a dense cloud actually work.
  function pick(px, py, radius) {
    if (state.s.cellsOn === false) return -1;
    const r = radius == null ? 6 : radius;
    ensurePick(pickW, pickH);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, pickW, pickH);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthMask(true); gl.disable(gl.BLEND);
    drawScene(true);
    const cx = Math.floor(px), cy = Math.floor(py);
    const x0 = Math.max(0, cx - r), y0 = Math.max(0, cy - r);
    const w = Math.min(pickW, cx + r + 1) - x0, h = Math.min(pickH, cy + r + 1) - y0;
    let best = -1, bestD = Infinity;
    if (w > 0 && h > 0) {
      const buf = new Uint8Array(w * h * 4);
      gl.readPixels(x0, pickH - y0 - h, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
        const o = ((h - 1 - j) * w + i) * 4;               // flip Y back to top-left
        const id = buf[o] | (buf[o + 1] << 8) | (buf[o + 2] << 16);
        if (id === 0) continue;
        const dx = (x0 + i) - cx, dy = (y0 + j) - cy, d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = id - 1; }
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return best;
  }
  // 0..1 is the reveal; up to 2 brings in the home section's extra cells (upload, data.hx)
  function setProgress(v) { invalidate(); progress = Math.max(0, Math.min(2, v)); }
  function getProgress() { return progress; }
  // exposed for tests: how far through a view dissolve we are (1 = settled)
  function getMorph() { return morph; }
  // Is the view mid-transition — INCLUDING a change made since the last frame that has not
  // been picked up yet? syncMorph runs inside render(), so a caller earlier in the frame
  // asking `morph < 1` is told "settled" on the very frame a morph begins. That is exactly
  // the frame a caller wants to stand down on, and it let one 1.3M-point pick pass through
  // per transition (measured: test/morph-budget.mjs).
  function isMorphing() {
    return morph < 1 || (curSig !== null && sigOf(state.s) !== curSig);
  }
  function getLod() { return lod; }
  return { look, swapSlots, dropAlt, hasAlt, init, upload, setGene, setGeneRange, setLabeling, setCodes, setPalette, setProgress, getProgress, resize, render, pick, getMorph, isMorphing, getLod, settleMorph, invalidate, setViewShift, scaleUm, setSplit, getSplit, needsFrame, whyFrame, setOnWake, adaptDpr, setDpr, getDpr, get N() { return N; }, get gl() { return gl; }, FADE };
})();
