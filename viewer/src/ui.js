/* Panels: color-by / caste / region multi-select / sliders / auto-orbit, grouped
   cell-type legend with hover-isolate, gene picker, and the click inspector.
   Mutates BIMP.state; calls the renderer only where a GPU upload is needed. */
window.BIMP = window.BIMP || {};
BIMP.ui = (function () {
  const { palette, state, camera, decode } = BIMP;
  const s = state.s;
  let M, DATA, R;
  const $ = (id) => document.getElementById(id);

  function build(opts) {
    M = opts.manifest; DATA = opts.data; R = opts.renderer;
    palette.configureNames(M);
    buildModeSeg(); buildLabelSeg(); buildCasteSeg(); buildRegionPicker(); buildSliders(); buildSpin(); buildUmapToggle();
    shapeTree(); composeCodes();     // uploads codes + palette, builds the legend
    buildGenePicker();
    placeLegend(); watchTitle(); buildLayers(); buildSheet(); buildTier();
    // a device on the lite tier keeps it: no full setData follows, so read the address here
    // (after boot.js's applyUrlParams, which runs right after build)
    if (BIMP.tier === "lite" || DATA.n >= (M.n || DATA.n)) setTimeout(() => { if (!_urlRead) readUrl(); }, 0);
    window.addEventListener("resize", syncInspectorLane);
    $("i-close").addEventListener("click", () => inspect(-1));   // the panel can get tall — always offer a way out
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && s.selected >= 0) inspect(-1); });
    // gene-modal dismissal (✕, backdrop click, Esc)
    $("gm-close").addEventListener("click", closeGeneModal);
    $("gene-modal").addEventListener("click", (e) => { if (e.target.id === "gene-modal") closeGeneModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeGeneModal(); });
    // glossary ⓘ — delegated, so a dot added to body.html later needs no wiring here.
    // CAPTURE phase, and that is the whole trick: one dot sits inside the Show-UMAP
    // toggle, which listens for clicks itself. A bubbling listener on document runs
    // AFTER the toggle's, so stopPropagation would come too late and asking what the
    // UMAP is would also switch it on. Capturing runs document→target, so this fires
    // first and the click never reaches the control underneath.
    document.addEventListener("click", (e) => {
      const dot = e.target.closest && e.target.closest("[data-info]");
      if (!dot) return;
      e.stopPropagation(); e.preventDefault();
      openInfo(dot.dataset.info);
    }, true);
    $("if-close").addEventListener("click", closeInfo);
    $("info-modal").addEventListener("click", (e) => { if (e.target.id === "info-modal") closeInfo(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeInfo(); });
  }

  // ---- the cell-type tree ------------------------------------------------------------
  // ONE legend for both annotation levels of the canonical cluster vote: each coarse
  // group is a row, and it
  // expands to the fine-scale types the vote decided inside it. Colour follows the level
  // that is showing, group by group — a collapsed group is one colour, an expanded one
  // shows its types. Kenyon cells are the exception: the panel cannot name their subtypes
  // per cluster, so the vote holds them at the group, and their children are instead the
  // RESTRICTED transfer (fig:kc_subtypes panel B): Tangram re-run on the Kenyon-compartment
  // cells against the five named snRNA-seq Kenyon populations.
  //
  // Large-type then opens one level further, to fig:kc_subtypes' three de novo subclusters S1–S3,
  // each Large-type cell taking the subcluster its projection puts most weight on.
  //
  // Every level is folded into ONE per-cell code, composed here and uploaded to the
  // renderer's cell-type attribute. uPalette, uVisible, uIso, the inspector and the
  // prevalence denominator are all indexed by it, so none of them needs to know which
  // level is on screen:
  //   [0, G)             coarse group g, while its group is collapsed
  //   [G, G+F)           fine type f (manifest.cellTypes), while its group is expanded
  //   [G+F, G+F+K)       restricted Kenyon type k (manifest.kcTypes), while Kenyon cells is
  //                      expanded; Kenyon-voted cells outside the compartments keep the
  //                      generic fine "Kenyon cells" code, shown as their own child
  //   [G+F+K, +3)        Large-type subcluster S1-S3 (argmax of ltkc1..3), while Large-type is
  const MAXC = 32;          // uPalette / uVisible length in the shader
  let TR = null;            // tree shape for this manifest + tier
  const opened = new Set(); // codes of the expanded rows (groups, and Large-type)
  let codes = null, codesVer = 0;
  function shapeTree() {
    const canon = M.canonTypes || [], fine = M.cellTypes || [];
    const flat = !canon.length || !DATA.ctCanon;   // an export without groups: one level
    const G = flat ? fine.length : canon.length, F = fine.length;
    const names = [], raws = [], cols = [], byCode = {};
    const residualGlia = fine.some((n) => n !== "Glia" && palette.poolOf(n) === "Glia");
    const node = (code, raw, name, color, count) => {
      names[code] = name; raws[code] = raw; cols[code] = color;
      return (byCode[code] = { code, raw, name, color, count: count || 0, kids: [], kind: null });
    };
    const groups = flat
      ? fine.map((n, f) => node(f, n, palette.displayName(n, "types"), M.subtypeColors[f], M.cellTypeCounts[f]))
      : canon.map((n, g) => node(g, n, palette.groupName(n), M.canonColors[g], M.canonCounts[g]));
    if (!flat) fine.forEach((n, f) => {
      const k = node(G + f, n, palette.displayName(n, "types"),
        palette.fineColor(n, M.subtypeColors[f], residualGlia), M.cellTypeCounts[f]);
      k.resid = canon.indexOf(n) >= 0;            // "Glia", "Kenyon cells": the group's remainder
      const g = canon.indexOf(palette.poolOf(n));
      if (g >= 0 && k.count > 0) groups[g].kids.push(k);
    });
    const byCount = (a, b) => (a.resid - b.resid) || (b.count - a.count);
    groups.forEach((gr) => { gr.kids.sort(byCount); gr.kind = gr.kids.length >= 2 ? "fine" : null; });
    // Kenyon cells → the restricted transfer, when this export carries it
    const kt = M.kcTypes, kcG = flat ? -1 : canon.indexOf("Kenyon cells");
    const K = kt && kt.names ? kt.names.length : 0;
    const lk = M.ltkc, hasLt = !!(lk && lk.colors && DATA.ltkc1 && DATA.ltkc2 && DATA.ltkc3);
    let kcOff = -1, ltOff = -1, ltK = -1;
    if (K && DATA.kctype && kcG >= 0 && G + F + K + 3 <= MAXC) {
      kcOff = G + F;
      ltK = kt.names.indexOf("Large-type Kenyon cells");
      if (hasLt && ltK >= 0) ltOff = G + F + K;
      const fKC = fine.indexOf("Kenyon cells"), resid = fKC >= 0 ? G + fKC : kcG;
      // Counts are the Kenyon-VOTED cells in each call, from the data: a child row has to be
      // a part of its parent. (manifest kcTypes.counts covers every compartment cell,
      // including the few the vote calls glia or Other, so it would not add up to the row.)
      const cnt = new Float64Array(K + 1), lt = new Float64Array(3), cc = DATA.ctCanon, kc = DATA.kctype;
      const l1 = DATA.ltkc1, l2 = DATA.ltkc2, l3 = DATA.ltkc3;
      for (let i = 0; i < DATA.n; i++) {
        if (cc[i] !== kcG) continue;
        const t = kc[i] > 0 && kc[i] <= K ? kc[i] : 0; cnt[t]++;
        if (ltOff >= 0 && t === ltK + 1) { const a = l1[i], b = l2[i], c = l3[i]; if (a + b + c > 0) lt[a >= b && a >= c ? 0 : b >= c ? 1 : 2]++; }
      }
      const scale = (M.n || DATA.n) / DATA.n;     // the lite tier is a sample of the atlas
      const gr = groups[kcG];
      // manifest order, which is the figures': Large-type, Small-type, Class II lead
      gr.kids = kt.names.map((n, k) => {
        // the tree row carries the figure's short name under its "Kenyon cells" parent; the
        // inspector and the tooltip keep the full one
        const sh = (kt.short || [])[k] || n;
        const c = node(kcOff + k, n, sh, palette.kcColor(n, kt.colors[k]), Math.round(cnt[k + 1] * scale));
        names[kcOff + k] = n; c.short = sh; c.title = n; return c;
      }).filter((c) => c.count > 0);
      const r = { code: resid, raw: "Kenyon cells", name: "Kenyon cells (outside the compartments)", resid: true,
        color: resid !== kcG ? palette.KC_OUTSIDE : cols[resid], count: Math.round(cnt[0] * scale), kids: [], kind: null };
      if (resid !== kcG) { names[resid] = r.name; byCode[resid] = r; cols[resid] = r.color; }
      if (r.count > 0) gr.kids.push(r);
      gr.kind = "kc";
      // Large-type → S1-S3 (fig:kc_subtypes), an ordinary level: filter, isolate, colour
      const L = ltOff >= 0 ? byCode[kcOff + ltK] : null;
      if (L) {
        L.kids = [0, 1, 2].map((j) => node(ltOff + j, lk.labels[j], "Subcluster " + lk.short[j], lk.colors[j], Math.round(lt[j] * scale)))
          .filter((c) => c.count > 0);
        L.kids.forEach((c) => { names[c.code] = "Large-type " + c.raw.replace("Subcluster ", "S"); });
        L.kind = L.kids.length ? "lt" : null;
      }
    }
    [...opened].forEach((c) => { if (!byCode[c] || !byCode[c].kind) opened.delete(c); });
    // the row order: biggest first, Other last (it is a quality-control outcome, not a type)
    const order = groups.filter((gr) => gr.count > 0)
      .sort((a, b) => ((a.raw === "Mixed") - (b.raw === "Mixed")) || (b.count - a.count));
    TR = { flat, G, F, K, kcG, kcOff, ltOff, ltK, groups, order, names, raws, cols, byCode };
  }
  function composeCodes() {
    if (!TR) return;
    const n = DATA.n;
    if (TR.flat) codes = DATA.ct;
    else {
      const how = new Uint8Array(TR.G);                 // 0 collapsed · 1 fine · 2 restricted KC
      TR.groups.forEach((gr, g) => { if (opened.has(g)) how[g] = gr.kind === "kc" ? 2 : gr.kind ? 1 : 0; });
      if (!codes || codes.length !== n || codes === DATA.ct) codes = new Uint8Array(n);
      const cc = DATA.ctCanon, ct = DATA.ct, kc = DATA.kctype, G = TR.G, off = TR.kcOff - 1;
      const kg = TR.kcG >= 0 ? TR.groups[TR.kcG] : null;
      const resid = kg && kg.kind === "kc" ? kg.kids.find((c) => c.resid) : null;
      const rc = resid ? resid.code : TR.kcG;
      const ltT = TR.ltOff >= 0 && opened.has(TR.kcOff + TR.ltK) ? TR.ltK + 1 : -1, lo = TR.ltOff;
      const l1 = DATA.ltkc1, l2 = DATA.ltkc2, l3 = DATA.ltkc3;
      const H = shownHist(), ca = DATA.caste, rg = DATA.region;
      for (let i = 0; i < n; i++) {
        const g = cc[i], h = how[g];
        if (h === 0) codes[i] = g;
        else if (h === 1) codes[i] = G + ct[i];
        else {
          const t = kc[i];
          if (t === 0) codes[i] = rc;
          else if (t !== ltT) codes[i] = off + t;
          else {
            const a = l1[i], b = l2[i], c = l3[i];
            codes[i] = a + b + c > 0 ? lo + (a >= b && a >= c ? 0 : b >= c ? 1 : 2) : off + t;
          }
        }
        H[((ca[i] ? 256 : 0) + rg[i]) * MAXC + codes[i]]++;   // the shown-cell histogram, same pass
      }
    }
    if (TR.flat) { const H = shownHist(), ca = DATA.caste, rg = DATA.region; for (let i = 0; i < n; i++) H[((ca[i] ? 256 : 0) + rg[i]) * MAXC + codes[i]]++; }
    codesVer++;
    const pal = [];
    for (let c = 0; c < MAXC; c++) pal.push(TR.cols[c] || "#8A93A6");
    R.setPalette(pal);
    if (R.setCodes) R.setCodes(codes); else R.setLabeling(opened.size ? "subtypes" : "canonical");
    s.labeling = opened.size ? "subtypes" : "canonical";   // kept for anything still reading it
    s.iso = -1;
    refreshLegend();
    updateStat();
  }
  // a row's cells are on screen if any of what it currently shows is
  function anyVis(nd) { return opened.has(nd.code) && nd.kind ? nd.kids.some(anyVis) : s.visible[nd.code] > 0.5; }
  function setDeep(nd, v) { s.visible[nd.code] = v; nd.kids.forEach((k) => setDeep(k, v)); }
  function setOpen(code, on) {
    const nd = TR.byCode[code];
    if (!nd || !nd.kind || opened.has(code) === on) return;
    if (on) { const v = s.visible[code]; nd.kids.forEach((k) => setDeep(k, v)); opened.add(code); }
    else { s.visible[code] = anyVis(nd) ? 1 : 0; opened.delete(code); }
  }
  // expand / collapse every group (not Large-type's subclusters): the old Coarse / Fine
  // switch, still pressed by the tour and by ?label=
  function applyLabeling(which) {
    TR.groups.forEach((gr, g) => setOpen(g, which === "subtypes"));
    s.selected = -1; hideInspector();
    composeCodes();
  }

  // Called when the full tier (with genes) replaces the lite tier.
  function setData(data) {
    DATA = data;
    shapeTree(); composeCodes();   // a new tier may bring kctype, and its cells are new ones
    _umapCloud = null;   // full tier brings real UMAP coords → rebuild the cloud
    _mk = _mkTarget = null; _mkIdx = -1; _paintSig = null;
    // Different tier, different cells: every cached "nothing changed" answer is about the
    // old index space and has to be thrown away, or the prevalence would stay the preview's.
    _prevSig = "";
    syncUmapToggle();    // the full tier is where umapx arrives, if it arrives at all
    if (_umapOn) toggleUmapPanel(true);
    $("gene-hint").style.display = "none";
    if (s.mode === 3 && s.gene >= 0) selectGene(s.gene);  // now that genes exist
    scheduleCastePrev();
    if (!_urlRead && (DATA.n >= (M.n || DATA.n) || BIMP.tier === "lite")) readUrl();   // cell and gene indices are the full tier's
    syncTier();
  }

  // ---- the address: every piece of state this file owns, read once and kept current ----
  // Keys (short and stable; boot.js and histology.js own theirs — camera, sections, stains):
  //   caste  worker | both            (queen = absent)       read by boot.js applyUrlParams
  //   color  region | gene | conf     (cell types = absent). NOT "view": histology.js writes
  //          view=2d for its 2D mode; boot.js still reads an old link's view=gene etc.
  //   gene   gene SYMBOL, else its id  (a bare number is the old index form, still read)
  //   open   expanded tree rows, by code: "4.0.14" (codes index this export's manifest)
  //   hide   hidden tree rows, by code, same form
  //   reg    region mask (bits of manifest.regionDefs)
  //   cell   selected cell, a full-tier index
  //   cells  0 = the Cells layer off
  //   umap   1 = the UMAP panel on
  // Nothing is written until the address has been read (after the full tier is in), so the
  // start-up state can never overwrite a link someone pasted.
  let _urlRead = false;
  function geneIndexOf(v) {
    if (v == null || v === "") return -1;
    if (/^\d+$/.test(v)) return +v < (M.genes || []).length ? +v : -1;
    const i = geneIndexBySymbol(v);
    return i >= 0 ? i : (M.genes || []).findIndex((g) => g.id === v);
  }
  function readUrl() {
    _urlRead = true;
    const P = new URLSearchParams(location.search);
    const codesOf = (k) => (P.get(k) || "").split(".").filter((x) => /^\d+$/.test(x)).map(Number).filter((n) => n < MAXC);
    if (P.has("open") || P.has("hide")) {
      codesOf("open").sort((a, b) => a - b).forEach((c) => setOpen(c, true));   // groups before Large-type
      codesOf("hide").forEach((c) => { s.visible[c] = 0; });
      composeCodes();
    }
    const col = P.get("color");
    if (col) { const b = document.querySelector(`#mode-seg [data-mode="${col}"]`); if (b && b.getAttribute("aria-pressed") !== "true") b.click(); }
    if (P.has("reg")) { s.regionMask = (parseInt(P.get("reg"), 10) || 0) & 0xff; syncRegionUI(); }
    const gi = geneIndexOf(P.get("gene"));
    if (gi >= 0 && s.mode === 3) selectGene(gi);
    let ci = P.has("cell") ? parseInt(P.get("cell"), 10) : -1;
    if (ci >= 0 && DATA.fullIndex && DATA.n < (M.n || DATA.n)) ci = DATA.fullIndex.indexOf(ci);   // -1: not in the sample
    if (ci >= 0 && ci < DATA.n) {
      s.spin = false; const sp = $("tg-spin"); if (sp) sp.setAttribute("aria-pressed", "false");
      inspect(ci);
    }
    if (P.get("umap") === "1" && DATA.umapx) toggleUmapPanel(true);
    updateStat();
    writeUrl();
  }
  function writeUrl() {
    const U = BIMP.url;
    if (!_urlRead || !U || !TR) return;
    U.set("caste", s.casteFilter === 1 ? "worker" : s.casteFilter < 0 ? "both" : null);
    U.set("color", { 2: "conf", 3: "gene", 4: "region" }[s.mode] || null);
    const g = s.mode === 3 && s.gene >= 0 && M.genes ? M.genes[s.gene] : null;
    U.set("gene", g ? (g.symbol || g.id) : null);
    U.set("open", opened.size ? [...opened].sort((a, b) => a - b).join(".") : null);
    const hid = [];
    const walk = (nd) => { if (nd.kind && opened.has(nd.code)) nd.kids.forEach(walk); else if (s.visible[nd.code] < 0.5) hid.push(nd.code); };
    TR.order.forEach(walk);
    U.set("hide", hid.length ? hid.join(".") : null);
    U.set("reg", s.regionMask || null);
    // always the FULL-tier index, so a link from a phone's sample opens the same cell anywhere
    const fi = DATA.fullIndex && DATA.n < (M.n || DATA.n) && s.selected >= 0 ? DATA.fullIndex[s.selected] : s.selected;
    U.set("cell", s.selected >= 0 ? fi : null);
    U.set("cells", s.cellsOn === false ? "0" : null);
    U.set("umap", _umapOn ? "1" : null);
    // replaced by the keys above: an old link's label/region/select are read once, then dropped
    U.set("label", null); U.set("region", null); U.set("select", null);
  }

  function seg(id, cb) {
    const el = $(id);
    el.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => {
        el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        cb(b);
      }));
  }
  function buildModeSeg() {
    seg("mode-seg", (b) => {
      // mode 1 colours by the hand-drawn anatomy (flat caste colours would repeat what the
      // Queen/Worker/Both filter shows). The shader's caste branch stays so ?mode=caste
      // links keep working.
      s.mode = { celltype: 0, caste: 1, conf: 2, gene: 3, region: 4, ltkc: 5 }[b.dataset.mode];
      setTimeout(writeUrl, 0);   // after selectGene below has picked the gene
      const geneMode = s.mode === 3, ramp = s.mode === 2 || s.mode === 3;
      $("gene-wrap").style.display = geneMode ? "flex" : "none";
      $("leg-body").style.display = ramp ? "none" : "block";
      refreshLegend();
      $("colorbar").style.display = ramp ? "flex" : "none";
      $("cb-cap").textContent = s.mode === 3
        ? (s.gene >= 0 ? "expression · " + geneLabel(M.genes[s.gene]) : "select a gene →")
        : "Tangram mapping confidence";
      // same scale, two meanings — point its ⓘ at whichever one is on screen
      $("cb-info").dataset.info = s.mode === 3 ? "genes" : "confidence";
      $("leg-title").textContent = ["Cell types", "Caste", "Confidence", "Gene", "Regions", "Cell types"][s.mode];
      // the header's ⓘ explains whatever the legend shows; on a ramp the scale carries its own
      const li = $("leg-info");
      li.hidden = ramp;
      li.dataset.info = ["labeling", "caste", "confidence", "genes", "regions", "ltkc"][s.mode];
      scheduleCastePrev();       // shows in gene mode, hides itself everywhere else
      if (geneMode && s.gene < 0) selectGene(0);
    });
  }
  function buildLabelSeg() {
    seg("label-seg", (b) => applyLabeling(b.dataset.v));
  }
  function buildCasteSeg() {
    seg("caste-seg", (b) => {
      const prev = s.casteFilter;
      s.casteFilter = { both: -1, queen: 0, worker: 1 }[b.dataset.v];
      // "both" splits the two brains side-by-side, so the camera has to pull back for them
      // to fit. EASE it over the same time the split itself takes — assigning the radius
      // outright made queen→both lurch: the zoom snapped while the brains were still sliding
      // apart. Skipped while a scripted tour flight owns the camera, so the tour still wins.
      if (camera && camera.flyTo && !(camera.isFlying && camera.isFlying())) {
        if (s.casteFilter < 0 && prev >= 0) camera.flyTo({ radius: Math.max(camera.cam.radius, FIT.both) }, 550);
        else if (s.casteFilter >= 0 && prev < 0) camera.flyTo({ radius: FIT.one }, 550);
      }
      s.selected = -1; hideInspector(); updateStat();
      $("caste-hint").classList.toggle("show", s.casteFilter < 0);   // only meaningful side by side
    });
  }

  // ---- region multi-select (hand-labeled anatomical masks) ----
  // Bits from manifest.regionDefs; toggling ORs/clears the bit in s.regionMask
  // (0 = all). Grouped by structure (Mushroom body / Optic lobe).
  // Same colour the shader paints each calyx compartment with (the SI figures' palette).
  const kcSwatch = (key) => palette.KC_REGION[key] || null;
  function buildRegionPicker() {
    const wrap = $("region-list");
    const groups = {};
    M.regionDefs.forEach((d) => { (groups[d.group] = groups[d.group] || []).push(d); });
    Object.keys(groups).forEach((group) => {
      const gh = document.createElement("div"); gh.className = "reg-group";
      gh.textContent = palette.regionGroupLabel(group); gh.title = "Toggle all in " + palette.regionGroupLabel(group);
      gh.addEventListener("click", () => {
        const bits = groups[group].reduce((m, d) => m | (1 << d.bit), 0);
        const allOn = groups[group].every((d) => s.regionMask & (1 << d.bit));
        s.regionMask = allOn ? (s.regionMask & ~bits) : (s.regionMask | bits);
        syncRegionUI(); updateStat();
      });
      // a one-mask group headed by its own name ("Antennal lobe" over Antennal lobe) says nothing
      const solo = groups[group].length === 1 && palette.regionLabel(groups[group][0]) === palette.regionGroupLabel(group);
      if (!solo) wrap.appendChild(gh);
      groups[group].forEach((d) => {
        const it = document.createElement("label"); it.className = "reg-item"; it.dataset.bit = d.bit;
        const sw = kcSwatch(d.key);   // the three KC compartments are shaded apart in the view
        it.innerHTML = `<span class="reg-box"></span>` +
          (sw ? `<span class="reg-dot" style="background:${sw}"></span>` : "") +
          // under "Kenyon cell compartments" the word compartment is already said once
          `<span class="reg-nm" title="${palette.regionLabel(d)}">${palette.regionLabel(d).replace(/ compartment$/, "")}</span>` +
          `<span class="reg-ct">${(d.count || 0).toLocaleString()}</span>`;
        it.addEventListener("click", (e) => {
          e.preventDefault();
          s.regionMask ^= (1 << d.bit);
          syncRegionUI(); updateStat();
        });
        wrap.appendChild(it);
      });
    });
    $("region-all").addEventListener("click", () => { s.regionMask = 0; syncRegionUI(); updateStat(); });
    syncRegionUI();
  }
  function syncRegionUI() {
    if (s.mode === 4) syncRegionLegend();
    document.querySelectorAll("#region-list .reg-item").forEach((it) => {
      const on = (s.regionMask & (1 << +it.dataset.bit)) !== 0;
      it.classList.toggle("on", on);
    });
    $("region-all").classList.toggle("on", s.regionMask === 0);
  }

  function slider(id, fmt, cb) {
    const wrap = $(id), inp = wrap.querySelector("input"), val = wrap.querySelector(".val");
    const upd = () => {
      const v = parseFloat(inp.value); val.textContent = fmt(v);
      inp.style.setProperty("--fill", ((v - inp.min) / (inp.max - inp.min) * 100) + "%");
      cb(v);
    };
    inp.addEventListener("input", upd); upd();
  }
  function buildSliders() {
    slider("sl-size", (v) => v.toFixed(1), (v) => (s.size = v));
    slider("sl-z", (v) => v.toFixed(1) + "×", (v) => (s.z = v));
    slider("sl-op", (v) => v.toFixed(2), (v) => (s.opacity = v));
  }
  function buildUmapToggle() {
    const el = $("tg-umap");
    const toggle = () => toggleUmapPanel(!_umapOn);
    el.addEventListener("click", toggle);
    el.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } });
    syncUmapToggle();
  }
  // Only the Pages build exports per-cell UMAP coordinates; the single-file artifact's
  // payload leaves them out to stay under the size limit. Without them this toggle is a
  // control that visibly does nothing — it flips, and no panel ever appears — and since the
  // glossary it also carries an ⓘ describing something you cannot get to. Hide it in builds
  // that have no UMAP rather than leave a dead switch on screen. Re-checked in setData
  // because the lite tier arrives first and the coordinates come with the full one.
  function syncUmapToggle() {
    const el = $("tg-umap");
    if (el) el.style.display = DATA && DATA.umapx ? "" : "none";
  }
  function buildSpin() {
    const el = $("tg-spin");
    const toggle = () => { s.spin = !s.spin; el.setAttribute("aria-pressed", String(s.spin)); };
    el.addEventListener("click", toggle);
    el.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } });
    el.setAttribute("aria-pressed", String(s.spin));
  }

  // ---- legend: the tree in type view, a key in the others ----
  const fmt = (n) => (n || 0).toLocaleString();
  // what the tree holds, said in words: groups, and the named fine types under them at every
  // level (vote types, the restricted Kenyon types, S1-S3), not counting Other or remainders
  function treeCounts() {
    let g = 0, t = 0;
    const walk = (nd) => { if (!nd.resid || nd.raw === "Glia") t++; nd.kids.forEach(walk); };
    TR.order.forEach((gr) => { if (gr.raw === "Mixed") return; g++; if (gr.kind) gr.kids.forEach(walk); else t++; });
    return { g, t };
  }
  function legendSummary() {
    const el = $("leg-sum"); if (!el) return;
    if (s.mode === 0 && TR) {
      const c = treeCounts();
      el.textContent = TR.flat ? c.t + " types" : c.g + " groups · " + c.t + " types";
      $("stat-types").textContent = String(c.t);
    } else if (s.mode === 4) el.textContent = (M.regionDefs || []).length + " masks";
    else el.textContent = "";
  }
  // motion helpers: the branch being opened right now (only it unfolds; a rebuild must not
  // replay every open branch), the legend's last mode (it cross-fades only when that changes)
  let _enterCode = -1, _legMode = -1;
  const REDUCED = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion:reduce)").matches);
  function refreshLegend() {
    if (_legMode !== s.mode) {
      const lb = $("leg-body");
      if (lb && _legMode !== -1) { lb.classList.remove("swap"); void lb.offsetWidth; lb.classList.add("swap"); }
      _legMode = s.mode;
    }
    if (s.mode === 4) buildRegionLegend(); else buildTreeLegend();
    legendSummary(); syncShowAll(); updatePeekKey();
  }
  function rowHTML(color, name, count, hollow) {
    return `<span class="dot${hollow ? " hollow" : ""}" style="--c:${color}"></span>` +
      `<span class="nm" title="${name.replace(/<[^>]*>/g, "").replace(/"/g, "&quot;")}">${name}</span>` + (count != null ? `<span class="ct">${fmt(count)}</span>` : "");
  }
  // A row is a filter (click), a preview (hover isolates), and — for a row with types under
  // it — a branch (the chevron, or → / ← on the keyboard).
  function treeRow(nd, depth) {
    const open = !!nd.kind && opened.has(nd.code);
    const r = document.createElement("div");
    r.className = "node d" + depth + (open ? " open" : "") + (nd.resid ? " resid" : "");
    r.tabIndex = 0; r.setAttribute("role", "treeitem"); r.dataset.code = nd.code;
    if (nd.kind) r.setAttribute("aria-expanded", String(open));
    r.innerHTML = (nd.kind
      ? `<button class="chev" tabindex="-1" aria-label="${open ? "Collapse" : "Expand"} ${nd.name}"></button>`
      : `<span class="chev none"></span>`) + rowHTML(nd.color, nd.name, nd.count, open);
    const toggle = () => {
      if (open) setDeep(nd, anyVis(nd) ? 0 : 1); else s.visible[nd.code] = s.visible[nd.code] > 0.5 ? 0 : 1;
      syncTreeOff(); updateStat();
    };
    const expand = (on) => {
      const done = () => {
        setOpen(nd.code, on); _enterCode = on ? nd.code : -1; composeCodes(); _enterCode = -1;
        const el = document.querySelector(`#leg-body .node[data-code="${nd.code}"]`); if (el) el.focus();
      };
      // folding: let the branch fold back into its row first, then rebuild without it
      const kids = !on && r.nextElementSibling && r.nextElementSibling.classList.contains("kids") ? r.nextElementSibling : null;
      if (kids && !REDUCED()) {
        // fold to nothing (height and fade), so the rows below rise with it instead of jumping
        kids.style.height = kids.offsetHeight + "px"; void kids.offsetHeight;
        kids.classList.add("closing"); kids.style.height = "0px";
        setTimeout(done, 170);
      } else done();
    };
    r.addEventListener("click", (e) => {
      if (e.target.closest(".chev:not(.none)")) { expand(!open); return; }
      if (e.target.closest("a,[data-info]")) return;
      toggle();
    });
    r.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      else if (nd.kind && e.key === "ArrowRight" && !open) { e.preventDefault(); expand(true); }
      else if (nd.kind && e.key === "ArrowLeft" && open) { e.preventDefault(); expand(false); }
    });
    // an expanded row has no single code to isolate, so it previews nothing
    r.addEventListener("mouseenter", () => { s.iso = open ? -1 : nd.code; });
    r.addEventListener("mouseleave", () => { s.iso = -1; });
    return r;
  }
  function syncTreeOff() {
    document.querySelectorAll("#leg-body .node[data-code]").forEach((el) => {
      const nd = TR.byCode[+el.dataset.code]; if (nd) el.classList.toggle("off", !anyVis(nd));
    });
  }
  function noteRow(html) { const d = document.createElement("div"); d.className = "tree-note"; d.innerHTML = html; return d; }
  function appendNode(parent, nd, depth) {
    parent.appendChild(treeRow(nd, depth));
    if (!nd.kind || !opened.has(nd.code)) return;
    const kids = document.createElement("div"); kids.className = "kids" + (nd.code === _enterCode ? " enter" : ""); kids.setAttribute("role", "group");
    if (nd.kind === "kc") kids.appendChild(noteRow(`Restricted transfer · compartments only<button class="info-dot" data-info="kctypes" aria-label="About the restricted Kenyon transfer">ⓘ</button>`));
    if (nd.kind === "lt") kids.appendChild(noteRow(`Large-type subclusters · strongest per cell<button class="info-dot" data-info="ltkc" aria-label="About the Large-type subclusters">ⓘ</button>`));
    nd.kids.forEach((k) => appendNode(kids, k, depth + 1));
    parent.appendChild(kids);
  }
  function buildTreeLegend() {
    const body = $("leg-body"); body.innerHTML = "";
    if (!TR) return;
    const tree = document.createElement("div"); tree.className = "tree"; tree.setAttribute("role", "tree");
    TR.order.forEach((gr) => appendNode(tree, gr, 0));
    body.appendChild(tree);
    syncTreeOff();
    // the branch just opened unfolds to its height, pushing the rows below down smoothly
    const ent = tree.querySelector(".kids.enter");
    if (ent && !REDUCED()) {
      const h = ent.scrollHeight;
      ent.style.height = "0px"; void ent.offsetHeight; ent.style.height = h + "px";
      setTimeout(() => { ent.style.height = ""; }, 230);
    }
  }
  // Region view: the eight masks in the figures' colours. Clicking one isolates it,
  // exactly like the region picker (they stay in sync).
  function buildRegionLegend() {
    const body = $("leg-body"); body.innerHTML = "";
    (M.regionDefs || []).forEach((d) => {
      const el = document.createElement("div");
      el.className = "node"; el.dataset.regbit = d.bit; el.tabIndex = 0;
      el.innerHTML = rowHTML(palette.KC_REGION[d.key] || "#8A93A6", palette.regionLabel(d), d.count);
      const go = () => { s.regionMask ^= (1 << d.bit); syncRegionUI(); syncRegionLegend(); updateStat(); };
      el.addEventListener("click", go);
      el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      body.appendChild(el);
    });
    syncRegionLegend();
  }
  function syncRegionLegend() {
    document.querySelectorAll("#leg-body .node[data-regbit]").forEach((c) => {
      c.classList.toggle("off", s.regionMask !== 0 && !(s.regionMask & (1 << +c.dataset.regbit)));
    });
  }
  // "Show all" (in the legend header) is there only while something is filtered out; the
  // rest of the time that corner says how many groups and types the legend holds
  function syncShowAll() {
    const a = $("leg-all"); if (!a) return;
    let off = false;
    if (s.mode === 4) off = s.regionMask !== 0;
    else if (s.mode === 0 && TR) off = TR.order.some((gr) => (opened.has(gr.code) ? gr.kids : [gr]).some((n) => s.visible[n.code] < 0.5));
    a.hidden = !off;
  }
  // "Show all" (in the legend header): every type back on, or every region in region view
  function showAll() {
    if (s.mode === 4) { s.regionMask = 0; syncRegionUI(); refreshLegend(); updateStat(); return; }
    s.visible.fill(1);
    refreshLegend();
    updateStat();
  }
  document.addEventListener("click", (e) => { if (e.target && e.target.id === "leg-all") showAll(); });
  document.addEventListener("keydown", (e) => {
    if (e.target && e.target.id === "leg-all" && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); showAll(); }
  });

  // ---- layers: Cells and Histology, each one click or one key (C / H) ----
  // Histology is BIMP.histo's (src/histology.js): this only presses its setOn and listens to
  // onChange, which also fires when its manifest arrives (available) — so the button
  // appears only in a build that ships sections.
  function buildLayers() {
    const bc = $("ly-cells"), bh = $("ly-histo");
    if (!bc || !bh) return;
    const H = () => BIMP.histo || BIMP.histology;
    const syncH = () => {
      const h = H(), ok = !!(h && h.available);
      bh.hidden = !ok;
      bh.setAttribute("aria-pressed", String(!!(ok && h.isOn())));
      // its settings (stain, strength, section) belong next to its switch; histology.js finds
      // them by id, so moving the node is enough. The panel's own switch then duplicates this
      // one and is hidden in CSS (#controls #histo-grp).
      document.body.classList.toggle("histo-on", !!(ok && h.isOn()));
      sumH();
    };
    const setCells = (on) => {
      s.cellsOn = on; bc.setAttribute("aria-pressed", String(on));
      if (!on) { s.hover = -1; hideTip(); if (s.selected >= 0) inspect(-1); }
      scheduleShown();
      if (R.invalidate) R.invalidate();   // cellsOn is not in the renderer's redraw signature
    };
    // ?cells=0 deep-links the stain alone; the switch keeps the address in step (replaceState,
    // so it never adds a history entry)
    bc.addEventListener("click", () => { setCells(s.cellsOn === false); writeUrl(); });
    if (new URLSearchParams(location.search).get("cells") === "0") setCells(false);
    bh.addEventListener("click", () => { const h = H(); if (h && h.available) h.setOn(!h.isOn()); syncH(); });
    // the disclosure's one line: what is shown and how strongly, readable in the peek
    const sumH = () => {
      const el = $("lh-sum"), h = H(); if (!el || !h || !h.available) return;
      const labs = (h.getStains ? h.getStains() : []).map((k) => {
        const l = document.querySelector(`.hs-chip[data-key="${k}"] .hs-lab`); return l ? l.textContent : k;
      });
      const op = h.getOpacity ? h.getOpacity() : null;
      el.innerHTML = `<span>Histology · <b>${(labs.join(", ") || "DAPI").replace(/</g, "&lt;")}</b>` + (op != null ? ` · strength <span class="m">${op.toFixed(2)}</span>` : "") + `</span>`;
    };
    const lh = $("layers-histo");
    if (lh) {
      lh.addEventListener("input", sumH); lh.addEventListener("click", () => setTimeout(sumH, 0));
      if (PHONE && PHONE.matches) lh.open = false;   // phones: one line until asked
      const onMq = () => { lh.open = !(PHONE && PHONE.matches); };
      if (PHONE) PHONE.addEventListener ? PHONE.addEventListener("change", onMq) : PHONE.addListener(onMq);
      // opening it from the peek opens the sheet, so the settings have room
      lh.addEventListener("toggle", () => {
        const sh = $("controls");
        if (lh.open && PHONE && PHONE.matches && sh && !sh.classList.contains("sheet-open")) {
          sh.classList.add("sheet-open"); document.body.classList.add("sheet-open");
        }
      });
    }
    let tries = 0;
    const hook = () => {
      const h = H();
      if (h && h.onChange) { h.onChange(syncH); syncH(); return; }
      if (++tries < 60) setTimeout(hook, 500);
    };
    hook();
    document.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test((e.target || {}).tagName || "")) return;
      const k = e.key.toLowerCase();
      if (k === "c") { e.preventDefault(); bc.click(); }
      else if (k === "h" && !bh.hidden) { e.preventDefault(); bh.click(); }
    });
  }

  // ---- the phone sheet: peek ↔ open, by the handle (drag or tap) or a swipe on the peek ----
  // boot.js owns #sheet-close / #sheet-toggle and the "sheet-open" classes; this adds the drag
  // and the peek height. The peek ends under Color by, and its height is published as
  // --sheet-peek so the histology rail can sit above it.
  const PHONE = window.matchMedia ? window.matchMedia("(max-width:600px)") : null;
  function buildSheet() {
    const sh = $("controls"), hd = $("sheet-hd");
    if (!sh || !hd || !PHONE) return;
    const isOpen = () => sh.classList.contains("sheet-open");
    const setOpenSheet = (on) => {
      sh.classList.toggle("sheet-open", on); document.body.classList.toggle("sheet-open", on);
      if (!on) sh.scrollTop = 0;
    };
    const measure = () => {
      if (!PHONE.matches) return;
      const pk = $("peek-key"), mode = $("mode-seg"), ls = $("lh-sum");
      const top = sh.getBoundingClientRect().top;
      let end = document.body.classList.contains("histo-on") && ls && ls.offsetParent ? ls : (pk && pk.offsetParent ? pk : mode);
      // budget: the peek is a quarter-ish of the screen. Caste outranks colour, so when both do
      // not fit (a short phone), the peek ends after Queen | Worker | Both and Color by sits
      // just below the fold
      if (end && end.getBoundingClientRect().bottom - top + 12 > 0.3 * innerHeight) {
        const hint = $("caste-hint"); end = hint && hint.offsetParent && hint.classList.contains("show") ? hint : $("caste-seg");
      }
      const peek = end ? Math.ceil(end.getBoundingClientRect().bottom - top + 12) : 160;
      document.documentElement.style.setProperty("--sheet-peek", peek + "px");
      // the first measurement releases the sheet (CSS keeps it off-screen until then); a frame
      // later, so the new peek is applied before it starts to move
      if (!document.documentElement.classList.contains("sheet-ready"))
        requestAnimationFrame(() => document.documentElement.classList.add("sheet-ready"));
      phoneFit();
    };
    // 2D/3D (histology.js's #view-mode) joins the layer row on a phone, so it lives in the peek
    let vmHome = null;
    const placeViewMode = () => {
      const vm = $("view-mode"), row = document.querySelector("#layers .layer-row");
      if (!vm || !row) return false;
      if (PHONE.matches) { if (vm.parentNode !== row) { vmHome = vmHome || { p: vm.parentNode, n: vm.nextSibling }; row.appendChild(vm); } }
      else if (vmHome && vm.parentNode === row) vmHome.p.insertBefore(vm, vmHome.n);
      return true;
    };
    let vmTries = 0;
    const waitVm = () => { if (!placeViewMode() && ++vmTries < 80) setTimeout(waitVm, 500); else measure(); };
    waitVm();
    const onMq = () => { placeViewMode(); measure(); if (!PHONE.matches) setOpenSheet(false); };
    PHONE.addEventListener ? PHONE.addEventListener("change", onMq) : PHONE.addListener(onMq);
    window.addEventListener("resize", measure);
    if (window.ResizeObserver) { const ro = new ResizeObserver(measure); ro.observe($("mode-seg")); ro.observe($("caste-grp")); }
    const h = BIMP.histo || BIMP.histology; if (h && h.onChange) h.onChange(() => requestAnimationFrame(measure));
    // drag: from the handle always; from anywhere on the peek while closed (a tap there is
    // still a tap — the drag starts only past 8 px of vertical travel)
    let drag = null;
    const yClosed = () => sh.getBoundingClientRect().height - (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sheet-peek")) || 160);
    sh.addEventListener("pointerdown", (e) => {
      if (!PHONE.matches) return;
      const onHd = !!e.target.closest("#sheet-hd") && !e.target.closest("#sheet-close");
      if (!onHd && isOpen()) return;                     // open: the content scrolls
      drag = { id: e.pointerId, y0: e.clientY, t0: performance.now(), base: isOpen() ? 0 : yClosed(), live: false, onHd };
    });
    sh.addEventListener("pointermove", (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      const dy = e.clientY - drag.y0;
      if (!drag.live) {
        if (Math.abs(dy) < 8) return;
        drag.live = true; sh.classList.add("dragging");
        try { sh.setPointerCapture(e.pointerId); } catch (err) { /* already released */ }
      }
      const y = Math.max(0, Math.min(yClosed(), drag.base + dy));
      sh.style.transform = `translateY(${y}px)`;
      e.preventDefault();
    });
    const end = (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      const d = drag; drag = null;
      sh.classList.remove("dragging"); sh.style.transform = "";
      if (!d.live) { if (d.onHd && e.type === "pointerup") setOpenSheet(!isOpen()); return; }   // tap on the handle
      const dy = e.clientY - d.y0, v = dy / Math.max(1, performance.now() - d.t0);           // px per ms
      const y = d.base + dy, open = Math.abs(v) > 0.4 ? v < 0 : y < yClosed() / 2;
      setOpenSheet(open);
      // the click that would follow a drag must not press whatever it ended on
      const swallow = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
      sh.addEventListener("click", swallow, { capture: true, once: true });
      setTimeout(() => sh.removeEventListener("click", swallow, { capture: true }), 50);
    };
    sh.addEventListener("pointerup", end);
    sh.addEventListener("pointercancel", end);

    // the inspector is the sheet's content on a phone (CSS: body.inspecting), so it lives in it
    let inHome = null;
    const placeInspector = () => {
      const ins = $("inspector"); if (!ins) return;
      if (PHONE.matches) { if (ins.parentNode !== sh) { inHome = inHome || { p: ins.parentNode, n: ins.nextSibling }; sh.insertBefore(ins, hd.nextSibling); } }
      else if (inHome && ins.parentNode === sh) inHome.p.insertBefore(ins, inHome.n);
    };
    placeInspector();
    PHONE.addEventListener ? PHONE.addEventListener("change", placeInspector) : PHONE.addListener(placeInspector);

    // fit the brain to a portrait screen, and centre it between the header and the peek
    const fit = () => { phoneFit(); };
    fit(); window.addEventListener("resize", fit);
    // re-centre when what covers the bottom changes (inspector, tour, sheet)
    new MutationObserver(() => requestAnimationFrame(() => requestAnimationFrame(fit)))
      .observe(document.body, { attributes: true, attributeFilter: ["class"] });
    PHONE.addEventListener ? PHONE.addEventListener("change", fit) : PHONE.addListener(fit);

    // the one top-right control
    const ob = $("ovf-btn"), om = $("ovf-menu");
    if (ob && om) {
      const show = (on) => { om.hidden = !on; ob.setAttribute("aria-expanded", String(on)); };
      ob.addEventListener("click", (e) => { e.stopPropagation(); show(om.hidden); });
      document.addEventListener("click", (e) => { if (!om.hidden && !om.contains(e.target)) show(false); });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") show(false); });
      om.addEventListener("click", (e) => {
        const b = e.target.closest("[data-act]"); if (!b) return;
        const act = b.dataset.act;
        if (act === "tour") { show(false); if (BIMP.tour) BIMP.tour.start(); }
        else if (act === "reset") { show(false); const r = $("btn-reset"); if (r) r.click(); }
        else if (act === "home") { show(false); const h = document.querySelector("#masthead .home"); if (h) h.click(); }
        else if (act === "share") {
          const done = () => { b.textContent = "Link copied"; setTimeout(() => { b.textContent = "Copy link to this view"; show(false); }, 900); };
          if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(location.href).then(done, done); else done();
        }
      });
    }
  }
  // Camera distances. Desktop values are the ones the viewer always used; on a portrait phone
  // the horizontal field is the limit, so every distance scales with how much narrower the
  // screen is (the brain is about twice as wide as it is tall), and "Both" stacks the brains
  // (queen above) instead of putting them side by side, where each got a third of the width.
  const FIT = { home: 4.2, one: 2.7, both: 4.6 };
  let _fitF = 1;
  function phoneFit() {
    const ph = !!(PHONE && PHONE.matches), W = innerWidth, H = innerHeight, portrait = ph && H > W * 1.15;
    const f = ph ? Math.max(1, 0.6 / (W / H)) : 1;
    FIT.home = 4.2 * f; FIT.one = 2.7 * f; FIT.both = portrait ? 3.9 * f : 4.6 * f;
    if (camera.HOME) camera.HOME.radius = FIT.home;
    if (f !== _fitF && camera.cam) { camera.cam.radius *= f / _fitF; _fitF = f; }
    if (R.setSplit) R.setSplit(portrait ? 0.62 : 1.15, portrait ? [0, -1] : [1, 0]);
    // lens shift (NDC, up): centre the brain between the header and whatever covers the
    // bottom — the peek, the inspector, or the tour card (an OPEN sheet is a place you work
    // in, so the brain is left where the peek put it)
    const hdH = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--hd")) || 48;
    let cover = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sheet-peek")) || 0;
    const ins = $("inspector"), tour = $("tour");
    if (ph && document.body.classList.contains("inspecting") && ins) cover = Math.max(cover, H - $("controls").getBoundingClientRect().top);
    if (ph && document.body.classList.contains("touring") && tour) cover = Math.max(cover, H - tour.getBoundingClientRect().top);
    if (R.setViewShift) R.setViewShift(ph ? Math.max(0, (cover - hdH) / H) : 0);
    const ch = $("caste-hint");
    if (ch && ch.children.length === 2) {
      ch.children[0].textContent = portrait ? "▲ queen" : "◀ queen";
      ch.children[1].textContent = portrait ? "worker ▼" : "worker ▶";
    }
  }

  // ---- phones: what the colours mean, in one line under the mode tabs ----
  function updatePeekKey() {
    const el = $("peek-key"); if (!el || !M) return;
    let html = "";
    const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    if (s.mode === 0 && TR) {
      html = TR.order.map((gr) => `<span class="k" style="--c:${gr.color}"><i></i>${esc(gr.name)}</span>`).join("");
    } else if (s.mode === 3) {
      const g = s.gene >= 0 && M.genes ? M.genes[s.gene] : null;
      html = g ? `<b>${esc(geneLabel(g))}</b><span class="bar"></span><span>${g.kind === "inferred" ? "inferred" : "measured"}</span>` : "pick a gene ↑";
    } else if (s.mode === 2) html = `<span>mapping confidence</span><span class="bar"></span>`;
    else if (s.mode === 4) html = (M.regionDefs || []).slice(0, 4).map((d) => `<span class="k" style="--c:${palette.KC_REGION[d.key] || "#8A93A6"}"><i></i>${esc(palette.regionLabel(d).replace(/ compartment$/, ""))}</span>`).join("") + "<span>…</span>";
    else html = "";
    if (el._html !== html) { el._html = html; el.innerHTML = html; }   // re-render (and re-fade) only on a real change
  }

  // ---- the lite tier: say so, and offer the rest (boot.js BIMP.loadFullDetail) ----
  // The View panel's row was the only place that said so, folded away (and the count under the
  // stage is hidden on phones), so a phone showed a sample with nothing saying it was one. A chip
  // on the stage now says it, with the way to load the rest.
  function syncTier() {
    const row = $("tier-row"); if (!row) return;
    const lite = BIMP.tier === "lite" && typeof BIMP.loadFullDetail === "function";
    row.hidden = !lite;
    if (lite) $("tier-msg").textContent = "Showing a " + (DATA.n || 0).toLocaleString() + "-cell sample" + (BIMP.tierWhy ? " (" + BIMP.tierWhy + ")" : "");
    let chip = $("tier-chip");
    if (lite && !chip) {
      chip = document.createElement("div"); chip.id = "tier-chip";
      chip.innerHTML = '<span class="tc-msg"></span><button type="button" class="tc-load"></button>';
      ($("app") || document.body).appendChild(chip);
      chip.querySelector(".tc-load").addEventListener("click", () => { const b = $("tier-load"); if (b) b.click(); });
    }
    if (chip) {
      chip.hidden = !lite;
      if (lite) {
        chip.querySelector(".tc-msg").textContent = (DATA.n || 0).toLocaleString() + "-cell sample";
        const HB = window.__BIMP_HERO__, mb = HB && HB.baseBytes ? " (" + Math.round(HB.baseBytes / 1e6) + " MB)" : "";
        const b = $("tier-load"), busy = b && b.getAttribute("aria-busy") === "true";
        chip.querySelector(".tc-load").textContent = busy ? "loading…" : "Load all " + ((M.n || 0) / 1e6).toFixed(1) + "M" + mb;
      }
    }
  }
  function buildTier() {
    const b = $("tier-load"); if (!b) return;
    b.textContent = "Load all " + (M.n || 0).toLocaleString();
    b.addEventListener("click", () => {
      if (b.getAttribute("aria-busy") === "true") return;
      b.setAttribute("aria-busy", "true"); const label = b.textContent; b.textContent = "loading…"; syncTier();
      BIMP.loadFullDetail().then(() => { syncTier(); }, (e) => {
        b.textContent = "Try again"; $("tier-msg").textContent = (e && e.message) || "Couldn't load all cells";
      }).finally(() => { b.removeAttribute("aria-busy"); if (b.textContent === "loading…") b.textContent = label; });
    });
    // boot.js clears the selection for the swap and restores it as a full-tier index after:
    // show that cell again (inspector rows and the address both follow)
    window.addEventListener("bimp:tier", () => { syncTier(); if (s.selected >= 0) inspect(s.selected); else writeUrl(); });
    syncTier();
  }

  // ---- layout: the legend's place depends on the screen ----
  // Desktop: the left pane, directly under the title (its top follows the title's height).
  // Tablet and phone have no left pane, so the legend joins the controls column / bottom
  // sheet, right under the colour mode it explains. It used to be hidden there outright,
  // which also hid the gene colour bar and the per-caste prevalence that live inside it.
  const NARROW = window.matchMedia ? window.matchMedia("(max-width:820px)") : null;
  function placeLegend() {
    const leg = $("legend"), ctl = $("controls"), app = $("app");
    if (!leg || !ctl || !app) return;
    if (NARROW && NARROW.matches) { const after = $("gene-wrap"); if (leg.previousElementSibling !== after) ctl.insertBefore(leg, after.nextSibling); }
    else if (leg.parentNode !== app) app.insertBefore(leg, $("inspector"));
  }
  if (NARROW) (NARROW.addEventListener ? NARROW.addEventListener("change", placeLegend) : NARROW.addListener(placeLegend));
  function watchTitle() {
    const t = $("title"); if (!t) return;
    const set = () => document.documentElement.style.setProperty("--title-h", Math.round(t.getBoundingClientRect().height) + "px");
    set();
    if (window.ResizeObserver) new ResizeObserver(set).observe(t);
  }

  // ---- gene picker ----
  function geneLabel(g) { return g.symbol || g.id.replace("bimp_v4.3_", "bimp·"); }
  // the gene's annotation id as the OGS writes it (BIMP_v4.3_002131), shown under its name
  function geneIdText(g) { return String(g.id).replace(/^bimp_v/i, "BIMP_v"); }
  // the row's description, readable: the bee annotation without isoform suffixes or the
  // name-builder's working notes, and the fly gene, when there is one, in plain words at the end
  function geneDesc(g) {
    let d = String(g.desc || "")
      .replace(/\s*isoform\s+[\w-]+$/i, "")
      .replace(/;\s*closest fly gene.*$/i, "")   // the name-builder's non-reciprocal guess stays out of the row
      .replace(/^unnamed in the bee annotation;\s*fly ortholog\s+[^()]*(\(([^)]*)\))?.*$/i, (m, p, name) => name ? name : "unnamed")
      .replace(/\s*\(fly [^)]*\)\s*/i, " ")   // a curated name's note on its fly ortholog: the fly gene follows anyway
      .trim();
    if (d.toLowerCase() === (g.symbol || "").toLowerCase()) d = "";
    // the fly gene only where the bee annotation names nothing: there it is the one clue to what
    // the gene is (named genes stay findable by their fly names in search)
    const unnamed = /^(unnamed in the bee annotation|uncharacterized)/i.test(String(g.desc || "")) || !g.symbol;
    const fly = unnamed && g.fly && (g.fly.toLowerCase() !== (g.symbol || "").toLowerCase()) ? `fly gene ${g.fly}` : "";
    return [d, fly].filter(Boolean).join(" · ");
  }
  function geneIndexBySymbol(sym) {
    const gs = M.genes || []; const t = String(sym).toLowerCase();
    for (let i = 0; i < gs.length; i++) if ((gs[i].symbol || "").toLowerCase() === t) return i;
    return -1;
  }
  // ---- gene details modal (opened by the ⓘ on each row; row-click still colors) ----
  function openGeneModal(i) {
    const g = M.genes[i]; if (!g) return;
    $("gm-kind").textContent = g.kind === "inferred"
      ? (g.flat ? "inferred gene · low spatial signal" : "inferred gene") : "measured gene";
    $("gm-name").textContent = geneLabel(g);
    $("gm-desc").textContent = g.desc || "—";
    const rows = [
      ["bimp id", g.id],
      ["symbol", g.symbol || "—"],
      ["source", g.kind === "inferred" ? "Tangram-inferred (not directly measured)" : "measured · Xenium panel"],
      ["expression max", g.hi != null ? g.hi.toFixed(1) + " (99.5th pct)" : "—"],
      ["spatial variance", g.var != null ? g.var.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—"],
    ];
    if (g.kind === "inferred") rows.push(["spatial signal", g.flat ? "low (near-flat)" : "resolved"]);
    $("gm-rows").innerHTML = rows.map(([k, v]) =>
      `<div class="gm-row"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
    $("gm-color").onclick = () => {
      closeGeneModal();
      const gb = document.querySelector('#mode-seg [data-mode="gene"]'); if (gb) gb.click();
      selectGene(i);
    };
    $("gene-modal").classList.add("show");
  }
  function closeGeneModal() { $("gene-modal").classList.remove("show"); }

  // ---- glossary (the ⓘ next to each section heading) ----
  // The panel is dense with terms a visitor has no reason to know — Tangram, mapping
  // confidence, inferred genes — and most of them were only ever explained in a title=
  // tooltip, which is invisible on touch and easy to miss on a mouse. Each entry says
  // what the thing IS and, where it matters, what it does NOT license you to conclude.
  const INFO = {
    atlas: { kind: "how this was built", name: "Xenium &amp; Tangram", body: `
      <p><b>Xenium</b> is an imaging-based spatial method: it counts transcripts for a fixed
      panel of <b>100 genes</b> inside intact tissue sections, so every cell keeps the position
      it had in the brain. What it does not give you is the rest of the transcriptome.</p>
      <p><b>Tangram</b> closes that gap. It aligns a dissociated single-nucleus RNA-seq
      reference onto the spatial data, learning where in the section each reference nucleus
      most likely sits. That alignment is what carries <b>cell-type labels</b> — and expression
      for genes the panel never measured — onto the Xenium cells you see here.</p>
      <p>Sections were then stacked back into the volume, giving <b>1,300,192</b> placed cells.</p>` },

    colormodes: { kind: "color by", name: "The four color modes", body: `
      <p><b>Type</b> — the cell type, by cluster vote over the Tangram calls (see the ⓘ on the legend).</p>
      <p><b>Region</b> — which hand-drawn anatomical mask the cell falls inside.</p>
      <p><b>Conf.</b> — how decisive Tangram's mapping was for that cell. See the ⓘ on the
      scale that appears under the legend in this mode.</p>
      <p><b>Gene</b> — expression of one gene, measured or inferred.</p>` },

    confidence: { kind: "color scale", name: "Mapping confidence", body: `
      <p>A per-cell score from Tangram describing <b>how decisively that cell's measured
      profile matched a single reference cell type</b>. In this atlas it runs from
      <b>0.10 to 1.00</b>. Low means the 100-gene panel could not cleanly separate the
      candidate types for that cell.</p>
      <div class="cav">It describes the <b>mapping</b>, not the truth. A confidently mapped
      cell can still be labelled wrongly — for instance if its real type is absent from the
      reference. Read low confidence as "the panel could not tell", not as "this cell is odd".</div>` },

    labeling: { kind: "cell types", name: "Groups and fine-scale types", body: `
      <p>Cell types are decided per <i>cluster</i>, not per cell, from expression alone: a vote
      over the Tangram calls of each cluster's cells, with no anatomy involved.</p>
      <p>Each row is one of the manuscript's seven <b>coarse groups</b> (Kenyon cells, projection
      neurons, other neurons, photoreceptors, glia, hemocytes, optic lobe cells); groups that
      never win a cluster vote on this panel do not appear. Open a group (▸) to see the
      <b>fine-scale types</b> the vote named inside it, where the data decide one: the type holds
      at least half the cluster, or leads the runner-up 1.5-fold. The rest stays at the group,
      shown as its remainder, e.g. <i>Glia (unresolved)</i>. A cluster undecided at both levels is
      <b>Other</b>: a quality-control outcome, not a cell type.</p>
      <p>Colour follows what is open: a closed group is one colour, an open one shows its types.
      Click a row to hide or show it; hover to see it alone.</p>
      <div class="cav">Kenyon cell subtypes are never voted: the 101-gene panel carries too few of
      their markers to name them per cluster. Where the export includes it, opening
      <b>Kenyon cells</b> shows a separate, restricted transfer instead. See the ⓘ there.</div>` },

    kctypes: { kind: "Kenyon cell types", name: "The restricted transfer", body: `
      <p>Tangram was run again on the cells inside the three hand-drawn <b>Kenyon cell
      compartments</b> only, against just the five named snRNA-seq Kenyon populations
      (Large-type, Small-type, Class&nbsp;II, Middle-type and other Kenyon cells), as in the
      manuscript's Kenyon cell figure, panel&nbsp;B. Each compartment cell gets its most probable
      population; the inspector shows that call with its renormalised probability.</p>
      <p>Here the rows split the cells the cluster vote calls <i>Kenyon cells</i>. Those outside
      the three compartments were not reclassified and are listed as such.</p>
      <div class="cav">Every compartment cell is forced onto one of the five, so a confident call
      is not by itself evidence of identity. What the manuscript rests on is that each type
      dominates the compartment of its name in both brains (58–74% of its cells). Read single cells
      with that in mind.</div>` },

    ltkc: { kind: "Large-type subclusters", name: "S1–S3", body: `
      <p>Three subclusters found de novo within the snRNA-seq <i>Large-type</i> Kenyon cells
      (the Kenyon-cell subtype figure), projected by Tangram onto the Kenyon-compartment cells. Each
      Large-type cell here is shown as the subcluster its projection weights most; the inspector
      gives that share.</p>
      <div class="cav">The projection is a probability spread, not a call, and many cells are
      close to even between two subclusters: a cell's colour is its strongest subcluster, not a
      firm identity. The manuscript reports the caste comparison on this projection as
      descriptive only.</div>` },

    // Counted from the manifest, and the mapping sentence follows manifest.inferredMapping,
    // which the export sets to "canonical" once the inferred genes come from the same Tangram
    // run as the cell types (gene-projection session). Until then they come from an earlier run.
    genes: { kind: "gene panel", name: "Measured vs inferred", body: () => {
      const gs = M.genes || [], nm = gs.filter((g) => g.kind !== "inferred").length, ni = gs.length - nm;
      const same = M.inferredMapping === "canonical";
      return `
      <p><span class="gg gg-m"></span> <b>Measured</b> (${nm.toLocaleString()} genes) were counted directly
      in the tissue by the Xenium panel. These are observations.</p>
      <p><span class="gg gg-i"></span> <b>Inferred</b> (${ni.toLocaleString()} genes) were <b>never measured
      here</b>. Tangram predicted them for each cell from the snRNA-seq reference, ${same
        ? "through the same Tangram mapping that assigned the cell types"
        : "from an earlier Tangram mapping of the same reference, not the one that assigned the cell types"}.</p>
      <div class="cav">An inferred gene is a prediction, not a measurement, and it inherits
      every assumption of the mapping. It is at its most trustworthy for genes with sharp,
      cell-type-specific expression, and at its weakest for subtle or graded patterns.
      Genes marked with a <b>dashed ring</b> <span class="gg gg-f"></span> carry little spatial signal at all.</div>`; } },

    caste: { kind: "caste", name: "Queen and worker", body: `
      <p><i>Bombus impatiens</i> queens and workers are the same species and largely the same
      genome; the interest is in what differs between them. <b>Both</b> places the two brains
      side by side — queen on the left, worker on the right.</p>
      <div class="cav">This atlas contains <b>one queen brain and one worker brain</b>. The
      caste axis is therefore also the individual axis: any difference you see here mixes
      caste with the ordinary variation between two bees. Treat contrasts in this viewer as
      something to look into, not as a result.</div>` },

    regions: { kind: "anatomical region", name: "The hand-drawn masks", body: `
      <p>Eight structures were outlined by hand on the sections — the three <b>Kenyon cell
      compartments</b> of the mushroom body, each named for the population that dominates it
      (Large-type = non-compact, Small-type = inner compact, Class&nbsp;II = outer compact),
      the three <b>optic lobe</b> neuropils (lamina, medulla, lobula), the <b>antennal
      lobe</b>, and the <b>central complex</b>.</p>
      <p>These are drawn boundaries, not derived from the expression data, which is what makes
      them a useful independent check on where a cell type landed.</p>
      <div class="cav">The masks cover only these eight structures. A cell in none of them is
      not unclassified — it simply sits outside the regions that were outlined.</div>` },

    prevalence: { kind: "caste comparison", name: "Cells expressing", body: `
      <p>The share of cells that carry this gene above the colour-window floor, computed
      separately for each brain. <b>Shares, not counts</b>, so the fact that one brain
      contributed more cells than the other cancels out. Both rows always refer to the whole
      brain even when you are only <i>looking</i> at one caste, and both follow the region
      and cell-type filters — so you can ask the question inside the mushroom body alone.</p>
      <p>This row exists because <b>the colours cannot answer it</b>. Across the 100 measured
      genes, a caste difference is almost entirely a difference in how many cells carry the
      transcript, not in how much they carry: among the cells that do express it, the two
      castes' averages differ by about a percent. Rescaling the colour window rescales both
      brains identically, so no setting of it makes a caste difference appear.</p>
      <div class="cav">This atlas contains <b>one queen brain and one worker brain</b>, so a
      ratio here mixes caste with the ordinary difference between two individual bees. It is
      a description of these two brains, not a tested result. See the ⓘ beside <b>Caste</b>.</div>` },

    umap: { kind: "umap", name: "The UMAP panel", body: `
      <p>A 2D embedding of the cells' own <b>measured 100-gene profiles</b>: cells that sit
      near each other here have similar expression. The marker follows whichever cell you
      point at, so you can watch a run through the anatomy trace a path through expression
      space — and see where the two disagree.</p>
      <div class="cav">UMAP position is not anatomy, and the distance between two far-apart
      clusters carries no meaning. Only local neighbourhoods are interpretable.</div>` },
  };
  function openInfo(key) {
    const e = INFO[key]; if (!e) return;
    $("if-kind").textContent = e.kind;
    $("if-name").innerHTML = e.name;
    $("if-body").innerHTML = typeof e.body === "function" ? e.body() : e.body;
    $("if-body").scrollTop = 0;
    $("info-modal").classList.add("show");
    $("if-close").focus();
  }
  function closeInfo() { $("info-modal").classList.remove("show"); }

  const FOCAL = new Set(["oct-β2R", "FTZ-F1β", "oct-β1R", "GlyRα1", "HSP90", "CaMKII", "ERR1", "sNPF", "Kr-h1"]);
  const GENE_CAP = 60;   // render at most this many rows (2000+ genes → search-driven)
  // the search text of every gene, built once: the list is filtered on each keystroke and the
  // gene set runs to thousands, so the string building must not happen per keystroke
  // Common names people search by that the annotation does not carry. SEARCH TEXT ONLY: the
  // labels shown stay the export's. Keyed by gene id; add more here.
  const GENE_ALIASES = {
    "bimp_v4.3_004765": "tbh tyramine beta-hydroxylase tyrosine beta hydroxylase tβh",   // labelled dopamine beta-hydroxylase
    "bimp_v4.3_009248": "allatostatin ast",
    "bimp_v4.3_000471": "allatotropin at",
    "bimp_v4.3_009257": "tk tachykinin",
    "bimp_v4.3_005730": "snpf short neuropeptide f",
  };
  // label, description, aliases, and the id with and without its "bimp_v4.3_" prefix, so
  // "009248" finds bimp_v4.3_009248; all lower-cased (the query is too)
  const hayOf = (x) => (geneLabel(x) + " " + (x.desc || "") + " " + (GENE_ALIASES[x.id] || "") + " " +
    (x.aliases || []).join(" ") + " " + (x.fly || "") + " " +
    x.id + " " + String(x.id).replace(/^bimp_v[\d.]+_/i, "")).toLowerCase();
  let _hay = null;
  function geneHay(g, i) {
    if (!_hay || _hay.length !== (M.genes || []).length) _hay = (M.genes || []).map(hayOf);
    return i != null ? _hay[i] : hayOf(g);
  }
  // how well a gene answers a query, for ordering the hits (lower is better): its own name,
  // then an exact alias or fly ortholog (export's gene_names.py: Met, gce, mblk-1...), then a
  // name starting with the query, then a word in its text that does, then any substring.
  // Without it "met" lists hundreds of methyltransferases ahead of Met itself.
  const namesOf = (x) => [geneLabel(x), ...(x.aliases || []), ...String(x.fly || "").split("/"),
    x.id, String(x.id).replace(/^bimp_v[\d.]+_/i, "")]
    .map((t) => String(t).toLowerCase());
  let _names = null;
  function geneRank(i, q) {
    const genes = M.genes || [];
    if (!_names || _names.length !== genes.length) _names = genes.map(namesOf);
    const nm = _names[i];
    if (nm[0] === q) return 0;
    if (nm.includes(q)) return 1;
    if (nm.some((t) => t.startsWith(q))) return 2;
    return (" " + geneHay(genes[i], i)).includes(" " + q) ? 3 : 4;
  }
  // where a gene's values come from, as one glyph before its name (key under the filter):
  // filled = measured on the Xenium panel, ring = Tangram-inferred, dashed ring = inferred
  // with little spatial signal. The full wording is the row's tooltip.
  function geneSource(g) {
    if (g.kind !== "inferred") return { cls: "gg-m", say: "measured on the Xenium panel" };
    return g.flat ? { cls: "gg-f", say: "Tangram-inferred, low spatial signal" } : { cls: "gg-i", say: "Tangram-inferred (not measured here)" };
  }
  function geneRow(g, i) {
    const it = document.createElement("div"); it.className = "gene-item"; it.dataset.i = i;
    it.setAttribute("aria-selected", String(i === s.gene));
    const sub = geneDesc(g), src = geneSource(g);
    it.innerHTML =
      `<i class="gg ${src.cls}" title="${src.say[0].toUpperCase() + src.say.slice(1)}"></i>` +
      `<div class="gi-main"><span class="gi-name">${geneLabel(g)}${FOCAL.has(g.symbol) ? `<span class="rank" title="discussed in the manuscript">★</span>` : ""}</span>` +
      `<span class="gi-id">${geneIdText(g)}</span>` +
      (sub ? `<span class="gi-desc">${sub}</span>` : "") + `</div>` +
      `<button class="gi-info" tabindex="-1" aria-label="Gene details">ⓘ</button>`;
    // no tooltip over the whole row: it repeated the name, id and description already shown; the
    // source wording sits on the glyph alone
    it.addEventListener("click", () => selectGene(i));
    it.addEventListener("mouseenter", () => prefetchGene(i));   // smart prefetch on hover
    it.querySelector(".gi-info").addEventListener("click", (e) => { e.stopPropagation(); openGeneModal(i); });
    return it;
  }
  let geneFilter = "all";   // "all" · "measured" · "inferred"
  function geneMatchesFilter(g) {
    if (geneFilter === "measured") return g.kind !== "inferred";
    if (geneFilter === "inferred") return g.kind === "inferred";
    return true;
  }
  function renderGeneList(q) {
    const list = $("gene-list"); list.innerHTML = "";
    const genes = M.genes || [];
    // indices passing filter+query
    const idxs = [];
    for (let i = 0; i < genes.length; i++) {
      if (!geneMatchesFilter(genes[i])) continue;
      if (q && !geneHay(genes[i], i).includes(q)) continue;
      idxs.push(i);
    }
    // On the empty "All" view, surface some inferred genes so their badges are
    // visible up front (otherwise the 100 measured genes fill the whole cap).
    let order = idxs;
    if (q) {
      const rank = new Map(idxs.map((i) => [i, geneRank(i, q)]));
      order = idxs.slice().sort((a, b) => rank.get(a) - rank.get(b) || a - b);
    }
    if (!q && geneFilter === "all") {
      const meas = idxs.filter((i) => genes[i].kind !== "inferred");
      const inf = idxs.filter((i) => genes[i].kind === "inferred");
      order = meas.slice(0, 40).concat(inf.slice(0, 20)).concat(meas.slice(40));
    }
    const shown = Math.min(order.length, GENE_CAP);
    for (let k = 0; k < shown; k++) list.appendChild(geneRow(genes[order[k]], order[k]));
    const more = $("gene-more");
    more.textContent = idxs.length > shown ? `${shown} of ${idxs.length.toLocaleString()} — search or use the filter to see more` : "";
    more.style.display = idxs.length > shown ? "block" : "none";
  }
  function buildGenePicker() {
    // measured / inferred / all filter (inferred genes — and their badges — are only
    // discoverable via this, since the 100 measured genes head the list).
    seg("gene-filter", (b) => { geneFilter = b.dataset.v; renderGeneList($("gene-search").value.toLowerCase().trim()); });
    // color-scale max slider: user override of the auto window (kills the wash-out
    // when a few high-prediction cells dominate an inferred gene)
    // Both ends of the colour window are adjustable: the floor hides low-expression cells,
    // the ceiling stops a few high cells washing everyone else out. Double-click either to
    // go back to the automatic window.
    const scHi = $("cb-hi"), scLo = $("cb-lo");
    const GAP = 0.04;   // keeps the two thumbs far enough apart to grab, on a phone too
    const onDrag = (e) => {
      userScale = true;
      let lo = parseFloat(scLo.value), hi = parseFloat(scHi.value);
      if (lo > hi - GAP) {                       // the dragged thumb stops at the other one
        if (e && e.target === scLo) lo = Math.max(0, hi - GAP); else hi = Math.min(1, lo + GAP);
      }
      setScaleUI(hi, false, lo);
    };
    scHi.addEventListener("input", onDrag);
    scLo.addEventListener("input", onDrag);
    const toAuto = () => { userScale = false; setScaleUI(geneAutoHi, true, 0); };
    scHi.addEventListener("dblclick", toAuto);
    scLo.addEventListener("dblclick", toAuto);
    renderGeneList("");
    let t = null;
    $("gene-search").addEventListener("input", (e) => {
      clearTimeout(t);
      const q = e.target.value.toLowerCase().trim();
      t = setTimeout(() => {
        renderGeneList(q);
        // smart prefetch: if the search narrows to a single gene, start loading it
        const genes = M.genes || [];
        if (q.length >= 2) {
          const hits = [];
          for (let i = 0; i < genes.length && hits.length < 2; i++) if (geneMatchesFilter(genes[i]) && geneHay(genes[i], i).includes(q)) hits.push(i);
          if (hits.length === 1) prefetchGene(hits[0]);
        }
      }, 90);
    });
  }
  // fetch a gene's data (fetch mode) without selecting it — warms the cache on hover
  // / autocomplete so click is instant.
  const _prefetching = new Set();
  function prefetchGene(i) {
    const g = M.genes[i];
    if (!g || !g.file || M.mode !== "fetch" || !BIMP.fullReady) return;
    if ((DATA.genes && DATA.genes[i]) || _prefetching.has(i)) return;
    _prefetching.add(i);
    decode.fetchGene((M.genesBase || window.__BIMP_SRC__) + g.file, M.gzip, { bits: g.bits, n: M.n }).then((arr) => {
      (DATA.genes || (DATA.genes = []))[i] = arr;
      if (s.gene === i && s.mode === 3) { R.setGene(arr); $("gene-hint").style.display = "none"; }
    }).catch(() => {}).finally(() => _prefetching.delete(i));
  }
  function selectGene(i) {
    s.gene = i;
    userScale = false;   // each gene re-adopts its own auto scale
    document.querySelectorAll("#gene-list .gene-item").forEach((it) =>
      it.setAttribute("aria-selected", String(+it.dataset.i === i)));
    const g = M.genes[i];
    const tag = g.kind === "inferred" ? (g.flat ? " · inferred (low signal)" : " · inferred") : "";
    $("cb-cap").textContent = "expression · " + geneLabel(g) + tag;
    updatePeekKey(); writeUrl();
    const hint = $("gene-hint");
    if (DATA.genes && DATA.genes[i]) {
      applyGene(DATA.genes[i]); hint.style.display = "none";
    } else if (M.mode === "fetch" && M.genes[i].file && BIMP.fullReady) {
      // GitHub-Pages fetch mode: lazy-load this gene's file (matches the full 1.3M
      // base), then cache + apply.
      hint.style.display = "block"; hint.textContent = "loading gene…";
      decode.fetchGene((M.genesBase || window.__BIMP_SRC__) + M.genes[i].file, M.gzip, { bits: M.genes[i].bits, n: M.n }).then((arr) => {
        (DATA.genes || (DATA.genes = []))[i] = arr;
        if (s.gene === i) { applyGene(arr); hint.style.display = "none"; }
      }).catch(() => { hint.textContent = "gene load failed"; });
    } else { hint.style.display = "block"; hint.textContent = "loading full atlas — gene expression shortly…"; }
  }
  // Upload the gene + adopt the auto color-scale max (unless the user pinned it).
  let userScale = false, geneAutoHi = 1, geneLoUser = 0;   // colour-window floor (user-set)
  function applyGene(arr) {
    const r = R.setGene(arr);
    geneAutoHi = (r && r.hi) || 1;
    if (!userScale) setScaleUI(geneAutoHi, true, 0);
    scheduleCastePrev();
  }

  // ---- prevalence per caste ------------------------------------------------------
  // The colour window cannot answer "how does this gene differ between the castes", and no
  // choice of lo/hi ever will, because the difference is not in the colour. Over the
  // measured genes, a caste difference in the per-caste MEAN is carried by prevalence —
  // the fraction of cells carrying any transcript — while the mean among EXPRESSING
  // cells barely moves.
  // A shared window rescales both halves identically; a per-caste window would rescale them
  // to look the SAME and erase the signal outright. So the honest display is the number.
  //
  // Percentages, not counts, so the 686k/614k difference in brain size cancels. The
  // denominator is every cell currently passing the region mask and the cell-type
  // filters — the same predicates the shader uses (see visOf/uVisible in shaders.js) minus
  // the caste filter, since hiding one caste must not empty its own row.
  let _prevSig = "", _prevTimer = 0;
  function scheduleCastePrev() {
    // Debounced: this is a full pass over 1.3M cells, and dragging the colour floor would
    // otherwise fire one per input event. Never runs from the frame loop.
    if (_prevTimer) clearTimeout(_prevTimer);
    _prevTimer = setTimeout(() => { _prevTimer = 0; updateCastePrev(); }, 90);
  }
  function updateCastePrev() {
    const box = $("cb-caste");
    if (!box) return;
    const arr = s.mode === 3 && s.gene >= 0 && DATA.genes ? DATA.genes[s.gene] : null;
    if (!arr || !DATA.caste || !DATA.n) { box.classList.remove("show"); return; }
    box.classList.add("show");
    const cc = activeCodes();
    const sig = [s.gene, s.regionMask, geneLoUser.toFixed(3), codesVer,
      Array.prototype.join.call(s.visible, "")].join("|");
    if (sig === _prevSig) return;
    _prevSig = sig;
    // "Expressing" means above the colour-window FLOOR, not merely nonzero, so the number
    // keeps describing what is actually lit on screen as you raise the floor. At the
    // default floor of 0 the two coincide.
    const thr = geneLoUser * 255;
    const rm = s.regionMask, vis = s.visible, caste = DATA.caste, reg = DATA.region;
    let dQ = 0, eQ = 0, dW = 0, eW = 0;
    for (let i = 0, n = DATA.n; i < n; i++) {
      if (rm !== 0 && (reg[i] & rm) === 0) continue;
      if (vis[cc[i]] < 0.5) continue;
      if (caste[i] === 0) { dQ++; if (arr[i] > thr) eQ++; }
      else { dW++; if (arr[i] > thr) eW++; }
    }
    const pQ = dQ ? eQ / dQ : 0, pW = dW ? eW / dW : 0, top = Math.max(pQ, pW, 1e-9);
    const pct = (p, d) => (d ? (p * 100).toFixed(1) + "%" : "—");
    $("cbc-q").textContent = pct(pQ, dQ);
    $("cbc-w").textContent = pct(pW, dW);
    $("cbc-q-bar").style.width = (100 * pQ / top) + "%";
    $("cbc-w-bar").style.width = (100 * pW / top) + "%";
    const note = $("cbc-note");
    const where = rm !== 0 ? ", within the selected regions" : "";
    const hi = pQ >= pW ? "queen" : "worker";
    if (!dQ || !dW) {
      note.innerHTML = "no cells of one caste in this selection";
    } else if (!eQ && !eW) {
      note.innerHTML = "not detected in this selection";
    } else if (!eQ || !eW) {
      // One side is exactly zero, so a ratio would divide by nothing and print a number
      // with no meaning. Say what actually happened instead.
      note.innerHTML = "detected only in the <b>" + hi + "</b>" + where;
    } else {
      const ratio = Math.max(pQ, pW) / Math.min(pQ, pW);
      // Deliberately phrased as a count of cells, not as "expression": that is what was
      // measured. The n=1 caveat lives in the ⓘ beside the heading.
      note.innerHTML = ratio < 1.05
        ? "about the same share of cells in both brains"
        : "<b>" + ratio.toFixed(1) + "×</b> the share of cells in the " + hi + where;
    }
  }
  function setScaleUI(hi, auto, lo) {
    lo = Math.max(0, Math.min(lo == null ? geneLoUser : lo, hi - 0.04));
    geneLoUser = lo;
    $("cb-lo").value = lo; $("cb-hi").value = hi;
    const fill = document.querySelector("#cb-scale .r2-fill");
    if (fill) { fill.style.left = (lo * 100) + "%"; fill.style.width = ((hi - lo) * 100) + "%"; }
    $("cb-scale").querySelector(".val").textContent =
      auto ? "auto" : (lo > 0 ? lo.toFixed(2) + " – " : "≤ ") + hi.toFixed(2);
    // the colourbar's end labels show the window actually in force
    const g = s.gene >= 0 && M.genes ? M.genes[s.gene] : null;
    const unit = (v) => (g ? (v * g.hi).toFixed(1) : v.toFixed(2));
    const loLab = $("cb-lo-lab"); if (loLab) loLab.textContent = lo > 0 ? unit(lo) : "low";
    $("cb-hi-lab").textContent = auto ? "high" : unit(hi);
    R.setGeneRange(lo, hi);
    scheduleCastePrev();   // "expressing" is defined against the floor, so it moves with it
  }

  // ---- stat + inspector + tooltip ----
  // The headline "cells mapped" is the full-atlas total — a fixed fact about the
  // dataset, NOT a live filtered count (filtering to Queen etc. shouldn't make the
  // atlas look smaller than it is).
  // the per-cell codes the renderer is drawing (see the tree, above)
  function activeCodes() { return codes && codes.length === DATA.n ? codes : (DATA.ctCanon || DATA.ct); }
  function updateStat() {
    $("stat-n").textContent = (M.nTotal || M.n).toLocaleString();
    syncShowAll(); scheduleShown(); writeUrl();
    // every caller of this is a change to the region mask or the cell-type filters, which
    // are exactly the predicates the prevalence denominator is built from
    scheduleCastePrev();
  }

  // className() returns the DATA name of what the cell is shown as (inspect() compares it
  // to the raw call); shownClass() is the same in display wording.
  // the HUD's one figure: cells on screen, after every filter (the total when nothing is off)
  let _shownT = 0;
  // Counted from a histogram over (caste, region byte, code), filled during composeCodes' own
  // pass over the cells. A filter change then sums ~16k bins instead of walking 1.3M cells:
  // the walk took ~34 ms and landed inside the caste dissolve as a dropped frame.
  let _hist = null;
  function shownHist() { if (!_hist) _hist = new Uint32Array(2 * 256 * MAXC); else _hist.fill(0); return _hist; }
  function scheduleShown() { countShown(); }
  function countShown() {
    const el = $("hud-cells"); if (!el || !DATA || !DATA.n || !_hist) return;
    if (s.cellsOn === false) { el.textContent = "cells hidden"; return; }
    const vis = s.visible, rm = s.regionMask, cf = s.casteFilter, H = _hist;
    let n = 0;
    for (let c = 0; c < 2; c++) {
      if (cf >= 0 && c !== cf) continue;
      for (let r = 0; r < 256; r++) {
        if (rm !== 0 && (r & rm) === 0) continue;
        const base = (c * 256 + r) * MAXC;
        for (let k = 0; k < MAXC; k++) if (vis[k] >= 0.5) n += H[base + k];
      }
    }
    const scale = (M.n || DATA.n) / DATA.n;
    // a sample's count is scaled up to the whole atlas: say it is an estimate
    el.textContent = (BIMP.tier === "lite" ? "≈ " : "") + Math.round(n * scale).toLocaleString() + " cells" + (BIMP.tier === "lite" ? " (from a sample)" : "");
  }
  function className(idx) { return TR.raws[activeCodes()[idx]]; }
  function shownClass(idx) { return TR.names[activeCodes()[idx]]; }
  const tip = () => $("tip");
  function showTip(idx, px, py) {
    if (idx < 0) { tip().style.opacity = 0; return; }
    $("tip-ct").textContent = DATA.caste[idx] === 0 ? "QUEEN" : "WORKER";
    $("tip-nm").textContent = shownClass(idx);
    tip().style.left = px + "px"; tip().style.top = py + "px"; tip().style.opacity = 1;
  }
  function hideTip() { tip().style.opacity = 0; }
  function hideInspector() { $("inspector").classList.remove("show"); syncInspectorLane(); }
  // The inspector and the controls panel share the right-hand column. Since the UMAP
  // plot made the inspector tall, publish its height so the controls can cap themselves
  // to the space left over (see `body.inspecting #controls`) instead of being overlapped.
  function syncInspectorLane() {
    const el = $("inspector"), on = el.classList.contains("show");
    const h = on ? Math.round(el.getBoundingClientRect().height) : 0;
    document.body.classList.toggle("inspecting", on);
    document.documentElement.style.setProperty("--insp-h", h + "px");
  }
  function inspect(idx) {
    if (idx < 0) { s.selected = -1; hideInspector(); writeUrl(); return; }
    s.selected = idx; setTimeout(writeUrl, 0);
    const caste = DATA.caste[idx];
    const conf = M.confRange[0] + (DATA.conf[idx] / 255) * (M.confRange[1] - M.confRange[0]);
    $("i-type").textContent = shownClass(idx);
    // raw per-cell Tangram call (noisy argmax) shown for transparency when it
    // differs from the displayed class.
    const raw = DATA.ctRaw ? (M.rawNames || M.cellTypes)[DATA.ctRaw[idx]] : null;
    const showRaw = raw && raw !== className(idx);
    $("i-subtype-row").style.display = showRaw ? "flex" : "none";
    if (showRaw) $("i-subtype").textContent = palette.displayName(raw, "types");
    // its group, whenever the row above names something finer than the group
    const showCanon = !TR.flat && DATA.ctCanon && activeCodes()[idx] >= TR.G;
    $("i-canon-row").style.display = showCanon ? "flex" : "none";
    if (showCanon) $("i-canon").textContent = TR.names[DATA.ctCanon[idx]];
    // the restricted Kenyon transfer (fig:kc_subtypes panel B), for any cell inside the compartments,
    // whether or not the legend is showing it: the call and its renormalised probability
    const kt = DATA.kctype ? DATA.kctype[idx] : 0, KT = M.kcTypes;
    const showKc = kt > 0 && KT && KT.names && kt <= KT.names.length;
    $("i-kc-row").style.display = showKc ? "flex" : "none";
    if (showKc) $("i-kc").textContent = ((KT.short || KT.names)[kt - 1]) +
      (DATA.kcprob ? " · " + (DATA.kcprob[idx] / 255).toFixed(2) : "");
    // and, for a Large-type call, fig:kc_subtypes' subcluster: the strongest of S1-S3 and its share
    const lk = M.ltkc, a = DATA.ltkc1, b = DATA.ltkc2, c = DATA.ltkc3;
    const sum = showKc && KT.names[kt - 1] === "Large-type Kenyon cells" && lk && a ? a[idx] + b[idx] + c[idx] : 0;
    $("i-lt-row").style.display = sum > 0 ? "flex" : "none";
    if (sum > 0) {
      const v = [a[idx], b[idx], c[idx]], j = v.indexOf(Math.max(...v));
      $("i-lt").textContent = lk.short[j] + " · " + (v[j] / sum).toFixed(2);
    }
    $("i-id").textContent = "#" + idx.toLocaleString();
    $("i-caste").innerHTML =
      `<span class="castetag" style="background:${caste === 0 ? "rgba(245,197,66,.18)" : "rgba(47,211,198,.18)"};` +
      `color:${caste === 0 ? "#FBE4A3" : "#B4F0EA"}">${caste === 0 ? "queen" : "worker"}</span>`;
    $("i-conf").textContent = conf.toFixed(3);
    // region membership of this cell
    const bits = DATA.region[idx];
    const regs = M.regionDefs.filter((d) => bits & (1 << d.bit)).map(palette.regionLabel);
    $("i-region").textContent = regs.length ? regs.join(", ") : "—";
    if (s.mode === 3 && s.gene >= 0 && DATA.genes && DATA.genes[s.gene]) {
      const g = M.genes[s.gene];
      $("i-gene-row").style.display = "flex";
      $("i-gene-k").textContent = geneLabel(g);
      $("i-gene").textContent = (DATA.genes[s.gene][idx] / 255 * g.hi).toFixed(2);
    } else $("i-gene-row").style.display = "none";
    drawUmap(idx);
    $("inspector").classList.add("show");
    requestAnimationFrame(syncInspectorLane);   // measure after the row show/hide relayout
  }

  // ---- UMAP mini-plot: where the picked cell sits in the atlas's own embedding ----
  // (uint8 per-cell coords in the full tier; the faint cloud is drawn once and cached.)
  const UMW = 216, UMH = 150, UMPAD = 7;
  let _umapCloud = null;
  // The live marker is animated: hover picks land ~90ms apart and in jumps, so the marker
  // eases toward its target every frame instead of teleporting between cells.
  let _mk = null, _mkTarget = null, _mkIdx = -1, _umapOn = false, _paintSig = null;
  function umapPt(i) {
    const ux = DATA.umapx[i] / 255, uy = DATA.umapy[i] / 255;
    return [UMPAD + ux * (UMW - 2 * UMPAD), UMPAD + (1 - uy) * (UMH - 2 * UMPAD)];   // flip Y
  }
  function buildUmapCloud() {
    const off = document.createElement("canvas"); off.width = UMW; off.height = UMH;
    const g = off.getContext("2d"); g.globalAlpha = 0.38;   // quiet: the context, not the subject
    const n = DATA.n, step = Math.max(1, Math.floor(n / 45000));   // cap the backdrop ~45k dots
    const cols = M.canonColors || [];
    for (let i = 0; i < n; i += step) {
      const p = umapPt(i);
      g.fillStyle = cols[DATA.ctCanon ? DATA.ctCanon[i] : 0] || "#8A93A6";
      g.fillRect(p[0], p[1], 1.1, 1.1);
    }
    return off;
  }
  // One cached bitmap of the embedding, and a single ring on the cell in question. Repainted
  // only when that cell changes: no per-frame easing (the glide cost a repaint every frame
  // and read as lag, not motion).
  function paintUmap(cv, at) {
    const g = cv.getContext("2d");
    if (!_umapCloud) _umapCloud = buildUmapCloud();
    g.clearRect(0, 0, UMW, UMH); g.globalAlpha = 1; g.drawImage(_umapCloud, 0, 0);
    if (at) {
      g.beginPath(); g.arc(at[0], at[1], 4.5, 0, 6.283);
      g.strokeStyle = "rgba(11,11,10,.85)"; g.lineWidth = 3; g.stroke();     // a dark halo so the ring reads on any colour
      g.strokeStyle = "#f7f8f6"; g.lineWidth = 1.25; g.stroke();
    }
  }
  function drawUmap(idx) {
    const box = $("umap-box"), cv = $("umap-mini");
    if (!cv || !DATA.umapx) { if (box) box.classList.remove("show"); return; }
    box.classList.add("show");
    // Paint once now so the panel is correct the instant the inspector opens; from here the
    // frame loop owns it (tickUmap) and the marker follows the pointer.
    _paintSig = null;
    paintUmap(cv, idx >= 0 ? umapPt(idx) : null);
  }
  // Is the inspector's copy on screen? It gets the same live marker as the standalone panel.
  // This is the one most visitors ever see — it appears the moment you click a cell, whereas
  // the panel below needs the Show-UMAP toggle — and it used to redraw ONLY on click, so
  // hovering around after clicking left it frozen on the cell you had already picked.
  // Gated on the INSPECTOR, not just the box: closing the inspector leaves #umap-box's own
  // .show in place (hideInspector never touches it), so checking the box alone would keep
  // this tracking the pointer into a canvas nobody can see.
  function miniOn() {
    const box = $("umap-box"), ins = $("inspector");
    return !!(DATA.umapx && box && ins && box.classList.contains("show") && ins.classList.contains("show"));
  }
  // ---- live UMAP: follows whatever the pointer is over, easing between cells ----
  function toggleUmapPanel(on) {
    _umapOn = on;
    const p = $("umap-panel");
    p.classList.toggle("show", on && !!DATA.umapx);
    document.body.classList.toggle("umap-on", on && !!DATA.umapx);   // the legend makes room
    if (on) document.documentElement.style.setProperty("--umap-h", Math.ceil(p.getBoundingClientRect().height) + "px");
    $("tg-umap").setAttribute("aria-pressed", String(on));
    writeUrl();
    if (on) { _paintSig = null; tickUmap(); }
  }
  // called every frame from the render loop; costs one comparison unless the cell changed
  function tickUmap() {
    updateScaleBar();
    if (!DATA.umapx) return;
    const mini = miniOn();
    if (!_umapOn && !mini) return;
    const idx = s.hover >= 0 ? s.hover : s.selected;
    const sig = idx + "|" + (_umapOn ? 1 : 0) + (mini ? 1 : 0);
    if (sig === _paintSig) return;
    _paintSig = sig;
    const nm = idx >= 0 ? shownClass(idx) : "hover a cell";
    $("umap-live-lab").textContent = nm;
    const ml = $("umap-mini-lab"); if (ml) ml.textContent = s.hover >= 0 ? nm : "selected";
    const at = idx >= 0 ? umapPt(idx) : null;
    if (_umapOn) paintUmap($("umap-live"), at);
    if (mini) paintUmap($("umap-mini"), at);
  }
  // ---- scale bar: µm per screen pixel at the orbit target's depth ----
  // The camera is a 48° perspective (renderer.js); a world unit is renderer.scaleUm() µm. At
  // the target's distance (the orbit radius) the visible height spans 2·r·tan(24°) units over
  // the canvas height, so that is the one depth where the bar is exact (said in its tooltip).
  const SB_STEPS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000];
  let _sbSig = "";
  function updateScaleBar() {
    const el = $("scalebar"), cv = $("gl");
    if (!el || !cv || !camera.cam || !R.scaleUm) return;
    const r = camera.cam.radius, H = cv.clientHeight || innerHeight;
    const sig = r.toFixed(4) + "|" + H;
    if (sig === _sbSig) return;
    _sbSig = sig;
    const umPerPx = (2 * r * Math.tan(24 * Math.PI / 180) / H) * R.scaleUm();
    const maxPx = innerWidth < 600 ? 90 : 120;
    let L = SB_STEPS[0];
    for (const v of SB_STEPS) if (v / umPerPx <= maxPx) L = v;
    el.style.setProperty("--sb-w", Math.round(L / umPerPx) + "px");
    el.querySelector(".sb-lab").textContent = L >= 1000 ? (L / 1000) + " mm" : L + " µm";
  }

  // rebuildTree: re-read manifest.kcTypes / data.kctype (an export or test that adds them late)
  function rebuildTree() { shapeTree(); composeCodes(); }
  function setGroupOpen(name, on) {
    const g = (M.canonTypes || []).indexOf(name); if (g < 0 || !TR) return false;
    setOpen(g, on !== false); composeCodes(); return opened.has(g);
  }
  function fitScale() { return FIT.home / 4.2; }
  return { fitScale, refit: phoneFit, build, setData, updateStat, showTip, hideTip, inspect, selectGene, syncRegionUI, applyLabeling, rebuildTree, setGroupOpen, geneIndexBySymbol, openGeneModal, prefetchGene, syncInspectorLane, tickUmap, openInfo, INFO };
})();
