/* Central view state + derived counts. Single source of truth; UI mutates it,
   renderer reads it each frame. */
window.BIMP = window.BIMP || {};
BIMP.state = (function () {
  const s = {
    mode: 0,            // 0 celltype · 1 caste · 2 conf · 3 gene
    labeling: "canonical", // "canonical" (coarse anatomy-vote) · "subtypes" (per-cluster subtype vote)
    casteFilter: 0,     // default queen (single caste, not overlaid) · -1 both · 1 worker
    regionMask: 0,      // 0 = all regions; else bitmask of selected hand-labeled regions
    gene: -1,           // index into manifest.genes when mode===3
    size: 2.6, z: 1.0, opacity: 0.85,   // z = 1× is true 10 µm scale (thin stack)
    spin: true,
    visible: new Float32Array(32).fill(1),
    iso: -1,            // hover-isolate a single cell type (transient preview)
    hover: -1,
    selected: -1,       // persistently highlighted picked cell (current tier index)
  };
  // Hover, held. Picking runs at ~11 Hz (a GPU pass is not free), and a pick whose ray
  // happens to land between two cells returns nothing. Writing that straight into s.hover
  // made the 3-D highlight, the tooltip and the UMAP marker all blink off and back on
  // several times a second while the pointer sat still over a dense part of the atlas.
  // A miss is only meaningful if it PERSISTS, so forgive a few in a row.
  //
  // Counted in PICKS, not milliseconds. Picking is what produces misses, so that is the
  // clock the forgiveness belongs on: a time-based hold expires between picks on any
  // machine whose frames are slower than the hold — which would make this BLINK MORE than
  // doing nothing on exactly the slow devices that need it most (measured: a headless run
  // at ~1 fps blanked 20 of 24 gaps with a 320 ms hold, and 0 with this).
  // pointerleave and drags clear it outright: there the pointer really has gone, and
  // holding would leave a stale highlight behind.
  function makeHoverHold(maxMisses) {
    const MAX = maxMisses == null ? 3 : maxMisses;
    let last = -1, misses = 0;
    return {
      report(idx) {
        if (idx >= 0) { last = idx; misses = 0; }
        else if (++misses > MAX) last = -1;
        return last;
      },
      get() { return last; },
      clear() { last = -1; misses = 0; },
    };
  }
  return { s, makeHoverHold };
})();
