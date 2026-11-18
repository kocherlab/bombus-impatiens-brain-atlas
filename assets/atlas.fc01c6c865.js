/* ---- palette.js ---- */
/* Palette — anatomy → hue family. Hue encodes real brain anatomy, not decoration.
   Kenyon cells = golds · optic lobe = cyans/blues · glia = violets ·
   neurons = greens · immune = coral. Caste: queen gold / worker teal. */
window.BIMP = window.BIMP || {};
BIMP.palette = (function () {
  const HEX = {
    0:"#6FD36A", 1:"#B06CE0", 2:"#B8641C", 3:"#8E5AD8", 4:"#FF7A6B", 5:"#F5C542",
    6:"#C77D1A", 7:"#4FBF7A", 8:"#FAD87A", 9:"#22D3EE", 10:"#2BB3E0", 11:"#3B95D6",
    12:"#4F7BE0", 13:"#37C6B0", 14:"#1FB2A6", 15:"#5AA9E6", 16:"#2FA8E8", 17:"#46C7D8",
    18:"#6D8FE8", 19:"#2CD1B4", 20:"#3FA0DC", 21:"#58D6C4", 22:"#9C7A3A", 23:"#9A79C4",
    24:"#98D98A", 25:"#D06AC8", 26:"#7BA9FF", 27:"#E8A22E", 28:"#6E4FD0",
    29:"#3E4552",   // "Mixed" — clusters below the subtype purity floor (dim slate)
  };
  // Grouping into anatomical families (ids are tangram_cell_type category order).
  const FAMILIES = [
    { name: "Mushroom body · Kenyon cells", ids: [5, 27, 6, 8, 2, 22] },
    { name: "Optic lobe",                   ids: [16, 17, 12, 11, 9, 10, 15, 20, 19, 13, 21, 14, 18, 26] },
    { name: "Glia",                         ids: [1, 3, 23, 25, 28] },
    { name: "Projection & other neurons",   ids: [7, 0, 24] },
    { name: "Immune",                       ids: [4] },
    { name: "Unresolved",                   ids: [29] },   // Mixed (subtype purity floor)
  ];
  function hexToRgb(h) {
    return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  }
  // The three Kenyon-cell calyx compartments, colored exactly as in the manuscript's
  // exploded-regions SI figure (scripts/regions/si_exploded_regions.py REGION_PALETTE), so the
  // viewer and the paper agree. Applied in cell-type view to cells inside each mask.
  const KC_REGION = {
    in_kc_ic: "#B07AA1", in_kc_nc: "#F28E2B", in_kc_oc: "#EDC948",
    in_lamina: "#59A14F", in_medulla: "#4E79A7", in_lobula: "#E15759",
    // Same swatches the paper uses for the two regions added in 2026-08/09
    // (nine_region_common.REGION_COLORS), so the viewer and the figures agree.
    in_al: "#76B7B2", in_central_complex: "#3D2C8D",
  };
  const KC_ORDER = ["in_kc_ic", "in_kc_nc", "in_kc_oc"];
  const UNREGIONED = "#39404F";   // cells outside every hand-drawn mask, in region view
  const QUEEN = hexToRgb("#F5C542");
  const WORKER = hexToRgb("#2FD3C6");

  // Display names — the manuscript's wording, applied where a name is SHOWN, never to
  // the data values (manifest names stay the export's raw strings, so comparisons and a
  // regenerated manifest both keep working). Mirrors src/bimp_st/
  // cluster_labels get_pooled_display / get_canonical_display / MIXED_DISPLAY /
  // DISPLAY_NAME_OVERRIDES and nine_region_common.DISPLAY, and the manuscript;
  // identity for any name already correct.
  //   "groups" — the seven pooled groups (or a hierarchical-vote export, where some
  //              clusters keep a fine name): "(all subtypes)" marks a pool.
  //   "types"  — a canonical-17 type (the Kenyon split, and the raw per-cell call): the
  //              generic "Kenyon cells" entry is the reference's unclassified residual.
  // "Mixed" (no decided label at either level) is shown as "Other", as in the paper.
  const GROUP_DISPLAY = {
    "Kenyon cells": "Kenyon cells (all subtypes)",
    "Glia": "Glia (all subtypes)",
    "Projection neurons": "Projection neurons (all subtypes)",
    "Optic lobe": "Optic lobe cells",
    "Mixed": "Other",
  };
  const TYPE_DISPLAY = {
    "Kenyon cells": "Kenyon cells (unclassified)",
    "Glia": "Glia (all subtypes)",
    "Projection neurons": "Projection neurons (all subtypes)",
    "Optic lobe": "Optic lobe cells",
    "Mixed": "Other",
  };
  // Once a glial subtype is named beside it (hierarchical_vote export), the pooled
  // "Glia" is the residual, not the whole lineage: DISPLAY_NAME_OVERRIDES says
  // "Glia (unresolved)". Decided from the manifest at load, so either export reads right.
  const FINE_GLIA = ["Astrocyte-like glia", "Cortex glia", "Perineurial glia",
                     "Subperineurial glia", "Other glia"];
  function configureNames(M) {
    const shown = [].concat(M.canonTypes || [], M.cellTypes || []);
    const residual = shown.some((n) => FINE_GLIA.indexOf(n) >= 0);
    GROUP_DISPLAY.Glia = TYPE_DISPLAY.Glia = residual ? "Glia (unresolved)" : "Glia (all subtypes)";
  }
  // The three Kenyon cell compartments are named for the population that dominates
  // each (manuscript Fig 1B / Supp. Methods; nine_region_common.DISPLAY): Large-type =
  // non-compact, Small-type = inner compact, Class II = outer compact. Keyed on the
  // mask column, not the label, so it holds for any manifest vintage.
  const REGION_DISPLAY = {
    in_kc_nc: "Large-type compartment", in_kc_ic: "Small-type compartment",
    in_kc_oc: "Class II compartment",
    in_lamina: "Lamina", in_medulla: "Medulla", in_lobula: "Lobula",
    in_al: "Antennal lobe", in_central_complex: "Central complex",
  };
  const REGION_GROUP_DISPLAY = { "Mushroom body · Kenyon cells": "Mushroom body · Kenyon cell compartments" };
  function displayName(name, vocab) {
    if (name == null) return name;
    const map = vocab === "groups" ? GROUP_DISPLAY : TYPE_DISPLAY;
    return map[name] || name;
  }
  // ---- the cell-type hierarchy (manuscript: coarse groups and fine-scale types) ----
  // POOL_MAP from src/bimp_st/cluster_labels.py: every fine label to its group. OL_1..OL_13
  // pool to "Optic lobe"; a name that is already a group ("Glia", "Kenyon cells", "Mixed")
  // maps to itself, which makes it that group's residual child.
  const POOL_MAP = {
    "Large-type Kenyon cells": "Kenyon cells", "Middle-type Kenyon cells": "Kenyon cells",
    "Small-type Kenyon cells": "Kenyon cells", "Class II Kenyon cells": "Kenyon cells",
    "Other Kenyon cells": "Kenyon cells", "Kenyon cells": "Kenyon cells",
    "Anterodorsal projection neurons": "Projection neurons",
    "Lateral projection neurons": "Projection neurons",
    "Other neurons": "Other neurons", "Photoreceptors": "Photoreceptors",
    "Astrocyte-like glia": "Glia", "Cortex glia": "Glia", "Perineurial glia": "Glia",
    "Subperineurial glia": "Glia", "Other glia": "Glia", "Hemocytes": "Hemocytes",
  };
  function poolOf(name) { return POOL_MAP[name] || (/^OL_\d+$/.test(name) ? "Optic lobe" : name); }
  // In the tree the group row IS "all subtypes", so it carries the bare name; the residual
  // child is the one that needs qualifying (TYPE_DISPLAY: "Glia (unresolved)").
  const TREE_GROUP = { "Optic lobe": "Optic lobe cells", "Mixed": "Other" };
  function groupName(name) { return TREE_GROUP[name] || name; }
  // cluster_labels.DISPLAY_COLOR_OVERRIDES: the export's residual "Glia" swatch is exactly
  // Perineurial glia's, so the two drew identically once both are on screen.
  const FINE_COLOR = { "Glia": "#9BC4C4" };
  function fineColor(name, hex, residual) { return residual && FINE_COLOR[name] ? FINE_COLOR[name] : hex; }
  // The restricted Kenyon types on the DARK stage. The manuscript figure's five are all ambers
  // and browns; Large-type, Small-type and Class II (the three it leads with) keep its colours
  // exactly, and only these two DEVIATE from the figure, apart by lightness, so that the five
  // can be told apart over black (approved 2026-10-16). The export's manifest is untouched.
  const KC_COLOR = { "Middle-type Kenyon cells": "#E8D9A8", "Other Kenyon cells": "#7D6E62" };
  function kcColor(name, hex) { return KC_COLOR[name] || hex; }
  // Kenyon-voted cells outside the compartments were never reclassified: a cool slate, outside
  // the Kenyon warm family, instead of the export's pale "Kenyon cells" sand (now Middle-type's).
  const KC_OUTSIDE = "#5A6170";
  function regionLabel(d) { return REGION_DISPLAY[d.key] || d.label; }
  function regionGroupLabel(g) { return REGION_GROUP_DISPLAY[g] || g; }

  return { HEX, FAMILIES, hexToRgb, QUEEN, WORKER, KC_REGION, KC_ORDER, UNREGIONED,
           displayName, regionLabel, regionGroupLabel, configureNames, poolOf, groupName, fineColor, kcColor, KC_OUTSIDE };
})();

;
/* ---- mat4.js ---- */
/* Minimal column-major 4x4 matrix helpers (perspective, lookAt, multiply). */
window.BIMP = window.BIMP || {};
BIMP.mat4 = (function () {
  function persp(fovy, asp, near, far) {
    const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = t / asp; o[5] = t; o[10] = (far + near) / (near - far); o[11] = -1;
    o[14] = 2 * far * near / (near - far);
    return o;
  }
  function lookAt(e, c, up) {
    let z0 = e[0] - c[0], z1 = e[1] - c[1], z2 = e[2] - c[2];
    const zl = Math.hypot(z0, z1, z2) || 1; z0 /= zl; z1 /= zl; z2 /= zl;
    let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
    const xl = Math.hypot(x0, x1, x2) || 1; x0 /= xl; x1 /= xl; x2 /= xl;
    const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
    const o = new Float32Array(16);
    o[0] = x0; o[1] = y0; o[2] = z0; o[4] = x1; o[5] = y1; o[6] = z1; o[8] = x2; o[9] = y2; o[10] = z2;
    o[12] = -(x0 * e[0] + x1 * e[1] + x2 * e[2]);
    o[13] = -(y0 * e[0] + y1 * e[1] + y2 * e[2]);
    o[14] = -(z0 * e[0] + z1 * e[1] + z2 * e[2]);
    o[15] = 1;
    return o;
  }
  function mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
    return o;
  }
  return { persp, lookAt, mul };
})();

;
/* ---- shaders.js ---- */
/* GLSL ES 3.0 shaders for the point cloud.
   Position is int16-quantized in [-32767,32767] over a normalized cube; the
   vertex shader maps it back to [-1,1], applies Z-exaggeration, sizes points by
   perspective distance, and colors by the active mode. A parallel pick pass
   writes vertex id as RGB for O(1) GPU cell selection. */
window.BIMP = window.BIMP || {};
BIMP.shaders = (function () {
  const VS = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in vec3 aPos;
layout(location=1) in float aCt;
layout(location=2) in float aCaste;
layout(location=3) in float aConf;
layout(location=4) in float aGene;
layout(location=5) in float aRegion;
layout(location=6) in float aBirth;   // per-cell reveal time in [0,1], spatially ordered
layout(location=7) in float aGenePrev;// the outgoing gene's values, for a color dissolve
layout(location=8) in vec3 aLt;      // Figure 6: Large-type subclusters S1-S3, projected probabilities
uniform mat4 uMVP, uMV;
uniform vec3 uPalette[32];
uniform vec3 uQueen, uWorker, uBg;
uniform int uMode;          // 0 celltype · 1 caste · 2 conf · 3 gene
uniform int uCasteFilter;   // -1 both · 0 queen · 1 worker
// the view state being morphed AWAY from, plus the 0→1 morph clock (see visOf/colorOf)
uniform int uModePrev, uCasteFilterPrev, uRegionMaskPrev;
uniform float uGeneLoPrev, uGeneHiPrev, uMorph;
uniform float uSplit;       // in "both" mode, offset queen/worker ±uSplit along uSplitDir
uniform vec2 uSplitDir;     // (1,0) side by side; (0,-1) queen above worker (portrait phones)
uniform float uScaffold;
uniform float uBirthHash;  // 1: cells arrive in a per-cell random order (an even sprinkle), not outward from the centre   // home page: cells outside the region mask stay as a faint grey scaffold instead of hiding
uniform int uRegionMask;    // 0 = all regions · else bitmask; show if bits overlap
uniform int uIso;           // -1 none · else isolate this cell type (hover)
uniform float uVisible[32];
uniform float uZ, uSize, uDpr, uProgress, uFade, uGeneLo, uGeneHi;
uniform float uFogRef;      // camera→target distance; fog is measured from there (see vFog)
uniform float uCellW;       // a cell's diameter in world units (µm / the export's µm-per-unit scale)
uniform float uPxK;         // viewport pixels per world unit at distance 1 (height / 2·tan(fov/2))
uniform float uCellTrue;    // 0 = the home page's look (old fixed sizing), 1 = the atlas (cell-true dots)
uniform float uLOD;         // 1 = every cell; <1 = keep that fraction while the camera moves
uniform vec4 uSlab;         // 2D section view (histology.js): [z queen, z worker, half thickness, -], data units; z<=0 = off
uniform int uKcBits[3];     // region bits for KC inner-compact / non-compact / outer-compact
uniform int uRegBits[6];    // all six hand-drawn masks, in manifest order,
uniform vec3 uRegCol[6];    // with the manuscript's colours
uniform vec3 uRegNone;      // cells outside every mask
uniform vec3 uLtCol[3];     // S1-S3 colours, as in the figure
uniform int uHover, uSelected, uPick;
uniform int uPass;          // 0 = one pass (everything) · 1 = solid, writes depth · 2 = faint, does not
// the home page's look (renderer.js "look"): ink over the cell colours, a slow per-cell wander
// in screen space, and a push away from the pointer; all zero in the atlas
uniform float uInk, uWob, uTime, uAspect;
uniform vec3 uInkCol;
uniform vec4 uPtr;          // ndc x, ndc y, radius, strength
out vec4 vColor; out float vFog; out float vReveal; out float vVis; out float vEmph;
out float vSparse; out float vPS; out float vFade; flat out int vId;
vec3 magma(float t){
  const vec3 c0=vec3(-0.002136,-0.000750,-0.005386);
  const vec3 c1=vec3(0.251661,0.677523,2.494027);
  const vec3 c2=vec3(8.353717,-3.577720,0.314468);
  const vec3 c3=vec3(-27.66873,14.26473,-13.64921);
  const vec3 c4=vec3(52.17614,-27.94361,12.94417);
  const vec3 c5=vec3(-50.76852,29.04658,4.234153);
  const vec3 c6=vec3(18.65570,-11.48977,-5.601962);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))),0.0,1.0);
}
// ---- morphing between two view states -------------------------------------------
// Changing the view (recolor, isolate a region, split by caste) is a MOVE, not a cut:
// the same cells are on screen before and after, so each one interpolates from its old
// look to its new one. Every cell evaluates both states and mixes by uMorph (0=old,
// 1=new). At rest uMorph==1 and the old-state terms fall out, so steady-state rendering
// is exactly as if this machinery weren't here.
// visibility under a given caste filter + region mask (1 = shown)
float visOf(int cf, int rm, int caste, int region){
  if(cf>=0 && caste!=cf) return 0.0;
  if(rm!=0 && (region & rm)==0) return 0.0;
  return 1.0;
}
// In region view the cells inside no hand-drawn mask are the overwhelming majority, so
// at full strength they simply bury the coloured regions behind a grey wall. Push them
// back hard — alpha and size — the same way near-zero cells are pushed back in gene view.
float regionFadeOf(int mode, int region){
  if(mode==5) return ((region & uKcBits[1])!=0 && aLt.x+aLt.y+aLt.z>0.0) ? 1.0 : 0.10;
  if(mode!=4) return 1.0;
  for(int i=0;i<6;i++) if(uRegBits[i]!=0 && (region & uRegBits[i])!=0) return 1.0;
  return 0.10;
}
// the side-by-side x-offset applied in "Both" mode
float splitOf(int cf, int caste){ return cf<0 ? (caste==0 ? -uSplit : uSplit) : 0.0; }
// color for one state; emph is the gene-expression weight (1 outside gene mode)
vec3 colorOf(int mode, int ct, int caste, int region, float gene, float lo, float hi, out float emph){
  emph=1.0;
  // Cell type: the code is composed on the CPU from the tree legend (ui.js), so a group,
  // a fine-scale type and a restricted-transfer Kenyon type are all just a palette index.
  // (The calyx is no longer repainted by compartment here: the Region view shows that.)
  if(mode==0) return uPalette[ct];
  if(mode==1) return (caste==0)?uQueen:uWorker;
  if(mode==2) return magma(aConf);
  if(mode==4){   // anatomy: the six hand-drawn masks, everything else pushed back
    for(int i=0;i<6;i++) if(uRegBits[i]!=0 && (region & uRegBits[i])!=0) return uRegCol[i];
    return uRegNone;
  }
  if(mode==5){   // Figure 6: a probability blend over the Large-type compartment only, not a call
    float sum=aLt.x+aLt.y+aLt.z;
    if((region & uKcBits[1])==0 || sum<=0.0) return uRegNone;
    return (aLt.x*uLtCol[0]+aLt.y*uLtCol[1]+aLt.z*uLtCol[2])/sum;
  }
  float t=clamp((gene-lo)/max(hi-lo,1e-4),0.0,1.0);   // gene: value drives color AND emphasis
  emph=t;
  return magma(t);
}
void main(){
  int ct=int(aCt+0.5); int caste=int(aCaste+0.5); int region=int(aRegion+0.5);
  // Spatially-ordered build-up: aBirth is SMALL near the brain core and LARGE at the
  // shell (computed CPU-side, radial-from-centroid + jitter), so as uProgress climbs a
  // coherent front sweeps outward through the volume. reveal fades+grows each cell in.
  // (A uniform/random reveal instead saturates density in the first ~15% — perceived
  // overplotting is inverse-logarithmic — so it reads as a snap. A travelling front is
  // what actually reads as "the brain assembling.")
  // Motion LOD: while the camera is moving, keep only a deterministic fraction of cells.
  // At this density perceived overplotting is inverse-logarithmic, so dropping ~40% is
  // barely visible — but it removes ~40% of the fragments, which is what costs frames.
  // The hash is per cell, so WHICH cells go is stable. What is NOT stable is the
  // threshold: uLOD tracks camera speed and moves every frame, so a hard cut popped whole
  // cells in and out continuously — that was the flicker while dragging or spinning.
  // Fade over a band instead: across LOD_BAND of the threshold a cell dims and shrinks
  // away rather than vanishing, so a drifting threshold is a drift in brightness.
  // The pick pass runs at uLOD=1, so hovering and clicking always see every cell.
  float lodK = 1.0;
  if(uLOD < 0.999){
    uint h = uint(gl_VertexID)*747796405u + 2891336453u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    float hv = float(h & 0xFFFFu)/65536.0;
    // The band sits ABOVE the threshold, not below it: everything under uLOD stays at
    // full strength (so nothing is dimmed when the view is settled and uLOD is 1), and
    // the cells just past it fade out over the next 0.14 instead of being cut.
    lodK = 1.0 - smoothstep(uLOD, uLOD + 0.14, hv);
    if(lodK <= 0.002){ gl_Position=vec4(2.0,2.0,2.0,1.0); gl_PointSize=0.0; return; }
  }
  // the page load grows the brain out from its centre (aBirth, spatial); the full cells the home
  // story swaps in arrive in a per-cell random order instead, so the brain just thickens evenly
  uint hb = uint(gl_VertexID)*2654435761u + 374761393u; hb ^= hb >> 15; hb *= 2246822519u; hb ^= hb >> 13;
  // (cells already on screen are born at -1 and keep that: hashing them too blanked the home
  // brain for a moment at the swap to the full atlas, then grew it back from black)
  float birth = aBirth < 0.0 ? aBirth : mix(aBirth, float(hb & 0xFFFFu) / 65536.0, uBirthHash);
  float reveal = smoothstep(birth, birth + uFade, uProgress);
  // A cell that is filtered out in one state but not the other fades across the morph
  // instead of blinking; one that is hidden in both is skipped entirely.
  // uMorph>=1.0 is a UNIFORM condition (same for every vertex in the draw), so the
  // settled path costs exactly what it did before this machinery existed.
  bool morphing = uMorph < 1.0;
  float visNow = visOf(uCasteFilter, uRegionMask, caste, region);
  float vis = morphing
    ? mix(visOf(uCasteFilterPrev, uRegionMaskPrev, caste, region), visNow, uMorph)
    : visNow;
  // the home page's scaffold: the region mask highlights rather than filters, so the whole brain
  // stays in view, faint and grey, around the highlighted part (hl: how highlighted, morphed)
  float hl = 1.0;
  if(uScaffold > 0.0){
    hl = vis;
    float cv = visOf(uCasteFilter, 0, caste, region);
    vis = morphing ? mix(visOf(uCasteFilterPrev, 0, caste, region), cv, uMorph) : cv;
  }
  // A cell that is arriving or leaving does so at its OWN moment within the dissolve (a
  // per-cell hash), over a short band, rather than every such cell sharing one alpha. A
  // million cells at a common 30% alpha sum into a translucent x-ray of outlines, and then
  // snap solid as they cross the depth-pass threshold; this way the set in transit is mostly
  // solid cells, present or not yet, on the same eased clock for leaving and arriving alike.
  if(morphing && vis > 0.0 && vis < 1.0){
    uint hd = uint(gl_VertexID)*2654435761u + 1013904223u;
    hd ^= hd >> 16; hd *= 2246822519u; hd ^= hd >> 13;
    float hv2 = float(hd & 0xFFFFu)/65536.0;
    vis = smoothstep(0.0, 1.0, (vis*1.12 - hv2)/0.12);
  }
  bool hide = reveal <= 0.0
    || uVisible[ct] < 0.5
    || vis <= 0.002
    || (uIso>=0 && ct!=uIso);
  if(uSlab.z > 0.0){   // a slab of the depth: one section per brain (2D, hard edge), or the home story's cut (soft edge, uSlab.w)
    float dz = abs(aPos.z/32767.0 - (caste==0 ? uSlab.x : uSlab.y)) - uSlab.z;
    if(uSlab.w <= 0.0){ if(dz > 0.0) hide = true; }
    else if(dz > 0.0){
      // past the cut, a cell fades with its distance from the section, so the brain thins
      // smoothly toward the plane (a per-cell on/off here read as twinkling)
      float sf = 1.0 - smoothstep(0.0, uSlab.w, dz);
      vis *= sf; if(sf < 0.004) hide = true;
    }
  }
  if(hide){ gl_Position=vec4(2.0,2.0,2.0,1.0); gl_PointSize=0.0; return; }
  vec3 p=vec3(aPos.x, aPos.y, aPos.z*uZ)/32767.0;
  // the two brains slide apart / back together rather than teleporting
  // Enter / update / exit, with one refinement: a BRAIN moves as one body. What matters
  // for an arriving cell is whether ITS CASTE was already on screen — if it was, the cell
  // travels with the rest of that brain even though it is only now fading in; if the whole
  // caste is new (queen-only → both), it appears at its destination instead of crawling out
  // of the other brain. Without the caste test, clearing a region filter and splitting at
  // the same time (tour stop 6 → 7) left the optic lobe sliding along on its own while the
  // rest of its brain materialised at the destination.
  float v0 = visOf(uCasteFilterPrev, uRegionMaskPrev, caste, region);
  float v1 = visOf(uCasteFilter, uRegionMask, caste, region);
  bool casteWasVis = (uCasteFilterPrev < 0) || (caste == uCasteFilterPrev);
  float posMix = (v1 < 0.5) ? 0.0                       // leaving: fade out where it stood
               : (v0 > 0.5) ? uMorph                    // present in both: travel
               : (casteWasVis ? uMorph : 1.0);          // arriving: with its brain, or in place
  p.xy += uSplitDir * (morphing ? mix(splitOf(uCasteFilterPrev, caste), splitOf(uCasteFilter, caste), posMix)
                                 : splitOf(uCasteFilter, caste));
  vec4 mv=uMV*vec4(p,1.0);
  gl_Position=uMVP*vec4(p,1.0);
  if(uPick==0 && (uWob>0.0 || uPtr.w>0.0) && gl_Position.w>0.0){
    vec2 asp=vec2(uAspect,1.0), ndc=gl_Position.xy/gl_Position.w;
    // seeded by where the cell is, not its index: the home sample and the full atlas number the
    // same cell differently, and every cell jumped to a new wobble at the swap between them
    float hv=fract(sin(dot(aPos.xy, vec2(0.0127, 0.0311)))*43758.5453), q=uTime*(0.35+0.5*hv)+hv*6.2832;
    ndc+=uWob*vec2(sin(q)+0.5*sin(q*2.3+1.7), cos(q*0.9)+0.5*cos(q*1.9+0.4))/asp;
    vec2 dd=(ndc-uPtr.xy)*asp; float r=length(dd);
    if(r<uPtr.z && r>1e-4){ float f=1.0-r/uPtr.z; ndc+=(dd/r)*uPtr.w*f*f*uPtr.z/asp; }
    gl_Position.xy=ndc*gl_Position.w;
  }
  float dist=max(-mv.z,0.05);
  // A dot is a cell: its true diameter projected at this depth. Zoomed out that is under a
  // pixel, so a floor keeps the brain solid — the old fixed look (uSize·3.4/dist), capped at
  // about two pixels so it no longer balloons as you zoom in. Past the cap the dots grow
  // only as the cells do. uSize (the slider, 2.6 by default) scales both.
  float k = uSize/2.6;
  float physical = k*uCellW*uPxK/dist;
  float floorPx = min(uDpr*uSize*3.4/dist, 2.0*uDpr*k);
  gl_PointSize = clamp(mix(uDpr*uSize*3.4/dist, max(physical, floorPx), uCellTrue), 1.0, 96.0);
  // grow each cell in from nothing, with a brief size "pop" right at birth so the moving
  // front is visible (motion + transient size/brightness are what sell "assembling").
  float pop = 1.0 + 0.5*exp(-pow((reveal-0.5)*4.0, 2.0));
  gl_PointSize *= reveal * pop;
  vId=gl_VertexID;
  // Cross-dissolve the color: the outgoing look (previous mode / previous gene values,
  // held in aGenePrev) blended into the incoming one. emph<1 only in gene mode; it drives
  // alpha+size so a sparse signal isn't buried under a sea of near-black zeros.
  float e0, e1;
  vec3 c1=colorOf(uMode, ct, caste, region, aGene, uGeneLo, uGeneHi, e1);
  vec3 col=c1; float emph=e1;
  float sparse=uMode==3?1.0:0.0;
  float fade=regionFadeOf(uMode, region);
  if(morphing){
    vec3 c0=colorOf(uModePrev, ct, caste, region, aGenePrev, uGeneLoPrev, uGeneHiPrev, e0);
    // a cell leaving the view keeps the colours it had, and one arriving comes in with its new
    // ones; only a cell in both views changes colour (otherwise the leaving cells flashed the
    // next view's colours on their way out)
    float vP = visOf(uCasteFilterPrev, uRegionMaskPrev, caste, region);
    float cm = (vP > 0.5 && visNow < 0.5) ? 0.0 : ((vP < 0.5 && visNow > 0.5) ? 1.0 : uMorph);
    col=mix(c0,c1,cm);
    emph=mix(e0,e1,cm);
    sparse=mix(uModePrev==3?1.0:0.0, sparse, cm);   // sparse-gene treatment ramps too
    fade=mix(regionFadeOf(uModePrev, region), fade, cm);
  }
  if(uScaffold > 0.0){
    float g = mix(1.0, hl, uScaffold);
    // one small dot size for the scaffold at any zoom: sized by distance, its grey brightened as
    // the story moved in and dimmed as it moved out, like the lighting changing
    gl_PointSize = mix(1.7*uDpr, gl_PointSize, g);
    col = mix(vec3(0.46, 0.48, 0.53), col, g);   // the scaffold's grey (its level on screen is this times its alpha, see FS)
    fade = mix(0.34, fade, g);                   // one faintness for every scaffold cell, whatever its region (the region view's own push-back hid most of them)
  }
  // ---- which pass does this cell belong to? -----------------------------------------
  // Depth writes are all-or-nothing. A cell rendered at a tenth alpha still claimed the
  // depth buffer and erased whatever sat behind it, so in region view the grey wash
  // blanked the very regions it was supposed to sit behind — and which of two overlapping
  // cells won depended on nothing but vertex-buffer order, which is why a sub-pixel turn
  // of the camera made the whole cloud crawl. So the atlas is drawn in two passes: cells
  // at full strength write depth, deliberately-faded ones only test against it.
  // The test is on the FADES ONLY, not on uOpacity — otherwise turning the global opacity
  // slider down would tip every cell into the no-depth pass at once.
  float sparseMul = mix(1.0, mix(0.05, 1.0, sqrt(max(emph,0.0))), sparse);
  float relA = reveal * vis * fade * sparseMul * lodK;
  // On the home page a highlighted region greys back into the scaffold with one strength for
  // every cell, so all of them crossed the solid/faint line on the same frame near the end of
  // the fade, and the region snapped out. There each cell crosses at its own level instead; at
  // rest nothing changes (highlighted cells sit at 1, the scaffold at 0.34, below every level).
  float solidAt = 0.4;
  if(uScaffold > 0.0){
    uint hs = uint(gl_VertexID)*2246822519u + 3266489917u; hs ^= hs >> 15; hs *= 2654435761u; hs ^= hs >> 13;
    solidAt = 0.36 + 0.6*float(hs & 0xFFFFu)/65536.0;
  }
  bool solid = relA >= solidAt || gl_VertexID==uSelected || gl_VertexID==uHover;
  if(uPass==1 && !solid){ gl_Position=vec4(2.0,2.0,2.0,1.0); gl_PointSize=0.0; return; }
  if(uPass==2 && solid){ gl_Position=vec4(2.0,2.0,2.0,1.0); gl_PointSize=0.0; return; }
  if(uPick==0) gl_PointSize*=mix(0.55,1.0,fade);   // and shrink them, so they stop occluding
  if(uPick==0) gl_PointSize*=mix(1.0, mix(0.5,1.25,emph), sparse);   // shrink ~zero cells
  if(uPick==0) gl_PointSize*=mix(0.6,1.0,lodK);    // LOD-thinned cells shrink away, not blink
  if(uPick==0 && gl_VertexID==uSelected){ col=vec3(1.0); gl_PointSize=max(gl_PointSize*3.0,10.0); }
  else if(uPick==0 && gl_VertexID==uHover){ col=mix(col,vec3(1.0),0.6); gl_PointSize*=2.2; }
  if(uPick==0) col=mix(col,uInkCol,uInk);   // the home page's ink, easing into the cell colours
  vColor=vec4(col,1.0);
  vReveal=reveal;          // birth clock only — the FS spark keys off it
  vVis=vis*lodK;           // filtered-out cells fade out rather than vanish; so do thinned ones
  vEmph=emph;
  vSparse=sparse;
  vFade=fade;
  vPS=gl_PointSize;   // the fragment shader needs it to keep the sprite edge ~1px at any zoom
  // Depth fog measured from the FOCAL PLANE (the orbit target, uFogRef units away), not
  // from the camera: it should say "this cell is behind that one" — a property of the
  // cloud, not of the zoom. Everything at or in front of the target is unfogged; only the
  // far half hazes, over ~1 brain-radius. Absolute-distance fog dimmed the entire atlas
  // the moment you pulled back (~50% background mixed into every cell at the default
  // orbit, ~70% in the pulled-back "Both" view) — that's what read as "not bright enough".
  vFog=clamp((dist-uFogRef)/2.2,0.0,0.5);
}`;

  const FS = `#version 300 es
precision highp float;
precision highp int;
in vec4 vColor; in float vFog; in float vReveal; in float vVis; in float vEmph;
in float vSparse; in float vPS; in float vFade; flat in int vId;
uniform int uPick; uniform int uMode; uniform float uOpacity; uniform vec3 uBg;
uniform float uSpark, uFogK; // uFogK: depth fog strength (eased off while the page dims, so the cells keep their contrast)       // 0 during the home <-> atlas move: the newborn flash is for the viewer's own build
out vec4 frag;
uniform int uPass; uniform float uScaffold;   // shared with the vertex stage
void main(){
  vec2 d=gl_PointCoord-vec2(0.5);
  float r=length(d)*2.0;
  if(r>1.0) discard;
  if(uPick==1){
    int id=vId+1;
    frag=vec4(float(id&255)/255.0, float((id>>8)&255)/255.0, float((id>>16)&255)/255.0, 1.0);
    return;
  }
  // The sprite is a solid disc with a ~1.5px antialiased rim. The rim width is derived
  // from the point size (r spans 0→1 over vPS/2 pixels, so one pixel is 2/vPS in r), which
  // keeps a zoomed-in cell CRISP — the old falloff was a fixed fraction of the sprite, so
  // a 64px point turned into a 22px-wide gradient, i.e. a blurry blob. Tiny points keep a
  // soft edge (the clamp), which is what antialiases them.
  float aa=2.0/max(vPS,2.0);
  float w=clamp(1.5*aa,0.05,0.85);
  float core=1.0-smoothstep(1.0-w,1.0,r);
  // the halo is a look, not a shape — fade it out as the point grows so big dots read as
  // discs rather than smudges
  float glow=(1.0-smoothstep(0.0,1.0,r))*0.32*clamp(8.0/max(vPS,1.0),0.15,1.0);
  float a=(core+glow)*uOpacity*vReveal*vVis*vFade;     // born / filtered / outside every mask
  vec3 c=mix(vColor.rgb, vColor.rgb*1.4, glow*0.7);    // brighten core → bloom feel
  // Sparse-gene legibility: in gene mode, near-zero cells (dark magma) are made nearly
  // transparent so they stop occluding the rare expressers, and high-expression cells
  // are lifted brighter so the signal pops out of the dark sea. Inert in other modes.
  // vSparse ramps this in/out with the morph, so entering gene view dims the near-zero
  // cells gradually instead of the whole cloud lurching darker in one frame.
  a *= mix(1.0, mix(0.05, 1.0, sqrt(vEmph)), vSparse);   // sqrt: near-zero stays suppressed,
  c += vColor.rgb * vEmph * 0.4 * vSparse;               // mid-expression stays legible
  // leading-edge spark: a cell flashes brighter the instant it's born, then settles to
  // its true color. Once the build completes vReveal==1 everywhere → spark==0 (no residue).
  float spark=smoothstep(0.0,0.5,vReveal)*(1.0-smoothstep(0.5,1.0,vReveal));
  c += spark*0.6*uSpark;
  c=mix(c, uBg, vFog*uFogK);                                 // depth fog into the void
  frag=vec4(c, a);
  // the home scaffold's faint pass blends with MAX (renderer.js): premultiplied here, a pixel takes
  // the brighter of what is there and this dot, so a thousand overlapping grey dots in a packed
  // region read as the same grey as one, not a bright cloud that shifted as regions changed
  if(uScaffold > 0.0 && uPass == 2) frag = vec4(c * a, a);
}`;
  return { VS, FS };
})();

;
/* ---- camera.js ---- */
/* Orbit camera — spherical coords around a pannable target. */
window.BIMP = window.BIMP || {};
BIMP.camera = (function () {
  // radius 4.2 @ 48° FOV shows a visible half-height of ~1.9 units at the target, so
  // the brain (~±1.0 after bbox-centered fit-to-±0.9) clears the frame with margin on
  // tall/portrait windows too, even with the downward orbit tilt.
  const HOME = { theta: 0.9, phi: 1.15, radius: 4.2, target: [0, 0, 0] };
  const cam = { theta: HOME.theta, phi: HOME.phi, radius: HOME.radius, target: [0, 0, 0], minR: 0.3, maxR: 14 };   // 0.3: close enough that a dot is a cell several pixels wide
  function eye() {
    const { radius: r, phi, theta, target: t } = cam;
    return [
      t[0] + r * Math.sin(phi) * Math.cos(theta),
      t[1] + r * Math.cos(phi),
      t[2] + r * Math.sin(phi) * Math.sin(theta),
    ];
  }
  function orbit(dx, dy) {
    cam.theta += dx * 0.006;
    cam.phi = Math.max(0.15, Math.min(Math.PI - 0.15, cam.phi - dy * 0.006));
  }
  // Pan the target in the camera's screen plane (right / up vectors), scaled by
  // distance so a drag moves the same on-screen amount at any zoom.
  function pan(dx, dy) {
    const e = eye();
    let fx = cam.target[0] - e[0], fy = cam.target[1] - e[1], fz = cam.target[2] - e[2];
    const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
    // right = forward × worldUp
    let rx = fy * 0 - fz * 1, ry = fz * 0 - fx * 0, rz = fx * 1 - fy * 0;
    const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
    // up = right × forward
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    // world units per CSS pixel at the target's depth (the renderer's 48° field over the view's
    // height), so what is under the pointer stays under it, as on a map; a fixed 0.0016 matched
    // the screen only at one window size
    const gl = typeof document !== "undefined" && document.getElementById("gl");
    const k = 2 * cam.radius * Math.tan(24 * Math.PI / 180) / Math.max(200, (gl && gl.clientHeight) || 900);
    cam.target[0] += (-rx * dx + ux * dy) * k;
    cam.target[1] += (-ry * dx + uy * dy) * k;
    cam.target[2] += (-rz * dx + uz * dy) * k;
  }
  function zoom(sign) { cam.radius = Math.max(cam.minR, Math.min(cam.maxR, cam.radius * (1 + sign * 0.08))); }
  function spin(dt) { cam.theta += dt * 0.12; }
  function reset() {
    cam.theta = HOME.theta; cam.phi = HOME.phi; cam.radius = HOME.radius;
    cam.target[0] = cam.target[1] = cam.target[2] = 0;
  }

  // ---- scripted camera flights (used by the guided tour) ----
  // flyTo eases theta/phi/radius/target from the current pose to a goal over `dur` ms;
  // tick() advances it each frame and returns true while a flight is in progress, so the
  // frame loop can suspend auto-orbit. Any user drag calls cancelFly() to hand back control.
  let fly = null;
  const snap = () => ({ theta: cam.theta, phi: cam.phi, radius: cam.radius, target: cam.target.slice() });
  function flyTo(goal, dur) {
    const from = snap();
    const to = {
      theta: goal.theta != null ? goal.theta : from.theta,
      phi: goal.phi != null ? goal.phi : from.phi,
      radius: goal.radius != null ? goal.radius : from.radius,
      target: goal.target ? goal.target.slice() : from.target.slice(),
    };
    let dth = to.theta - from.theta;            // shortest angular path
    while (dth > Math.PI) dth -= 2 * Math.PI;
    while (dth < -Math.PI) dth += 2 * Math.PI;
    to.theta = from.theta + dth;
    fly = { from, to, t: 0, dur: Math.max(0.2, (dur || 1100) / 1000) };
  }
  function tick(dt) {
    if (!fly) return false;
    fly.t += dt;
    const u = Math.min(1, fly.t / fly.dur);
    const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;   // easeInOutCubic
    const L = (a, b) => a + (b - a) * e;
    cam.theta = L(fly.from.theta, fly.to.theta);
    cam.phi = L(fly.from.phi, fly.to.phi);
    cam.radius = L(fly.from.radius, fly.to.radius);
    for (let k = 0; k < 3; k++) cam.target[k] = L(fly.from.target[k], fly.to.target[k]);
    if (u >= 1) fly = null;
    return true;
  }
  function cancelFly() { fly = null; }
  function isFlying() { return !!fly; }
  return { cam, eye, orbit, pan, zoom, spin, reset, flyTo, tick, cancelFly, isFlying, HOME };
})();

;
/* ---- decode.js ---- */
/* Decode a tier payload: base64 → gzip → planar typed-array views.
   Two tiers share this: LITE (core attrs, no genes — fast first paint) and
   FULL (core + all genes). Returns typed arrays over the decoded buffer. */
window.BIMP = window.BIMP || {};
BIMP.decode = (function () {
  function b64ToBytes(b64) {
    const bin = atob(b64), len = bin.length, out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  async function gunzip(bytes) {
    if (typeof DecompressionStream === "undefined") return bytes.buffer;
    const ds = new DecompressionStream("gzip");
    const stream = new Response(new Blob([bytes]).stream().pipeThrough(ds));
    return stream.arrayBuffer();
  }
  // Build typed-array views over an already-decompressed ArrayBuffer.
  function views(buf, tier, geneCount) {
    const o = tier.offsets, n = tier.n;
    const out = {
      n,
      x: new Int16Array(buf, o.x[0], n),
      y: new Int16Array(buf, o.y[0], n),
      z: new Int16Array(buf, o.z[0], n),
      ct: new Uint8Array(buf, o.celltype[0], n),          // coherent subtype (cluster-vote)
      ctCanon: o.celltypecanon ? new Uint8Array(buf, o.celltypecanon[0], n) : null,
      ctRaw: o.celltyperaw ? new Uint8Array(buf, o.celltyperaw[0], n) : null,  // raw per-cell call (inspector)
      caste: new Uint8Array(buf, o.caste[0], n),
      conf: new Uint8Array(buf, o.conf[0], n),
      region: o.region ? new Uint8Array(buf, o.region[0], n) : new Uint8Array(n),
      // home tier only: 1 = one of the home section's cells beyond the sample (build-pages hero)
      hx: o.hx ? new Uint8Array(buf, o.hx[0], n) : null,
      // per-cell UMAP position (uint8, 0-255 over the embedding bbox) — full tier only
      umapx: o.umapx ? new Uint8Array(buf, o.umapx[0], n) : null,
      umapy: o.umapy ? new Uint8Array(buf, o.umapy[0], n) : null,
      // Figure 6's Large-type subclusters S1-S3, projected per cell (0-255 probabilities)
      ltkc1: o.ltkc1 ? new Uint8Array(buf, o.ltkc1[0], n) : null,
      ltkc2: o.ltkc2 ? new Uint8Array(buf, o.ltkc2[0], n) : null,
      ltkc3: o.ltkc3 ? new Uint8Array(buf, o.ltkc3[0], n) : null,
      // Fig 5B's restricted Kenyon transfer: 0 = outside the three Kenyon compartments,
      // 1..5 = manifest.kcTypes.names order; kcprob = the winning renormalised probability ×255
      kctype: o.kctype ? new Uint8Array(buf, o.kctype[0], n) : null,
      kcprob: o.kcprob ? new Uint8Array(buf, o.kcprob[0], n) : null,
      genes: [],
    };
    for (let i = 0; i < (geneCount || 0); i++) {
      if (o["gene" + i]) out.genes.push(new Uint8Array(buf, o["gene" + i][0], n));
    }
    return out;
  }
  // Decompress a base64 blob and return both the raw buffer (for caching) and views.
  async function decodeBlob(b64, tier, geneCount, gz) {
    const bytes = b64ToBytes(b64);
    const buf = gz ? await gunzip(bytes) : bytes.buffer;
    return { buf, data: views(buf, tier, geneCount) };
  }
  // ---- fetch mode (GitHub Pages: data served as separate files) ----
  // Retries a failed or stalled request (network error, 5xx, 408/429) with backoff, so one
  // dropped connection on a phone doesn't leave the atlas half-loaded. A 4xx is final.
  // opts.timeout (ms) aborts an attempt that has taken too long; opts.onRetry(n, err) reports.
  async function fetchBuf(url, gz, opts) {
    opts = opts || {};
    const tries = opts.retries == null ? 3 : opts.retries;
    for (let k = 0; ; k++) {
      const ac = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = ac && opts.timeout ? setTimeout(() => ac.abort(), opts.timeout) : 0;
      try {
        const resp = await fetch(url, ac ? { signal: ac.signal } : undefined);
        if (!resp.ok) {
          const err = new Error("fetch " + url + " → " + resp.status);
          err.final = resp.status < 500 && resp.status !== 408 && resp.status !== 429;
          throw err;
        }
        const bytes = new Uint8Array(await resp.arrayBuffer());
        return gz ? await gunzip(bytes) : bytes.buffer;
      } catch (err) {
        if (err.final || k >= tries) throw err;
        if (opts.onRetry) try { opts.onRetry(k + 1, err); } catch (e) { /* reporting only */ }
        await new Promise((r) => setTimeout(r, 600 * Math.pow(2, k)));
      } finally { if (timer) clearTimeout(timer); }
    }
  }
  // Gene files hold every full-tier cell. When the drawn tier is the lite one (boot.js device
  // tier), setGather(fullIndex) makes fetchGene return only the drawn cells, in drawn order.
  let gather = null;
  function setGather(idx) { gather = idx || null; }
  function gatherOut(arr) {
    if (!gather) return arr;
    const out = new Uint8Array(gather.length);
    for (let j = 0; j < gather.length; j++) out[j] = arr[gather[j]];
    return out;
  }
  // the lite tier's full-tier indices (build-pages.mjs liteIndex): Uint16 deltas → Uint32
  async function fetchLiteIndex(url, n) {
    const d = new Uint16Array(await fetchBuf(url, true, { timeout: 30000 }));
    if (d.length !== n) throw new Error("lite index: " + d.length + " ≠ " + n);
    const idx = new Uint32Array(n);
    for (let j = 0, p = -1; j < n; j++) { p += d[j]; idx[j] = p; }
    return idx;
  }
  // base.bin → views over all cells (core attrs only, no genes)
  async function fetchBase(url, tier, gz, opts) {
    return views(await fetchBuf(url, gz, opts), tier, 0);
  }
  // one gene file → per-cell Uint8Array (0-255). Measured genes are 8-bit; inferred
  // genes are 4-bit packed (2 cells/byte) — unpack to 0-255 for the shared color path.
  async function fetchGene(url, gz, opts) {
    const buf = await fetchBuf(url, gz);
    const bytes = new Uint8Array(buf);
    if (opts && opts.bits === 4) {
      const n = (opts.n != null) ? opts.n : bytes.length * 2;
      const out = new Uint8Array(n);
      for (let i = 0; i < bytes.length; i++) {
        const hi = (bytes[i] >> 4) & 0xf, lo = bytes[i] & 0xf;
        if (i * 2 < n) out[i * 2] = hi * 17;         // 0-15 → 0-255
        if (i * 2 + 1 < n) out[i * 2 + 1] = lo * 17;
      }
      return gatherOut(out);
    }
    return gatherOut(bytes);
  }
  return { decodeBlob, views, b64ToBytes, fetchBase, fetchGene, fetchBuf, fetchLiteIndex, setGather };
})();

;
/* ---- state.js ---- */
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

;
/* ---- renderer.js ---- */
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
  // The six hand-drawn masks and their manuscript colours, for the "Region" colour mode.
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
    // location 8: Figure 6's Large-type subcluster projection (S1-S3 probabilities), or zeros
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
  // Swap the cell-type coloring between canonical (manuscript vote) and raw subtypes.
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

;
/* ---- histology.js ---- */
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
  // caste: left out rather than shipped misaligned. A manifest may also carry its own
  // {"exclude": {"worker": [i, ...]}}; both lists apply.
  //
  // Empty since the 2026-10-16 per-section refit (histo_refit.py). The first export registered
  // each brain with ONE translation, which held for the queen but left 27 of 35 worker sections
  // 10-65 px (25-170 um) off with visible rotation, so they were listed here. Re-checked on the
  // refit (cells onto image, per section): queen 30/30 and worker 35/35 at 0 px best shift,
  // best scale 1.00, correlation 0.79-0.90 (queen) and 0.86-0.91 (worker).
  const EXCLUDE = { queen: [], worker: [] };
  // Composite colours, in the usual fluorescence assignments: nuclei blue, membrane green,
  // RNA red, interior protein yellow. All four are pure-hue and bright, so they stay apart from
  // one another on the dark stage, and they sit under the cells rather than mixing with the
  // cell palette (teal / magenta / orange / slate), whose hues they avoid where it matters.
  const STAIN_COL = { dapi: "#4F86FF", boundary: "#3FE07A", rna18s: "#FF4F4F", smavim: "#FFD23F" };
  const SPARE_COL = ["#4F86FF", "#3FE07A", "#FF4F4F", "#FFD23F"];
  // Display levels per stain: black point, white point, gamma (lo / hi / g).
  // The export stretches each stain to its brain's 0.5–99.7 % range, so over the tissue the
  // 99.8th percentile sits at ~0.98 for every stain (measured, every 4th section of both brains)
  // and the white point stays at the top: lowering it is what clipped DAPI's nuclei to flat
  // white specks. Black points sit just above the empty background, so faint neuropil signal
  // is not crushed; a gamma below 1 lifts dim nuclei and texture; the shader's soft shoulder
  // keeps the bright end from clipping when the strength (gain) goes up. Boundary keeps a
  // higher floor: its median over tissue is 0.11, a diffuse haze rather than structure.
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
  // the same absorbed from the paper (light stage). A program per number of stains on, so a
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
    } catch (e) { return; }          // no histology shipped: no control, silently
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
    // light stage → stains absorb from the paper (MIN); dark stage → stains emit (MAX)
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

;
/* ---- ui.js ---- */
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
  // ONE legend for both annotation levels of the manuscript's cluster vote (Supplementary
  // Methods, "Cell type annotation by cluster vote"): each coarse group is a row, and it
  // expands to the fine-scale types the vote decided inside it. Colour follows the level
  // that is showing, group by group — a collapsed group is one colour, an expanded one
  // shows its types. Kenyon cells are the exception: the panel cannot name their subtypes
  // per cluster, so the vote holds them at the group, and their children are instead the
  // manuscript's RESTRICTED transfer (Fig. 5B): Tangram re-run on the Kenyon-compartment
  // cells against the five named snRNA-seq Kenyon populations.
  //
  // Large-type then opens one level further, to Figure 6's three de novo subclusters S1–S3,
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
      // manifest order, which is the manuscript's: Large-type, Small-type, Class II lead
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
      // Large-type → S1-S3 (Figure 6), an ordinary level: filter, isolate, colour
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
    // superseded by the keys above: an old link's label/region/select are read once, then dropped
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
      // mode 1 (flat caste colours) is retired — it painted every cell one bulk colour,
      // which the Queen/Worker/Both filter already tells you. Its slot now colours by the
      // hand-drawn anatomy, which is real per-cell information. The shader branch stays so
      // old ?mode=caste links keep working.
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
  // Same colour the shader paints each calyx compartment with (manuscript SI palette).
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
    if (nd.kind === "lt") kids.appendChild(noteRow(`Fig. 6 subclusters · strongest per cell<button class="info-dot" data-info="ltkc" aria-label="About the Large-type subclusters">ⓘ</button>`));
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
  // Region view: the eight masks in the manuscript's colours. Clicking one isolates it,
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
      (manuscript Fig.&nbsp;6), projected by Tangram onto the Kenyon-compartment cells. Each
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
  // choice of lo/hi ever will, because the difference is not in the colour. Measured over
  // all 100 measured genes in this export: of the 61 with any caste shift, the median
  // |log2 fold| of the per-caste MEAN is 0.184, of which prevalence — the fraction of cells
  // carrying any transcript — accounts for 0.165; the mean among EXPRESSING cells differs by
  // a median |log2| of 0.018, i.e. about a percent. sNPF is typical: 19.1% of queen cells vs
  // 12.3% of worker cells express it, but those that do average 190 vs 184 out of 255.
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
  // to the raw call); shownClass() is the same in the manuscript's wording, for display.
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
    // the restricted Kenyon transfer (Fig. 5B), for any cell inside the compartments,
    // whether or not the legend is showing it: the call and its renormalised probability
    const kt = DATA.kctype ? DATA.kctype[idx] : 0, KT = M.kcTypes;
    const showKc = kt > 0 && KT && KT.names && kt <= KT.names.length;
    $("i-kc-row").style.display = showKc ? "flex" : "none";
    if (showKc) $("i-kc").textContent = ((KT.short || KT.names)[kt - 1]) +
      (DATA.kcprob ? " · " + (DATA.kcprob[idx] / 255).toFixed(2) : "");
    // and, for a Large-type call, Figure 6's subcluster: the strongest of S1-S3 and its share
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

;
/* ---- tour.js ---- */
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

;
/* ---- boot.js ---- */
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
    // the histology layer's control appears only if data/histo/ was shipped (Pages build)
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
  // from there down, so in flight the paper can fade while the stage above it stays put
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
  // coloured as in the manuscript: the Kenyon cell compartments, the Large-type subclusters
  // (Fig. 6), the optic lobe neuropils, then front-on and cut down to a section. Every change is
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
      S.goals = [   // mode: 0 cell type · 4 region (manuscript colours) · 5 Large-type subclusters
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
    // NOT USED (2026-10-19: any correction of the reader's scroll read as the page fighting them).
    // As the reader's scroll dies away, the step they are arriving at eases into place on a soft
    // spring that picks up their own speed, so it reads as one motion, not a stop and a second
    // push: its text at the middle beside the brain on wide screens; on a phone, centred in the
    // space under the pinned brain. It prefers the step in the direction of travel, only reaches
    // for one that is already close, never moves under a finger, and not with reduced motion.
    // (CSS scroll-snap did this with a hard jump; waiting for a dead stop felt like a jolt.)
    let idleT = 0, spring = null, setTo = null;   // setTo: the spring's own last position, so its scroll events are not taken for the reader's
    let vel = 0, lastY = 0, lastT = 0, finger = false;
    // where the scroll would come to rest on its own (as a phone pages): the step nearest that
    // point, so the spring only finishes the reader's own motion at about the reader's own speed
    function settleTarget(rest) {
      if (narrow.matches && !fig) return null;
      const mid = narrow.matches ? (fig.getBoundingClientRect().bottom + innerHeight) / 2 : innerHeight / 2;
      const ds = steps.map((el) => { const r = el.getBoundingClientRect(); return (r.top + r.height / 2) - mid; })
        .filter((d) => Math.abs(d - rest) < innerHeight * 0.16)   // only when it would stop close by: a nudge, never a pull.sort((a, b) => Math.abs(a - rest) - Math.abs(b - rest));
      return ds.length && Math.abs(ds[0]) > 2 ? ds[0] : null;
    }
    function startSpring() {
      if (finger || still.matches) return;
      const v0 = vel * 1000;   // px/s, the reader's own speed carries into the spring
      const d = settleTarget(v0 * 0.35); if (d === null) return;   // ~where momentum alone would stop
      const to = intro.scrollTop + d; let x = intro.scrollTop, v = v0, last = performance.now();
      const K = 34, C = 2 * Math.sqrt(K) * 0.9;   // soft, a touch underdamped: settles in about half a second
      spring = () => {
        const now = performance.now(); let dt = Math.min(0.05, (now - last) / 1000); last = now;
        while (dt > 0) { const h = Math.min(dt, 1 / 120); v += ((to - x) * K - v * C) * h; x += v * h; dt -= h; }
        setTo = Math.round(x); intro.scrollTop = x;
        if (Math.abs(to - x) < 0.5 && Math.abs(v) < 4) { setTo = Math.round(to); intro.scrollTop = to; spring = null; return; }
        requestAnimationFrame(() => spring && spring());
      };
      requestAnimationFrame(spring);
    }
    const stopSpring = () => { spring = null; clearTimeout(idleT); };
    if (steps.length && intro) {
      intro.addEventListener("scroll", () => {
        measure();
        if (setTo !== null && Math.abs(intro.scrollTop - setTo) <= 1) return;   // the spring moving it
        setTo = null;
        const now = performance.now(), y = intro.scrollTop, dt = now - lastT;
        if (lastT && dt > 0 && dt < 120) vel += ((y - lastY) / dt - vel) * 0.35; else vel = 0;
        lastY = y; lastT = now;
        // take over while the reader's own momentum is fading, before it stops dead, so the motion
        // never comes to rest in between; a scroll that simply ends is caught by the timer
        // (no settling: scrolling stays entirely the reader's, as in a newspaper's scrolling
        // graphics; startSpring is kept but not called)
      }, { passive: true });
      intro.addEventListener("touchstart", () => { finger = true; stopSpring(); }, { passive: true });
      intro.addEventListener("touchend", () => { finger = false; }, { passive: true });
      intro.addEventListener("touchcancel", () => { finger = false; }, { passive: true });
      ["wheel", "pointerdown", "keydown"].forEach((t) => intro.addEventListener(t, stopSpring, { passive: true }));
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
  //   2. the lights go down, quickly, over the whole page (never a dark shape on the paper);
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
      // off screen or hidden: no animation, but still repaint the paper a few times a second — the
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

;