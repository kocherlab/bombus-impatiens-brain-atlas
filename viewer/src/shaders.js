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
layout(location=8) in vec3 aLt;      // fig:kc_subtypes: Large-type subclusters S1-S3, projected probabilities
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
  if(mode==5){   // fig:kc_subtypes: a probability blend over the Large-type compartment only, not a call
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
