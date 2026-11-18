/* Does a view change DISSOLVE, or blink through empty?
 *
 * Compiles the real src/shaders.js in headless WebGL2, draws a handful of cells, and
 * sweeps uMorph 0→1 across three transitions the guided tour actually performs:
 *
 *   recolor  — cell-type view → gene view (same cells, colours interpolate)
 *   filter   — everything → one region isolated (the rest fade, not blink)
 *   split    — one caste → both (the brains slide apart)
 *
 * Asserts, at every step of the sweep:
 *   • the frame is never empty — a cell on screen at morph 0 and at morph 1 is on
 *     screen throughout (that was the "-> empty -> next thing" bug),
 *   • the colour path is monotone from the old colour to the new one (a dissolve,
 *     not a cut), and mid-sweep is genuinely intermediate,
 *   • morph==1 renders EXACTLY what the pre-morph shader would have (no drift).
 *
 *   node viewer/test/morph.mjs
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const SRC = process.env.BIMP_SRC || join(APP, "src");   // negative controls point this at a broken copy
const OUT = join(process.env.BIMP_TEST_TMP || "/tmp", "tmp", "morph-page.html");

const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
if (!CHROMIUM) { console.error("Chromium not found in the Playwright cache"); process.exit(1); }

const shaderSrc = readFileSync(join(SRC, "shaders.js"), "utf8");
const mat4Src = readFileSync(join(SRC, "mat4.js"), "utf8");   // the app's own matrices, not hand-rolled

const page = `<!doctype html><meta charset=utf-8><title>MORPH</title><canvas id=c width=256 height=256></canvas>
<script>window.BIMP={};</script>
<script>${mat4Src}</script>
<script>${shaderSrc}</script>
<script>
const S=BIMP.shaders, gl=document.getElementById('c').getContext('webgl2',{antialias:false,alpha:false,preserveDrawingBuffer:true});
function mk(){const p=gl.createProgram();
  for(const [t,src] of [[gl.VERTEX_SHADER,S.VS],[gl.FRAGMENT_SHADER,S.FS]]){
    const sh=gl.createShader(t); gl.shaderSource(sh,src); gl.compileShader(sh);
    if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS)) throw new Error('compile: '+gl.getShaderInfoLog(sh));
    gl.attachShader(p,sh);}
  gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error('link: '+gl.getProgramInfoLog(p));
  return p;}
const prog=mk(); gl.useProgram(prog);
const u=n=>gl.getUniformLocation(prog,n);

// 4 cells: two queen / two worker, two in region bit0 / two in bit3, spread across x
// 4 transition-test cells, then a 400-cell grid used only by the LOD checks
const N4=4, NG=400, N=N4+NG;
const pos=new Float32Array(N*3), ct=new Float32Array(N), caste=new Float32Array(N),
      conf=new Float32Array(N), gene=new Float32Array(N), genePrev=new Float32Array(N),
      region=new Float32Array(N), birth=new Float32Array(N);
pos.set([-9000,0,0, -3000,0,0, 3000,0,0, 9000,0,0]);
ct.set([0,1,2,3]); caste.set([0,0,1,1]); conf.set([0.2,0.5,0.7,0.9]);
gene.set([0.05,0.9,0.3,0.7]); genePrev.set([0.9,0.05,0.7,0.3]);
region.set([1,8,1,8]);        // bit0/bit3, CROSS-CUTTING caste (else a caste filter hides it)
for(let i=0;i<NG;i++){        // spread wide so no two grid cells overlap on screen
  const k=N4+i, gx=i%20, gy=(i/20)|0;
  pos[k*3]=(gx-9.5)*3200; pos[k*3+1]=(gy-9.5)*3200; pos[k*3+2]=0;
  ct[k]=i%4; caste[k]=0; conf[k]=0.5; gene[k]=0.5; genePrev[k]=0.5; region[k]=1;
}
function buf(loc,arr,size){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0);}
const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
buf(0,pos,3); buf(1,ct,1); buf(2,caste,1); buf(3,conf,1); buf(4,gene,1); buf(5,region,1);
buf(6,birth,1); buf(7,genePrev,1);

// camera built exactly as the renderer does: perspective * lookAt, head-on from +z
const M=BIMP.mat4;
const view=M.lookAt([0,0,4],[0,0,0],[0,1,0]);
const proj=M.persp(48*Math.PI/180, 1, 0.05, 60);
gl.uniformMatrix4fv(u('uMVP'),false,M.mul(proj,view));
gl.uniformMatrix4fv(u('uMV'),false,view);
const pal=new Float32Array(32*3);
for(let i=0;i<32;i++){pal[i*3]=(i%3===0)?1:0.1; pal[i*3+1]=(i%3===1)?1:0.1; pal[i*3+2]=(i%3===2)?1:0.15;}
gl.uniform3fv(u('uPalette'),pal);
gl.uniform3fv(u('uQueen'),[0.96,0.77,0.26]); gl.uniform3fv(u('uWorker'),[0.18,0.83,0.78]);
gl.uniform3f(u('uBg'),0.027,0.035,0.055);
gl.uniform1fv(u('uVisible'),new Float32Array(32).fill(1));
gl.uniform1f(u('uZ'),1); gl.uniform1f(u('uSize'),9); gl.uniform1f(u('uDpr'),1);
gl.uniform1f(u('uProgress'),2); gl.uniform1f(u('uFade'),0.14);
gl.uniform1f(u('uOpacity'),0.95); gl.uniform1f(u('uSplit'),1.15);
gl.uniform1i(u('uIso'),-1); gl.uniform1i(u('uHover'),-1); gl.uniform1i(u('uSelected'),-1);
gl.uniform1i(u('uPick'),0); gl.uniform1f(u('uFogRef'),4.0); gl.uniform1f(u('uLOD'),1.0);
gl.uniform1iv(u('uKcBits[0]'), new Int32Array([0,0,0]));
// the SI figures' compartment colours (bimp_st.style): ic purple, nc orange, oc yellow
const KC=[[0.690,0.478,0.631],[0.949,0.557,0.169],[0.929,0.788,0.282]];
gl.uniform3fv(u('uKcCol[0]'), new Float32Array(KC.flat()));

// Column profile of lit pixels — used to see WHERE each brain is during the split.
function columns(st,morph){
  draw(st,morph);
  const px=new Uint8Array(256*256*4); gl.readPixels(0,0,256,256,gl.RGBA,gl.UNSIGNED_BYTE,px);
  const col=new Array(256).fill(0);
  for(let y=0;y<256;y++) for(let x=0;x<256;x++){
    const i=(y*256+x)*4, d=Math.abs(px[i]-7)+Math.abs(px[i+1]-9)+Math.abs(px[i+2]-14);
    if(d>18) col[x]++;
  }
  return col;
}
function draw(st,morph){
  gl.uniform1i(u('uMode'),st.mode); gl.uniform1i(u('uModePrev'),st.modeP);
  gl.uniform1i(u('uCasteFilter'),st.caste); gl.uniform1i(u('uCasteFilterPrev'),st.casteP);
  gl.uniform1i(u('uRegionMask'),st.region); gl.uniform1i(u('uRegionMaskPrev'),st.regionP);
  gl.uniform1f(u('uGeneLo'),0); gl.uniform1f(u('uGeneHi'),1);
  gl.uniform1f(u('uGeneLoPrev'),0); gl.uniform1f(u('uGeneHiPrev'),1);
  gl.uniform1f(u('uMorph'),morph);
  gl.uniform1iv(u('uKcBits[0]'), new Int32Array(st.kc||[0,0,0]));
  gl.uniform1f(u('uLOD'), st.lod==null?1.0:st.lod);
  gl.uniform1i(u('uPick'), st.pick?1:0);
  gl.viewport(0,0,256,256); gl.clearColor(0.027,0.035,0.055,1);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.depthMask(true); gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
  gl.drawArrays(gl.POINTS, 0, st.all ? N : N4);
  const px=new Uint8Array(256*256*4); gl.readPixels(0,0,256,256,gl.RGBA,gl.UNSIGNED_BYTE,px);
  let lit=0,R=0,G=0,B=0;
  for(let i=0;i<px.length;i+=4){
    const d=Math.abs(px[i]-7)+Math.abs(px[i+1]-9)+Math.abs(px[i+2]-14);
    if(d>18){lit++;R+=px[i];G+=px[i+1];B+=px[i+2];}
  }
  return {lit, r:lit?R/lit:0, g:lit?G/lit:0, b:lit?B/lit:0};
}
// Exact survivor count: the pick pass writes each cell's id as a colour, so counting
// distinct non-empty colours counts cells — immune to overlap, unlike lit pixels.
function cellsDrawn(lod){
  gl.uniform1i(u('uMode'),0); gl.uniform1i(u('uModePrev'),0);
  gl.uniform1i(u('uCasteFilter'),0); gl.uniform1i(u('uCasteFilterPrev'),0);
  gl.uniform1i(u('uRegionMask'),0); gl.uniform1i(u('uRegionMaskPrev'),0);
  gl.uniform1f(u('uMorph'),1); gl.uniform1f(u('uLOD'),lod); gl.uniform1i(u('uPick'),1);
  gl.viewport(0,0,256,256); gl.clearColor(0,0,0,0);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.depthMask(true); gl.disable(gl.BLEND);
  gl.drawArrays(gl.POINTS,0,N);
  const px=new Uint8Array(256*256*4); gl.readPixels(0,0,256,256,gl.RGBA,gl.UNSIGNED_BYTE,px);
  const ids=new Set();
  for(let i=0;i<px.length;i+=4){ const id=px[i]|(px[i+1]<<8)|(px[i+2]<<16); if(id>0) ids.add(id); }
  gl.uniform1i(u('uPick'),0); gl.uniform1f(u('uLOD'),1.0);
  return ids.size;
}
const cases={
  // celltype → gene: identical cells, colour only
  recolor:{mode:3,modeP:0,caste:0,casteP:0,region:0,regionP:0},
  // no filter → isolate region bit0: half the cells leave
  filter :{mode:0,modeP:0,caste:0,casteP:0,region:1,regionP:0},
  // queen only → both castes side by side: cells arrive and the brains slide apart
  split  :{mode:0,modeP:0,caste:-1,casteP:0,region:0,regionP:0},
};
const out={};
// baseline: the state every case starts from (queen, no region mask) = 2 of the 4 cells.
// The filter/split cases are asserted against it, so a case that changes nothing fails.
out.baseline=draw({mode:0,modeP:0,caste:0,casteP:0,region:0,regionP:0},1);
for(const [name,st] of Object.entries(cases)){
  const steps=[];
  for(let i=0;i<=10;i++) steps.push(draw(st,i/10));
  out[name]=steps;
}
// ---- motion LOD: thins the cloud while moving, restores it exactly when settled ----
// Uses many cells so the kept fraction is measurable. Counts DISTINCT lit columns, which
// is a proxy for how many of the points survived the hash cull.
out.lod={ full: cellsDrawn(1.0), k60: cellsDrawn(0.6), k30: cellsDrawn(0.3), restored: cellsDrawn(1.0) };

// ---- enter/exit: the ARRIVING caste must not crawl out of the other brain ----
// queen-only → both. Worker cells (right-hand brain) are arriving: they must appear at
// their destination the whole way through, never sliding out from the centre.
const splitSt={mode:0,modeP:0,caste:-1,casteP:0,region:0,regionP:0};
out.enter={};
for(const mth of [0.2,0.45,0.7,1.0]){   // from where the arriving cells are first above the visibility floor
  const col=columns(splitSt,mth);
  // centre of mass of lit pixels strictly right of centre = where the arriving brain sits
  let num=0,den=0;
  for(let x=129;x<256;x++){ num+=x*col[x]; den+=col[x]; }
  out.enter['m'+mth]= den? +(num/den).toFixed(1) : -1;
}

// ---- cohesion: tour stop 6 → 7 (region filter clears AND the caste splits) ----
// Two QUEEN cells, one inside the filtered region and one outside, stacked vertically so
// they can be told apart by row. The one outside is only arriving — but it belongs to a
// brain that was already on screen, so both must travel together. If they separate, the
// optic lobe is sliding along without the rest of its brain.
const savedPos=pos.slice(), savedRegion=region.slice(), savedCaste=caste.slice();
pos.set([0,7000,0, 0,-7000,0, 0,7000,0, 0,-7000,0]);          // two rows, same x
region.set([1,8,1,8]); caste.set([0,0,0,0]);                   // all queen
buf(0,pos,3); buf(5,region,1); buf(2,caste,1);
out.cohesion={};
for(const mth of [0.25,0.5,0.75,1.0]){
  draw({mode:0,modeP:0,caste:-1,casteP:0,region:0,regionP:1},mth);
  const px=new Uint8Array(256*256*4); gl.readPixels(0,0,256,256,gl.RGBA,gl.UNSIGNED_BYTE,px);
  const cx=(y0,y1)=>{let num=0,den=0;
    for(let y=y0;y<y1;y++) for(let x=0;x<256;x++){
      const i=(y*256+x)*4, d=Math.abs(px[i]-7)+Math.abs(px[i+1]-9)+Math.abs(px[i+2]-14);
      if(d>18){num+=x;den++;}}
    return den?num/den:-1;};
  out.cohesion['m'+mth]=[+cx(0,128).toFixed(1), +cx(128,256).toFixed(1)];   // in-region vs arriving
}
pos.set(savedPos); region.set(savedRegion); caste.set(savedCaste);
buf(0,pos,3); buf(5,region,1); buf(2,caste,1);

// ---- the three KC compartments must render as visibly different shades ----
// same cell type, differing only in which calyx mask it carries
const kcSt={mode:0,modeP:0,caste:-2,casteP:-2,region:0,regionP:0,kc:[1,2,4],all:false};
out.kc={};
for(const [name,bits] of [['ic',1],['nc',2],['oc',4],['none',0]]){
  region.set([bits,bits,bits,bits]); buf(5,region,1);
  out.kc[name]=draw(kcSt,1);
}
region.set([1,8,1,8]); buf(5,region,1);   // restore

// morph==1 must equal the plain current state (prev == current, morph irrelevant)
const ref=draw({mode:3,modeP:3,caste:0,casteP:0,region:0,regionP:0},1);
const at1=draw(cases.recolor,1);
out.settled={ref,at1};
document.title='MORPHDATA '+JSON.stringify(out);
</script>`;
writeFileSync(OUT, page);

const dom = execFileSync(CHROMIUM, ["--headless", "--disable-gpu", "--no-sandbox",
  "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--virtual-time-budget=8000",
  "--dump-dom", "file://" + OUT], { encoding: "utf8", maxBuffer: 1 << 26 });
const m = dom.match(/MORPHDATA (\{.*?\})<\/title>/s);
if (!m) { console.error("no data — the page did not run:\n" + dom.slice(0, 1200)); process.exit(1); }
const data = JSON.parse(m[1].replace(/&quot;/g, '"'));

const fails = [];
const ok = (c, msg) => { if (!c) fails.push(msg); };
const dist = (a, b) => Math.hypot(a.r - b.r, a.g - b.g, a.b - b.b);

// the four test cells must all be rendering, or the filter/split cases prove nothing
const perCell = data.baseline.lit / 2;      // the baseline state shows 2 of the 4 cells
ok(data.baseline.lit > 40, `baseline only lit ${data.baseline.lit}px — cells are missing from the frame`);
console.log(`  baseline 2 cells = ${data.baseline.lit} lit px (${perCell.toFixed(0)}/cell)`);

for (const name of ["recolor", "filter", "split"]) {
  const st = data[name], first = st[0], last = st[st.length - 1];
  const minLit = Math.min(...st.map((s) => s.lit));
  const floor = Math.min(first.lit, last.lit) * 0.6;      // a dissolve may dip, never empty
  ok(minLit > 0, `${name}: the frame went EMPTY mid-transition`);
  ok(minLit >= floor, `${name}: lit pixels collapsed to ${minLit} (ends ${first.lit}/${last.lit})`);
  // colour travels from the old look to the new one, and mid-sweep is genuinely between
  const mid = st[5];
  ok(dist(mid, first) > 1 && dist(mid, last) > 1, `${name}: mid-sweep is not an intermediate look`);
  // monotone-ish: distance to the destination never increases by much
  let backtrack = 0;
  for (let i = 1; i < st.length; i++) {
    const d = dist(st[i], last) - dist(st[i - 1], last);
    if (d > 2) backtrack += d;
  }
  ok(backtrack < 12, `${name}: colour path backtracks (${backtrack.toFixed(1)}) — not a clean dissolve`);
  if (name === "filter") ok(last.lit < first.lit * 0.75, `filter: isolating a region removed nothing (${first.lit}→${last.lit})`);
  if (name === "split") ok(last.lit > first.lit * 1.5, `split: showing both castes added nothing (${first.lit}→${last.lit})`);
  console.log(`  ${name.padEnd(8)} lit ${first.lit}→${last.lit} (min ${minLit})  ` +
    `rgb ${first.r.toFixed(0)},${first.g.toFixed(0)},${first.b.toFixed(0)} → ` +
    `${mid.r.toFixed(0)},${mid.g.toFixed(0)},${mid.b.toFixed(0)} → ` +
    `${last.r.toFixed(0)},${last.g.toFixed(0)},${last.b.toFixed(0)}`);
}
// ---- motion LOD: thins the cloud while moving, restores every cell when settled ----
// The cull is SOFT (see the LOD_BAND fade in shaders.js — a hard cut made cells pop in and
// out as the threshold tracked camera speed). So the cells still drawing SOME pixels run
// past uLOD by the width of the band: those in it are dimmed and shrunk, not gone. This
// counts anything that renders at all, so expect uLOD + BAND, not uLOD. The gradualness
// itself is what test/flicker.mjs measures.
const LOD_BAND = 0.14;
const L = data.lod, f60 = L.k60 / L.full, f30 = L.k30 / L.full;
console.log(`  lod      ${L.full} cells → 0.6 keeps ${L.k60} (${(f60*100).toFixed(0)}%) → ` +
  `0.3 keeps ${L.k30} (${(f30*100).toFixed(0)}%) → restored ${L.restored}`);
ok(Math.abs(f60 - (0.6 + LOD_BAND)) < 0.08, `lod 0.6 kept ${(f60*100).toFixed(0)}% of cells (want ~${((0.6+LOD_BAND)*100).toFixed(0)}%)`);
ok(Math.abs(f30 - (0.3 + LOD_BAND)) < 0.08, `lod 0.3 kept ${(f30*100).toFixed(0)}% of cells (want ~${((0.3+LOD_BAND)*100).toFixed(0)}%)`);
ok(L.restored === L.full, `settling did not restore every cell (${L.restored} vs ${L.full})`);

// ---- enter/exit ----
const E = data.enter, xs = [E['m0.2'], E['m0.45'], E['m0.7'], E.m1];
console.log(`  enter    arriving brain centre-of-mass px: ${xs.join(' → ')} (should barely move)`);
ok(xs.every((x) => x > 0), "the arriving caste never appeared");
ok(Math.abs(xs[0] - xs[3]) < 12,
  `the arriving brain travelled ${Math.abs(xs[0]-xs[3]).toFixed(0)}px across the split — it is sliding out of the other brain`);

// ---- cohesion (stop 6 → 7) ----
const C = data.cohesion;
const rows = ["m0.25", "m0.5", "m0.75", "m1"].map((k) => C[k]);
console.log(`  cohesion in-region vs arriving x: ` +
  rows.map(([a, b]) => `${a}/${b}`).join("  "));
for (const [k, [a, b]] of Object.entries(C)) {
  ok(a > 0 && b > 0, `cohesion ${k}: one of the two halves never drew`);
  ok(Math.abs(a - b) < 6,
    `cohesion ${k}: the region-filtered cells sit ${Math.abs(a-b).toFixed(0)}px from the rest of their own brain — the brain is not moving as one body`);
}

// ---- KC compartments: coloured by compartment, matching the SI figures ----
const K = data.kc, d = (a, b) => Math.hypot(a.r-b.r, a.g-b.g, a.b-b.b);
console.log(`  kc       ic ${K.ic.r.toFixed(0)},${K.ic.g.toFixed(0)},${K.ic.b.toFixed(0)}  ` +
  `nc ${K.nc.r.toFixed(0)},${K.nc.g.toFixed(0)},${K.nc.b.toFixed(0)}  ` +
  `oc ${K.oc.r.toFixed(0)},${K.oc.g.toFixed(0)},${K.oc.b.toFixed(0)}  ` +
  `none ${K.none.r.toFixed(0)},${K.none.g.toFixed(0)},${K.none.b.toFixed(0)}`);
ok(d(K.ic, K.nc) > 12, `inner-compact and non-compact look the same (Δ${d(K.ic,K.nc).toFixed(1)})`);
ok(d(K.oc, K.nc) > 12, `outer-compact and non-compact look the same (Δ${d(K.oc,K.nc).toFixed(1)})`);
ok(d(K.ic, K.oc) > 12, `inner- and outer-compact look the same (Δ${d(K.ic,K.oc).toFixed(1)})`);
ok(d(K.nc, K.none) > 12, `a calyx cell was left at the flat cell-type colour (Δ${d(K.nc,K.none).toFixed(1)})`);
// hues must be the SI figures', not just "different": ic is the purple one (most blue),
// oc the yellow one (least red-dominant), nc the orange one in between
ok(K.ic.b > K.nc.b && K.ic.b > K.oc.b, `inner-compact is not the purple of the SI figure`);
ok(K.oc.g > K.nc.g, `outer-compact is not the yellow of the SI figure`);

const { ref, at1 } = data.settled;
ok(ref.lit === at1.lit && dist(ref, at1) < 0.5,
  `settled frame drifted: ${JSON.stringify(at1)} vs reference ${JSON.stringify(ref)}`);
console.log(`  settled  morph==1 matches the un-morphed render exactly (lit ${at1.lit})`);

if (fails.length) { for (const f of fails) console.error("✗ " + f); process.exit(1); }
console.log("✓ view changes dissolve: never empty, colours interpolate, settled state unchanged");
