/* Why does the atlas SHIMMER when it spins, and BLINK under the pointer?
 *
 * Three separate mechanisms, measured separately, all in real headless WebGL2 using
 * src/shaders.js and src/mat4.js unmodified:
 *
 *   shimmer  — a dense cloud is drawn at pose θ and again at θ+0.002 rad (ONE frame of
 *              the idle auto-orbit; every point moves well under a pixel). Any pixel
 *              that changes a lot between those two frames is not motion, it is a
 *              rendering artifact. Reported as a fraction of lit pixels.
 *
 *   faint    — depth writes are all-or-nothing, so a cell drawn at a tenth alpha still
 *              claimed the depth buffer and erased whatever sat behind it, with vertex
 *              order alone deciding which cell won. Probe: region view's grey context
 *              wash laid over a coloured cell, in BOTH buffer orders.
 *
 *   lodPop   — motion LOD culls a deterministic hash fraction. The FRACTION moves every
 *              frame with camera speed, so a hard threshold pops whole cells in and out.
 *              Probe: step uLOD by 0.01 and measure the largest per-cell brightness jump.
 *              A hard cull gives 1.0 (full → gone in one step).
 *
 * The GL draw state is replicated from renderer.js; a source guard below fails if the
 * renderer's state and this test's copy drift apart.
 *
 *   node viewer/test/flicker.mjs
 *   BIMP_SRC=/path/to/old/src node viewer/test/flicker.mjs    # negative control
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const SRC = process.env.BIMP_SRC || join(APP, "src");
const TMP = join(process.env.BIMP_TEST_TMP || "/tmp", "tmp");
// keyed on SRC so a negative-control run can't overwrite the page a normal run is loading
const OUT = join(TMP, "flicker-page-" + createHash("sha1").update(SRC).digest("hex").slice(0, 8) + ".html");

const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
if (!CHROMIUM) { console.error("Chromium not found in the Playwright cache"); process.exit(1); }

const shaderSrc = readFileSync(join(SRC, "shaders.js"), "utf8");
const mat4Src = readFileSync(join(SRC, "mat4.js"), "utf8");
const rendererSrc = readFileSync(join(SRC, "renderer.js"), "utf8");

// Source guard: this test hardcodes the renderer's draw state, so if the renderer's
// state changes the measurements stop meaning anything. Fail loudly instead.
const STATE_RE = /gl\.enable\(gl\.DEPTH_TEST\);\s*gl\.depthMask\(true\);\s*gl\.enable\(gl\.BLEND\)/;
if (!STATE_RE.test(rendererSrc)) {
  console.error("✗ renderer.js draw state no longer matches this test's copy (DEPTH_TEST + depthMask(true) + BLEND).");
  console.error("  Update the draw() helper in test/flicker.mjs to match, then re-run.");
  process.exit(1);
}

const page = `<!doctype html><meta charset=utf-8><title>FLICKER</title>
<canvas id=c width=320 height=320></canvas>
<script>window.BIMP={};</script>
<script>${mat4Src}</script>
<script>${shaderSrc}</script>
<script>
const S=BIMP.shaders;
const cv=document.getElementById('c');
// antialias mirrors renderer.js; preserveDrawingBuffer so we can read after the fact
const gl=cv.getContext('webgl2',{antialias:true,alpha:false,premultipliedAlpha:false,preserveDrawingBuffer:true});
const W=320,H=320;
function mk(){const p=gl.createProgram();
  for(const [t,src] of [[gl.VERTEX_SHADER,S.VS],[gl.FRAGMENT_SHADER,S.FS]]){
    const sh=gl.createShader(t); gl.shaderSource(sh,src); gl.compileShader(sh);
    if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS)) throw new Error('compile: '+gl.getShaderInfoLog(sh));
    gl.attachShader(p,sh);}
  gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error('link: '+gl.getProgramInfoLog(p));
  return p;}
const prog=mk(); gl.useProgram(prog);
const u=n=>gl.getUniformLocation(prog,n);

// ---- scene ---------------------------------------------------------------------
// A dense ball of cells (the shimmer probe) + a 400-cell non-overlapping grid (the LOD
// probe) + 2 hand-placed cells (the halo probe). All in one buffer; each probe draws
// only its own slice via drawArrays offsets.
const NBALL=24000, NGRID=400, NHALO=4;
const OFF_BALL=0, OFF_GRID=NBALL, OFF_HALO=NBALL+NGRID, N=NBALL+NGRID+NHALO;
const pos=new Float32Array(N*3), ct=new Float32Array(N), caste=new Float32Array(N),
      conf=new Float32Array(N), gene=new Float32Array(N), genePrev=new Float32Array(N),
      region=new Float32Array(N), birth=new Float32Array(N);
// deterministic PRNG — the same cloud every run
let _s=12345; const rnd=()=>{_s=(_s*1664525+1013904223)>>>0; return _s/4294967296;};
for(let i=0;i<NBALL;i++){
  // uniform in a ball, radius 0.45 of the coordinate range
  let x,y,z,d; do{x=rnd()*2-1;y=rnd()*2-1;z=rnd()*2-1;d=x*x+y*y+z*z;}while(d>1);
  pos[i*3]=x*14000; pos[i*3+1]=y*14000; pos[i*3+2]=z*14000;
  ct[i]=i%6; caste[i]=0; conf[i]=0.5; gene[i]=0.5; genePrev[i]=0.5; region[i]=1;
}
for(let i=0;i<NGRID;i++){          // spread wide: no two grid cells share a pixel
  const k=OFF_GRID+i, gx=i%20, gy=(i/20)|0;
  pos[k*3]=(gx-9.5)*3200; pos[k*3+1]=(gy-9.5)*3200; pos[k*3+2]=0;
  ct[k]=i%4; caste[k]=0; conf[k]=0.5; gene[k]=0.5; genePrev[k]=0.5; region[k]=1;
}
// faint-occludes probe: [0] = a bright cell INSIDE a region mask, far from the camera;
// [1] = a context cell in NO mask (region view renders it at a tenth alpha), directly in
// front of it. The faint one must not erase the bright one.
{ const a=OFF_HALO, b=OFF_HALO+1;
  pos[a*3]=0; pos[a*3+1]=0; pos[a*3+2]=-6000;
  pos[b*3]=0; pos[b*3+1]=0; pos[b*3+2]=16000;
  ct[a]=0; ct[b]=1; caste[a]=caste[b]=0; conf[a]=conf[b]=0.5;
  gene[a]=gene[b]=0.5; genePrev[a]=genePrev[b]=0.5;
  region[a]=1;      // bit0 → inside uRegBits[0], coloured at full strength
  region[b]=2;      // bit1 → inside nothing, the grey context wash
  // ...and the SAME pair in the opposite buffer order. Nothing about the picture should
  // depend on which of two cells the vertex buffer happens to list first; with depth
  // writes on it does, so both orders are asserted.
  const c2=OFF_HALO+2, d2=OFF_HALO+3;
  pos[c2*3]=0; pos[c2*3+1]=0; pos[c2*3+2]=16000;    // grey, in front, listed FIRST
  pos[d2*3]=0; pos[d2*3+1]=0; pos[d2*3+2]=-6000;    // coloured, behind, listed second
  ct[c2]=1; ct[d2]=0; caste[c2]=caste[d2]=0; conf[c2]=conf[d2]=0.5;
  gene[c2]=gene[d2]=0.5; genePrev[c2]=genePrev[d2]=0.5;
  region[c2]=2; region[d2]=1; }
function buf(loc,arr,size){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0);}
gl.bindVertexArray(gl.createVertexArray());
buf(0,pos,3); buf(1,ct,1); buf(2,caste,1); buf(3,conf,1); buf(4,gene,1); buf(5,region,1);
buf(6,birth,1); buf(7,genePrev,1);

const M=BIMP.mat4;
const proj=M.persp(48*Math.PI/180, 1, 0.05, 60);
const pal=new Float32Array(32*3);
for(let i=0;i<32;i++){pal[i*3]=0.35+0.6*((i*37)%7)/6; pal[i*3+1]=0.3+0.65*((i*53)%5)/4; pal[i*3+2]=0.4+0.55*((i*29)%3)/2;}
gl.uniform3fv(u('uPalette'),pal);
gl.uniform3fv(u('uQueen'),[0.96,0.77,0.26]); gl.uniform3fv(u('uWorker'),[0.18,0.83,0.78]);
const BG=[0.027,0.035,0.055];
gl.uniform3f(u('uBg'),BG[0],BG[1],BG[2]);
gl.uniform1fv(u('uVisible'),new Float32Array(32).fill(1));
gl.uniform1f(u('uZ'),1); gl.uniform1f(u('uDpr'),1);
gl.uniform1f(u('uProgress'),2); gl.uniform1f(u('uFade'),0.14);
gl.uniform1f(u('uOpacity'),0.85); gl.uniform1f(u('uSplit'),1.15);
gl.uniform1i(u('uIso'),-1); gl.uniform1i(u('uHover'),-1); gl.uniform1i(u('uSelected'),-1);
gl.uniform1i(u('uPick'),0); gl.uniform1f(u('uLOD'),1.0);
gl.uniform1iv(u('uKcBits[0]'), new Int32Array([0,0,0]));
gl.uniform3fv(u('uKcCol[0]'), new Float32Array(9));
gl.uniform1iv(u('uRegBits[0]'), new Int32Array([1,0,0,0,0,0]));   // only bit0 is a real mask
gl.uniform3fv(u('uRegCol[0]'), new Float32Array([0.95,0.85,0.25, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]));
gl.uniform3fv(u('uRegNone'), new Float32Array([0.22,0.25,0.31]));
gl.uniform1i(u('uMode'),0); gl.uniform1i(u('uModePrev'),0);
gl.uniform1i(u('uCasteFilter'),0); gl.uniform1i(u('uCasteFilterPrev'),0);
gl.uniform1i(u('uRegionMask'),0); gl.uniform1i(u('uRegionMaskPrev'),0);
gl.uniform1f(u('uGeneLo'),0); gl.uniform1f(u('uGeneHi'),1);
gl.uniform1f(u('uGeneLoPrev'),0); gl.uniform1f(u('uGeneHiPrev'),1);
gl.uniform1f(u('uMorph'),1);

// ---- draw ----------------------------------------------------------------------
// State replicated from renderer.js drawScene() (guarded on the node side).
function draw(o){
  const theta=o.theta||0, radius=o.radius||4;
  gl.uniform1i(u('uMode'),o.mode||0); gl.uniform1i(u('uModePrev'),o.mode||0);
  const eye=[Math.sin(theta)*radius, 0, Math.cos(theta)*radius];
  const view=M.lookAt(eye,[0,0,0],[0,1,0]);
  gl.uniformMatrix4fv(u('uMVP'),false,M.mul(proj,view));
  gl.uniformMatrix4fv(u('uMV'),false,view);
  gl.uniform1f(u('uFogRef'),radius);
  gl.uniform1f(u('uSize'),o.size==null?2.6:o.size);
  gl.uniform1f(u('uLOD'),o.lod==null?1.0:o.lod);
  gl.viewport(0,0,W,H); gl.clearColor(BG[0],BG[1],BG[2],1);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.depthMask(o.noDepthWrite?false:true); gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
  // mirrors renderer.js drawScene(): one pass when no cell is deliberately faded,
  // otherwise solid-with-depth then faint-without.
  const uPass=u('uPass');
  if(o.twoPass && uPass){
    gl.uniform1i(uPass,1); gl.depthMask(true);  gl.drawArrays(gl.POINTS,o.first,o.count);
    gl.uniform1i(uPass,2); gl.depthMask(false); gl.drawArrays(gl.POINTS,o.first,o.count);
    gl.depthMask(true);
  } else {
    if(uPass) gl.uniform1i(uPass,0);
    gl.drawArrays(gl.POINTS, o.first, o.count);
  }
  const px=new Uint8Array(W*H*4); gl.readPixels(0,0,W,H,gl.RGBA,gl.UNSIGNED_BYTE,px);
  return px;
}
const bg8=[Math.round(BG[0]*255),Math.round(BG[1]*255),Math.round(BG[2]*255)];
const lum=(p,i)=>0.2126*p[i]+0.7152*p[i+1]+0.0722*p[i+2];
const isLit=(p,i)=>Math.abs(p[i]-bg8[0])+Math.abs(p[i+1]-bg8[1])+Math.abs(p[i+2]-bg8[2])>18;

// ---- probe 1: shimmer ----------------------------------------------------------
// One auto-orbit frame is 0.12 rad/s / 60 fps = 0.002 rad. Nothing moves a whole pixel.
function shimmer(opts){
  const A=draw(Object.assign({first:OFF_BALL,count:NBALL,theta:0.0},opts));
  const B=draw(Object.assign({first:OFF_BALL,count:NBALL,theta:0.002},opts));
  let lit=0, jumped=0, sum=0;
  for(let i=0;i<A.length;i+=4){
    const a=isLit(A,i), b=isLit(B,i);
    if(!a&&!b) continue;
    lit++;
    const d=Math.abs(lum(A,i)-lum(B,i));
    sum+=d;
    if(d>25) jumped++;         // a quarter-of-a-step change from a sub-pixel move
  }
  // mean brightness of the cloud, so a "fix" that just dims everything is visible
  let bright=0,bn=0;
  for(let i=0;i<A.length;i+=4) if(isLit(A,i)){bright+=lum(A,i);bn++;}
  return {lit, frac:lit?jumped/lit:0, meanDelta:lit?sum/lit:0, meanLum:bn?bright/bn:0, litPx:bn};
}

// ---- probe 2: a faint cell must not write depth ---------------------------------
// Region view: the bright masked cell behind, the tenth-alpha grey context cell in
// front. Depth writes are all-or-nothing, so the grey one erased the coloured one
// however transparent it was made — "grey goes over all of the points".
function faintOccludes(){
  const c=(W*(H/2)+W/2)*4;                       // both cells project to the centre
  const opt={mode:4,size:30,twoPass:true};
  const alone=draw(Object.assign({first:OFF_HALO,count:1},opt));
  const aloneL=lum(alone,c);
  const greyOnly=draw(Object.assign({first:OFF_HALO+1,count:1},opt));
  const both=draw(Object.assign({first:OFF_HALO,count:2},opt));       // coloured listed first
  const rev=draw(Object.assign({first:OFF_HALO+2,count:2},opt));      // grey listed first
  return {aloneL, greyL:lum(greyOnly,c), bothL:lum(both,c), revL:lum(rev,c)};
}

// ---- probe 3: LOD popping ------------------------------------------------------
// Per-cell brightness across a 0.01 step in uLOD. Cells are on a wide grid, so each
// owns its own pixels: sample every cell's centre by locating it once at lod 1.
function cellCentres(){
  const px=draw({first:OFF_GRID,count:NGRID,size:6});
  const seen=[];
  for(let y=0;y<H;y++) for(let x=0;x<W;x++){ const i=(y*W+x)*4; if(isLit(px,i)) seen.push([x,y,i]); }
  return seen;   // every lit pixel of the grid; per-cell grouping is not needed for the max-jump metric
}
function lodPop(){
  const pts=cellCentres();
  let worst=0, worstAt=0;
  let prev=draw({first:OFF_GRID,count:NGRID,size:6,lod:1.0});
  for(let step=1;step<=20;step++){
    const l=1.0-step*0.01;
    const cur=draw({first:OFF_GRID,count:NGRID,size:6,lod:l});
    for(const [,,i] of pts){
      const d=Math.abs(lum(prev,i)-lum(cur,i))/255;
      if(d>worst){worst=d; worstAt=l;}
    }
    prev=cur;
  }
  return {worstJump:worst, at:worstAt, sampled:pts.length};
}

const out={
  shimmer: shimmer({}),
  shimmerNoDepthWrite: shimmer({noDepthWrite:true}),   // diagnostic, not asserted
  faint: faintOccludes(),
  lod: lodPop(),
};
document.title='DONE';
const pre=document.createElement('pre'); pre.id='out'; pre.textContent=JSON.stringify(out);
document.body.appendChild(pre);
</script>`;

writeFileSync(OUT, page);
const dump = execFileSync(CHROMIUM, [
  "--headless", "--disable-gpu", "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
  "--no-sandbox", "--virtual-time-budget=30000", "--dump-dom", "file://" + OUT,
], { encoding: "utf8", maxBuffer: 1 << 28 });

const m = dump.match(/<pre id="out">([\s\S]*?)<\/pre>/);
if (!m) { console.error("✗ page produced no result — GL error?\n" + dump.slice(0, 3000)); process.exit(1); }
const r = JSON.parse(m[1]);

let fail = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "✓" : "✗"} ${name}  ${detail}`);
  if (!ok) fail++;
};

console.log("— flicker probes —");
check("shimmer: a sub-pixel turn does not repaint the cloud",
  r.shimmer.frac < 0.05,
  `${(r.shimmer.frac * 100).toFixed(1)}% of ${r.shimmer.lit} lit px jump >25/255 (mean Δ ${r.shimmer.meanDelta.toFixed(1)})`);
console.log(`  · diagnostic: same probe with depth writes off → ${(r.shimmerNoDepthWrite.frac * 100).toFixed(1)}%`);
console.log(`  · cloud mean luminance ${r.shimmer.meanLum.toFixed(1)} over ${r.shimmer.litPx} lit px`);

check("faint: the grey region wash does not blank the coloured cell behind it",
  r.faint.bothL > 0.6 * r.faint.aloneL && r.faint.revL > 0.6 * r.faint.aloneL,
  `coloured alone ${r.faint.aloneL.toFixed(1)} → behind the wash ${r.faint.bothL.toFixed(1)} / ${r.faint.revL.toFixed(1)} in the other buffer order (wash alone ${r.faint.greyL.toFixed(1)})`);

check("lod: thinning fades cells out instead of popping them",
  r.lod.worstJump < 0.35,
  `largest per-pixel jump across a 0.01 LOD step = ${(r.lod.worstJump * 100).toFixed(0)}% at lod ${r.lod.at.toFixed(2)}`);

console.log(fail ? `\n${fail} check(s) failed` : "\nall flicker checks passed");
process.exit(fail ? 1 : 0);
