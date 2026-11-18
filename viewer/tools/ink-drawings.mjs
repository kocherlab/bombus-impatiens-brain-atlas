/* Turn scanned ink drawings into ink-only art: paper becomes transparent, ink keeps its
 * tone, so a drawing sits directly on any ground (the light page, or the dark viewer with
 * a CSS invert) instead of arriving in a white box.
 *
 * Alpha is taken from lightness: anything lighter than PAPER is fully transparent, anything
 * darker than INK is fully opaque, linear in between. Existing transparency in the source is
 * respected (pixels are composited over white first). Runs in headless Chrome for its canvas.
 *
 *   node viewer/tools/ink-drawings.mjs <chrome-debug-port> <out-dir> <maxPx> <file>...
 */
import { readFileSync, writeFileSync } from "node:fs";
import { basename, extname } from "node:path";
const [port, outDir, maxPx, ...files] = process.argv.slice(2);
const PAPER = 228, INK = 70;
const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
const ws = new WebSocket(t.webSocketDebuggerUrl); await new Promise((r) => (ws.onopen = r));
let id = 0; const pend = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (x) => (await send("Runtime.evaluate", { expression: x, awaitPromise: true, returnByValue: true })).result?.result?.value;
for (const f of files) {
  const mime = /\.png$/i.test(f) ? "image/png" : "image/jpeg";
  const src = `data:${mime};base64,${readFileSync(f).toString("base64")}`;
  const out = await ev(`(async () => {
    const im = new Image(); im.src = ${JSON.stringify(src)}; await im.decode();
    const k = Math.min(1, ${+maxPx} / Math.max(im.width, im.height));
    const W = Math.round(im.width * k), H = Math.round(im.height * k);
    const c = new OffscreenCanvas(W, H), x = c.getContext("2d");
    x.drawImage(im, 0, 0, W, H);
    const d = x.getImageData(0, 0, W, H), p = d.data;
    let x0 = W, y0 = H, x1 = 0, y1 = 0;
    for (let i = 0; i < p.length; i += 4) {
      const a = p[i + 3] / 255;
      const L = (0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2]) * a + 255 * (1 - a);
      const v = Math.max(0, Math.min(1, (${PAPER} - L) / (${PAPER} - ${INK})));
      p[i] = p[i + 1] = p[i + 2] = 0; p[i + 3] = Math.round(v * 255);
      if (v > 0.15) { const q = i / 4, X = q % W, Y = (q / W) | 0; if (X < x0) x0 = X; if (X > x1) x1 = X; if (Y < y0) y0 = Y; if (Y > y1) y1 = Y; }
    }
    x.putImageData(d, 0, 0);
    const pad = 12; x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(W - 1, x1 + pad); y1 = Math.min(H - 1, y1 + pad);
    const c2 = new OffscreenCanvas(x1 - x0 + 1, y1 - y0 + 1); c2.getContext("2d").drawImage(c, -x0, -y0);
    const b = await c2.convertToBlob({ type: "image/webp", quality: 0.9 });
    const u8 = new Uint8Array(await b.arrayBuffer()); let s = ""; for (const ch of u8) s += String.fromCharCode(ch);
    return { w: c2.width, h: c2.height, b64: btoa(s) };
  })()`);
  const name = basename(f, extname(f)).toLowerCase().replace(/[^a-z0-9]+/g, "-") + ".webp";
  writeFileSync(`${outDir}/${name}`, Buffer.from(out.b64, "base64"));
  console.log(`✓ ${name}  ${out.w}×${out.h}`);
}
process.exit(0);
