// fraction of bright pixels in the canvas centre of a screenshot (BMP from sips)
import { readFileSync } from "node:fs";
const b = readFileSync(process.argv[2]); const off = b.readUInt32LE(10), w = b.readInt32LE(18), h = Math.abs(b.readInt32LE(22)), bpp = b.readUInt16LE(28) / 8;
const row = Math.ceil(w * bpp / 4) * 4; let lit = 0, n = 0;
for (let y = Math.floor(h * 0.3); y < h * 0.7; y += 2) for (let x = Math.floor(w * 0.3); x < w * 0.7; x += 2) { const o = off + y * row + x * bpp; n++; if (b[o] + b[o + 1] + b[o + 2] > 120) lit++; }
console.log(process.argv[2].split("/").pop(), (lit / n).toFixed(3));
