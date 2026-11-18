/* Build the self-contained artifact from src/ modules.
 *
 * Concatenation only — no transformation, no bundler (npm registry is unavailable
 * in this environment). Every module is a plain IIFE publishing to `window.BIMP`,
 * loaded in dependency order, so inlining is a reliable string join. The output is
 * the exact file the test harness loads, so `what we test == what we ship`.
 *
 *   node viewer/build.mjs            → dist/bimp_brain.html
 *
 * Run export_data.py first to (re)generate src/data.js.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, "src");
const read = (f) => readFileSync(join(SRC, f), "utf8");

// dependency order
const MODULES = [
  "palette.js", "mat4.js", "shaders.js", "camera.js",
  "decode.js", "state.js", "renderer.js", "ui.js", "tour.js", "boot.js",
];

const css = read("styles.css");
const body = read("body.html");
let dataLite, dataFull;
try {
  dataLite = read("data-lite.js");
  dataFull = read("data-full.js");
} catch {
  console.error("✗ src/data-lite.js / data-full.js missing — run: PYTHONPATH=src .venv/bin/python viewer/export_data.py");
  process.exit(1);
}
const scripts = MODULES.map((m) => `<script>\n${read(m)}\n</script>`).join("\n");

// Order matters for progressive paint: manifest + LITE payload, then all code
// (boot decodes lite and paints), then the heavy FULL blob LAST — it streams in
// while the lite view is already interactive, then hot-swaps.
const html = `<title>Bombus impatiens Brain Atlas — 3D</title>
<style>
${css}
</style>
${body}
<script>
${dataLite}
</script>
${scripts}
<script>
${dataFull}
</script>
`;

mkdirSync(join(HERE, "dist"), { recursive: true });
const out = join(HERE, "dist", "bimp_brain.html");
writeFileSync(out, html);
console.log(`✓ wrote ${out}  (${(html.length / 1e6).toFixed(2)} MB)`);
