/* Headless FE test harness — zero npm dependencies.
 *
 * The npm registry is unavailable in this environment, so instead of
 * Vitest/Playwright we drive the already-installed Chromium (bundled with the
 * Playwright browser cache) directly via its CLI:
 *
 *   • loads dist/bimp_brain.html?selftest via file:// with software WebGL,
 *   • the page runs BIMP.selfTest() (real GL: renders, reads pixels, GPU-picks,
 *     recolors by gene) and writes a JSON verdict + a screenshot,
 *   • we parse the DOM dump, print per-check pass/fail, and exit non-zero on
 *     failure or on any console error.
 *
 *   node viewer/test/run.mjs            # assumes dist is built
 *   node viewer/test/run.mjs --build    # build first
 *
 * Emits test/last-screenshot.png for visual inspection.
 */
import { execFileSync } from "node:child_process";
import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ART = join(APP, "dist", "bimp_brain.html");
const SHOT = join(HERE, "last-screenshot.png");
const DOM = join(HERE, "last-dom.html");

const CHROMIUM_CANDIDATES = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
];

function findChromium() {
  for (const c of CHROMIUM_CANDIDATES) if (existsSync(c)) return c;
  // glob-ish fallback: any chrome-headless-shell under the cache
  throw new Error("Chromium not found in Playwright cache. Looked at:\n" + CHROMIUM_CANDIDATES.join("\n"));
}

function run() {
  if (process.argv.includes("--build")) {
    execFileSync(process.execPath, [join(APP, "build.mjs")], { stdio: "inherit" });
  }
  if (!existsSync(ART)) { console.error("✗ " + ART + " not built. Run with --build."); process.exit(1); }

  const chromium = findChromium();
  const url = "file://" + ART + "?selftest";
  const flags = [
    "--headless=new", "--no-sandbox", "--disable-gpu-sandbox",
    "--use-gl=angle", "--use-angle=swiftshader",
    "--window-size=1280,860",
    "--virtual-time-budget=20000",
    "--enable-logging=stderr", "--v=0",
    "--screenshot=" + SHOT,
    "--dump-dom",
    url,
  ];

  let dom = "", stderr = "";
  try {
    dom = execFileSync(chromium, flags, { encoding: "utf8", maxBuffer: 256 * 1024 * 1024, stdio: ["ignore", "pipe", "pipe"] });
  } catch (e) {
    dom = (e.stdout || "").toString(); stderr = (e.stderr || "").toString();
  }
  writeFileSync(DOM, dom);

  // extract the self-test verdict
  const m = dom.match(/<div id="selftest-result"[^>]*>(\{.*?\})<\/div>/s);
  if (!m) {
    console.error("✗ no self-test result found in DOM. Title:", (dom.match(/<title>(.*?)<\/title>/) || [])[1]);
    console.error("  (page may have failed to boot — check test/last-dom.html)");
    const err = (dom.match(/Failed: [^<]*/) || [])[0];
    if (err) console.error("  loader said:", err);
    process.exit(1);
  }
  let verdict;
  try { verdict = JSON.parse(m[1]); } catch (e) { console.error("✗ bad JSON:", m[1].slice(0, 200)); process.exit(1); }

  // console errors from the page (SwiftShader prints benign warnings; only flag Error/GL_ERROR)
  const consoleErrors = stderr.split("\n").filter((l) =>
    /:ERROR:|Uncaught|GL_INVALID|shader:|link:/.test(l) && !/SwiftShader|Fontconfig|GPU stall|gpu_/.test(l));

  console.log("\n  BIMP 3D viewer — headless self-test\n  " + "─".repeat(40));
  for (const c of verdict.checks || []) {
    console.log(`  ${c.pass ? "✓" : "✗"}  ${c.name.padEnd(18)} ${c.info}`);
  }
  if (verdict.error) console.log("  ✗  boot error: " + verdict.error);
  if (consoleErrors.length) {
    console.log("  " + "─".repeat(40));
    console.log("  console errors:");
    consoleErrors.slice(0, 10).forEach((l) => console.log("    " + l.trim()));
  }
  console.log("  " + "─".repeat(40));
  console.log("  screenshot: " + SHOT);
  const ok = verdict.ok && consoleErrors.length === 0;
  console.log(ok ? "\n  ✓ PASS\n" : "\n  ✗ FAIL\n");
  process.exit(ok ? 0 : 1);
}
run();
