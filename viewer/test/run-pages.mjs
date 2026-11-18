/* Headless test for the GitHub-Pages (fetch-mode) build. Serves viewer/pages/
 * over http (fetch doesn't work on file://) and drives the installed Chromium with
 * software WebGL — same self-test as the artifact, but over the full 1.3M-cell
 * base.bin + a lazily-fetched gene. Zero npm deps.
 *
 *   node viewer/test/run-pages.mjs [--build]
 */
import { execFileSync, spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ROOT = join(APP, "pages");
const SHOT = join(HERE, "pages-screenshot.png");
const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);

const TYPES = { ".html": "text/html", ".json": "application/json", ".bin": "application/octet-stream" };

async function main() {
  if (process.argv.includes("--build")) execFileSync(process.execPath, [join(APP, "build-pages.mjs")], { stdio: "inherit" });
  if (!existsSync(join(ROOT, "index.html"))) { console.error("✗ pages/index.html missing (run --build)"); process.exit(1); }
  if (!existsSync(join(ROOT, "data", "manifest.json"))) { console.error("✗ pages/data/manifest.json missing (run export_data.py --pages)"); process.exit(1); }
  if (!CHROMIUM) { console.error("✗ Chromium not found"); process.exit(1); }

  const server = createServer((req, res) => {
    let p = normalize(decodeURIComponent(req.url.split("?")[0])).replace(/^(\.\.[/\\])+/, "");
    if (p === "/" || p === "\\") p = "/index.html";
    const file = join(ROOT, p);
    if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    const ext = file.slice(file.lastIndexOf("."));
    res.writeHead(200, { "content-type": TYPES[ext] || "application/octet-stream" });
    res.end(readFileSync(file));
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}/?selftest`;

  // Chromium is spawned ASYNCHRONOUSLY, and it has to be: the page it loads is served by
  // the http server above, running in THIS process. execFileSync blocks the event loop, so
  // the server could never accept the connection — the browser sat waiting for bytes that
  // could not arrive until the browser exited. Nothing ever rendered and nothing ever
  // reported; both sides idled at 0% CPU until the budget ran out. (run.mjs may keep its
  // execFileSync: it loads a self-contained file:// page and needs no server.)
  let dom = "";
  try {
    dom = await new Promise((resolve, reject) => {
      const child = spawn(CHROMIUM, [
        "--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
        "--window-size=1280,860", "--virtual-time-budget=45000",
        "--screenshot=" + SHOT, "--dump-dom", url,
      ], { stdio: ["ignore", "pipe", "pipe"] });
      const out = [];
      child.stdout.on("data", (c) => out.push(c));
      child.stderr.resume();                 // drain, or a full pipe would stall the child
      child.on("error", reject);
      // Parse whatever --dump-dom produced whatever the exit code: Chromium exits non-zero
      // in plenty of ways that still leave a perfectly good DOM on stdout.
      child.on("close", () => resolve(Buffer.concat(out).toString("utf8")));
    });
  } finally {
    server.close();
  }

  const m = dom.match(/<div id="selftest-result"[^>]*>(\{.*?\})<\/div>/s);
  if (!m) {
    console.error("✗ no self-test result. title:", (dom.match(/<title>(.*?)<\/title>/) || [])[1]);
    console.error("  loader:", (dom.match(/Failed: [^<]*|loading[^<]*|ready[^<]*/) || [])[0]);
    process.exit(1);
  }
  const v = JSON.parse(m[1]);
  console.log("\n  BIMP 3D viewer — GitHub-Pages (fetch) self-test\n  " + "─".repeat(44));
  for (const c of v.checks || []) console.log(`  ${c.pass ? "✓" : "✗"}  ${c.name.padEnd(18)} ${c.info}`);
  if (v.error) console.log("  ✗ error: " + v.error);
  console.log("  screenshot: " + SHOT);
  console.log(v.ok ? "\n  ✓ PASS\n" : "\n  ✗ FAIL\n");
  process.exit(v.ok ? 0 : 1);
}
main();
