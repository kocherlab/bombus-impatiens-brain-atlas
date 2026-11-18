/* Screenshot the GitHub-Pages build over local http (fetch mode). Captures the
 * lite tier's fast paint. Usage: node viewer/test/shot-pages.mjs [budgetMs]
 */
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize, extname } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(dirname(HERE), "pages");
const SHOT = join(HERE, "pages-shot.png");
const budget = process.argv[2] || "9000";
const CH = ["chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell",
  "chromium-1228/chrome-linux64/chrome"]
  .map((p) => join(process.env.HOME || "", ".cache/ms-playwright", p)).find(existsSync);
const TYPES = { ".html": "text/html", ".json": "application/json", ".bin": "application/octet-stream" };

const server = createServer((req, res) => {
  let p = normalize(decodeURIComponent(req.url.split("?")[0])).replace(/^(\.\.[/\\])+/, "");
  if (p === "/" || p === "\\") p = "/index.html";
  const f = join(ROOT, p);
  if (!f.startsWith(ROOT) || !existsSync(f) || statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": TYPES[extname(f)] || "application/octet-stream" });
  res.end(readFileSync(f));
});
server.listen(0, "127.0.0.1", () => {
  const port = server.address().port;
  try {
    execFileSync(CH, ["--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
      "--window-size=1280,860", "--virtual-time-budget=" + budget, "--screenshot=" + SHOT,
      `http://127.0.0.1:${port}/?nointro&spin=0&theta=0.7&phi=1.35&radius=2.6`], { stdio: "ignore" });
    console.log("✓ " + SHOT);
  } catch (e) { console.error("✗ " + e.message); }
  server.close();
});
