// Local preview of viewer/pages on 127.0.0.1:8765. Missing /data/* files (gene
// files not exported locally) are fetched from the live site and passed through.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, extname, normalize } from "node:path";

const ROOT = process.argv[2];
const LIVE = "https://kocherlab.github.io/bombus-impatiens-brain-atlas";
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".webp": "image/webp",
  ".png": "image/png", ".jpg": "image/jpeg", ".bin": "application/octet-stream", ".woff2": "font/woff2" };

createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  let file = normalize(join(ROOT, path));
  if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; }
  try {
    if ((await stat(file)).isDirectory()) file = join(file, "index.html");
    const body = await readFile(file);
    res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream", "cache-control": "no-cache" });
    res.end(body);
  } catch {
    if (path.startsWith("/data/")) {
      try {
        const r = await fetch(LIVE + path);
        res.writeHead(r.status, { "content-type": r.headers.get("content-type") || "application/octet-stream" });
        res.end(Buffer.from(await r.arrayBuffer()));
        return;
      } catch { /* fall through */ }
    }
    res.writeHead(404).end("not found");
  }
}).listen(8765, "127.0.0.1", () => console.log("serving " + ROOT + " on http://127.0.0.1:8765/"));
