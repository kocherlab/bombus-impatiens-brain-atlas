/* Capture clean screenshots of each color mode (intro dismissed, spin off) for
 * visual QA. Uses the same installed Chromium as run.mjs. No npm deps.
 *
 *   node viewer/test/shots.mjs [--build]
 *
 * Writes test/shot-<view>.png for celltype / caste / conf / gene.
 */
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ART = join(APP, "dist", "bimp_brain.html");
const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);

if (!CHROMIUM) { console.error("✗ Chromium not found"); process.exit(1); }
if (process.argv.includes("--build")) execFileSync(process.execPath, [join(APP, "build.mjs")], { stdio: "inherit" });
if (!existsSync(ART)) { console.error("✗ dist not built (use --build)"); process.exit(1); }

const views = [
  { name: "default", q: "theta=0.9&phi=1.05" },
  { name: "celltype-mb", q: "view=celltype&region=mb&theta=0.9&phi=1.05" },
  { name: "caste-both", q: "view=caste&caste=both&theta=0.9&phi=1.05" },
  { name: "gene", q: "view=gene&gene=0&theta=0.9&phi=1.05" },
];
for (const v of views) {
  const shot = join(HERE, "shot-" + v.name + ".png");
  const url = `file://${ART}?nointro&spin=0&${v.q}`;
  execFileSync(CHROMIUM, [
    "--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
    "--window-size=1280,860", "--virtual-time-budget=8000",
    "--screenshot=" + shot, url,
  ], { stdio: "ignore", maxBuffer: 64 * 1024 * 1024 });
  console.log("✓ " + shot);
}
