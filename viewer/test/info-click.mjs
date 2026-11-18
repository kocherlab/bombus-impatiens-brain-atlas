/* Do the ⓘ dots actually open the glossary in the BUILT app?
 *
 * test/info-modal.mjs pairs the keys against the table without a browser; this drives the
 * real artifact, so the delegation, the modal and the dismissal are covered too.
 *
 * The check worth having here is the Show-UMAP dot. That one sits INSIDE a control which
 * listens for its own clicks, so a bubbling delegate would fire after the toggle and
 * asking what the UMAP is would also switch it on. The handler captures instead; this
 * proves it, by asserting the toggle's aria-pressed is unchanged across the click.
 *
 *   node viewer/test/info-click.mjs
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = dirname(HERE);
const ART = process.env.BIMP_ART || join(APP, "dist", "bimp_brain.html");   // override for negative controls
const TMP = join(process.env.BIMP_TEST_TMP || "/tmp", "tmp");
const DRIVER = join(TMP, "info-click.html");

const CHROMIUM = [
  join(process.env.HOME || "", ".cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"),
  join(process.env.HOME || "", ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"),
].find(existsSync);
if (!CHROMIUM) { console.error("Chromium not found in the Playwright cache"); process.exit(1); }
if (!existsSync(ART)) { console.error("✗ dist not built — run: node viewer/build.mjs"); process.exit(1); }

const art = readFileSync(ART, "utf8");
const probe = `
<script>
(async function(){
  const rep = (o) => { const p = document.createElement('pre'); p.id='probe';
    p.textContent = JSON.stringify(o); document.body.appendChild(p); };
  try {
    await BIMP.ready;
    // Timer, not rAF: every frame here is a full point cloud through swiftshader and the
    // probe would starve waiting on one. Nothing below needs a painted frame.
    await new Promise(r => setTimeout(r, 300));
    const modal = document.getElementById('info-modal');
    const esc = () => document.dispatchEvent(new KeyboardEvent('keydown', {key:'Escape'}));
    const seen = [];
    for (const dot of document.querySelectorAll('[data-info]')) {
      const key = dot.dataset.info;
      // the toggle this dot may or may not be sitting inside
      const host = dot.closest('.toggle');
      const before = host ? host.getAttribute('aria-pressed') : null;
      dot.click();
      const opened = modal.classList.contains('show');
      const title = (document.getElementById('if-name').textContent || '').trim();
      const bodyLen = (document.getElementById('if-body').textContent || '').trim().length;
      const after = host ? host.getAttribute('aria-pressed') : null;
      esc();
      seen.push({ key, opened, title, bodyLen, closed: !modal.classList.contains('show'),
                  host: !!host, flipped: host ? before !== after : false });
    }
    rep({ ok:true, seen });
  } catch (e) { rep({ ok:false, error: String(e && e.message || e) }); }
})();
</script>`;
writeFileSync(DRIVER, art + probe);

let dump = "";
try {
  dump = execFileSync(CHROMIUM, [
    "--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
    "--window-size=1280,860", "--virtual-time-budget=25000", "--dump-dom",
    "file://" + DRIVER,
  ], { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
} catch (e) { dump = (e.stdout || "").toString(); }

const m = dump.match(/<pre id="probe">([\s\S]*?)<\/pre>/);
if (!m) { console.error("✗ probe never reported — page failed to load?\n" + dump.slice(0, 1500)); process.exit(1); }
const r = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">"));
if (!r.ok) { console.error("✗ probe errored: " + r.error); process.exit(1); }

let fail = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "✓" : "✗"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) fail++;
};

console.log("— glossary dots in the built app —");
check("every dot was found and clicked", r.seen.length >= 8, r.seen.length + " dots");

const dead = r.seen.filter((d) => !d.opened || !d.title || d.bodyLen < 80);
check("every dot opens a filled card", dead.length === 0,
  dead.length ? "empty: " + dead.map((d) => d.key).join(", ")
              : r.seen.map((d) => d.key + "(" + d.bodyLen + ")").join(" "));

const stuck = r.seen.filter((d) => !d.closed);
check("Escape closes the card", stuck.length === 0,
  stuck.length ? "stuck open: " + stuck.map((d) => d.key).join(", ") : "all " + r.seen.length + " closed");

// the capture-phase guard
const nested = r.seen.filter((d) => d.host);
check("a dot inside a toggle does not flip it", nested.length > 0 && nested.every((d) => !d.flipped),
  nested.length ? nested.map((d) => d.key + (d.flipped ? " FLIPPED" : " ok")).join(", ")
                : "no nested dot found — the capture-phase guard is untested");

console.log(fail ? `\n${fail} check(s) failed` : "\nthe glossary opens");
process.exit(fail ? 1 : 0);
