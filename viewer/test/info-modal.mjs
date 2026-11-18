/* Does every ⓘ in the markup actually open something?
 *
 * The dots are wired by delegation on [data-info], so a typo'd key is silent: the button
 * renders, the click fires, openInfo() finds nothing and returns, and the user gets a dead
 * icon with no error anywhere. This pairs body.html against the INFO table and fails on a
 * mismatch in EITHER direction — a dot with no entry, or an entry no dot can reach.
 *
 * Pure Node; neither file needs a DOM to inspect.
 *
 *   node viewer/test/info-modal.mjs
 */
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = process.env.BIMP_SRC || join(dirname(HERE), "src");

// ui.js destructures BIMP.* at IIFE time and touches the DOM only inside functions,
// so stub siblings are enough to get at the INFO table without a browser.
const ctx = vm.createContext({});
ctx.window = ctx;
ctx.BIMP = { palette: {}, state: { s: {} }, camera: {}, decode: {} };
ctx.document = { getElementById: () => null, addEventListener: () => {} };
vm.runInContext(readFileSync(join(SRC, "ui.js"), "utf8"), ctx);
const INFO = ctx.BIMP.ui.INFO;

const html = readFileSync(join(SRC, "body.html"), "utf8");
const inMarkup = [...html.matchAll(/data-info="([^"]+)"/g)].map((m) => m[1]);

let fail = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "✓" : "✗"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) fail++;
};

check("the INFO table loaded", !!INFO && typeof INFO === "object",
  INFO ? Object.keys(INFO).length + " entries" : "missing");
if (!INFO) process.exit(1);

check("body.html has ⓘ dots at all", inMarkup.length > 0, inMarkup.length + " in markup");

// every dot resolves
const dead = inMarkup.filter((k) => !INFO[k]);
check("every data-info key has an entry", dead.length === 0,
  dead.length ? "dead: " + dead.join(", ") : inMarkup.join(", "));

// ...and every entry is reachable. cb-info is retargeted at runtime by the mode switch,
// so keys it can be set to count as reachable even though body.html holds only one.
const runtime = [...readFileSync(join(SRC, "ui.js"), "utf8")
  .matchAll(/dataset\.info\s*=\s*[^;]*?"([^"]+)"\s*:\s*"([^"]+)"/g)].flatMap((m) => [m[1], m[2]]);
// ...and dots ui.js renders itself (the tree's note rows carry their own ⓘ)
const rendered = [...readFileSync(join(SRC, "ui.js"), "utf8").matchAll(/data-info="([^"$]+)"/g)].map((m) => m[1]);
const reachable = new Set([...inMarkup, ...runtime, ...rendered]);
const orphan = Object.keys(INFO).filter((k) => !reachable.has(k));
check("every entry is reachable from some dot", orphan.length === 0,
  orphan.length ? "orphaned: " + orphan.join(", ") : Object.keys(INFO).length + " entries reachable");

// the fields openInfo() reads must all be present, or the card renders blank/undefined
const bad = Object.entries(INFO).filter(([, e]) => !e.kind || !e.name || !e.body);
check("every entry has kind, name and body", bad.length === 0,
  bad.length ? "incomplete: " + bad.map(([k]) => k).join(", ") : "");

// The user-visible claims that are cheap to get wrong later: these numbers are pinned to
// the manifest elsewhere in the app, and prose drifts silently when data is re-exported.
const conf = INFO.confidence.body;
check("the confidence entry states the real range", /0\.10/.test(conf) && /1\.00/.test(conf),
  "manifest confRange = 0.096…0.99994");
// the gene counts and the mapping sentence are computed from the manifest when the card
// opens (the gene set grows with each projection), so the entry is a function: check that
// it counts by kind and keys the mapping claim on manifest.inferredMapping, with no number
// or mapping claim hard-coded
const gsrc = String(INFO.genes.body);
check("the gene entry counts measured/inferred from the manifest", typeof INFO.genes.body === "function" &&
  /kind/.test(gsrc) && !/2,062|2062/.test(gsrc), "body() reads M.genes");
check("the gene entry's mapping claim follows manifest.inferredMapping", /inferredMapping/.test(gsrc) && /canonical/.test(gsrc),
  "\"same mapping\" only when inferredMapping === \"canonical\"");
check("the caste entry states one brain per caste", /one queen brain and one worker brain/i.test(INFO.caste.body),
  "obs/brain has exactly 2 categories");

console.log(fail ? `\n${fail} check(s) failed` : "\nthe glossary is wired up");
process.exit(fail ? 1 : 0);
