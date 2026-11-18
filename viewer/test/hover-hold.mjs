/* Does a missed pick blank the hover?
 *
 * Picking runs at ~11 Hz, and a pick that lands between two cells returns -1. Feeding
 * that straight to the UI made the 3-D highlight, the tooltip and the UMAP marker blink
 * off and back on several times a second with the pointer sitting still. state.js holds
 * the last hit through a stray miss; a sustained run of misses still expires it.
 *
 * Pure Node — state.js needs no DOM.
 *
 *   node viewer/test/hover-hold.mjs
 *   BIMP_SRC=/path/to/old/src node viewer/test/hover-hold.mjs   # negative control
 */
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = process.env.BIMP_SRC || join(dirname(HERE), "src");

// state.js is browser code: it assigns window.BIMP and then reads the bare global BIMP.
// A vm context whose `window` IS the global reproduces that faithfully.
const ctx = vm.createContext({});
ctx.window = ctx;
vm.runInContext(readFileSync(join(SRC, "state.js"), "utf8"), ctx);
const state = ctx.BIMP.state;

let fail = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "✓" : "✗"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) fail++;
};

if (typeof state.makeHoverHold !== "function") {
  console.log("✗ state.makeHoverHold is missing — a stray missed pick will blank the hover");
  process.exit(1);
}

const h = state.makeHoverHold(3);
check("a hit is reported straight through", h.report(42) === 42);
check("one missed pick keeps the last cell", h.report(-1) === 42, "a stray miss must not blink");
check("two in a row still keep it", h.report(-1) === 42);
check("three in a row still keep it", h.report(-1) === 42);
check("a fourth consecutive miss lets it go", h.report(-1) === -1, "a run of misses is real");

// a hit resets the count — the forgiveness is for STRAY misses, not a budget
const h2 = state.makeHoverHold(3);
h2.report(5); h2.report(-1); h2.report(-1);
check("a new hit resets the miss count", h2.report(9) === 9);
check("and the next miss is forgiven again", h2.report(-1) === 9);

// leaving the canvas must not wait for a run of misses
const h3 = state.makeHoverHold(3);
h3.report(7);
h3.clear();
check("clear() drops it at once", h3.get() === -1, "pointerleave / drag start");

// The hold must NOT be a wall-clock timer: on a slow device picks are rare, and a
// time-based hold expired between them and blinked MORE than doing nothing.
const h4 = state.makeHoverHold(3);
h4.report(11);
check("holding does not decay with time, only with missed picks", h4.get() === 11,
  "no wall-clock expiry");

console.log(fail ? `\n${fail} check(s) failed` : "\nhover hold behaves");
process.exit(fail ? 1 : 0);
