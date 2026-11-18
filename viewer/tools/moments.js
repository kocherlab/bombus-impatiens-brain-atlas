// in-page: rAF frame-time traces around the moments users see (boot-probe --after)
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const out = {};
  async function trace(name, act, ms) {
    await sleep(1200);                     // settle (and let the loop fall asleep)
    const ft = []; let last = 0, on = true;
    const loop = (t) => { if (last) ft.push(t - last); last = t; if (on) requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    await sleep(50);
    const t0 = performance.now(); await act(); const actMs = performance.now() - t0;
    await sleep(ms); on = false;
    const s = ft.slice().sort((a, b) => b - a);
    out[name] = { actMs: +actMs.toFixed(1), frames: ft.length, worst: s.slice(0, 3).map((x) => +x.toFixed(1)), over33: ft.filter((x) => x > 33.4).length, over50: ft.filter((x) => x > 50).length };
  }
  const click = (sel) => () => { const e = document.querySelector(sel); if (e) e.click(); };
  await trace("caste→worker", click("#caste-seg [data-v=worker]"), 1500);
  await trace("caste→queen", click("#caste-seg [data-v=queen]"), 1500);
  await trace("caste→both", click("#caste-seg [data-v=both]"), 1500);
  await trace("caste→queen2", click("#caste-seg [data-v=queen]"), 1500);
  await trace("gene mode", click("#mode-seg [data-mode=gene]"), 1500);
  await trace("type mode", click("#mode-seg [data-mode=celltype]"), 1500);
  if (BIMP.histology && BIMP.histology.available) {
    await trace("histology on", () => BIMP.histology.setOn(true), 3000);
    await trace("2D enter", () => BIMP.histology.enter2D(), 2500);
    await trace("2D exit", () => BIMP.histology.exit2D(), 2500);
    await trace("histology off", () => BIMP.histology.setOn(false), 1500);
  }
  out.tier = BIMP.tier;
  return JSON.stringify(out);
})()
