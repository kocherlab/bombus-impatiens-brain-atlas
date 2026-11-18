// in-page: home page → atlas hand-off, frame-time trace (boot-probe --after, on the home URL)
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 100 && !(window.BIMP && BIMP.renderer && BIMP.renderer.N > 0); i++) await sleep(100);
  await sleep(1500);
  const way = document.querySelector("#enter-tour") || document.querySelector("#masthead .open-atlas");
  way.dispatchEvent(new PointerEvent("pointerover", { bubbles: true }));      // intent, as a hover would
  await sleep(600);
  const ft = []; let last = 0, on = true;
  const loop = (t) => { if (last) ft.push([+(t - last).toFixed(1), +(BIMP.transition ? BIMP.transition.tp : 0).toFixed(3)]); last = t; if (on) requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
  const t0 = performance.now(); way.click();
  for (let i = 0; i < 150 && !(BIMP.transition && BIMP.transition.tp >= 1); i++) await sleep(50);
  const doneMs = performance.now() - t0; await sleep(1500); on = false;
  const firstMove = ft.findIndex((f) => f[1] > 0);
  const s = ft.map((f) => f[0]).sort((a, b) => b - a);
  return JSON.stringify({ tier: BIMP.tier, clickToMoveFrames: firstMove, handoffMs: Math.round(doneMs), frames: ft.length, worst: s.slice(0, 4), over33: s.filter((x) => x > 33.4).length, over50: s.filter((x) => x > 50).length,
    worstAt: ft.filter((f) => f[0] > 33.4).slice(0, 6) });
})()
