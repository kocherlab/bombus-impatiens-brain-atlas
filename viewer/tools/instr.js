(() => { const R = BIMP.renderer; window.__inR = false; const r0 = R.render; R.render = function () { __inR = true; try { return r0.apply(this, arguments); } finally { __inR = false; } };
  const st = () => new Error().stack.split("\n").slice(2, 6).map((l) => l.trim().replace(/\(?https?:\/\/[^ )]*\//g, "")).join(" | ");
  window.__glog = []; const P = WebGL2RenderingContext.prototype;
  ["clear", "drawArrays", "drawElements", "bindFramebuffer", "viewport"].forEach((m) => { const f = P[m]; P[m] = function () { if (!__inR && __glog.length < 60) __glog.push(m + " " + st()); return f.apply(this, arguments); }; });
  const cv = R.gl.canvas; const d = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, "width");
  Object.defineProperty(cv, "width", { configurable: true, get() { return d.get.call(this); }, set(v) { __glog.push("canvas.width=" + v + " " + st()); d.set.call(this, v); } });
  return 0; })()
