/* Minimal column-major 4x4 matrix helpers (perspective, lookAt, multiply). */
window.BIMP = window.BIMP || {};
BIMP.mat4 = (function () {
  function persp(fovy, asp, near, far) {
    const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = t / asp; o[5] = t; o[10] = (far + near) / (near - far); o[11] = -1;
    o[14] = 2 * far * near / (near - far);
    return o;
  }
  function lookAt(e, c, up) {
    let z0 = e[0] - c[0], z1 = e[1] - c[1], z2 = e[2] - c[2];
    const zl = Math.hypot(z0, z1, z2) || 1; z0 /= zl; z1 /= zl; z2 /= zl;
    let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
    const xl = Math.hypot(x0, x1, x2) || 1; x0 /= xl; x1 /= xl; x2 /= xl;
    const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
    const o = new Float32Array(16);
    o[0] = x0; o[1] = y0; o[2] = z0; o[4] = x1; o[5] = y1; o[6] = z1; o[8] = x2; o[9] = y2; o[10] = z2;
    o[12] = -(x0 * e[0] + x1 * e[1] + x2 * e[2]);
    o[13] = -(y0 * e[0] + y1 * e[1] + y2 * e[2]);
    o[14] = -(z0 * e[0] + z1 * e[1] + z2 * e[2]);
    o[15] = 1;
    return o;
  }
  function mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
    return o;
  }
  return { persp, lookAt, mul };
})();
