/* Orbit camera — spherical coords around a pannable target. */
window.BIMP = window.BIMP || {};
BIMP.camera = (function () {
  // radius 4.2 @ 48° FOV shows a visible half-height of ~1.9 units at the target, so
  // the brain (~±1.0 after bbox-centered fit-to-±0.9) clears the frame with margin on
  // tall/portrait windows too, even with the downward orbit tilt.
  const HOME = { theta: 0.9, phi: 1.15, radius: 4.2, target: [0, 0, 0] };
  const cam = { theta: HOME.theta, phi: HOME.phi, radius: HOME.radius, target: [0, 0, 0], minR: 0.3, maxR: 14 };   // 0.3: close enough that a dot is a cell several pixels wide
  function eye() {
    const { radius: r, phi, theta, target: t } = cam;
    return [
      t[0] + r * Math.sin(phi) * Math.cos(theta),
      t[1] + r * Math.cos(phi),
      t[2] + r * Math.sin(phi) * Math.sin(theta),
    ];
  }
  function orbit(dx, dy) {
    cam.theta += dx * 0.006;
    cam.phi = Math.max(0.15, Math.min(Math.PI - 0.15, cam.phi - dy * 0.006));
  }
  // Pan the target in the camera's screen plane (right / up vectors), scaled by
  // distance so a drag moves the same on-screen amount at any zoom.
  function pan(dx, dy) {
    const e = eye();
    let fx = cam.target[0] - e[0], fy = cam.target[1] - e[1], fz = cam.target[2] - e[2];
    const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
    // right = forward × worldUp
    let rx = fy * 0 - fz * 1, ry = fz * 0 - fx * 0, rz = fx * 1 - fy * 0;
    const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
    // up = right × forward
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    // world units per CSS pixel at the target's depth (the renderer's 48° field over the view's
    // height), so what is under the pointer stays under it, as on a map; a fixed 0.0016 matched
    // the screen only at one window size
    const gl = typeof document !== "undefined" && document.getElementById("gl");
    const k = 2 * cam.radius * Math.tan(24 * Math.PI / 180) / Math.max(200, (gl && gl.clientHeight) || 900);
    cam.target[0] += (-rx * dx + ux * dy) * k;
    cam.target[1] += (-ry * dx + uy * dy) * k;
    cam.target[2] += (-rz * dx + uz * dy) * k;
  }
  function zoom(sign) { cam.radius = Math.max(cam.minR, Math.min(cam.maxR, cam.radius * (1 + sign * 0.08))); }
  function spin(dt) { cam.theta += dt * 0.12; }
  function reset() {
    cam.theta = HOME.theta; cam.phi = HOME.phi; cam.radius = HOME.radius;
    cam.target[0] = cam.target[1] = cam.target[2] = 0;
  }

  // ---- scripted camera flights (used by the guided tour) ----
  // flyTo eases theta/phi/radius/target from the current pose to a goal over `dur` ms;
  // tick() advances it each frame and returns true while a flight is in progress, so the
  // frame loop can suspend auto-orbit. Any user drag calls cancelFly() to hand back control.
  let fly = null;
  const snap = () => ({ theta: cam.theta, phi: cam.phi, radius: cam.radius, target: cam.target.slice() });
  function flyTo(goal, dur) {
    const from = snap();
    const to = {
      theta: goal.theta != null ? goal.theta : from.theta,
      phi: goal.phi != null ? goal.phi : from.phi,
      radius: goal.radius != null ? goal.radius : from.radius,
      target: goal.target ? goal.target.slice() : from.target.slice(),
    };
    let dth = to.theta - from.theta;            // shortest angular path
    while (dth > Math.PI) dth -= 2 * Math.PI;
    while (dth < -Math.PI) dth += 2 * Math.PI;
    to.theta = from.theta + dth;
    fly = { from, to, t: 0, dur: Math.max(0.2, (dur || 1100) / 1000) };
  }
  function tick(dt) {
    if (!fly) return false;
    fly.t += dt;
    const u = Math.min(1, fly.t / fly.dur);
    const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;   // easeInOutCubic
    const L = (a, b) => a + (b - a) * e;
    cam.theta = L(fly.from.theta, fly.to.theta);
    cam.phi = L(fly.from.phi, fly.to.phi);
    cam.radius = L(fly.from.radius, fly.to.radius);
    for (let k = 0; k < 3; k++) cam.target[k] = L(fly.from.target[k], fly.to.target[k]);
    if (u >= 1) fly = null;
    return true;
  }
  function cancelFly() { fly = null; }
  function isFlying() { return !!fly; }
  return { cam, eye, orbit, pan, zoom, spin, reset, flyTo, tick, cancelFly, isFlying, HOME };
})();
