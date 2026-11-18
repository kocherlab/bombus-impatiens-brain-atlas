/* Decode a tier payload: base64 → gzip → planar typed-array views.
   Two tiers share this: LITE (core attrs, no genes — fast first paint) and
   FULL (core + all genes). Returns typed arrays over the decoded buffer. */
window.BIMP = window.BIMP || {};
BIMP.decode = (function () {
  function b64ToBytes(b64) {
    const bin = atob(b64), len = bin.length, out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  async function gunzip(bytes) {
    if (typeof DecompressionStream === "undefined") return bytes.buffer;
    const ds = new DecompressionStream("gzip");
    const stream = new Response(new Blob([bytes]).stream().pipeThrough(ds));
    return stream.arrayBuffer();
  }
  // Build typed-array views over an already-decompressed ArrayBuffer.
  function views(buf, tier, geneCount) {
    const o = tier.offsets, n = tier.n;
    const out = {
      n,
      x: new Int16Array(buf, o.x[0], n),
      y: new Int16Array(buf, o.y[0], n),
      z: new Int16Array(buf, o.z[0], n),
      ct: new Uint8Array(buf, o.celltype[0], n),          // coherent subtype (cluster-vote)
      ctCanon: o.celltypecanon ? new Uint8Array(buf, o.celltypecanon[0], n) : null,
      ctRaw: o.celltyperaw ? new Uint8Array(buf, o.celltyperaw[0], n) : null,  // raw per-cell call (inspector)
      caste: new Uint8Array(buf, o.caste[0], n),
      conf: new Uint8Array(buf, o.conf[0], n),
      region: o.region ? new Uint8Array(buf, o.region[0], n) : new Uint8Array(n),
      // home tier only: 1 = one of the home section's cells beyond the sample (build-pages hero)
      hx: o.hx ? new Uint8Array(buf, o.hx[0], n) : null,
      // per-cell UMAP position (uint8, 0-255 over the embedding bbox) — full tier only
      umapx: o.umapx ? new Uint8Array(buf, o.umapx[0], n) : null,
      umapy: o.umapy ? new Uint8Array(buf, o.umapy[0], n) : null,
      // fig:kc_subtypes' Large-type subclusters S1-S3, projected per cell (0-255 probabilities)
      ltkc1: o.ltkc1 ? new Uint8Array(buf, o.ltkc1[0], n) : null,
      ltkc2: o.ltkc2 ? new Uint8Array(buf, o.ltkc2[0], n) : null,
      ltkc3: o.ltkc3 ? new Uint8Array(buf, o.ltkc3[0], n) : null,
      // fig:kc_subtypes panel B's restricted Kenyon transfer: 0 = outside the three Kenyon compartments,
      // 1..5 = manifest.kcTypes.names order; kcprob = the winning renormalised probability ×255
      kctype: o.kctype ? new Uint8Array(buf, o.kctype[0], n) : null,
      kcprob: o.kcprob ? new Uint8Array(buf, o.kcprob[0], n) : null,
      genes: [],
    };
    for (let i = 0; i < (geneCount || 0); i++) {
      if (o["gene" + i]) out.genes.push(new Uint8Array(buf, o["gene" + i][0], n));
    }
    return out;
  }
  // Decompress a base64 blob and return both the raw buffer (for caching) and views.
  async function decodeBlob(b64, tier, geneCount, gz) {
    const bytes = b64ToBytes(b64);
    const buf = gz ? await gunzip(bytes) : bytes.buffer;
    return { buf, data: views(buf, tier, geneCount) };
  }
  // ---- fetch mode (GitHub Pages: data served as separate files) ----
  // Retries a failed or stalled request (network error, 5xx, 408/429) with backoff, so one
  // dropped connection on a phone doesn't leave the atlas half-loaded. A 4xx is final.
  // opts.timeout (ms) aborts an attempt that has taken too long; opts.onRetry(n, err) reports.
  async function fetchBuf(url, gz, opts) {
    opts = opts || {};
    const tries = opts.retries == null ? 3 : opts.retries;
    for (let k = 0; ; k++) {
      const ac = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = ac && opts.timeout ? setTimeout(() => ac.abort(), opts.timeout) : 0;
      try {
        const resp = await fetch(url, ac ? { signal: ac.signal } : undefined);
        if (!resp.ok) {
          const err = new Error("fetch " + url + " → " + resp.status);
          err.final = resp.status < 500 && resp.status !== 408 && resp.status !== 429;
          throw err;
        }
        const bytes = new Uint8Array(await resp.arrayBuffer());
        return gz ? await gunzip(bytes) : bytes.buffer;
      } catch (err) {
        if (err.final || k >= tries) throw err;
        if (opts.onRetry) try { opts.onRetry(k + 1, err); } catch (e) { /* reporting only */ }
        await new Promise((r) => setTimeout(r, 600 * Math.pow(2, k)));
      } finally { if (timer) clearTimeout(timer); }
    }
  }
  // Gene files hold every full-tier cell. When the drawn tier is the lite one (boot.js device
  // tier), setGather(fullIndex) makes fetchGene return only the drawn cells, in drawn order.
  let gather = null;
  function setGather(idx) { gather = idx || null; }
  function gatherOut(arr) {
    if (!gather) return arr;
    const out = new Uint8Array(gather.length);
    for (let j = 0; j < gather.length; j++) out[j] = arr[gather[j]];
    return out;
  }
  // the lite tier's full-tier indices (build-pages.mjs liteIndex): Uint16 deltas → Uint32
  async function fetchLiteIndex(url, n) {
    const d = new Uint16Array(await fetchBuf(url, true, { timeout: 30000 }));
    if (d.length !== n) throw new Error("lite index: " + d.length + " ≠ " + n);
    const idx = new Uint32Array(n);
    for (let j = 0, p = -1; j < n; j++) { p += d[j]; idx[j] = p; }
    return idx;
  }
  // base.bin → views over all cells (core attrs only, no genes)
  async function fetchBase(url, tier, gz, opts) {
    return views(await fetchBuf(url, gz, opts), tier, 0);
  }
  // one gene file → per-cell Uint8Array (0-255). Measured genes are 8-bit; inferred
  // genes are 4-bit packed (2 cells/byte) — unpack to 0-255 for the shared color path.
  async function fetchGene(url, gz, opts) {
    const buf = await fetchBuf(url, gz);
    const bytes = new Uint8Array(buf);
    if (opts && opts.bits === 4) {
      const n = (opts.n != null) ? opts.n : bytes.length * 2;
      const out = new Uint8Array(n);
      for (let i = 0; i < bytes.length; i++) {
        const hi = (bytes[i] >> 4) & 0xf, lo = bytes[i] & 0xf;
        if (i * 2 < n) out[i * 2] = hi * 17;         // 0-15 → 0-255
        if (i * 2 + 1 < n) out[i * 2 + 1] = lo * 17;
      }
      return gatherOut(out);
    }
    return gatherOut(bytes);
  }
  return { decodeBlob, views, b64ToBytes, fetchBase, fetchGene, fetchBuf, fetchLiteIndex, setGather };
})();
