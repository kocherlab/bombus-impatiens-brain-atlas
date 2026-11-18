#!/usr/bin/env python
"""Export the 3D brain atlas to a self-contained JS data payload.

Reads the 1.3M-cell Tangram h5ad and writes ``src/data.js`` — a single script
that sets ``window.__BIMP__`` with:

  * a gzip+base64 binary blob (16-bit quantized xyz + uint8 category/gene attrs),
  * a manifest (category names, exact full-atlas counts, palettes, gene list).

The blob is decoded in the browser with ``DecompressionStream('gzip')`` (no deps).
Everything is inlined so the built page works under a strict-CSP artifact host.

Run from repo root (or a worktree — data is resolved from the main checkout):
    PYTHONPATH=src .venv/bin/python viewer/export_data.py [--sample N] [--genes K]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import gene_names  # viewer/gene_names.py, beside this file
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


# The viewer reads obs/tangram_cell_type and shows those names on the site, so
# it reads the canonical Tangram object (bimp_st.paths.CANONICAL_TANGRAM), whose
# cell-type column names are keyed by name, not by position.
TANGRAM_REL = "results/tangram/combined_clustered_g2_R2_s42_tangram.h5ad"


def data_root(repo: Path) -> Path:
    """Checkout that holds results/ (gitignored, absent inside a worktree)."""
    rel = TANGRAM_REL
    if (repo / rel).exists():
        return repo
    from bimp_st.site import get as site_path

    other = site_path("data_repo")  # a second checkout's data (bimp_st.site)
    return other if (other / rel).exists() else repo


H5AD = data_root(REPO) / TANGRAM_REL

# Hand-curated short symbols for the focal genes. Preferred over the
# derived protein name when present, and these genes are always included in
# the export.
#
# Sync target: src/bimp_st/genes.py's GENE_SYMBOLS, which is the superset
# (12 entries; these 9 plus `to`/`shakB`/`gukh`, which are caste-DE volcano tag
# labels rather than focal genes, so they are deliberately not pinned here).
# scripts/paper/check_gene_symbol_sync.py checks the shared ids.
GENE_SYMBOLS = {
    "bimp_v4.3_009307": "oct-β2R",
    "bimp_v4.3_005257": "FTZ-F1β",
    "bimp_v4.3_009612": "oct-β1R",
    "bimp_v4.3_006191": "GlyRα1",
    "bimp_v4.3_004137": "HSP90",
    "bimp_v4.3_005222": "CaMKII",
    "bimp_v4.3_008309": "ERR1",
    "bimp_v4.3_005730": "sNPF",
    "bimp_v4.3_005433": "Kr-h1",
}
# Full protein descriptions for all 100 panel genes.
PANEL_MAP = data_root(REPO) / "data/panel/np31_to_v43_mapping.csv"
# Tangram-imputed full-transcriptome projection (measured 100 + ~2062 inferred).
IMPUTED_H5AD = data_root(REPO) / "results/tangram/combined_clustered_imputed.h5ad"
# fig:kc_subtypes' Large-type subcluster projection
# (scripts/kenyon/project_large_type_subclusters.py;
# read the same way as scripts/kenyon/fig_kc_subtypes.py load_ltkc()). Cluster ids,
# labels and colours are snrna_kc_subclusters.py's CLUSTERS / LABELS / COLORS.
LTKC_H5AD = (
    data_root(REPO) / "results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad"
)
LTKC_CLUSTERS = ["0", "1", "2"]
LTKC_COLORS = {"0": "#0B775E", "1": "#5B4C7C", "2": "#F2300F"}
# The restricted Kenyon-type transfer (scripts/kenyon/project_kc_populations.py), as
# fig_kc_subtypes.py load_kc5pop() reads it: TYPE_ORDER, TYPE_SHORT, colours from
# cluster_labels.
KC5POP_H5AD = data_root(REPO) / "results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad"
KC5_TYPES = [
    "Large-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Middle-type Kenyon cells",
    "Other Kenyon cells",
]
KC5_SHORT = {
    "Large-type Kenyon cells": "Large-type",
    "Small-type Kenyon cells": "Small-type",
    "Class II Kenyon cells": "Class II",
    "Middle-type Kenyon cells": "Middle-type",
    "Other Kenyon cells": "Other KC",
}


def _kc5_color(name):
    from bimp_st.cluster_labels import get_color

    return get_color(name)


def write_imputed_genes(out, measured_meta, descriptions, imputed_h5ad=IMPUTED_H5AD):
    """Append Tangram-inferred genes: one lazy gzipped uint8 file per gene, tagged
    'inferred', with a sparsity flag. Skips genes already present as measured.
    Returns the combined gene list (measured first, then inferred)."""
    import csv as _csv  # noqa

    imputed_h5ad = Path(imputed_h5ad)
    if not imputed_h5ad.exists():
        print(f"  (imputed h5ad not found at {imputed_h5ad}; skipping inferred genes)")
        return measured_meta
    import time

    have = {g["id"] for g in measured_meta}
    n_measured = len(measured_meta)
    print(f"  inferred genes from {imputed_h5ad}", flush=True)
    with h5py.File(imputed_h5ad, "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else x for x in f["var/_index"][:]]
        X = f["X"]  # (N, G) float, projected onto all cells
        n, G = X.shape
        new_gi = [gi for gi, gid in enumerate(ids) if gid not in have]

        # Fast path: if a sidecar with the exact per-gene metadata exists (from a
        # prior full run), reuse the already-written gene files and rebuild only the
        # manifest — skips the ~14 min matrix reload when just positions changed.
        sidecar = out / "inferred_meta.json"
        existing = sorted((out / "genes").glob("i*.b4"))
        if sidecar.exists() and len(existing) == len(new_gi):
            saved = json.loads(sidecar.read_text())
            # same count is not enough: a different imputed object must not reuse these
            # files
            if [g["id"] for g in saved] == [ids[gi] for gi in new_gi]:
                print(
                    f"  reusing {len(existing)} inferred gene files via "
                    "sidecar (skipping matrix load)",
                    flush=True,
                )
                return list(measured_meta) + saved

        # Load the whole matrix ONCE into RAM in row-blocks (single-column reads on a
        # 22GB chunked h5ad are pathologically slow). float32 → ~11GB; if that's too
        # much, fall back to per-column. Then quantize per gene from memory.
        t0 = time.time()
        print(
            f"  loading imputed matrix ({n}x{G} → {n * G * 4 / 1e9:.0f}GB float32)…",
            flush=True,
        )
        try:
            M = np.empty((n, G), np.float32)
            step = 100000
            for r in range(0, n, step):
                M[r : r + step] = X[r : r + step]
            print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
        except MemoryError:
            print(
                "  (not enough RAM for full matrix; falling back to per-column reads)"
            )
            M = None

        combined = list(measured_meta)
        out_i = n_measured
        for gi in new_gi:
            gid = ids[gi]
            col = M[:, gi] if M is not None else np.asarray(X[:, gi], dtype=np.float32)
            hi = float(np.percentile(col, 99.5)) or 1.0
            # spatial informativeness: low coefficient of variation ⇒ ~flat gene
            mean = float(col.mean())
            cv = float(np.sqrt(col.var()) / (mean + 1e-12))
            # 4-bit quantization, packed 2 values/byte — imputed values are smooth
            # small probabilities, so 16 levels is plenty and keeps the payload ~40MB.
            q = np.clip(np.round(col / hi * 15), 0, 15).astype(np.uint8)
            if q.size % 2:  # pad to even for packing
                q = np.append(q, 0)
            packed = (q[0::2] << 4) | q[1::2]
            fn = f"genes/i{out_i:04d}.b4"
            (out / fn).write_bytes(gzip.compress(packed.astype(np.uint8).tobytes(), 6))
            desc = descriptions.get(gid)
            combined.append(
                {
                    "id": gid,
                    "symbol": GENE_SYMBOLS.get(gid)
                    or (short_symbol(desc) if desc else None),
                    "desc": desc,
                    "hi": hi,
                    "file": fn,
                    "kind": "inferred",
                    "bits": 4,
                    "flat": cv < 1.0,  # low-variance ⇒ little spatial signal
                }
            )
            out_i += 1
            if (out_i - n_measured) % 250 == 0:
                print(
                    f"  imputed genes: {out_i - n_measured}/{len(new_gi)} written…",
                    flush=True,
                )
    n_flat = sum(1 for g in combined if g.get("flat"))
    # sidecar so a later position-only rebuild can reuse these files without the reload
    (out / "inferred_meta.json").write_text(json.dumps(combined[n_measured:]))
    print(
        f"  inferred genes: {len(combined) - n_measured} written "
        f"(4-bit; {n_flat} flagged low-signal)"
    )
    return combined


def load_descriptions() -> dict:
    """v4.3 id (lowercase) → clean protein description, from the panel map."""
    import csv
    import re

    out = {}
    if not PANEL_MAP.exists():
        return out
    with open(PANEL_MAP, newline="") as fh:
        for row in csv.DictReader(fh):
            vid = (row.get("v43_id") or "").strip().lower()
            desc = (row.get("description") or "").strip()
            m = re.search(r"\[protein=(.*?)\]", desc)
            if vid and m:
                out[vid] = m.group(1).strip()
    return out


def short_symbol(protein: str) -> str:
    """Derive a compact display name from a protein description."""
    import re

    s = re.sub(r"\s*isoform\s+[\w-]+$", "", protein)  # drop 'isoform X1'
    s = re.sub(r"\s*[_,]\s*$", "", s).strip()
    if len(s) > 26:
        s = s[:25].rstrip() + "…"
    return s


def cats(f, path):
    return [
        x.decode() if isinstance(x, bytes) else x for x in f[f"{path}/categories"][:]
    ]


def q16(a: np.ndarray) -> np.ndarray:
    return np.round(a * 32767).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample",
        type=int,
        default=180_000,
        help="full-tier rendered cell sample size (kept so the single-file "
        "artifact stays < 16 MB)",
    )
    ap.add_argument(
        "--lite",
        type=int,
        default=100_000,
        help="lite-tier (fast first paint) cell count",
    )
    ap.add_argument(
        "--genes",
        type=int,
        default=100,
        help="number of genes to include (focal first, then by variance)",
    )
    ap.add_argument(
        "--pages",
        action="store_true",
        help="export ALL 1.3M cells to fetched binaries (pages/data/) for a "
        "static host instead of the inline artifact",
    )
    ap.add_argument(
        "--imputed-genes",
        action="store_true",
        help="also export the Tangram-inferred transcriptome (~2062 extra genes) "
        "as lazy per-gene files (Pages only)",
    )
    ap.add_argument(
        "--imputed-h5ad",
        type=Path,
        default=IMPUTED_H5AD,
        help="Tangram-imputed object the inferred genes are read from (default: "
        "the manuscript's combined_clustered_imputed.h5ad)",
    )
    ap.add_argument(
        "--genes-base-url",
        default="./data/",
        help="base URL the viewer fetches per-gene files from (manifest "
        "genesBase); point it at another host when the gene files are served "
        "outside GitHub Pages",
    )
    ap.add_argument(
        "--inferred-mapping",
        default=None,
        help='manifest inferredMapping: "canonical" when --imputed-h5ad comes '
        "from the same Tangram mapping that assigned the cell types "
        "(paths.CANONICAL_TANGRAM); omit for an older imputation",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    with h5py.File(H5AD, "r") as f:
        ct_names = cats(f, "obs/tangram_cell_type")
        ct_full = f["obs/tangram_cell_type/codes"][:].astype(np.int32)
        caste_full = f["obs/brain/codes"][:].astype(np.uint8)  # 0 queen, 1 worker
        conf = f["obs/tangram_confidence"][:].astype(np.float32)
        z = f["obs/z_level"][:].astype(np.float32)
        xy = f["obsm/spatial_aligned"][:].astype(np.float32)
        # Per-cell UMAP (the atlas's own 100-gene-panel embedding) → the "where does
        # this
        # cell sit" panel. Same row order as everything else here.
        umap_full = (
            f["obsm/X_umap"][:].astype(np.float32) if "obsm/X_umap" in f else None
        )
        gene_ids = [x.decode() for x in f["var/_index"][:]]
        n_total = int(len(ct_full))
        # Hand-labeled anatomical region masks → one bit per region in a per-cell
        # bitmask (uint8). Grouped Mushroom body (in_kc_*) / Optic lobe (visual
        # neuropils) so the UI can offer multi-select. These are defined as
        # regions by the masks, not by the dispersed st-KC cell-type label.
        # The antennal lobe and the central complex are NOT in this h5ad: it
        # carries the six flags labeling.py baked in, and the two newer regions
        # live in the nine-region crop sidecars. Join them on
        # (slice_name, cell_id), never positionally -- QC drops cells, so the
        # row orders differ.
        _extra = {}
        try:
            from types import SimpleNamespace as _NS

            import pandas as _pd

            from bimp_st.nine_region_common import (
                DESCRIPTIVE_REGIONS,
                attach_region_flags,
            )

            def _cat(g):
                cats = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in g["categories"][:]
                ]
                return np.asarray(cats)[g["codes"][:]]

            _obs = _pd.DataFrame(
                {
                    "slice_name": _cat(f["obs/slice_name"]),
                    "cell_id": _cat(f["obs/cell_id"]),
                }
            )
            _ns = _NS(obs=_obs)
            # Join ALL of them, not just the two newer ones. The six baked
            # into this h5ad are frozen at whatever labeling.py last wrote, and
            # they have since moved: the reclassification took 119 cells out of
            # the lobula, so the stored 5,227 is stale against the current
            # 5,108. Taking six from the frozen copy and two from the sidecars
            # would put two vintages in one bitmask.
            _want = list(DESCRIPTIVE_REGIONS)
            attach_region_flags(_ns, regions=_want)
            for r in _want:
                col = f"in_{r}"
                if col in _ns.obs:
                    # pd.NA means "never traced here", which is not membership.
                    _extra[col] = _ns.obs[col].to_numpy(dtype=bool, na_value=False)
            print(f"  joined {sorted(_extra)} from the nine-region sidecars")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not join the newer region flags: {exc}")

        def _mask(col):
            if col in _extra:
                return _extra[col]
            return (
                f["obs/" + col][:].astype(bool)
                if "obs/" + col in f
                else np.zeros(n_total, bool)
            )

        # The three calyx masks are Kenyon cell COMPARTMENTS, each named for
        # the population that dominates it: Large-type (non-compact),
        # Small-type (inner compact), Class II (outer compact), as in
        # bimp_st.nine_region_common.DISPLAY and fig:atlas_overview panel B.
        # src/palette.js REGION_DISPLAY applies the same labels at load, keyed on
        # the column, so any
        # exported manifest shows them. Cell-type names stay
        # the raw data values here ("Optic lobe", "Mixed", ...); their display
        # wording is applied at display time by palette.displayName.
        REGION_DEFS = [
            ("in_kc_ic", "Small-type compartment", "Mushroom body · Kenyon cells"),
            ("in_kc_nc", "Large-type compartment", "Mushroom body · Kenyon cells"),
            ("in_kc_oc", "Class II compartment", "Mushroom body · Kenyon cells"),
            ("in_lamina", "Lamina", "Optic lobe"),
            ("in_medulla", "Medulla", "Optic lobe"),
            ("in_lobula", "Lobula", "Optic lobe"),
            ("in_al", "Antennal lobe", "Antennal lobe"),
            ("in_central_complex", "Central complex", "Central complex"),
        ]
        # Eight regions is exactly the width of the uint8 bitmask below. A
        # ninth needs uint16 here AND in the reader, so fail rather than
        # silently drop the top bit.
        assert len(REGION_DEFS) <= 8, (
            f"{len(REGION_DEFS)} regions will not fit in the uint8 region "
            "bitmask; widen the dtype in export_data.py and in src/renderer.js"
        )
        region_bits_full = np.zeros(n_total, np.uint8)
        for bit, (col, lab, _) in enumerate(REGION_DEFS):
            m = _mask(col)
            # An all-zero mask ships a toggle that selects nothing, which looks
            # like "no cells there" rather than "this never loaded".
            if not m.any():
                raise SystemExit(
                    f"region mask {col} ({lab}) is empty -- it is either absent "
                    "from the h5ad or the sidecar join failed. Refusing to ship "
                    "a filter that silently selects nothing."
                )
            region_bits_full |= m.astype(np.uint8) << bit
            print(f"  bit {bit}  {col:<20}{int(m.sum()):>9,} cells")

        # Cell-type labels: the canonical hierarchical cluster vote
        # (bimp_st.cluster_labels.hierarchical_vote). Expression only, no anatomy
        # anywhere. A cluster takes a FINE atlas type where the data decide one
        # (purity >= 0.5, or a 1.5x lead), else its pooled group if that is
        # decided, else Mixed (shown as "Other"). Kenyon cell subtypes are held at
        # "Kenyon cells" whatever the vote returns (CAP_TO_POOLED): the panel
        # cannot name a Kenyon subtype per cell, so
        # the atlas shows no per-cell Large/Small/Middle/Class II calls.
        # The viewer's two labelings are the vote's two levels:
        #   "Atlas types" = the vote as written;  "Groups" = POOL_MAP of it (seven +
        # Other).
        leiden_codes = f["obs/leiden/codes"][:]
        leiden_cats = [
            x.decode() if isinstance(x, bytes) else x
            for x in f["obs/leiden/categories"][:]
        ]
        import anndata as _ad
        import pandas as _pd

        from bimp_st.cluster_labels import (
            CANONICAL_ORDER,
            POOLED_ORDER,
            get_canonical_color,
            get_pooled,
            get_pooled_color,
            hierarchical_vote,
        )

        _obs = _pd.DataFrame(
            {
                "leiden": _pd.Categorical.from_codes(leiden_codes, leiden_cats),
                "tangram_cell_type": _pd.Categorical.from_codes(ct_full, ct_names),
            }
        )
        _A = _ad.AnnData(X=np.zeros((n_total, 1), np.float32), obs=_obs)
        hierarchical_vote(
            _A, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
        )
        vote = _A.obs["cluster_vote"].astype(str).values
        present = set(vote)
        subtype_names = (
            [nm for nm in CANONICAL_ORDER if nm in present]
            + [nm for nm in POOLED_ORDER if nm in present and nm not in CANONICAL_ORDER]
            + ["Mixed"]
        )
        unknown = present - set(subtype_names)
        assert not unknown, (
            f"hierarchical_vote emitted names outside the vocabulary: {unknown}"
        )
        sub_idx = {nm: i for i, nm in enumerate(subtype_names)}
        sub_full = np.array([sub_idx[v] for v in vote], dtype=np.uint8)
        canon_order = list(POOLED_ORDER) + ["Mixed"]
        canon_idx = {name: i for i, name in enumerate(canon_order)}
        canon_full = np.array(
            [
                canon_idx.get(
                    "Mixed" if v == "Mixed" else get_pooled(v), canon_idx["Mixed"]
                )
                for v in vote
            ],
            dtype=np.uint8,
        )
        # "Mixed" (displayed "Other") = undecided clusters. Dim slate so the decided
        # classes read on the dark stage.
        canon_colors = [
            get_pooled_color(nm) if nm != "Mixed" else "#3E4552" for nm in canon_order
        ]
        subtype_colors = [
            (
                "#3E4552"
                if nm == "Mixed"
                else get_canonical_color(nm)
                if nm in CANONICAL_ORDER
                else get_pooled_color(nm)
            )
            for nm in subtype_names
        ]
        for nm in subtype_names:
            print(f"  {nm:<34}{int((sub_full == sub_idx[nm]).sum()):>9,}")

        # fig:kc_subtypes (fig_kc_subtypes.py, panels F-H): the three Large-type
        # subclusters (S1-S3, de novo on the CB reference's own Large-type nuclei)
        # PROJECTED by Tangram onto every cell of the three Kenyon compartments --
        # a per-cell probability blend, not a call. Joined to atlas cells by
        # (brain, z_level, aligned x/y): the projection object carries the atlas's
        # own aligned coordinates, and the atlas has duplicate obs_names, so names
        # cannot be the key.
        brain_cats = [
            x.decode() if isinstance(x, bytes) else x
            for x in f["obs/brain/categories"][:]
        ]
        kc_any = np.zeros(n_total, bool)
        for col, _, _ in REGION_DEFS:
            if col.startswith("in_kc_"):
                kc_any |= _mask(col).astype(bool)

        def key(c, zz, xx, yy):
            return (
                int(c),
                int(round(float(zz) * 10)),
                int(round(float(xx) * 10)),
                int(round(float(yy) * 10)),
            )

        where = {
            key(caste_full[i], z[i], xy[i, 0], xy[i, 1]): i for i in np.where(kc_any)[0]
        }

        def kc_projection(path, columns, label):
            """Per-cell renormalised probabilities of a Tangram run restricted to the
            three Kenyon compartments, joined to atlas cells by (brain, z_level,
            aligned x/y) -- the projection objects carry the atlas's own aligned
            coordinates, and the atlas has duplicate obs_names, so names cannot be
            the key. Returns (rows, probs) or None."""
            if not path.exists():
                print(f"  (no {path.name}; the {label} layer is left empty)")
                return None
            L = _ad.read_h5ad(path)
            pr = L.obsm["tangram_ct_pred"]
            pr = (
                pr
                if isinstance(pr, _pd.DataFrame)
                else _pd.DataFrame(np.asarray(pr), columns=columns)
            )
            pr.columns = [str(c) for c in pr.columns]
            probs = (
                pr[columns]
                .div(pr[columns].sum(axis=1), axis=0)
                .fillna(1.0 / len(columns))
                .values
            )
            lb = np.array([brain_cats.index(b) for b in L.obs["brain"].astype(str)])
            lz = L.obs["z_level"].values.astype(np.float32)
            lxy = np.asarray(L.obsm["spatial"], dtype=np.float32)
            rows = np.array(
                [
                    where.get(key(lb[j], lz[j], lxy[j, 0], lxy[j, 1]), -1)
                    for j in range(L.n_obs)
                ]
            )
            rate = float((rows >= 0).mean()) if L.n_obs else 0.0
            print(
                f"  {label}: {int((rows >= 0).sum()):,} / {L.n_obs:,} cells "
                f"joined ({rate:.1%})"
            )
            assert rate >= 0.98, (
                f"{label} did not join to the atlas by (brain, z, x, y); "
                "check that its obsm['spatial'] is the aligned frame"
            )
            ok = rows >= 0
            return rows[ok], probs[ok]

        lt_full = np.zeros((n_total, 3), np.uint8)
        got = kc_projection(LTKC_H5AD, LTKC_CLUSTERS, "Large-type S1-S3 projection")
        if got:
            rows, probs = got
            lt_full[rows] = np.round(probs * 255).astype(np.uint8)

        # The three Kenyon cell TYPES (fig:kc_subtypes panel B): a RESTRICTED Tangram
        # reclassification -- query = every cell of the three Kenyon compartments,
        # reference =
        # only the five named snRNA-seq Kenyon populations -- read as
        # fig_kc_subtypes.py
        # load_kc5pop() reads it. The whole-atlas vote cannot name Kenyon subtypes per
        # cell
        # (the panel lacks their markers), this restricted run can. kctype 0 = not a
        # compartment cell; 1..5 = KC5_TYPES in order; kcprob = the winning probability.
        kctype_full = np.zeros(n_total, np.uint8)
        kcprob_full = np.zeros(n_total, np.uint8)
        got = kc_projection(KC5POP_H5AD, KC5_TYPES, "restricted Kenyon-type transfer")
        if got:
            rows, probs = got
            kctype_full[rows] = (probs.argmax(axis=1) + 1).astype(np.uint8)
            kcprob_full[rows] = np.round(probs.max(axis=1) * 255).astype(np.uint8)

        # Cell selection: all cells for the GitHub-Pages build (fetched binaries,
        # no size limit), else a stratified caste-balanced downsample for the inline
        # single-file artifact.
        if args.pages:
            idx = np.arange(n_total)
        else:
            per = args.sample // 2
            idx = np.concatenate(
                [
                    rng.choice(
                        np.where(caste_full == c)[0],
                        size=min(int((caste_full == c).sum()), per),
                        replace=False,
                    )
                    for c in (0, 1)
                ]
            )
            idx.sort()
        # gene expression: load full X once (fancy row-indexing on h5py is O(rows) slow)
        print("loading expression matrix…", flush=True)
        X = f["X"][:]  # (N,100) dense float32
        Xs = X[idx]

    n_ct = len(subtype_names)  # 29 tangram types + "Mixed"
    # legend counts reflect the coherent subtype vote (what the Subtypes view shows)
    full_counts = [int((sub_full == i).sum()) for i in range(n_ct)]

    PIXEL_UM = 0.2125  # Xenium pixel size; z_level is already µm (10 µm slices)
    x, y = xy[idx, 0].copy() * PIXEL_UM, xy[idx, 1].copy() * PIXEL_UM
    z, ct, caste, conf = z[idx], ct_full[idx], caste_full[idx], conf[idx]
    region_bits = region_bits_full[idx]
    canon = canon_full[idx]
    sub = sub_full[idx]
    umap_sel = umap_full[idx] if umap_full is not None else None
    n = len(idx)

    # Center each caste on its BOUNDING-BOX center (not the mean) so the brain sits
    # centered in the frame — the C-shape's mean is offset from its visual center,
    # which pushed a lobe off-screen. Overlaps queen & worker in one volume.
    for c in (0, 1):
        m = caste == c
        xlo, xhi = np.percentile(x[m], [0.5, 99.5])
        ylo, yhi = np.percentile(y[m], [0.5, 99.5])
        x[m] -= (xlo + xhi) / 2
        y[m] -= (ylo + yhi) / 2
    z_um = z - (z.min() + z.max()) / 2
    # normalize x, y AND z by the SAME (µm) scale so the z stack keeps its true
    # proportion — the brain is ~340 µm thick vs ~4 mm wide (a thin stack). Scale so
    # the FULL bounding box (both castes) fits within ±0.9, leaving margin so no lobe
    # touches the frame edge after the camera tilt.
    half_x = np.percentile(np.abs(x), 99.8)
    half_y = np.percentile(np.abs(y), 99.8)
    scale = float(max(half_x, half_y)) / 0.9
    xn = np.clip(x / scale, -1, 1)
    yn = np.clip(y / scale, -1, 1)
    zn = np.clip(z_um / scale, -1, 1)

    X16, Y16, Z16 = q16(xn), q16(yn), q16(zn)

    # Normalize the UMAP to [0,1] with a SHARED half-range (square aspect), centered on
    # the
    # robust bounding-box midpoint. Only a 210px scatter panel reads it, so uint8 is
    # plenty
    # and keeps the base.bin download small (int16 here nearly doubled the file).
    def u8(a):
        return np.clip(np.round((np.clip(a, -1, 1) + 1) / 2 * 255), 0, 255).astype(
            np.uint8
        )

    if umap_sel is not None:
        ucx = float(np.mean(np.percentile(umap_sel[:, 0], [0.5, 99.5])))
        ucy = float(np.mean(np.percentile(umap_sel[:, 1], [0.5, 99.5])))
        uxc, uyc = umap_sel[:, 0] - ucx, umap_sel[:, 1] - ucy
        uhalf = (
            float(
                max(np.percentile(np.abs(uxc), 99.5), np.percentile(np.abs(uyc), 99.5))
            )
            or 1.0
        )
        UMAPX, UMAPY = u8(uxc / uhalf), u8(uyc / uhalf)
    else:
        UMAPX = UMAPY = np.zeros(n, np.uint8)
    CT, CASTE, REGION = (
        ct.astype(np.uint8),
        caste.astype(np.uint8),
        region_bits.astype(np.uint8),
    )
    CANON, SUB = canon.astype(np.uint8), sub.astype(np.uint8)
    LT = lt_full[idx]
    KCT, KCP = kctype_full[idx], kcprob_full[idx]
    CONF = np.clip(
        np.round((conf - conf.min()) / (conf.max() - conf.min()) * 255), 0, 255
    ).astype(np.uint8)

    # Gene selection: always include the named focal genes (symbols), then fill up
    # to K with the most spatially-variable remaining genes. Named genes are listed
    # first so they head the picker; fillers follow by descending variance.
    var = Xs.var(0)
    by_var = list(np.argsort(var)[::-1])
    id_to_idx = {g: i for i, g in enumerate(gene_ids)}
    focal = [id_to_idx[g] for g in GENE_SYMBOLS if g in id_to_idx]
    focal.sort(key=lambda i: -var[i])
    chosen = list(focal)
    for gi in by_var:
        if len(chosen) >= args.genes:
            break
        if gi not in chosen:
            chosen.append(gi)
    # Pin the two focal neuropeptides to the very front of the picker, in
    # this order, ahead of the variance sort: tachykinins then sNPF (the focal
    # genes of fig:caste_de and fig:neuropeptide_spatial).
    PIN_FIRST = ["bimp_v4.3_009257", "bimp_v4.3_005730"]  # tachykinins, sNPF
    pin = [id_to_idx[g] for g in PIN_FIRST if g in id_to_idx and id_to_idx[g] in chosen]
    chosen = pin + [gi for gi in chosen if gi not in pin]

    descriptions = load_descriptions()
    gene_arrays, gene_meta = [], []
    for gi in chosen:
        col = Xs[:, gi]
        hi = float(np.percentile(col, 99.5)) or 1.0
        gene_arrays.append(np.clip(np.round(col / hi * 255), 0, 255).astype(np.uint8))
        gid = gene_ids[gi]
        desc = descriptions.get(gid)
        symbol = GENE_SYMBOLS.get(gid) or (short_symbol(desc) if desc else None)
        gene_meta.append(
            {
                "id": gid,
                "symbol": symbol,
                "desc": desc,
                "hi": hi,
                "var": float(var[gi]),
            }
        )

    def offsets_of(arrays):
        off, p = {}, 0
        for name, arr in arrays:
            off[name] = [p, arr.nbytes, str(arr.dtype)]
            p += arr.nbytes
        return off

    def pack(arrays):
        buf = b"".join(arr.tobytes() for _, arr in arrays)
        return base64.b64encode(gzip.compress(buf, 9)).decode(), offsets_of(arrays)

    core = [
        ("x", X16),
        ("y", Y16),
        ("z", Z16),
        ("celltype", SUB),
        ("celltyperaw", CT),
        ("celltypecanon", CANON),
        ("caste", CASTE),
        ("conf", CONF),
        ("region", REGION),
        ("umapx", UMAPX),
        ("umapy", UMAPY),
        ("ltkc1", np.ascontiguousarray(LT[:, 0])),
        ("ltkc2", np.ascontiguousarray(LT[:, 1])),
        ("ltkc3", np.ascontiguousarray(LT[:, 2])),
        ("kctype", KCT),
        ("kcprob", KCP),
    ]

    region_defs = [
        {
            "key": col,
            "label": lab,
            "group": grp,
            "bit": bit,
            "count": int(((region_bits_full >> bit) & 1).sum()),
        }
        for bit, (col, lab, grp) in enumerate(REGION_DEFS)
    ]
    # bee names, fly orthologs and search aliases (data/panel/bimp_v43_gene_names.csv);
    # applied again below once the inferred genes are appended
    gene_names.annotate(gene_meta)
    shared = dict(
        gzip=True,
        nTotal=n_total,
        cellTypes=subtype_names,
        cellTypeCounts=full_counts,
        subtypeColors=subtype_colors,
        rawNames=ct_names,  # 29 raw tangram names, for the inspector's per-cell call
        canonTypes=canon_order,
        canonColors=canon_colors,
        canonCounts=[int((canon_full == i).sum()) for i in range(len(canon_order))],
        castes=["queen", "worker"],
        casteTotal=[int((caste_full == 0).sum()), int((caste_full == 1).sum())],
        regionDefs=region_defs,
        confRange=[float(conf.min()), float(conf.max())],
        genes=gene_meta,
        zLevels=35,
        # µm per normalised unit (x, y and z were all divided by it above): the viewer's
        # scale bar and its cell-sized dots
        scaleUm=round(scale, 4),
        # fig:kc_subtypes' Large-type subclusters, projected (see LTKC_H5AD);
        # colours as in the figure
        ltkc=dict(
            labels=["Subcluster 1", "Subcluster 2", "Subcluster 3"],
            short=["S1", "S2", "S3"],
            colors=[LTKC_COLORS[c] for c in LTKC_CLUSTERS],
            n=int((lt_full.sum(1) > 0).sum()),
        ),
        # fig:kc_subtypes panel B's restricted Kenyon-type transfer (see KC5POP_H5AD)
        kcTypes=dict(
            names=KC5_TYPES,
            short=[KC5_SHORT[t] for t in KC5_TYPES],
            colors=[_kc5_color(t) for t in KC5_TYPES],
            counts=[int((kctype_full == i + 1).sum()) for i in range(len(KC5_TYPES))],
            n=int((kctype_full > 0).sum()),
        ),
    )

    # ---- GitHub-Pages build: ALL cells as fetched binary files (no size limit).
    # Tiered like the artifact: a small LITE base (~100k) fetches first for a snappy
    # paint, then the full 1.3M base streams in and swaps; genes are per-file lazy. --
    if args.pages:
        out = HERE / "pages" / "data"
        (out / "genes").mkdir(parents=True, exist_ok=True)
        (out / "base.bin").write_bytes(
            gzip.compress(b"".join(a.tobytes() for _, a in core), 6)
        )
        # lite base: caste-balanced ~args.lite subsample
        per = args.lite // 2
        lorder = np.concatenate(
            [
                rng.choice(
                    np.where(caste_full == c)[0],
                    size=min(int((caste_full == c).sum()), per),
                    replace=False,
                )
                for c in (0, 1)
            ]
        )
        lorder.sort()
        lite_core = [(name, arr[lorder]) for name, arr in core]
        (out / "base-lite.bin").write_bytes(
            gzip.compress(b"".join(a.tobytes() for _, a in lite_core), 6)
        )
        for i, a in enumerate(gene_arrays):
            (out / "genes" / f"g{i:03d}.bin").write_bytes(gzip.compress(a.tobytes(), 6))
            gene_meta[i] = dict(
                gene_meta[i], file=f"genes/g{i:03d}.bin", kind="measured"
            )
        # Tangram-INFERRED genes: the imputed h5ad projects the full snRNA
        # transcriptome (2162 genes) onto all cells — the measured 100 + ~2062
        # imputed-only. Write one lazy file per inferred gene, tagged; flag genes
        # that are mostly near-zero (sparse) as low-confidence.
        if args.imputed_genes:
            gene_meta = write_imputed_genes(
                out, gene_meta, descriptions, args.imputed_h5ad
            )
            gene_names.annotate(gene_meta)
        manifest = dict(
            version=f"pages-{n_total}-{len(gene_meta)}",
            mode="fetch",
            n=n_total,
            base="base.bin",
            offsets=offsets_of(core),
            genesBase=args.genes_base_url,
            **(
                {"inferredMapping": args.inferred_mapping}
                if args.inferred_mapping
                else {}
            ),
            lite=dict(
                file="base-lite.bin", n=int(len(lorder)), offsets=offsets_of(lite_core)
            ),
            **{**shared, "genes": gene_meta},
        )
        (out / "manifest.json").write_text(json.dumps(manifest))

        def mb(p):
            return p.stat().st_size / 1e6

        genes_mb = sum(g.stat().st_size for g in (out / "genes").glob("*.bin")) / 1e6
        print(
            f"wrote {out.relative_to(REPO)}  base.bin={mb(out / 'base.bin'):.1f}MB "
            f"base-lite.bin={mb(out / 'base-lite.bin'):.2f}MB  {len(chosen)} "
            f"genes={genes_mb:.1f}MB  ({n_total:,} cells)"
        )
        return

    # ---- inline single-file artifact: downsample + base64, LITE + FULL tiers ----
    b64_full, off_full = pack(
        core + [(f"gene{i}", a) for i, a in enumerate(gene_arrays)]
    )
    lite_n = min(args.lite, n)
    lorder = rng.permutation(n)[:lite_n]
    lorder.sort()
    b64_lite, off_lite = pack([(name, arr[lorder]) for name, arr in core])
    manifest = dict(
        version=f"v8-{n}-{lite_n}-{len(chosen)}-{args.seed}",
        n=int(n),
        full=dict(n=int(n), offsets=off_full),
        lite=dict(n=int(lite_n), offsets=off_lite),
        **shared,
    )
    # Two files so the build can put the LITE payload + all code BEFORE the heavy
    # FULL blob in the HTML. The browser stream-parses and paints the lite tier
    # after ~1 MB while the full blob (last in the file) keeps downloading.
    lite_out = HERE / "src" / "data-lite.js"
    lite_out.write_text(
        "// GENERATED by export_data.py — do not edit.\n"
        f"window.__BIMP__ = {json.dumps(manifest)};\n"
        f"window.__BIMP__.b64lite = {json.dumps(b64_lite)};\n"
    )
    full_out = HERE / "src" / "data-full.js"
    full_out.write_text(
        "// GENERATED by export_data.py — do not edit.\n"
        f"window.__BIMP__.b64 = {json.dumps(b64_full)};\n"
        "if (window.__BIMP_needFull__) window.__BIMP_needFull__();\n"
    )
    lite_mb, full_mb = lite_out.stat().st_size / 1e6, full_out.stat().st_size / 1e6
    n_named = sum(1 for g in gene_meta if g["symbol"])
    print(
        f"wrote data-lite.js ({lite_mb:.2f} MB, {lite_n:,} cells) + data-full.js "
        f"({full_mb:.2f} MB, {n:,} cells, {len(chosen)} genes, {n_named} named)"
    )


if __name__ == "__main__":
    main()
