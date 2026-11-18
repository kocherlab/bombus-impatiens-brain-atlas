"""Recompute anatomical region membership for the nine-region atlas.

Why this does not go through Snakemake
--------------------------------------
``label`` is the root of the pipeline DAG, so ``snakemake`` would rebuild
``align`` (PASTE3, GPU, ~6 h) and everything after it.  That alignment is
stochastic: re-running it would move every spatial result and
invalidate the canonical Tangram object, all to add polygons that do not move
a single cell.  Alignment, clustering and Tangram never read ``in_*`` columns
-- ``analysis_workflow.py`` explicitly carries them through concatenation --
so the labels are passengers and can be recomputed on their own.

The join cannot be positional
-----------------------------
Crop 1A holds more cells than it contributes to the aligned object: QC
drops some.  Slices are block-contiguous in the aligned object, which makes a
positional join *look* right while silently shifting labels within a block.
Everything here keys on (slice_name, cell_id).

Stages
------
``crops``   per-crop point-in-polygon -> one parquet sidecar per crop
``verify``  recompute the six OLD regions from the OLD annotations and assert
            cell-for-cell equality with the stored ``in_*`` columns.  This is
            the correctness proof for the machinery and for the switch to
            vectorised ``shapely.contains_xy``.
``join``    write the nine columns into copies of the atlas objects
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (  # noqa: E402
    BASE_REGIONS,
    BRAINS,
    REPO,
    UNTRACED_SECTIONS,
    all_regions,
    annotation_sources,
    crop_map,
    load_regions,
)

# The live vocabulary, discovered from the annotation files themselves.  A
# region drawn in QuPath appears here with no code change.
ALL_REGIONS = all_regions("v2")

OUT = REPO / "results/nine_region"


# --------------------------------------------------------------------------
# h5ad readers that avoid materialising X
# --------------------------------------------------------------------------
def _read_obs_col(h5obs, key):
    """Read one obs column from an open h5ad, handling categoricals."""
    if key not in h5obs:
        return None
    node = h5obs[key]
    if isinstance(node, h5py.Group):  # categorical
        cats = node["categories"][:]
        codes = node["codes"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        out = np.full(len(codes), None, dtype=object)
        ok = codes >= 0
        out[ok] = cats[codes[ok]]
        return out
    vals = node[:]
    if vals.dtype.kind == "S":
        return np.array([v.decode() for v in vals])
    return vals


def crop_cell_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cid = _read_obs_col(f["obs"], "cell_id")
        if cid is None:
            idx = f["obs"].attrs["_index"]
            cid = _read_obs_col(f["obs"], idx)
    return np.asarray(cid).astype(str)


def crop_spatial(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["obsm"]["spatial"][:, :2].astype(float)


def crop_stored_flags(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        for k in f["obs"].keys():
            if k.startswith("in_"):
                out[k[3:]] = np.asarray(_read_obs_col(f["obs"], k)).astype(bool)
    return out


# --------------------------------------------------------------------------
def find_crop_file(data_dir: Path, slide: str, suffix: str) -> Path | None:
    """Replicate label-all's matching: stem ends with _<suffix> and names the slide."""
    for f in sorted(data_dir.glob("*.h5ad")):
        n = f.name
        if "_clustered" in n or "_labeled" in n or n.startswith(("aligned", "test_")):
            continue
        if f.stem.endswith("_" + suffix) and f"__{slide}__" in n:
            return f
    return None


def membership(coords: np.ndarray, regions: dict) -> dict:
    """Vectorised point-in-polygon for every region present in this section."""
    from shapely import contains_xy

    x, y = coords[:, 0], coords[:, 1]
    return {
        name: np.asarray(contains_xy(geom, x, y), dtype=bool)
        for name, geom in regions.items()
    }


# --------------------------------------------------------------------------
def slide_region_drawn(
    brain: str, src: dict, use_old: bool = False, repair_invalid: bool = True
) -> dict:
    """(slide, region) -> was this region drawn anywhere on this slide?

    Absence of a region from a GeoJSON is ambiguous: it can mean the structure
    is genuinely not in that section, or that nobody traced it.  The two look
    identical in the file but are not symmetric in cost -- recording "never
    traced" as False manufactures a difference out of missing data, while
    recording "genuinely absent" as null only costs a little power.

    The disambiguation used here is the granularity at which annotation
    actually happens.  A region missing from EVERY section of a slide, while
    present on another slide of the same brain, is treated as not traced on
    that slide (null).  A region missing from SOME sections of a slide where
    it was traced is real anatomy (False) -- the central complex genuinely
    does not span every section.

    This is what caught the worker antennal lobe: it is drawn on all six
    annotated sections of slide 0068401 and on none of the eight sections of
    0069041, at a per-section density matching the queen's.
    """
    per_slide: dict = {}
    slides = set()
    for (slide, _section), s in sorted(src.items()):
        path = s.get("old") if use_old else (s.get("v2") or s.get("old"))
        if path is None:
            continue
        slides.add(slide)
        for r in load_regions(path, repair_invalid=repair_invalid):
            per_slide.setdefault((slide, r), 0)
            per_slide[(slide, r)] += 1

    drawn = {}
    for slide in slides:
        for r in ALL_REGIONS:
            drawn[(slide, r)] = per_slide.get((slide, r), 0) > 0
    return drawn


def stage_crops(
    brain: str,
    use_old: bool = False,
    outdir: Path | None = None,
    repair_invalid: bool = True,
):
    """Compute the nine flags for every crop of one brain.

    ``repair_invalid`` controls whether a self-intersecting hand-drawn polygon
    is repaired or dropped.  It exists so that the verify stage can reproduce
    the ORIGINAL loader's behaviour exactly; production runs repair.
    """
    data_dir = BRAINS[brain]["dir"] / "data"
    cm = crop_map(brain)
    src = annotation_sources(brain)
    drawn = slide_region_drawn(
        brain, src, use_old=use_old, repair_invalid=repair_invalid
    )

    notdrawn = sorted({(s, r) for (s, r), v in drawn.items() if not v})
    if notdrawn:
        print(
            f"  [{brain}] regions never traced on a whole slide -> recorded "
            f"as not-annotated (null), not as absent:",
            flush=True,
        )
        for s, r in notdrawn:
            print(f"     slide {s}: {r}", flush=True)
    outdir = outdir or (OUT / ("crops_old" if use_old else "crops") / brain)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (slide, section), suffixes in sorted(cm.items()):
        s = src.get((slide, section), {})
        # verify mode forces the ORIGINAL annotations so the comparison is like-for-like
        path = s.get("old") if use_old else (s.get("v2") or s.get("old"))
        if path is None:
            print(f"  !! {brain} {slide} R{section}: no annotation at all", flush=True)
            continue

        from_v2 = (not use_old) and s.get("v2") is not None
        # A region counts as annotated for this section when (a) the section's
        # source file covers that vocabulary at all, and (b) the region was
        # actually traced somewhere on this slide.  Condition (b) is what stops
        # a whole slide that nobody traced for a region from being recorded as
        # a slide where the region is absent.
        vocab = set(ALL_REGIONS) if from_v2 else set(BASE_REGIONS)
        annotated = {r for r in vocab if drawn.get((slide, r), False)}
        # Sections the slide-level rule cannot see: traced for a region
        # everywhere else on the slide, but demonstrably skipped here. See
        # nine_region_common.UNTRACED_SECTIONS for the evidence per entry.
        skipped = UNTRACED_SECTIONS.get((brain, slide, section), set()) & annotated
        if skipped:
            # A mask entry that names a region the section actually HAS would
            # throw away real tracing (a section masked and later traced would
            # have every cell's flag nulled). Fail rather than discard.
            drawn_here = {
                r
                for r in skipped
                if any(
                    rr == r for rr in load_regions(path, repair_invalid=repair_invalid)
                )
            }
            assert not drawn_here, (
                f"{brain} {slide} R{section}: UNTRACED_SECTIONS marks "
                f"{sorted(drawn_here)} as untraced, but polygons for it exist in "
                f"{path.name}. Remove the entry -- masking it would discard real "
                "annotation."
            )
            annotated -= skipped
            print(
                f"  [{brain}] {slide} R{section}: {sorted(skipped)} not traced on "
                f"this section -> null, not absent",
                flush=True,
            )

        # The six originals are traced on every section, so a slide where one
        # of them appears nowhere would mean the annotation set changed under
        # us. Assert that rather than forcing them into `annotated`: a blind
        # override would reintroduce, for the base six, exactly the
        # false-absence bug this masking exists to prevent, and the "recorded
        # as null" message printed above would then contradict what is written.
        undrawn_base = sorted(
            r for r in (vocab & set(BASE_REGIONS)) if not drawn.get((slide, r), False)
        )
        assert not undrawn_base, (
            f"{brain} slide {slide}: base region(s) {undrawn_base} are traced "
            "on no section of this slide. That is new; decide deliberately "
            "whether they are absent (False) or untraced (null) before "
            "re-running, rather than letting this default."
        )

        regions = load_regions(path, repair_invalid=repair_invalid)
        for suffix in suffixes:
            cf = find_crop_file(data_dir, slide, suffix)
            if cf is None:
                print(f"  !! no crop h5ad for {slide} R{section} {suffix}", flush=True)
                continue
            coords = crop_spatial(cf)
            cids = crop_cell_ids(cf)
            if len(np.unique(cids)) != len(cids):
                raise SystemExit(f"cell_id not unique within {cf.name}")

            hit = membership(coords, regions)
            df = pd.DataFrame({"cell_id": cids})
            for r in ALL_REGIONS:
                if r not in annotated:
                    df[f"in_{r}"] = pd.array([pd.NA] * len(df), dtype="boolean")
                else:
                    v = hit.get(r, np.zeros(len(df), dtype=bool))
                    df[f"in_{r}"] = pd.array(v, dtype="boolean")
            df["slice_name"] = cf.stem
            df.to_parquet(outdir / f"{cf.stem}.parquet", index=False)

            counts = {
                r: int(df[f"in_{r}"].sum(skipna=True))
                for r in ALL_REGIONS
                if r in annotated
            }
            rows.append(
                dict(
                    brain=brain,
                    slide=slide,
                    section=section,
                    crop=suffix,
                    n=len(df),
                    source=path.name,
                    source_set=path.parent.name,
                    **counts,
                )
            )
            print(
                f"  {cf.stem}: n={len(df):,} "
                + " ".join(f"{r}={counts.get(r, 'NA')}" for r in ALL_REGIONS),
                flush=True,
            )

    summ = pd.DataFrame(rows)
    summ.to_csv(outdir / "_summary.csv", index=False)
    print(f"\nwrote {len(rows)} crop sidecars -> {outdir}")
    return summ


def stage_verify(brain: str):
    """Prove the machinery: OLD annotations must reproduce the STORED flags exactly.

    ``repair_invalid=False`` is what makes this a proof rather than an
    approximation. The stored columns were produced by the original loader,
    which discarded self-intersecting polygons; recomputing WITH repair
    guarantees a difference of exactly those cells, and a genuine regression
    in the vectorised point-in-polygon join could then hide inside a diff that
    a maintainer has been taught to expect. Disabling repair asks the one
    question this stage exists to answer -- does the reimplementation
    reproduce the original six-region labels cell for cell -- and leaves the separate
    question of what repair changes to stage_join, which counts it explicitly.
    """
    tmp = OUT / "verify" / brain
    stage_crops(brain, use_old=True, outdir=tmp, repair_invalid=False)

    data_dir = BRAINS[brain]["dir"] / "data"
    bad, checked, total_cells = [], 0, 0
    for pq in sorted(tmp.glob("*.parquet")):
        cf = data_dir / f"{pq.stem}.h5ad"
        if not cf.exists():
            continue
        df = pd.read_parquet(pq)
        stored = crop_stored_flags(cf)
        if not stored:
            continue
        checked += 1
        total_cells += len(df)
        for r in BASE_REGIONS:
            if r not in stored:
                continue
            got = df[f"in_{r}"].fillna(False).to_numpy(dtype=bool)
            exp = stored[r]
            if len(got) != len(exp):
                bad.append((pq.stem, r, f"length {len(got)} vs {len(exp)}"))
                continue
            d = int((got != exp).sum())
            if d:
                bad.append(
                    (pq.stem, r, f"{d} cells differ ({100 * d / len(exp):.4f}%)")
                )

    print(f"\n=== VERIFY {brain}: {checked} crops, {total_cells:,} cells ===")
    if bad:
        print(f"MISMATCHES ({len(bad)}):")
        for s, r, m in bad[:40]:
            print(f"  {s} {r}: {m}")
    else:
        print(
            "EXACT MATCH on all six original regions -- machinery reproduces "
            "the stored labels cell-for-cell."
        )
    return bad


def stage_join(brain: str):
    """Write the nine flags into a copy of aligned_clustered.h5ad."""
    import anndata as ad

    src = BRAINS[brain]["dir"] / "data" / "aligned_clustered.h5ad"
    dst_dir = OUT / "atlas"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{brain}_aligned_clustered_9region.h5ad"

    print(f"loading {src} ...", flush=True)
    adata = ad.read_h5ad(src)

    key = (
        pd.Series(adata.obs["slice_name"].astype(str)).to_numpy()
        + "||"
        + pd.Series(adata.obs["cell_id"].astype(str)).to_numpy()
    )
    if len(np.unique(key)) != len(key):
        raise SystemExit("(slice_name, cell_id) is not unique in the atlas object")

    frames = [
        pd.read_parquet(p) for p in sorted((OUT / "crops" / brain).glob("*.parquet"))
    ]
    side = pd.concat(frames, ignore_index=True)
    side["_k"] = side["slice_name"].astype(str) + "||" + side["cell_id"].astype(str)
    side = side.drop_duplicates("_k").set_index("_k")

    idx = side.index.get_indexer(key)
    miss = int((idx < 0).sum())
    print(f"atlas cells {len(key):,}; matched {len(key) - miss:,}; unmatched {miss:,}")
    if miss:
        raise SystemExit("unmatched atlas cells -- refusing to write partial labels")

    # Preserve the original six-region columns BEFORE overwriting.  The verify stage
    # proved that the recomputation reproduces them exactly only with the
    # polygon repair DISABLED; the columns written below are the repaired
    # ones, which differ by the four polygons only the repair can read.
    # So this is a real change to the six original regions, and it is recorded
    # rather than assumed away.
    diffs = {}
    for r in BASE_REGIONS:
        col = f"in_{r}"
        if col not in adata.obs:
            continue
        prev = np.asarray(adata.obs[col]).astype(bool)
        adata.obs[f"{col}__prerepair"] = pd.array(prev, dtype="boolean")
        new = pd.array(side[col].to_numpy()[idx], dtype="boolean")
        d = int((np.asarray(new.fillna(False)).astype(bool) != prev).sum())
        if d:
            diffs[r] = d

    for r in ALL_REGIONS:
        vals = side[f"in_{r}"].to_numpy()[idx]
        adata.obs[f"in_{r}"] = pd.array(vals, dtype="boolean")

    adata.uns["nine_region_provenance"] = dict(
        vocabulary=list(ALL_REGIONS),
        repaired_invalid_polygons=True,
        changed_vs_published={k: int(v) for k, v in diffs.items()},
        note=(
            "in_<region> are the repaired labels; in_<region>__prerepair are "
            "the columns the published atlas carried. The difference is the "
            "self-intersecting polygons the original loader discarded."
        ),
    )
    if diffs:
        print(
            "  polygon repair changed the published six:",
            ", ".join(f"{k} {v:+,}" for k, v in diffs.items()),
        )

    adata.write_h5ad(dst)
    print(f"wrote {dst}")
    for r in ALL_REGIONS:
        c = adata.obs[f"in_{r}"]
        print(
            f"  {r}: {int(c.sum(skipna=True)):,} cells, {int(c.isna().sum()):,} "
            "not-annotated"
        )
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["queen", "worker"], required=True)
    ap.add_argument("--stage", choices=["crops", "verify", "join"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"crops": stage_crops, "verify": stage_verify, "join": stage_join}[a.stage](a.brain)


if __name__ == "__main__":
    main()
