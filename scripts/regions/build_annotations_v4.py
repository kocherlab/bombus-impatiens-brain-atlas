"""annotations_v4 = v3 + the antennal-lobe relabel, deduplicated and renamed.

Three things happen here, in this order:

  MERGE      antennal-lobe polygons from the uploaded relabel are added to the
             matching v3 file. Worker 0068401 R1 is deliberately NOT taken from
             the upload: the uploaded copy is the untouched staging
             file, in which the four polygons are still classed `lobula`, so
             applying it would undo the reclassification.

  DEDUPE     exact duplicate rings are dropped. The relabel upload
             contains 54 of them on 0069041 R5 and one on R2. Duplicates
             overlay perfectly, so they change no cell count and are invisible
             in every rendering -- they only inflate polygon counts and make a
             section look doubly annotated.

  RENAME     every polygon gets `b<i>_<region>[_<n>]`, where i indexes the brain
             section within the capture region (a capture region holds two to
             four) and n disambiguates repeats. The mixture inherited from
             QuPath -- `Selection 7`, `b0_c3_kc_oc_0`, `b0_al0` -- carried no
             information that the class and the geometry do not already carry.

             The ellipsoid and fan-shaped bodies keep names ending `_eb` / `_fb`
             AND get an explicit `subregion` property. nine_region_common.
             canonical_name() reads `subregion` first and falls back to parsing
             `_(eb|fb)\\b` out of the name, so writing both means neither the
             name nor the property alone is load-bearing.

Original names are preserved in `orig_name` so nothing is lost.

  python3 scripts/regions/build_annotations_v4.py --dry-run --upload DIR
  python3 scripts/regions/build_annotations_v4.py --apply --upload DIR
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (  # noqa: E402
    BRAINS,
    CENTRAL_COMPLEX,
    canonical_name,
)

PX = 0.2125
# The upload's copy of this file predates the reclassification; keep v3's.
SKIP_UPLOAD = {("0068401", 1)}
SHORT = {"ellipsoid_body": "eb", "fan_shaped_body": "fb", "antennal_lobe": "al"}


def feats(p):
    o = json.load(open(p))
    return o["features"] if isinstance(o, dict) else o


def ring_of(ft):
    g = ft.get("geometry") or {}
    if g.get("type") != "Polygon":
        return None
    return np.array(g["coordinates"][0], float) * PX


def key_of(ft):
    r = ring_of(ft)
    return None if r is None else tuple(np.round(r.ravel(), 3))


def crop_boxes(brain):
    """(slide, section) -> [(suffix, lo, hi)] for each brain section."""
    out = defaultdict(list)
    for f in sorted((Path(BRAINS[brain]["dir"]) / "data").glob("*.h5ad")):
        m = re.search(r"__(\d{7})__Region_(\d+)__", f.name)
        if not m:
            continue
        with h5py.File(f, "r") as h:
            xy = h["obsm"]["spatial"][:, :2].astype(float)
        out[(m.group(1), int(m.group(2)))].append(
            (f.stem.split("_")[-1], xy.min(0) - 150, xy.max(0) + 150)
        )
    for k in out:
        out[k].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    ap.add_argument(
        "--upload",
        type=Path,
        required=True,
        help="the relabelled GeoJSON set (relabeled_20260831_g/)",
    )
    a = ap.parse_args()

    grand = Counter()
    for brain in ("queen", "worker"):
        base = Path(BRAINS[brain]["annotations"])
        src, dst = base / "v3", base / "v4"
        boxes = crop_boxes(brain)
        if a.apply:
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)

        print(f"\n=== {brain} ===")
        print(f"  {'file':<52}{'kept':>6}{'added':>7}{'dup':>6}{'renamed':>9}")
        for p in sorted(src.glob("*.geojson")):
            m = re.search(r"__(\d{7})__Region_(\d+)__", p.name)
            if not m:
                continue
            slide, sec = m.group(1), int(m.group(2))
            out = list(feats(p))
            seen = {k for k in (key_of(f) for f in out) if k}

            # ---- merge the upload -------------------------------------
            added = 0
            # the upload names files by export time; match on slide + section
            up = next(
                (
                    q
                    for q in sorted(a.upload.glob("*.geojson"))
                    if (m2 := re.search(r"__(\d{7})__Region_(\d+)__", q.name))
                    and (m2.group(1), int(m2.group(2))) == (slide, sec)
                ),
                a.upload / p.name,
            )
            if up.exists() and (slide, sec) not in SKIP_UPLOAD:
                for ft in feats(up):
                    k = key_of(ft)
                    if k is None or k in seen:
                        continue
                    pr = ft.get("properties") or {}
                    if canonical_name(pr) is None:
                        continue
                    # The annotator marks an abandoned shape by putting
                    # "(empty)" in its name. Two such slivers on 0068401 R4
                    # were dropped when v3 was built -- both contain zero
                    # cells and both sit inside a lobula polygon that was
                    # kept -- so taking them back from the upload would
                    # silently reverse that.
                    if "(empty)" in str(pr.get("name") or ""):
                        continue
                    out.append(ft)
                    seen.add(k)
                    added += 1

            # ---- drop exact duplicates --------------------------------
            uniq, kseen, dup = [], set(), 0
            for ft in out:
                k = key_of(ft)
                if k is None:
                    uniq.append(ft)
                    continue
                if k in kseen:
                    dup += 1
                    continue
                kseen.add(k)
                uniq.append(ft)

            # ---- standardise names ------------------------------------
            counters: dict = defaultdict(int)
            renamed = 0
            for ft in uniq:
                pr = ft.setdefault("properties", {})
                cls = canonical_name(pr)
                if cls is None:
                    continue
                r = ring_of(ft)
                if r is None:
                    continue
                c = r.mean(0)
                idx = 0
                for j, (_suffix, lo, hi) in enumerate(boxes.get((slide, sec), [])):
                    if (c >= lo).all() and (c <= hi).all():
                        idx = j
                        break
                short = SHORT.get(cls, cls)

                # Write the resolved class back onto the polygon. The exports
                # arrive with one region spelled several ways -- Lamina and
                # lamina, Lobula and lobula and the long-standing Lobulla
                # typo, Central Complex and Central complex and central
                # complex. canonical_name() absorbs all of it, so nothing
                # downstream ever broke, but the files disagreeing with each
                # other is what a person editing them in QuPath sees and
                # copies. The central complex keeps its spaced spelling: it is
                # a SPLIT_CLASS whose substructure comes from the name or the
                # subregion property, not from the class string.
                disk_cls = "central complex" if cls == CENTRAL_COMPLEX else cls
                cur = pr.get("classification")
                if isinstance(cur, dict):
                    cur["name"] = disk_cls
                else:
                    pr["classification"] = disk_cls
                sub = pr.get("subregion")
                if not sub:
                    # Polygons arriving from QuPath carry the substructure only
                    # in their name (b0_eb, b1_fb). Recover it before renaming,
                    # or the rename destroys the one copy of that information.
                    mm = re.search(r"_(eb|fb)(?![a-z])", str(pr.get("name", "")))
                    if mm:
                        sub = {"eb": "ellipsoid_body", "fb": "fan_shaped_body"}[
                            mm.group(1)
                        ]
                if cls == "central_complex" or sub in (
                    "ellipsoid_body",
                    "fan_shaped_body",
                ):
                    # The two substructures are pooled for reporting, but the
                    # name still says which one it is and `subregion` still
                    # carries it explicitly, so un-pooling needs no re-naming.
                    sub = sub or cls
                    tag = SHORT.get(sub, sub)
                    n = counters[(idx, sub)]
                    counters[(idx, sub)] += 1
                    new = f"b{idx}_{tag}_{n}"
                    pr["subregion"] = sub
                else:
                    n = counters[(idx, cls)]
                    counters[(idx, cls)] += 1
                    new = f"b{idx}_{short}_{n}"
                old = str(pr.get("name") or "")
                if old != new:
                    if old and "orig_name" not in pr:
                        pr["orig_name"] = old
                    pr["name"] = new
                    renamed += 1

            print(f"  {p.name[:51]:<52}{len(uniq):>6}{added:>7}{dup:>6}{renamed:>9}")
            grand["kept"] += len(uniq)
            grand["added"] += added
            grand["dup"] += dup
            grand["renamed"] += renamed

            if a.apply:
                json.dump(
                    {"type": "FeatureCollection", "features": uniq},
                    open(dst / p.name, "w"),
                )

    print("\n=== totals ===")
    for k in ("kept", "added", "dup", "renamed"):
        print(f"  {k:<10}{grand[k]:>8,}")
    if a.dry_run:
        print("\ndry run — nothing written. Re-run with --apply.")
    else:
        print("\nwritten to data/annotations/*/v4/")


if __name__ == "__main__":
    main()
