"""Reclassify worker 0068401 R1's four `lobula` polygons as antennal lobe.

Four independent properties agree, and each is measured against the other 31
sections rather than asserted:

  pair spacing   R1's polygons sit 388 and 415 um apart. Worker antennal lobes
                 sit 401 um apart (IQR 383-435); worker lobulae sit 1,174 um
                 apart (IQR 1,059-1,226). The two distributions do not overlap
                 anywhere in either brain, and R1 is one of only two sections
                 falling on the wrong side.
  shape          aspect 1.01, 1.06, 1.01, 1.31. Worker antennal lobe is
                 1.23 +/- 0.16; worker lobula is 1.52 +/- 0.26.
  trace density  108-243 vertices. Real antennal lobes on this slide run
                 85-251; real lobulae run 34-58.
  position       the pairs sit within ~150 um of where section R5 -- which has
                 both classes drawn -- puts its antennal lobes.

The other section the spacing test flagged, R4, is a false positive: its tight
spacing comes from two slivers the annotator named "Selection 7 (empty)" and
"Selection 9 (empty)", and its real lobula polygons are ordinary. Those two
empty objects are dropped here; R4's classes are left alone.

R1 has no antennal lobe otherwise and no lamina at all, which fits a section
whose annotation pass was incomplete.

  python3 scripts/regions/relabel_worker_r1_antennal_lobe.py --dry-run
  python3 scripts/regions/relabel_worker_r1_antennal_lobe.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import BRAINS, canonical_name  # noqa: E402

PX = 0.2125
V3 = BRAINS["worker"]["annotations"] / "v3"
R1 = V3 / "output-XETG00151__0068401__Region_1__20260115__193900.geojson"
R4 = V3 / "output-XETG00151__0068401__Region_4__20260115__193900.geojson"


def centroid(ft):
    r = np.array(ft["geometry"]["coordinates"][0], dtype=float) * PX
    return float(r[:, 0].mean()), float(r[:, 1].mean())


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    # ---- R1: lobula -> al -------------------------------------------------
    obj = json.load(open(R1))
    feats = obj["features"] if isinstance(obj, dict) else obj
    lob = [
        ft for ft in feats if canonical_name(ft.get("properties", {}) or {}) == "lobula"
    ]
    lob.sort(key=lambda ft: centroid(ft)[1])
    if len(lob) != 4:
        raise SystemExit(f"expected 4 lobula polygons in R1, found {len(lob)}")

    print("worker 0068401 R1 — reclassify lobula -> antennal lobe")
    print(f"   {'was':<12}{'becomes':<12}{'centroid':>18}{'verts':>7}")
    for i, ft in enumerate(lob):
        brain_i, idx = divmod(i, 2)
        new_name = f"b{brain_i}_al{idx}"
        cx, cy = centroid(ft)
        old = (ft.get("properties") or {}).get("name")
        print(
            f"   {'lobula':<12}{'al':<12}{f'({cx:.0f}, {cy:.0f})':>18}"
            f"{len(ft['geometry']['coordinates'][0]):>7}   {old} -> {new_name}"
        )
        if a.apply:
            pr = ft.setdefault("properties", {})
            cl = pr.get("classification")
            if isinstance(cl, dict):
                cl["name"] = "al"
            else:
                pr["classification"] = {"name": "al"}
            pr["name"] = new_name
            pr["reclassified_from"] = "lobula"

    # ---- R4: drop the two objects the annotator marked empty --------------
    obj4 = json.load(open(R4))
    f4 = obj4["features"] if isinstance(obj4, dict) else obj4
    keep, dropped = [], []
    for ft in f4:
        nm = str((ft.get("properties") or {}).get("name") or "")
        (dropped if "(empty)" in nm else keep).append(ft)
    print(
        f"\nworker 0068401 R4 — drop objects the annotator marked empty: {len(dropped)}"
    )
    for ft in dropped:
        cx, cy = centroid(ft)
        print(f"   {(ft.get('properties') or {}).get('name')}  at ({cx:.0f}, {cy:.0f})")

    if a.apply:
        json.dump(obj, open(R1, "w"))
        if isinstance(obj4, dict):
            obj4["features"] = keep
        else:
            obj4 = keep
        json.dump(obj4, open(R4, "w"))
        print("\nwritten to data/annotations/worker/v3/")
    else:
        print("\ndry run -- nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
