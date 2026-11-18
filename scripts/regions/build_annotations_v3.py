"""Rebuild the annotation set as ORIGINAL LABELS + appended new regions.

The v2 set was produced by taking the 2026-08 export wholesale. This instead
starts from the ORIGINAL set (data/annotations/<brain>/v1/) -- the labels the original
six-region atlas was built on, untouched -- and appends only the three regions the new
drop adds: antennal lobe, ellipsoid body, fan-shaped body.

Why start from the original rather than the new export: the original is the
record the atlas's region flags were built from, and it has been verified to
reproduce those flags cell for cell on both brains. Anything the new export changed
about the six original regions should be a deliberate, visible decision rather
than a side effect of adopting a whole file.

What is appended, per section, from the new export:
  * every polygon whose class resolves to al / ellipsoid_body / fan_shaped_body
  * nothing else

What is NOT done:
  * no polygon is reclassified. The positional audit found no case meeting the
    bar: worker 0068401 R1's extra mid-section `lobula` clusters sit near the
    antennal lobe, but the strong one is 19 cells against a real-AL range of
    42-164, and R1's lobula is already bilaterally complete.
  * no geometry is edited. Self-intersecting polygons are repaired at READ
    time, not rewritten here.

  python3 scripts/regions/build_annotations_v3.py --dry-run
  python3 scripts/regions/build_annotations_v3.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import BRAINS, annotation_sources, canonical_name  # noqa: E402

NEW = {"al", "ellipsoid_body", "fan_shaped_body"}


def features(path):
    obj = json.load(open(path))
    return obj["features"] if isinstance(obj, dict) else obj


def ring_key(ft):
    g = ft.get("geometry", {}) or {}
    c = (g.get("coordinates") or [[]])[0]
    return (g.get("type"), tuple((round(x, 3), round(y, 3)) for x, y in c))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    grand = Counter()
    for brain in ("queen", "worker"):
        base = BRAINS[brain]["annotations"]
        out = base / "v3"
        if a.apply:
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)

        print(f"\n=== {brain} ===")
        print(f"{'section':<16}{'from original':>14}{'appended':>10}   detail")
        # Name the source set explicitly. The default would resolve to
        # v3 -- the directory this script just created empty --
        # and every append would silently be a no-op.
        for (slide, sec), s in sorted(
            annotation_sources(brain, newest_dir="v2").items()
        ):
            old_p, new_p = s.get("old"), s.get("v2")
            if old_p is None:
                print(f"  {slide} R{sec:<4} NO ORIGINAL -- skipped")
                continue

            kept = features(old_p)
            seen = {ring_key(f) for f in kept}

            appended, counts = [], Counter()
            if new_p is not None:
                for ft in features(new_p):
                    nm = canonical_name(ft.get("properties", {}) or {})
                    if nm in NEW and ring_key(ft) not in seen:
                        appended.append(ft)
                        seen.add(ring_key(ft))
                        counts[nm] += 1

            detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "-"
            print(f"  {slide} R{sec:<4}{len(kept):>12}{len(appended):>10}   {detail}")
            grand.update(counts)
            grand["kept"] += len(kept)

            if a.apply:
                obj = {"type": "FeatureCollection", "features": kept + appended}
                name = old_p.name
                json.dump(obj, open(out / name, "w"))

    print("\n=== totals ===")
    print(f"  polygons carried from the original set : {grand.pop('kept'):,}")
    for k, v in sorted(grand.items()):
        print(f"  appended {k:<18}{v:>6}")
    if a.dry_run:
        print("\ndry run -- nothing written. Re-run with --apply.")
    else:
        print("\nwritten to data/annotations/*/v3/")


if __name__ == "__main__":
    main()
