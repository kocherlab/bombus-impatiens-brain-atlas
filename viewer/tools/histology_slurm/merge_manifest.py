#!/usr/bin/env python3
"""Merge the stain and tile indexes into histo/manifest.json, and check the result.

Run after copying histo/, histo_stains/ and histo_tiles/ into the viewer's data dir
(stdlib only, seconds; safe to re-run -- it rewrites the same keys):

    python3 viewer/tools/histology_slurm/merge_manifest.py [viewer/pages/data]

Adds to histo/manifest.json:
  * "stains": [{key, label, path, small}] from histo_stains/stains.json, the non-DAPI
    paths prefixed "../histo_stains/" (they resolve relative to the manifest), and
    per section "stains": [keys];
  * "tiles": {path, tile, overlap} from histo_tiles/tiles.json (an absolute "path"
    already in the manifest, set by the deploy step, is kept), and per section
    "tiles": {stains, levels: [{l, um_per_px, w, h, nx, ny, have}]}.
Then checks that every file the manifest names exists and prints the counts
(expected: 30 queen + 35 worker sections, 17,128 tiles).
"""

import json
import sys
from pathlib import Path

data = Path(sys.argv[1] if len(sys.argv) > 1 else "viewer/pages/data")
man_p = data / "histo" / "manifest.json"
man = json.loads(man_p.read_text())
by = {c: {s["i"]: s for s in v["sections"]} for c, v in man["castes"].items()}

st_p = data / "histo_stains" / "stains.json"
if st_p.exists():
    st = json.loads(st_p.read_text())
    man["stains"] = [
        dict(
            s,
            path=s["path"] if s["key"] == "dapi" else "../histo_stains/" + s["path"],
            small=s["small"] if s["key"] == "dapi" else "../histo_stains/" + s["small"],
        )
        for s in st["stains"]
    ]
    for c, secs in st["sections"].items():
        for i, keys in secs.items():
            by[c][int(i)]["stains"] = keys
else:
    print(f"no {st_p}: stains not merged")

ti_p = data / "histo_tiles" / "tiles.json"
ti = {
    "tiles": {"path": "../histo_tiles/{caste}/{ii}/{key}/{l}/{tx}_{ty}.webp"}
}  # on-disk layout
if ti_p.exists():
    ti = json.loads(ti_p.read_text())
    # keep a tile host the deploy step already set (an absolute URL, e.g. the
    # Cloudflare worker); only the relative default comes from tiles.json
    host = man.get("tiles", {}).get("path", "")
    man["tiles"] = dict(
        ti["tiles"],
        **({"path": host} if host.startswith(("http://", "https://")) else {}),
    )
    for c, secs in ti["sections"].items():
        for i, e in secs.items():
            by[c][int(i)]["tiles"] = e
else:
    print(f"no {ti_p}: tiles not merged")

tmp = man_p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(man, indent=1, ensure_ascii=False))
tmp.replace(man_p)

# ---- check: every referenced file exists
base = data / "histo"
missing, n_tiles, n_stain = [], 0, 0
stains = man.get("stains", [])
for c, secs in by.items():
    for i, s in secs.items():
        for f in (s["file"], s["small"]):
            if not (base / f).exists():
                missing.append(f)
        ii = f"{i:02d}"
        for k in s.get("stains", []):
            sd = next((x for x in stains if x["key"] == k), None)
            if sd and k != "dapi":
                for f in (sd["path"], sd["small"]):
                    n_stain += 1
                    p = f.replace("{caste}", c).replace("{ii}", ii)
                    if not (base / p).exists():
                        missing.append(p)
        t = s.get("tiles")
        if t:
            for lv in t["levels"]:
                for j, ch in enumerate(lv["have"]):
                    if ch != "1":
                        continue
                    for k in lv.get("stains", t["stains"]):
                        n_tiles += 1
                        p = ti["tiles"]["path"].format(
                            caste=c,
                            ii=ii,
                            key=k,
                            l=lv["l"],
                            tx=j % lv["nx"],
                            ty=j // lv["nx"],
                        )
                        if not (base / p).exists():
                            missing.append(p)
counts = {c: len(v) for c, v in by.items()}
tiled = {c: sum(1 for s in v.values() if "tiles" in s) for c, v in by.items()}
print(f"sections {counts} (expected queen 30, worker 35); with tiles {tiled}")
print(
    f"stain images {n_stain} (expected 390); tiles {n_tiles:,} (expected "
    f"17,128); missing {len(missing)}"
)
for p in missing[:10]:
    print("  MISSING", p)
sys.exit(1 if missing else 0)
