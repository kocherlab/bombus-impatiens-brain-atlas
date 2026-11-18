#!/usr/bin/env python3
"""Split a high-resolution tile set into Cloudflare-sized shards and point the
manifest at them.

A free Cloudflare Worker holds at most 20,000 static files, so a tile set bigger
than that is cut into whole sections, one Worker per shard (`bimp-st-tiles-hq1`,
`-hq2`, ...). Run after
`export_tiles.py --stains dapi --upscale 8 --url-dir histo_tiles_hq \
    --out .../histo_tiles_hq`
and after `histology_slurm/merge_manifest.py` (stdlib only, safe to re-run):

    python3 viewer/tools/shard_hq_tiles.py [viewer/pages/data] \
        [--max-files 19000] \
        [--host 'https://bimp-st-tiles-hq{n}.kocherlab.workers.dev/']

Writes, next to histo_tiles_hq/:
  * histo_tiles_hq_shards/hq{n}/{caste}/{ii}/{stain}/{level}/{tx}_{ty}.webp
    (hard links; copied when the filesystem refuses), the folder
    `cloudflare.sh tiles-hq` uploads, one Worker each;
  * in histo/manifest.json, per section that has HQ tiles, `tiles` = the HQ entry
    plus a `path` template on that section's shard host. The viewer prefers a
    section's own `path` over the manifest-wide one; stains missing from an HQ
    section are simply not tiled there (the overview still shows them).
Checks that every tile the manifest names for an HQ section exists in its shard.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="?", type=Path, default=Path("viewer/pages/data"))
ap.add_argument(
    "--max-files",
    type=int,
    default=19000,
    help="per shard (Cloudflare free plan: 20,000)",
)
ap.add_argument("--host", default="https://bimp-st-tiles-hq{n}.kocherlab.workers.dev/")
a = ap.parse_args()

src = a.data / "histo_tiles_hq"
ti = json.loads((src / "tiles.json").read_text())
tpl = ti["tiles"]["path"].split("{caste}", 1)[1]  # /{ii}/{key}/{l}/{tx}_{ty}.webp
man_p = a.data / "histo" / "manifest.json"
man = json.loads(man_p.read_text())


def tiles_of(caste, i, e):
    ii = f"{int(i):02d}"
    for lv in e["levels"]:
        for j, ch in enumerate(lv["have"]):
            if ch == "1":
                for k in lv.get("stains", e["stains"]):
                    yield (
                        f"{caste}/{ii}/{k}/{lv['l']}/{j % lv['nx']}_"
                        f"{j // lv['nx']}.webp"
                    )


# whole sections per shard, in queen-then-worker order, each shard under the file cap
shards, cur, n = [], [], 0
for caste in ("queen", "worker"):
    for i, e in sorted(
        ti["sections"].get(caste, {}).items(), key=lambda kv: int(kv[0])
    ):
        files = list(tiles_of(caste, i, e))
        if len(files) > a.max_files:
            sys.exit(
                f"{caste} section {i} alone has {len(files)} tiles (> {a.max_files})"
            )
        if n + len(files) > a.max_files:
            shards.append(cur)
            cur, n = [], 0
        cur.append((caste, i, e, files))
        n += len(files)
if cur:
    shards.append(cur)

out = a.data / "histo_tiles_hq_shards"
if out.exists():
    shutil.rmtree(out)
by = {c: {s["i"]: s for s in v["sections"]} for c, v in man["castes"].items()}
missing = 0
for sn, shard in enumerate(shards, 1):
    host = a.host.format(n=sn)
    for caste, i, e, files in shard:
        for f in files:
            s, d = src / f, out / f"hq{sn}" / f
            if not s.exists():
                missing += 1
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        by[caste][int(i)]["tiles"] = dict(e, path=host + "{caste}" + tpl)
    print(
        f"hq{sn}: {sum(len(x[3]) for x in shard):,} files, {len(shard)} "
        f"sections -> {host}"
    )

if "tiles" not in man:  # the viewer needs the global block
    man["tiles"] = {k: v for k, v in ti["tiles"].items() if k != "path"}
    man["tiles"]["path"] = a.host.format(n=1) + "{caste}" + tpl
tmp = man_p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(man, indent=1, ensure_ascii=False))
tmp.replace(man_p)
print(f"{len(shards)} shard(s); missing tiles {missing}")
sys.exit(1 if missing else 0)
