#!/usr/bin/env bash
# Upload the atlas's two large data sets to Cloudflare (Workers static assets, free plan):
#
#   tiles  pages/data/histo_tiles  →  https://bimp-st-tiles.kocherlab.workers.dev   (~17k files, 353 MB)
#   genes  pages/data/genes        →  https://bimp-st-genes.kocherlab.workers.dev   (9,710 files, ~3.3 GB)
#
# Together they are far over GitHub Pages' 1 GB limit, so the site (deploy/gh-pages.sh) carries
# everything else and the manifests point at these hosts. A free Worker holds at most 20,000
# files of at most 25 MiB each, which is why the two sets are two Workers.
#
# Usage:  viewer/deploy/cloudflare.sh [tiles|genes|all|tiles-hq]      (default: all)
#   tiles-hq  pages/data/histo_tiles_hq_shards/hq<n>  →  https://bimp-st-tiles-hq<n>.kocherlab.workers.dev
#             (high-resolution DAPI tiles, <19,000 files per Worker; built by tools/shard_hq_tiles.py)
#
# Needs `npx wrangler login` once (a free Cloudflare account; the workers.dev subdomain of the
# account is "kocherlab"). If npm's cache in ~/.npm is owned by root, set NPM_CONFIG_CACHE to a
# directory you own.
#
# Each set is staged in a fresh temporary folder (an APFS clone on macOS, so no extra disk) with a
# _headers file that lets the github.io page read it (CORS) and caches it for a week, and
# wrangler is run from there. Never run wrangler from viewer itself: with no config file it
# "auto-configures" the folder (writes wrangler.jsonc, edits package.json) and deploys pages/.
set -euo pipefail
APP="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$APP/pages/data"
WHAT="${1:-all}"
DATE="2026-10-19"   # Workers compatibility date; static assets do not depend on it

stage() {   # stage <src dir> <subdir in the stage, or "">  → prints the stage path
  local src="$1" sub="$2" st
  st="$(mktemp -d "${TMPDIR:-/tmp}/bimp-cf.XXXXXX")"
  printf '/*\n  Access-Control-Allow-Origin: *\n  Cache-Control: public, max-age=604800\n' > "$st/_headers"
  local dst="$st${sub:+/$sub}"; mkdir -p "$(dirname "$dst")"
  cp -c -R "$src" "$dst" 2>/dev/null || cp -R "$src" "$dst"   # clonefile where the filesystem has it
  find "$st" -name '*.log' -delete
  echo "$st"
}

deploy() {   # deploy <worker name> <stage>
  ( cd "$2" && CI=1 WRANGLER_SEND_METRICS=false npx --yes wrangler deploy \
      --name "$1" --assets . --compatibility-date "$DATE" )
}

if [[ "$WHAT" == tiles || "$WHAT" == all ]]; then
  test -f "$DATA/histo_tiles/tiles.json" || { echo "no $DATA/histo_tiles/tiles.json (see tools/histology_slurm)"; exit 1; }
  # tiles.json paths are {caste}/{ii}/... relative to the host root, so the folder's contents go at the root
  st="$(stage "$DATA/histo_tiles" "")"; mv "$st/histo_tiles"/* "$st"/ && rmdir "$st/histo_tiles"
  deploy bimp-st-tiles "$st"; rm -rf "$st"
fi

if [[ "$WHAT" == tiles-hq ]]; then   # not part of "all": run tools/shard_hq_tiles.py first
  test -d "$DATA/histo_tiles_hq_shards" || { echo "no $DATA/histo_tiles_hq_shards (run tools/shard_hq_tiles.py)"; exit 1; }
  for sd in "$DATA"/histo_tiles_hq_shards/hq*; do   # one Worker per shard, bimp-st-tiles-hq<n>
    st="$(stage "$sd" "")"; mv "$st/$(basename "$sd")"/* "$st"/ && rmdir "$st/$(basename "$sd")"
    deploy "bimp-st-tiles-$(basename "$sd")" "$st"; rm -rf "$st"
  done
fi

if [[ "$WHAT" == genes || "$WHAT" == all ]]; then
  # every gene file the manifest lists must be here and non-empty before anything is uploaded
  python3 - "$DATA" <<'EOF'
import json, os, sys
d = sys.argv[1]; m = json.load(open(os.path.join(d, "manifest.json")))
p = lambda g: os.path.join(d, g["file"])
miss = [g["file"] for g in m["genes"] if not os.path.exists(p(g))]
bad = [g["file"] for g in m["genes"] if os.path.exists(p(g)) and not os.path.getsize(p(g))]
if miss or bad: sys.exit(f"{len(miss)} gene files missing, {len(bad)} empty (first: {(miss or bad)[:3]})")
print(f"{len(m['genes'])} gene files present")
EOF
  # manifest file paths are genes/<file>, so the stage keeps the genes/ folder
  st="$(stage "$DATA/genes" "genes")"
  deploy bimp-st-genes "$st"; rm -rf "$st"
fi
