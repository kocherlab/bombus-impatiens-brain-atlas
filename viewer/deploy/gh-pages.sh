#!/usr/bin/env bash
# Publish the site to the gh-pages branch, which is what serves https://kocherlab.github.io/bombus-impatiens-brain-atlas/
#
# Usage:  viewer/deploy/gh-pages.sh [--push]
#
# Builds pages/index.html, mirrors pages/ into a temporary worktree of origin/gh-pages, leaves out
# what Cloudflare serves (data/genes, data/histo_tiles: deploy/cloudflare.sh) and anything that is
# not site data (logs, per-section scratch, the histology export's registration record), points the manifests at the Cloudflare hosts, and
# commits. Without --push it stops there and prints the worktree, so the commit can be looked at
# first; with --push it pushes to origin gh-pages and removes the worktree.
#
# The source commit is named in the gh-pages commit message, so every deploy can be traced back.
set -euo pipefail
APP="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(git -C "$APP" rev-parse --show-toplevel)"
GENES_URL="${GENES_URL:-https://bimp-st-genes.kocherlab.workers.dev/}"
TILES_URL="${TILES_URL:-https://bimp-st-tiles.kocherlab.workers.dev/}"
PUSH=0; [[ "${1:-}" == --push ]] && PUSH=1

node "$APP/build-pages.mjs"
SRC="$(git -C "$REPO" rev-parse --short HEAD)"; DIRTY="$(git -C "$REPO" status --porcelain -- viewer | head -1)"

git -C "$REPO" fetch -q origin gh-pages
WT="$(mktemp -d "${TMPDIR:-/tmp}/bimp-ghp.XXXXXX")"
git -C "$REPO" worktree add -q --detach "$WT" origin/gh-pages

# mirror pages/ by clearing the worktree (all but its .git link) and copying, rather than rsync
# --delete with a protect filter, which macOS's built-in rsync ignores (it deleted the .git link)
find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
rsync -rlt \
  --exclude='data/genes/' --exclude='data/histo_tiles/' --exclude='data/histo_tiles_hq/' --exclude='data/histo_tiles_hq_shards/' --exclude='data/histo_persection/' \
  --exclude='data/histo/registration.json' --exclude='*.log' --exclude='probe-*.html' --exclude='index-probe.html' --exclude='vendor/' \
  "$APP/pages/" "$WT/"
touch "$WT/.nojekyll"

# the copies on gh-pages fetch genes and tiles from Cloudflare; the local files stay as they are
python3 - "$WT/data" "$GENES_URL" "$TILES_URL" <<'EOF'
import json, sys, os
d, genes, tiles = sys.argv[1:]
mp = os.path.join(d, "manifest.json"); m = json.load(open(mp))
m["genesBase"] = genes; json.dump(m, open(mp, "w"), separators=(",", ":"))
hp = os.path.join(d, "histo", "manifest.json")
if os.path.exists(hp):
    h = json.load(open(hp))
    if h.get("tiles"):
        tail = h["tiles"]["path"].split("{caste}", 1)[1]
        h["tiles"]["path"] = tiles + "{caste}" + tail
    json.dump(h, open(hp, "w"), separators=(",", ":"))
print("genesBase", genes, "| tiles", tiles)
EOF

du -sh "$WT"
git -C "$WT" add -A
git -C "$WT" commit -q -m "deploy webviz from $SRC${DIRTY:+ (with uncommitted changes)}" || { echo "nothing changed"; }
if (( PUSH )); then
  git -C "$WT" push origin HEAD:gh-pages
  git -C "$REPO" worktree remove --force "$WT"
  echo "pushed; GitHub Pages updates within a minute or two"
else
  echo "committed in $WT (not pushed). Push with: git -C $WT push origin HEAD:gh-pages"
fi
