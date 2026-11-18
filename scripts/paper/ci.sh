#!/usr/bin/env bash
# The checks CI runs, in one place so a local run is the same run. Needs only
# config/envs/ci.txt; no results/, data/ or atlas objects. With a manuscript
# checkout (the submodule, or BIMP_MANUSCRIPT) the registry and provenance checks
# also compare against the manuscript; without one they skip those parts.
#
#   uv venv -p 3.12 /tmp/ci && uv pip install --python /tmp/ci/bin/python -r config/envs/ci.txt
#   PY=/tmp/ci/bin/python bash scripts/paper/ci.sh [<base commit>]
#
# With a base commit, scripts/ files changed since it are linted too (the repo
# rule: scripts/ must pass `ruff check` on files you change).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-python}
BASE=${1:-}
export PYTHONPATH=src:scripts MPLBACKEND=Agg

run() {
    echo "== $*"
    "$@"
}

run "$PY" -m ruff check src scripts/paper tests
run "$PY" -m ruff format --check src
run "$PY" scripts/paper/leak_lint.py
if [ -n "$BASE" ] && git cat-file -e "${BASE}^{commit}" 2>/dev/null; then
    mapfile -t changed < <(
        git diff --name-only --diff-filter=ACMR "$BASE" HEAD -- scripts | grep '\.py$' || true
    )
    if ((${#changed[@]})); then
        run "$PY" -m ruff check "${changed[@]}"
    fi
fi
run "$PY" -m pytest -q tests
run "$PY" scripts/paper/verify_manuscript_figures.py --registry-only
run "$PY" scripts/paper/check_data_provenance.py --baseline scripts/paper/provenance_baseline.txt
