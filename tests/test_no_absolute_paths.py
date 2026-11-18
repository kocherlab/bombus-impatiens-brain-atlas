"""No machine-specific absolute paths in tracked code, pipelines or configs.

Off-repo inputs resolve through ``bimp_st.site`` keys and Slurm partitions live
in ``config/profiles/slurm/sbatch.env``. ``archive/``, ``docs/`` and ``vendor/``
are records, not code that runs, and are not checked; nor is a file carrying the
SKIP_MARKER line (a scanner whose own patterns name the paths it rejects).
"""
# path-scan: skip

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKED = re.compile(
    r"\.(py|sh|sbatch|smk|R|yaml|yml|toml|txt|tsv|cfg|mjs|js|html)$|(^|/)Snakefile$|\.md$|\.env$"
)
SKIPPED = re.compile(
    r"^(archive|docs|vendor)/"
    # the EGAPx run's parameter file, kept as the run wrote it (only the file
    # names in it are read)
    r"|^data/genome/egapx_run_params\.yaml$"
)
SKIP_MARKER = "\n# path-scan: skip\n"
ABSOLUTE = re.compile(r"/Genomics/|/home/\w|/Users/\w|/scratch/|/tigress/|/n/fs/")
SBATCH_SITE = re.compile(r"^#SBATCH\s+(--partition|-p\b|--qos|-q\b)", re.M)


def tracked() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.splitlines() if CHECKED.search(p) and not SKIPPED.search(p)]


def test_no_absolute_paths():
    hits = []
    for p in tracked():
        f = REPO / p
        if not f.is_file():
            continue
        text = f.read_text(errors="replace")
        if SKIP_MARKER in text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if ABSOLUTE.search(line):
                hits.append(f"{p}:{i}: {line.strip()[:120]}")
    assert not hits, "absolute paths (use bimp_st.site):\n" + "\n".join(hits)


def test_sbatch_headers_carry_no_site_options():
    hits = [
        p
        for p in tracked()
        if p.endswith((".sh", ".sbatch")) and SBATCH_SITE.search((REPO / p).read_text())
    ]
    assert not hits, "partition/QOS belong in config/profiles/slurm/sbatch.env: " + (
        ", ".join(hits)
    )
