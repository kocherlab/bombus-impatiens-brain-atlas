#!/usr/bin/env python
"""Build the manuscript PDF in a scratch directory, fetching any missing LaTeX
package from CTAN.

The cluster's TeX install is partial (no multirow, etc.), and Overleaf does the real
build, so this is a layout check only: it copies the manuscript submodule to a
scratch dir, runs pdflatex (no biber, so citations stay unresolved), and on
``File `x.sty' not found`` downloads the package's TDS zip from CTAN into a local
TEXMFHOME tree and retries. Prints the log's overfull/underfull warnings and the
page count.

    .venv/bin/python scripts/paper/build_manuscript_scratch.py <scratch dir> [passes]

Needs a copy of the manuscript already at ``<scratch dir>/ms_build`` (cp -r the
submodule).
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

scratch = Path(sys.argv[1])
passes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ms = scratch / "ms_build"
home = scratch / "texmf"
home.mkdir(exist_ok=True)
env = dict(
    os.environ, TEXMFHOME=str(home), TEXINPUTS=f"{ms}//:{home}//:", BSTINPUTS=f"{ms}//:"
)
ctx = None
if Path("/etc/pki/tls/certs/ca-bundle.crt").exists():
    import ssl

    ctx = ssl.create_default_context(cafile="/etc/pki/tls/certs/ca-bundle.crt")


def fetch(pkg: str) -> bool:
    for url in (
        f"https://mirrors.ctan.org/install/macros/latex/contrib/{pkg}.tds.zip",
        f"https://mirrors.ctan.org/install/macros/latex/required/{pkg}.tds.zip",
        f"https://mirrors.ctan.org/install/macros/latex/base/{pkg}.tds.zip",
    ):
        try:
            data = urllib.request.urlopen(url, timeout=60, context=ctx).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(home)
            print("fetched", pkg, flush=True)
            return True
        except Exception:
            continue
    print("could not fetch", pkg, flush=True)
    return False


def run() -> str:
    r = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory",
            str(ms),
            str(ms / "00_Article_Merge.tex"),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=900,
    )
    return r.stdout


done = 0
for _attempt in range(40):
    out = run()
    m = re.search(r"File `([^']+)\.(sty|cls)' not found", out)
    if m:
        if not fetch(m.group(1)):
            print(out[-800:])
            sys.exit(1)
        continue
    done += 1
    if done >= passes:
        break
log = (ms / "00_Article_Merge.log").read_text(errors="replace")
print("pages:", re.findall(r"Output written on .*?\((\d+) pages", log))
print(
    "overfull:",
    len(re.findall(r"Overfull", log)),
    "underfull:",
    len(re.findall(r"Underfull", log)),
)
print(
    "undefined refs:", len(re.findall(r"Reference `[^']+' on page \d+ undefined", log))
)
print("fatal:", re.findall(r"^! .*$", log, flags=re.M)[:5])
