#!/usr/bin/env python
"""Freeze the CellBender ambient profiles that ``snrna_controls.py`` reads.

CellBender was run (by I. Traniello) on each central-brain library behind
``BIMP_atlas_qw_CentralBrain.h5ad``; the atlas reads each run's
``*_halfLR_force_filtered.h5`` and the unfiltered sibling carries the learned
ambient profile (``global_latents/ambient_expression``). This copies the six
profiles, each normalised to sum to 1 over its features, into
``cellbender_ambient.csv.gz`` in ``bimp_st.paths.CASTE_DE_AMBIENT_DIR`` (index
gene id, columns ``Queen_t1`` ... ``Worker_t3``) and writes ``PROVENANCE.md``
with the md5 of every source file, so no script in this repository reads the
snRNA-seq analysis tree.

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/caste_de/freeze_cellbender_ambient.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from bimp_st.paths import CASTE_DE_AMBIENT_DIR  # noqa: E402
from bimp_st.site import get as site_path

REPO = Path(__file__).resolve().parents[2]
CB_ROOT = site_path("snrna_source") / "out/cellbender"
OUT = CASTE_DE_AMBIENT_DIR


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    import h5py

    OUT.mkdir(parents=True, exist_ok=True)
    out, rows = {}, []
    for t in (1, 2, 3):
        for caste in ("queen", "worker"):
            f = (
                CB_ROOT
                / f"Trial{t}"
                / caste
                / f"BIMP_{caste.capitalize()}_Trial{t}_CellbenderOut_halfLR_force.h5"
            )
            with h5py.File(f, "r") as h:
                amb = h["global_latents/ambient_expression"][()]
                ids = [x.decode() for x in h["matrix/features/id"][()]]
                ftype = {x.decode() for x in h["matrix/features/feature_type"][()]}
            assert ftype == {"Gene Expression"}, ftype
            v = pd.Series(amb, index=ids)
            v = v[~v.index.duplicated()]
            out[f"{caste.capitalize()}_t{t}"] = v / v.sum()
            rows.append((f, md5(f)))
            print(f.name, len(ids), "features", flush=True)
    dst = OUT / "cellbender_ambient.csv.gz"
    pd.DataFrame(out).to_csv(dst)
    lines = [
        "# CellBender ambient profiles (I. Traniello's runs), frozen",
        "",
        "Written by `scripts/caste_de/freeze_cellbender_ambient.py`; read by "
        "`scripts/caste_de/snrna_controls.py --stage contamination`.",
        "",
        f"`cellbender_ambient.csv.gz` md5 {md5(dst)}",
        "",
        "| source | md5 |",
        "|---|---|",
        *[f"| `{f}` | {m} |" for f, m in rows],
        "",
    ]
    (OUT / "PROVENANCE.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
