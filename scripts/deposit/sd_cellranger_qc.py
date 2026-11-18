#!/usr/bin/env python
"""Supplementary Data: Cell Ranger metrics and CellBender settings per snRNA library.

Rebuilds I. Traniello's ``cellranger_cellbender_output.csv`` (eight samples: queen
and worker central brain in trials 1-3, queen and worker optic lobe) from the raw
outputs on disk, read-only from the analysis directory:

- Cell Ranger ``multi`` (v10.0.0) per-sample
  ``outs/per_sample_outs/<sample>/metrics_summary.csv``: the 13 ``Cells``-category
  metrics, values as Cell Ranger wrote them;
- one CellBender ``remove-background`` (v0.3.0) run per sample, summarised from
  the command line in that run's ``.log``: ``--learning-rate`` below the default
  1e-4 -> "half learning rate", ``--expected-cells`` /
  ``--total-droplets-included`` -> "force cells", neither -> "default". The run
  is the one the clustering notebooks read: central brain, the ``_halfLR_force``
  runs (``00_QC_BIMP-Atlas_Integration_Clustering_splitTissues.ipynb``, which
  writes ``BIMP_atlas_adata_qw_CentralBrain.h5ad``); optic lobe, the default runs
  (``..._OpticLobe.ipynb``, which writes ``BIMP_adata_qw_OpticLobe.h5ad``).
  Caveat: ``..._jointTissue.ipynb`` reads the ``_force`` run ("force cells") for
  the optic-lobe queen instead; which run the repo's
  ``data/sn/BIMP_atlas_qw_OpticLobe.h5ad`` descends from is not established here.

Writes ``results/caste_programs/si_tables/cellranger_cellbender_output.csv``.

    PYTHONPATH=src:scripts .venv/bin/python scripts/deposit/sd_cellranger_qc.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from bimp_st.paths import CASTE_PROGRAMS_DIR  # noqa: E402
from bimp_st.site import get as site_path

ROOT = Path(__file__).resolve().parents[2]
OUT = CASTE_PROGRAMS_DIR / "si_tables"
OUT = OUT / "cellranger_cellbender_output.csv"

CB_DEFAULT_LR = 1e-4  # CellBender 0.3.0 --learning-rate default
# (tissue, run, Cell Ranger multi dir, CellBender dir, CellBender run suffix)
RUNS = [
    ("Central brain", f"Trial{t}", f"Trial{t}_centralBrain_out/"
     f"20260615_BIMP_QW_atlas_Trial{t}", f"Trial{t}", "_halfLR_force")
    for t in (1, 2, 3)
] + [
    ("Optic lobe", "OpticLobe", "opticLobe_out/20260615_BIMP_QW_atlas_OpticLobe",
     "opticLobe", ""),
]  # fmt: skip
METRICS = [
    "Cells",
    "Confidently mapped antisense",
    "Confidently mapped reads in cells",
    "Confidently mapped to exonic regions",
    "Confidently mapped to genome",
    "Confidently mapped to intergenic regions",
    "Confidently mapped to intronic regions",
    "Confidently mapped to transcriptome",
    "Mapped to genome",
    "Median UMI counts per cell",
    "Median genes per cell",
    "Number of reads in cells",
    "Total genes detected",
]


def cellranger_metrics(path: Path) -> dict[str, str]:
    """The ``Cells``-category rows of one per-sample metrics_summary.csv."""
    m = pd.read_csv(path, dtype=str)
    m = m[m.Category == "Cells"].set_index("Metric Name")["Metric Value"]
    return {k: m[k] for k in METRICS}


def cellbender_note(log: Path) -> str:
    """Summarise the non-default options of one CellBender run from its log."""
    cmd = next(line for line in log.read_text().splitlines() if " --input " in line)
    notes = []
    lr = re.search(r"--learning-rate\s+(\S+)", cmd)
    if lr and float(lr.group(1)) < CB_DEFAULT_LR:
        frac = float(lr.group(1)) / CB_DEFAULT_LR
        notes.append(
            {0.5: "half", 0.25: "quarter"}.get(frac, f"{frac:g}x") + " learning rate"
        )
    if "--expected-cells" in cmd or "--total-droplets-included" in cmd:
        notes.append("force cells")
    return " + ".join(notes) or "default"


def build() -> pd.DataFrame:
    """One row per sample, in the saved table's column names and order."""
    rows = []
    out = site_path("snrna_source") / "out"
    for tissue, run, cr_dir, cb_dir, suffix in RUNS:
        for caste in ("Queen", "Worker"):
            sample = f"{run}_BIMP_{caste}"
            ms = out / "aligned" / cr_dir / "outs/per_sample_outs" / sample
            log = (
                out / "cellbender" / cb_dir / caste.lower()
                / f"BIMP_{caste}_{run}_CellbenderOut{suffix}.log"
            )  # fmt: skip
            row = {"Tissue": tissue, "Run": run, "Sample ID": sample, "Caste": caste}
            row |= cellranger_metrics(ms / "metrics_summary.csv")
            row["Cellbender_Notes"] = cellbender_note(log)
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in METRICS:
        df[c] = pd.to_numeric(df[c])
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    df = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
