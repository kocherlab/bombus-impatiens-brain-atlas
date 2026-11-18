#!/usr/bin/env python
"""Supplementary Data: the read sets given to EGAPx for the BIMP v4.3 annotation.

Source: the EGAPx run's own parameter file, ``data/genome/egapx_run_params.yaml``
(the EGAPx run's ``nextflow/run_params.yaml``, 2025-07-15; EGAPx 0.4.0-alpha;
storage prefixes replaced by ``<LAB_DATA>`` and ``<SCRATCH>``). Its
``short_reads`` list (38 entries)
and ``long_reads`` list (two Iso-Seq FLNC FASTAs) are the inputs.

Two tables:

- ``short_reads()``: one row per short-read entry, in the yaml's order, with the
  author-supplied sheet's columns (``#, Source, Sample, R1 file, R2 file``);
  Source is "NCBI SRA" for SRR runs, "In-house" otherwise. Written with a BOM
  and CRLF like that sheet, which it reproduces byte for byte (checked against
  ``data/genome/egapx_rnaseq_samples.csv``).
- ``long_reads()``: one row per Iso-Seq set. The EGAPx input
  ``flnc-<i>.fa`` (the FLNC FASTAs named in run_params.yaml)
  was converted (bamtools 2.5.2) from
  the SMRT Link output ``flnc-<i>.bam``; barcode, PacBio sample
  name and read counts come from that run's ``isoseq_primers.report.json`` and
  run identifiers from the BAM ``@RG`` header, read-only from
  ``<isoseq_raw>``.

Writes ``results/genome/egapx_inputs/{egapx_rnaseq_samples,egapx_isoseq_flnc}.csv``.

    PYTHONPATH=src:scripts .venv/bin/python scripts/deposit/sd_egapx_inputs.py
"""

from __future__ import annotations

import gzip
import json
import re
import struct
from pathlib import Path

import pandas as pd
import yaml

from bimp_st.site import get as site_path

ROOT = Path(__file__).resolve().parents[2]
RUN_PARAMS = ROOT / "data" / "genome" / "egapx_run_params.yaml"
AUTHOR = ROOT / "data" / "genome" / "egapx_rnaseq_samples.csv"
ISOSEQ = site_path("isoseq_raw") / "Isoseq_analysis_13887"
OUT = ROOT / "results" / "genome" / "egapx_inputs"


def params() -> dict:
    return yaml.safe_load(RUN_PARAMS.read_text())["input"]


def short_reads() -> pd.DataFrame:
    rows = []
    for i, (name, files) in enumerate(params()["short_reads"], 1):
        r1, r2 = (Path(f).name for f in files)
        src = "NCBI SRA" if re.fullmatch(r"SRR\d+", name) else "In-house"
        rows.append((i, src, name, r1, r2))
    return pd.DataFrame(rows, columns=["#", "Source", "Sample", "R1 file", "R2 file"])


def bam_read_group(path: Path) -> dict[str, str]:
    """@RG fields of a BAM header (BGZF read with gzip; no samtools needed)."""
    with gzip.open(path) as f:
        if f.read(4) != b"BAM\1":
            raise ValueError(f"{path}: not a BAM")
        (n,) = struct.unpack("<i", f.read(4))
        header = f.read(n).decode()
    rg = next(ln for ln in header.splitlines() if ln.startswith("@RG"))
    tags = dict(t.split(":", 1) for t in rg.split("\t")[1:])
    ds = dict(kv.split("=", 1) for kv in tags.get("DS", "").split(";") if "=" in kv)
    return {**tags, **ds}


def long_reads() -> pd.DataFrame:
    rep = json.loads((ISOSEQ / "isoseq_primers.report.json").read_text())
    tab = {
        c["header"]: c["values"]
        for t in rep["tables"]
        if t["id"].endswith("primer_table")
        for c in t["columns"]
    }
    by_sample = {
        s: i for i, s in enumerate(tab["Bio Sample Name"]) if s.startswith("BioSample")
    }
    rows = []
    for name, (fa,) in params()["long_reads"]:
        k = int(re.fullmatch(r"flnc-(\d+)", name).group(1))
        rg = bam_read_group(ISOSEQ / f"flnc-{k}.bam")
        j = by_sample[rg["SM"]]
        rows.append(
            {
                "egapx_set": name,
                "egapx_input_fasta": Path(fa).name,
                "smrt_link_flnc_bam": f"flnc-{k}.bam",
                "pacbio_bio_sample": rg["SM"],
                "barcode": tab["Primer Name"][j].split("--")[0],
                "library": rg["LB"],
                "movie": rg["PU"],
                "run_id": rg.get("RUNID"),
                "smrt_cell_id": rg.get("SMRTCELLID"),
                "segmented_reads_with_primers": tab["Reads with 5' and 3' Primers"][j],
                "flnc_reads": tab["Non-Concatamer Reads with 5' and 3' Primers"][j],
                "flnc_polya_reads": tab[
                    "Non-Concatamer Reads with 5' and 3' Primers and Poly-A Tail"
                ][j],
            }
        )
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    """The short-read table (cnmf_gsea/compare_si_tables.py interface)."""
    return short_reads()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sr = short_reads()
    p = OUT / "egapx_rnaseq_samples.csv"
    p.write_bytes(("﻿" + sr.to_csv(index=False, lineterminator="\r\n")).encode("utf-8"))
    same = p.read_bytes() == AUTHOR.read_bytes()
    print(f"{p.name}: {len(sr)} rows; byte-identical to author sheet: {same}")
    if not same:
        raise SystemExit("short-read table differs from the author-supplied sheet")
    lr = long_reads()
    lr.to_csv(OUT / "egapx_isoseq_flnc.csv", index=False)
    print(lr.to_string())


if __name__ == "__main__":
    main()
