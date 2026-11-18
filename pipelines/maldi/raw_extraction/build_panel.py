#!/usr/bin/env python
"""Build the channel spec for PASTE alignment from Bombus_MALDI_targets.tsv.

- ALIGNMENT channels: every neuropeptide EXCEPT sNPF (BIMP_v4.3_005730) and
  Tk (BIMP_v4.3_009257), as (gene_isoform x adduct). Each channel carries the
  list of mature-fragment m/z for that adduct; per-pixel intensity = max over
  those fragment windows (aggregation='max').
- READOUT channels (collaborator, NOT used to drive alignment): sNPF_v1 @
  974.5891, Tk2 @ 993.4757.

Writes channel_spec.json: {channels: [{name, mz:[...], role}], ppm}
"""

import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
TARGETS = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(_REPO_ROOT / "data/panel/Bombus_MALDI_targets.tsv")
)
OUT = sys.argv[2] if len(sys.argv) > 2 else "channel_spec.json"
PPM = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

EXCLUDE_GENES = {"BIMP_v4.3_005730", "BIMP_v4.3_009257"}  # sNPF, Tk -> collaborator
ADDUCTS = [("H", "[M+H]+"), ("Na", "[M+Na]+"), ("K", "[M+K]+")]

t = pd.read_csv(TARGETS, sep="\t")
t["gene_base"] = t["gene_id"].str.extract(r"(BIMP_v4\.3_\d+)")

channels = {}
for _, row in t.iterrows():
    if row["gene_base"] in EXCLUDE_GENES:
        continue
    iso = row["gene_id"]  # e.g. BIMP_v4.3_000407-R1
    for _, col in ADDUCTS:
        name = f"{iso}_{col}"
        mz = float(row[col])
        channels.setdefault(name, set()).add(round(mz, 5))

spec = {"ppm": PPM, "channels": []}
for name in sorted(channels):
    spec["channels"].append(
        {"name": name, "mz": sorted(channels[name]), "role": "align"}
    )
# collaborator readout channels (their focal SCiLS m/z, not the sheet)
spec["channels"].append({"name": "sNPF_v1", "mz": [974.5891], "role": "readout"})
spec["channels"].append({"name": "Tk2", "mz": [993.4757], "role": "readout"})

import os

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(spec, open(OUT, "w"), indent=2)
n_align = sum(1 for c in spec["channels"] if c["role"] == "align")
n_mz = sum(len(c["mz"]) for c in spec["channels"])
print(f"wrote {OUT}")
print(f"  align channels: {n_align}  (the 'other' neuropeptides, gene-iso x adduct)")
print("  readout channels: 2  (sNPF_v1@974.5891, Tk2@993.4757)")
print(f"  total target m/z to extract: {n_mz}  at ±{PPM} ppm")
print("  example align channels:", [c["name"] for c in spec["channels"][:4]])
