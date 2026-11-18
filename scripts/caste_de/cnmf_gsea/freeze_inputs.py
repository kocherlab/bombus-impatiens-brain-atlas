#!/usr/bin/env python
"""Copy the inputs the in-repo pipeline needs from the snRNA-seq analysis tree.

Analysis design and source outputs: I. Traniello. The ONLY script in
``scripts/caste_de/cnmf_gsea/`` that reads the snRNA-seq analysis tree
(site key ``snrna_source``, :mod:`bimp_st.site`); every other step reads
the copies written here, under ``results/caste_programs/inputs/``:

  BIMP_v4.3.emapper.annotations   eggNOG-mapper GO annotation (reference dir)
  go-basic.obo                    the GO OBO the analysis used
  BIMP_lookup.csv                 gene annotation table (cell 4)
  ``cnmf_gsea_paths.SAVED``       saved outputs, for parity and --source saved:
    cnmf_joint/joint/             the saved joint cNMF run (consensus k = 40/50 files,
                                  k selection, overdispersed genes, replicate
                                  parameters, merged per-k spectra)
    de_all.csv.gz                 the saved DESeq2 table (cell 6 cache)
    joint_k50_caste.csv, joint_k40_caste.csv, joint_k50_go_labels.csv,
    gsea_all.csv.gz, prog_go_six.csv.gz, cnmf_chosen_k.json (record only)
    atlas_fingerprint.json        content hash of the source central-brain h5ad
                                  (``check_inputs.fingerprint``)
  PROVENANCE.md                   source path and md5 of every copy, releases

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/freeze_inputs.py
"""

import hashlib
import json
import re
import shutil

import anndata as ad
from check_inputs import fingerprint
from cnmf_gsea_paths import EMAPPER, GO_OBO, INPUTS, LOOKUP, SAVED, SAVED_DE

from bimp_st.site import get as site_path

OUT = INPUTS
BASE = site_path("snrna_source")
CA = BASE / "out/clustering_and_annotation"
JOINT = CA / "cnmf_joint/joint"
ATLAS = CA / "BIMP_atlas_adata_qw_CentralBrain.h5ad"
OBO = site_path("snrna_shared_scripts") / "go-basic.obo"
SRC_EMAPPER = (
    BASE / "src/reference_genome/BIMP_4.3_eggnog/Annotations/EggNOG/"
    "BIMP_v4.3.emapper.annotations"
)
SRC_LOOKUP = BASE / "src/reference_genome/tables/BIMP_lookup.csv"
TABLES = [
    CA / "joint_k50_caste.csv",
    CA / "joint_k40_caste.csv",
    CA / "joint_k50_go_labels.csv",
    CA / "gsea_all.csv.gz",
    BASE / "out/fig4_supp/prog_go_six.csv.gz",
    # chosen k of the per-cell-type cNMF; no step reads it (the joint run
    # fixes k = 50), copied so the record of the inputs is complete
    CA / "cnmf_chosen_k.json",
]
TMP_KEEP = re.compile(
    r"joint\.(nmf_params\.df\.npz|nmf_idvrun_params\.yaml|spectra\.k_\d+\.merged\.df\.npz)$"
)


def md5(p) -> str:
    """md5 of a file, read in 4 MB blocks."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def copy(src, dst, rows) -> None:
    """Copy src to dst (with mtime) and record (copy, source, md5 of source)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rows.append((dst.relative_to(OUT), src, md5(src)))


def main() -> None:
    rows = []
    copy(SRC_EMAPPER, EMAPPER, rows)
    copy(OBO, GO_OBO, rows)
    copy(SRC_LOOKUP, LOOKUP, rows)
    copy(CA / "de_all.csv.gz", SAVED_DE, rows)
    for t in TABLES:
        copy(t, SAVED / t.name, rows)
    for f in sorted(JOINT.iterdir()):
        if f.is_file():
            copy(f, SAVED / "cnmf_joint/joint" / f.name, rows)
    for f in sorted((JOINT / "cnmf_tmp").iterdir()):
        if TMP_KEEP.search(f.name):
            copy(f, SAVED / "cnmf_joint/joint/cnmf_tmp" / f.name, rows)
    fp = fingerprint(ad.read_h5ad(ATLAS))
    (SAVED / "atlas_fingerprint.json").write_text(json.dumps(fp, indent=1))
    rows.append((f"{SAVED.name}/atlas_fingerprint.json", ATLAS, md5(ATLAS)))

    emv = open(EMAPPER).readlines()[1].strip("# \n")
    ver = next(
        ln.split(":", 1)[1].strip()
        for ln in open(OBO)
        if ln.startswith("data-version:")
    )
    with open(OUT / "PROVENANCE.md", "w") as f:
        f.write(
            "# Inputs copied from the saved cNMF / GSEA analysis\n\n"
            "Written by `scripts/caste_de/cnmf_gsea/freeze_inputs.py`, the "
            "only pipeline script that reads the saved analysis directory. "
            "eggNOG-mapper "
            f"{emv}; GO OBO go-basic data-version {ver} (byte-identical to the "
            f"repo's `data/go_annotation/go-basic.obo`). `{SAVED.name}/` holds the "
            "saved outputs, read only by `parity.py` and by `--source saved` runs. "
            "`check_inputs.py` compares the repo atlas "
            f"(`bimp_st.paths.SN_CB`) with `{SAVED.name}/atlas_fingerprint.json` "
            "(result in `input_check.csv`).\n\n"
            "| copy | source | md5 of source |\n|---|---|---|\n"
        )
        for dst, src, h in rows:
            f.write(f"| `{dst}` | `{src}` | {h} |\n")
    print((OUT / "PROVENANCE.md").read_text())


if __name__ == "__main__":
    main()
