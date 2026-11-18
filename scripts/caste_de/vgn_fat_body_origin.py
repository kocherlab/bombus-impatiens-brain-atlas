"""Where the queen Vgn reads in the central-brain snRNA-seq come from.

The pseudobulk caste DE calls vitellogenin (Vgn, BIMP_v4.3_005500) strongly
queen-biased. This tabulates how those UMIs are distributed over nuclei:
how many nuclei carry most of them, whether those nuclei also carry fat-body
transcripts (a Kunitz-type protease inhibitor, BIMP_v4.3_000091, and
apolipophorin, BIMP_v4.3_009153), which cell type they were annotated as, and
how much Vgn each queen library holds per million UMIs.

Definitions
  central brain   ``region == "CentralBrain"`` (trials t1-t3)
  high nuclei     Vgn UMI >= 5
  fat-body nuclei high nuclei with Kunitz-inhibitor UMI >= 10
  per million     Vgn UMI / all UMI of the library x 1e6

Input: ``paths.SN_COMBINED`` (layers['counts']).

Outputs (results/handoff/vgn_fat_body/)
  summary.csv      one row per summary quantity (name, value)
  by_sample.csv    per library: nuclei, Vgn+ nuclei, Vgn UMI, per million
  high_nuclei.csv  the high nuclei with their marker UMIs and cell type

Run
    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/caste_de/vgn_fat_body_origin.py
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from bimp_st.paths import RESULTS, SN_COMBINED

OUT = RESULTS / "handoff" / "vgn_fat_body"
GENES = {
    "vgn": "BIMP_v4.3_005500",
    "kunitz": "BIMP_v4.3_000091",
    "apolpp": "BIMP_v4.3_009153",
}
HIGH, FAT = 5, 10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(SN_COMBINED)
    cb = (a.obs["region"] == "CentralBrain").to_numpy()
    X = sp.csr_matrix(a.layers["counts"])[cb]
    o = a.obs.loc[
        cb, ["sample", "caste", "trial", "major_cell_type", "cell_type"]
    ].copy()
    o["total"] = np.asarray(X.sum(1)).ravel()
    for k, g in GENES.items():
        o[k] = X[:, a.var_names.get_loc(g)].toarray().ravel()

    s = (
        o.groupby(["sample", "caste", "trial"], observed=True)
        .agg(
            nuclei=("vgn", "size"),
            vgn_pos=("vgn", lambda x: int((x > 0).sum())),
            vgn_umi=("vgn", "sum"),
            total_umi=("total", "sum"),
        )
        .reset_index()
    )
    s["vgn_per_million"] = s.vgn_umi / s.total_umi * 1e6
    s.to_csv(OUT / "by_sample.csv", index=False)

    q = o[o.caste == "Queen"]
    tot = q.vgn.sum()
    hi = q[q.vgn >= HIGH]
    fat = hi[hi.kunitz >= FAT]
    neg = q[q.vgn == 0]
    hi.sort_values("vgn", ascending=False).to_csv(OUT / "high_nuclei.csv")
    per = s[s.caste == "Queen"].set_index("trial").vgn_per_million
    rows = [
        ("queen_cb_nuclei", len(q)),
        ("queen_vgn_umi", tot),
        ("high_nuclei", len(hi)),
        ("high_share_of_queen_vgn", hi.vgn.sum() / tot),
        ("fat_body_nuclei", len(fat)),
        ("fat_body_share_of_queen_vgn", fat.vgn.sum() / tot),
        (
            "fat_body_lateral_pn",
            int((fat.cell_type == "Lateral projection neurons").sum()),
        ),
        ("fat_body_with_apolpp", int((fat.apolpp > 0).sum())),
        ("kunitz_pct_of_vgn_negative_queen", 100 * (neg.kunitz > 0).mean()),
        (
            "queen_cb_hemocyte_vgn_umi",
            int(q[q.major_cell_type == "Hemocytes"].vgn.sum()),
        ),
        (
            "worker_cb_hemocyte_vgn_umi",
            int(o[(o.caste != "Queen") & (o.major_cell_type == "Hemocytes")].vgn.sum()),
        ),
    ] + [(f"queen_{t}_vgn_per_million", float(v)) for t, v in per.items()]
    summ = pd.DataFrame(rows, columns=["name", "value"])
    summ.to_csv(OUT / "summary.csv", index=False)
    print(s.to_string(index=False))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
