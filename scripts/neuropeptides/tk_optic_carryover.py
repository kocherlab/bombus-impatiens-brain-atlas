"""Is the queen optic-lobe Tk signal Kenyon-cell carryover?

The optic-lobe snRNA-seq dissection (one library per caste) captured some
Kenyon cell nuclei, and Kenyon cells express Tk at high rates, so a queen
excess of Tk+ optic-lobe neurons could in principle be carryover (ambient
RNA or misassigned nuclei). Three checks:

1. Kenyon-cell share of each optic-lobe library and sequencing depth: carryover
   would need more Kenyon material in the queen library.
2. Kenyon-cell-enriched transcripts (defined in the central brain only) in
   Tk+ vs Tk- optic-lobe neurons: misassigned or contaminated nuclei would
   carry them along with Tk.
3. Tk+ rate within shared depth quintiles, so depth cannot explain the gap.

Writes results/handoff/tk_carryover/{summary,depth_bins,ol_subclusters}.csv.

    PYTHONPATH=src:scripts .venv/bin/python scripts/neuropeptides/tk_optic_carryover.py
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from bimp_st.paths import REPO, RESULTS

TK = "BIMP_v4.3_009257"  # tachykinin (np31_to_v43_mapping.csv)
SN = REPO / "data" / "sn" / "BIMP_atlas_qw_combined.h5ad"
OUT = RESULTS / "handoff" / "tk_carryover"
NEURONS = ["Kenyon cells", "Projection neurons", "Other neurons", "Photoreceptors"]


def main():
    a = ad.read_h5ad(SN)
    X = sp.csr_matrix(a.layers["counts"])
    o = a.obs
    tk = X[:, a.var_names.get_loc(TK)].toarray().ravel()
    tot = np.asarray(X.sum(1)).ravel()
    cb = (o.region == "CentralBrain").to_numpy()
    ol = (o.region == "OpticLobe").to_numpy()
    kc = (o.major_cell_type == "Kenyon cells").to_numpy()
    oln = (o.major_cell_type == "Optic lobe neurons").to_numpy()
    queen = (o.caste == "Queen").to_numpy()

    # Kenyon-enriched genes, from central-brain neurons only (independent of the
    # optic-lobe libraries): detected in >25% of KCs and >=5x other neurons
    i = np.where(cb & o.major_cell_type.isin(NEURONS).to_numpy())[0]
    ik = kc[i]
    dk = np.asarray((X[i[ik]] > 0).mean(0)).ravel()
    do = np.asarray((X[i[~ik]] > 0).mean(0)).ravel()
    cand = np.where((dk > 0.25) & (dk > 5 * do))[0]
    cand = cand[a.var_names[cand] != TK]
    kc_load = np.asarray((X[:, cand] > 0).sum(1)).ravel()

    rows, bins_out = [], []
    edges = np.quantile(tot[ol & oln], np.linspace(0, 1, 6))
    for caste, m in [("queen", queen), ("worker", ~queen)]:
        lib = ol & m
        n = ol & oln & m
        pos = tk[n] > 0
        rows.append(
            {
                "caste": caste,
                "ol_neurons": int(n.sum()),
                "tk_pct": 100 * pos.mean(),
                "tk_ge2_umi_pct": 100 * (tk[n] >= 2).mean(),
                "kc_nuclei": int((lib & kc).sum()),
                "kc_share_of_library_pct": 100 * (lib & kc).sum() / lib.sum(),
                "median_umi_ol_neuron": float(np.median(tot[n])),
                "kc_genes_tested": len(cand),
                "kc_gene_load_tk_pos": kc_load[n][pos].mean(),
                "kc_gene_load_tk_neg": kc_load[n][~pos].mean(),
                "kc_gene_load_kc_nuclei": kc_load[lib & kc].mean(),
            }
        )
        b = np.digitize(tot[n], edges[1:-1])
        for q in range(5):
            bins_out.append(
                {
                    "caste": caste,
                    "depth_quintile": q + 1,
                    "n": int((b == q).sum()),
                    "tk_pct": 100 * pos[b == q].mean(),
                }
            )

    sub = (
        pd.DataFrame(
            {
                "caste": o.caste[ol & oln].to_numpy(),
                "subcluster": o.cell_type[ol & oln].astype(str).to_numpy(),
                "tk_pos": tk[ol & oln] > 0,
            }
        )
        .groupby(["subcluster", "caste"])
        .tk_pos.agg(["mean", "size"])
        .reset_index()
    )
    sub["mean"] *= 100

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(bins_out).to_csv(OUT / "depth_bins.csv", index=False)
    sub.rename(columns={"mean": "tk_pct", "size": "n"}).to_csv(
        OUT / "ol_subclusters.csv", index=False
    )
    print(pd.DataFrame(rows).round(2).T.to_string())
    print(
        pd.DataFrame(bins_out)
        .pivot(index="depth_quintile", columns="caste", values="tk_pct")
        .round(1)
    )


if __name__ == "__main__":
    main()
