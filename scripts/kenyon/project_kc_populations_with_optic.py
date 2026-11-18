#!/usr/bin/env python
"""Restricted five-population Kenyon reclassification with the optic neuropils in the
same query.

Does the Kenyon result survive when the query is the mushroom-body compartments and the
lamina, lobula and medulla together, instead of the Kenyon compartments
alone? Same reference (the five central-brain Kenyon populations), marker
recipe (top 50 Wilcoxon genes per population), reference-density prior, cluster-sum
expression, 500 epochs, lr 0.1 and seed 42 as
``scripts/kenyon/project_kc_populations.py`` and
``project_kc_populations_optic_control.py``; only the query changes. Reports the call
shares and confidence per region and Cramer's V of call x region for the Kenyon and the
optic regions separately.

GPU: ``sbatch $BIMP_SBATCH_GPU --gres=gpu:1 -c 8 --mem=128G -t 04:00:00``
(site options in ``config/profiles/slurm/sbatch.env``). Writes only
``mb_ol_calls.parquet`` and ``mb_ol_summary.csv`` in
``bimp_st.paths.MB_OL_CALLS_DIR``.

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/kenyon/project_kc_populations_with_optic.py
"""

from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats.contingency import association
from tangram2 import mapping as tg
from tangram2.mapping import mapping_optimizer as mo
from tangram2.mapping import mapping_utils as mu

faulthandler.enable()
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.paths import CANONICAL_TANGRAM, MB_OL_CALLS_DIR, SN_CB  # noqa: E402

OUT = MB_OL_CALLS_DIR
CLUSTER_COL = "cell_type"
CLUSTERS = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
]
SHORT = {
    "Large-type Kenyon cells": "Large",
    "Middle-type Kenyon cells": "Middle",
    "Small-type Kenyon cells": "Small",
    "Class II Kenyon cells": "ClassII",
    "Other Kenyon cells": "Other",
}
DEVICE = os.environ.get("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
REGIONS = ["KC-IC", "KC-NC", "KC-OC", "Lamina", "Lobula", "Medulla"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("device", DEVICE, flush=True)
    ref = sc.read_h5ad(SN_CB)
    ref.obs[CLUSTER_COL] = ref.obs[CLUSTER_COL].astype(str)
    ref = ref[ref.obs[CLUSTER_COL].isin(CLUSTERS)].copy()
    assert ref.n_obs == 32672, ref.n_obs
    sp = sc.read_h5ad(CANONICAL_TANGRAM)
    sp.obs_names_make_unique()
    o = sp.obs
    la, lo, me = (
        o[c].astype(bool).values for c in ("in_lamina", "in_lobula", "in_medulla")
    )
    kic, knc, koc = (
        o[c].astype(bool).values for c in ("in_kc_ic", "in_kc_nc", "in_kc_oc")
    )
    kc, ol = kic | knc | koc, la | lo | me
    assert (kc & ol).sum() == 0
    q = kc | ol
    comp = np.select([kic, knc, koc, la, lo, me], REGIONS, default="")[q]
    print(
        "query cells",
        int(q.sum()),
        pd.Series(comp).value_counts().to_dict(),
        flush=True,
    )
    spq = sp[q].copy()
    del sp
    rl = {g.lower(): g for g in ref.var_names}
    sl = {g.lower(): g for g in spq.var_names}
    ov = sorted(set(rl) & set(sl))
    print("gene overlap", len(ov), flush=True)
    spq.var_names = [dict((sl[g], rl[g]) for g in ov).get(g, g) for g in spq.var_names]
    names = [rl[g] for g in ov]
    rs, ss = ref[:, names].copy(), spq[:, names].copy()
    del spq
    sc.tl.rank_genes_groups(rs, groupby=CLUSTER_COL, method="wilcoxon")
    mk = pd.DataFrame(rs.uns["rank_genes_groups"]["names"]).head(50)
    markers = [g for g in set(mk.values.flatten()) if g in names]
    print("markers", len(markers), flush=True)
    tg.pp_adatas(rs, ss, genes=markers)
    tr = list(rs.uns["training_genes"])
    d = np.asarray(ss.obs["rna_count_based_density"], dtype="float32")
    G = ss[:, tr].X
    G = np.asarray(G.toarray() if hasattr(G, "toarray") else G, dtype="float32")
    brain = ss.obs["brain"].astype(str).values
    a = mu.adata_to_cluster_expression(rs, CLUSTER_COL, scale=True, add_density=True)
    order = [str(c) for c in a.obs[CLUSTER_COL]]
    S = a[:, tr].X
    S = np.asarray(S.toarray() if hasattr(S, "toarray") else S, dtype="float32")
    dens = np.asarray(a.obs["cluster_density"], dtype="float32")
    idx = [order.index(c) for c in CLUSTERS]
    S, dens = S[idx], dens[idx]
    mapper = mo.Mapper(
        S=S,
        G=G,
        d=d,
        coeff=None,
        d_source=dens,
        lambda_d=1,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        device=DEVICE,
        random_state=SEED,
    )
    M, _ = mapper.train(learning_rate=0.1, num_epochs=500, print_each=100)
    probs = pd.DataFrame(M.T, index=ss.obs_names, columns=CLUSTERS)
    rn = probs.div(probs.sum(axis=1), axis=0)
    out = pd.DataFrame({"region": comp, "brain": brain}, index=ss.obs_names)
    out["call"] = [SHORT[c] for c in probs.idxmax(axis=1)]
    out["maxprob"] = rn.max(axis=1).values
    for c in CLUSTERS:
        out["p_" + SHORT[c]] = rn[c].values
    out.to_parquet(OUT / "mb_ol_calls.parquet")

    pct = (
        pd.crosstab(out.region, out.call, normalize="index").reindex(REGIONS)[
            list(SHORT.values())
        ]
        * 100
    )
    conf = (
        out.groupby("region")
        .maxprob.agg(
            median="median", frac_below_0p5=lambda x: (x < 0.5).mean(), n="size"
        )
        .reindex(REGIONS)
    )
    summ = pct.join(conf)
    for name, sel in (
        ("Kenyon regions", out.region.str.startswith("KC")),
        ("optic regions", ~out.region.str.startswith("KC")),
        ("all six", out.region != ""),
    ):
        ct = pd.crosstab(out.region[sel], out.call[sel])
        summ.loc[f"Cramers V, {name}", "median"] = association(
            ct.values, method="cramer"
        )
    summ.to_csv(OUT / "mb_ol_summary.csv")
    print(summ.round(4).to_string(), flush=True)
    print(
        out.groupby(["region", "brain"])
        .maxprob.median()
        .unstack()
        .reindex(REGIONS)
        .round(4),
        flush=True,
    )


if __name__ == "__main__":
    main()
