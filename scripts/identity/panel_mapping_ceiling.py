"""What can the 101-gene panel carry at all? The ceiling on any transfer.

Downstream checks (against anatomy, other runs, other vocabularies) cannot
separate a bad mapping from a sound mapping working through a panel that
cannot express the distinction asked of it. This measures the latter without
touching the spatial data: split the snRNA reference by cell, train a
classifier on one half using only the Xenium panel genes, and predict the
held-out half. The result is an upper bound for any transfer method: a type
the panel cannot separate here cannot be transferred well downstream.

Reported at canonical-17, the vocabulary the figures use.

Two controls, since raw accuracy on 17 imbalanced classes is not
interpretable on its own:
  - a stratified-random baseline at the same class frequencies;
  - the same classifier on all reference genes, which separates "the panel
    is too small" from "these types are not separable by expression at all".

Writes results/reframe/mapping_ceiling/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bimp_st.cluster_labels import CANONICAL_ORDER, get_canonical  # noqa: E402
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import PANEL_CROSSWALK  # noqa: E402

REF = REPO / "data" / "sn" / "BIMP_atlas_qw_combined.h5ad"
PANEL = PANEL_CROSSWALK
OUT = REPO / "results" / "reframe" / "mapping_ceiling"
SEED = 0
MIN_CELLS = 40


def _panel_genes(ref):
    """The Xenium panel's genes, as they are named in the reference."""
    ids = set()
    if PANEL.exists():
        m = read_panel_crosswalk()
        for c in m.columns:
            ids |= set(m[c].astype(str))
    have = set(map(str, ref.var_names))
    hits = sorted(have & ids)
    if len(hits) >= 50:
        return hits, "np31_to_v43_mapping.csv"
    # fall back to the spatial object's own var_names
    from bimp_st.paths import SPATIAL_COMBINED

    sp = ad.read_h5ad(SPATIAL_COMBINED, backed="r")
    names = [str(v) for v in sp.var_names]
    sp.file.close()
    return sorted(have & set(names)), "spatial object var_names"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading reference...", flush=True)
    ref = ad.read_h5ad(REF)
    print(f"  {ref.n_obs:,} nuclei x {ref.n_vars:,} genes", flush=True)

    panel, how = _panel_genes(ref)
    print(f"  panel genes found in the reference: {len(panel)} (via {how})")

    lab = ref.obs["cell_type"].astype(str).values
    y = np.array([get_canonical(v) for v in lab])
    keep = pd.Series(y).groupby(y).transform("size").values >= MIN_CELLS
    y = y[keep]
    print(
        f"  {keep.sum():,} nuclei across {len(set(y))} canonical types "
        f"(types under {MIN_CELLS} cells dropped)"
    )

    def dense(mat):
        return np.asarray(
            mat.todense() if hasattr(mat, "todense") else mat, dtype=np.float32
        )

    res = {}
    for tag, genes in (
        ("panel_only", panel),
        ("all_genes", list(map(str, ref.var_names))),
    ):
        X = dense(ref[keep, genes].X)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0)
        # one BLAS thread: lbfgs sums in a thread-count-dependent order otherwise
        with threadpool_limits(limits=1):
            clf.fit(sc.transform(Xtr), ytr)
            pred = clf.predict(sc.transform(Xte))
        acc = float((pred == yte).mean())
        bacc = float(balanced_accuracy_score(yte, pred))
        dum = DummyClassifier(strategy="stratified", random_state=SEED)
        dum.fit(Xtr, ytr)
        dpred = dum.predict(Xte)
        res[tag] = {
            "n_genes": len(genes),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "chance_accuracy": float((dpred == yte).mean()),
            "chance_balanced": float(balanced_accuracy_score(yte, dpred)),
        }
        print(f"\n=== {tag}: {len(genes)} genes ===")
        print(
            f"  accuracy {acc:.4f}   balanced {bacc:.4f}"
            f"   (stratified chance {res[tag]['chance_accuracy']:.4f} / "
            f"{res[tag]['chance_balanced']:.4f})"
        )
        if tag == "panel_only":
            tab = pd.crosstab(
                pd.Series(yte, name="true snRNA type"),
                pd.Series(pred, name="recovered from panel"),
            )
            o = [c for c in CANONICAL_ORDER if c in tab.index or c in tab.columns]
            tab = tab.reindex(
                index=[c for c in o if c in tab.index],
                columns=[c for c in o if c in tab.columns],
                fill_value=0,
            )
            tab.to_csv(OUT / "ceiling_confusion_canonical17.csv")
            per = pd.DataFrame(
                {
                    "n_test": tab.sum(1),
                    "recovered": np.diag(
                        tab.reindex(
                            index=tab.index, columns=tab.index, fill_value=0
                        ).values
                    ),
                }
            )
            per["recall"] = per.recovered / per.n_test.replace(0, np.nan)
            print("\n  per-type ceiling (recall from panel genes alone):")
            print(
                per.sort_values("recall", ascending=False).to_string(
                    float_format=lambda x: f"{x:.3f}"
                )
            )
            per.to_csv(OUT / "ceiling_per_type.csv")

    res["interpretation"] = (
        "panel_only is the ceiling on any spatial transfer through this "
        "panel; all_genes says whether a shortfall is the panel's or the "
        "biology's."
    )
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
