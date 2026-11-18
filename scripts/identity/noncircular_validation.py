"""Tangram-independent validation battery: is the atlas good, without circularity?

Cross-modal comparisons of ``obs['tangram_cell_type']`` against the snRNA
reference are partly circular, because those labels were assigned by Tangram
from that reference. These tests put no reference-derived label on the Xenium
side.

Allowed on the Xenium side:
  * raw / area-normalised panel expression                       (measured)
  * ``obs['in_lamina'|'in_medulla'|'in_lobula'|'in_kc_oc'|'in_kc_ic'|'in_kc_nc']``
    -- hand-drawn GeoJSON anatomy, never seen by Tangram          (measured)
  * ``obs['leiden']`` / ``obs['leiden_original']``                (see below)

Leiden provenance: ``combined_analysis.py::cluster``
(pipelines/combined_atlas/combined_analysis.py:123-170) does
normalize_total -> log1p -> highly_variable_genes(n_top=2000, batch_key='brain')
-> pca -> neighbors -> leiden, with no spatial coordinates, anatomy flags or
cell-type labels (the Harmony line is commented out). On a 101-gene panel the
HVG step selects everything, so Leiden is an expression-only Xenium partition.
This is asserted at runtime against the stored uns.

Tests

  A0  Panel ceiling. Logistic regression on Xenium panel expression -> calyx
      vs optic anatomy, blocked-section and cross-brain CV. Within-Xenium
      only; gives the ceiling against which A1 is read.

  A1  Anatomy test. Kenyon-cell markers are defined in the snRNA reference
      from its own labels, scored on Xenium cells, and tested for separating
      calyx from optic anatomy. The reference contributes only a gene list.
      The null is a 1000x abundance-matched random gene-set null rather than
      0.5, because the densely packed calyx inflates a naive AUROC.

  B   Unsupervised structure agreement. Cluster the snRNA reference de novo
      on the shared panel genes and take the Xenium de novo Leiden partition;
      no label crosses. Match centroids post hoc (Hungarian on centroid
      Spearman) and count how many distinct Xenium clusters the snRNA
      clusters map onto. Nulls: gene permutation (weak) and size-matched
      wrong partition (strong; wrong partitions can score higher in raw
      correlation, so the distinct-cluster count is the headline).

  C   Gene specificity per modality. Yanai tau for each panel gene against
      that modality's own partition, then correlate the two tau vectors.
      Nulls: gene permutation and size-matched wrong partition; the
      wrong-partition delta is reported.

  D   Composition in Aitchison geometry (circular; run for metric consistency
      only). Cosine on raw proportions, with one dominant category, mostly
      measures "both have many Kenyon cells", and the Methods use Aitchison
      geometry for scCODA, so this uses the project's CLR (imported from
      pipelines/wholebrain/queen/compositional_analysis.py: multiplicative
      zero replacement, delta 0.65 times the smallest non-zero value;
      Martin-Fernandez 2003),
      reporting Aitchison distance and CLR-space correlation beside cosine.
      The Xenium side still carries Tangram labels.

Provenance: the substrate conventions (area-normalised Xenium counts, CP10k +
log1p snRNA, np31->v43 gene bridge), ``tau_yanai``, the marker-score AUROC and
the wrong-partition null follow the Xenium-snRNA correspondence battery's
harness. The normalisation
guards in ``prep`` are from scripts/atlas/snrna_xenium_joint_embedding.py.

Known traps guarded:
  (a) the reference h5ads carry a stale ``var['highly_variable']`` that
      restricts PCA to 45 of 101 genes -- deleted before PCA;
  (b) they carry ``obsm['X_pca_harmony']``, which survives subsetting and can
      make two arms numerically identical -- inherited obsm keys are deleted
      and the recomputed embedding is asserted to differ;
  (c) combined_clustered.h5ad has no layers and its ``X`` is already log1p, so
      log-scale is detected before any normalisation.

Outputs to results/reframe/noncircular/:
    summary.csv              one row per reported number: value, null, verdict
    testA0_supervised.csv    per CV scheme
    testA1_markerscore.csv   per brain / per section, with matched null
    testA1_null_draws.csv
    testB_matching.csv       Hungarian matching + nulls
    testB_centroid_corr.csv  snRNA cluster x Xenium cluster correlation matrix
    testC_tau.csv            per-gene tau in both modalities, both pairings
    testC_summary.csv
    testD_clr.csv            cosine / CLR correlation / Aitchison distance
    sn_denovo_clusters.csv   snRNA de novo cluster per nucleus (audit trail)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
# Test D imports the project's CoDA module, which pulls in pyplot/seaborn at
# module scope; on a headless node that must not try to find a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.cluster_labels import (  # noqa: E402
    CANONICAL_ORDER,
    POOLED_ORDER,
    add_canonical,
    get_canonical,
    get_pooled,
)
from bimp_st.utils.gene_mapping import (  # noqa: E402
    load_gene_mapping,
    map_spatial_genes_to_v43,
)

SPATIAL = (
    REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
)
SN_CB = REPO / "data" / "sn" / "BIMP_atlas_qw_CentralBrain.h5ad"
SN_OL = REPO / "data" / "sn" / "BIMP_atlas_qw_OpticLobe.h5ad"
CODA_MODULE = REPO / "pipelines" / "wholebrain" / "queen" / "compositional_analysis.py"
OUT = REPO / "results" / "reframe" / "noncircular"
OUT.mkdir(parents=True, exist_ok=True)

CALYX_FLAGS = ["in_kc_oc", "in_kc_ic", "in_kc_nc"]
OPTIC_FLAGS = ["in_lamina", "in_medulla", "in_lobula"]
SEED = 0
N_PERM = 1000  # gene-permutation null draws
N_WRONGPART = 500  # size-matched wrong-partition draws
N_NULL_GENESETS = 1000  # abundance-matched gene-set draws for Test A1
SN_LEIDEN_RES = 1.0
# tau and centroid matching both depend on how finely each side is cut, so the
# snRNA de novo partition is swept rather than fixed at one arbitrary value.
SN_LEIDEN_RES_SWEEP = [0.3, 0.5, 1.0]
N_PCS = 40
XEN_NULL_SUBSAMPLE = 300_000  # for the Xenium-side wrong-partition null

rng = np.random.default_rng(SEED)
SUMMARY: list[dict] = []


def record(test, quantity, value, null_value, null_desc, verdict, note=""):
    """Every reported number carries its null and its circularity verdict."""
    SUMMARY.append(
        dict(
            test=test,
            quantity=quantity,
            value=value,
            null=null_value,
            null_description=null_desc,
            circularity=verdict,
            note=note,
        )
    )


def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def is_log_scale(M) -> bool:
    """True if a matrix looks log1p-transformed rather than integer counts."""
    d = dense(M[:5000] if M.shape[0] > 5000 else M)
    nz = d[d > 0]
    if nz.size == 0:
        return False
    s = nz[:100_000]
    return not np.allclose(s, np.round(s))


def prep(adata, tag: str):
    """counts -> CP10k -> log1p, with nothing inherited from the file.

    Vendored from scripts/atlas/snrna_xenium_joint_embedding.py; covers traps (a)-(c).
    """
    dropped = [k for k in ("X_pca", "X_pca_harmony", "X_umap") if k in adata.obsm]
    for k in dropped:
        del adata.obsm[k]
    if dropped:
        print(f"    [{tag}] dropped inherited obsm: {dropped}", flush=True)
    if "highly_variable" in adata.var.columns:
        n_hv = int(adata.var["highly_variable"].sum())
        del adata.var["highly_variable"]
        print(
            f"    [{tag}] dropped stale highly_variable ({n_hv} genes flagged "
            f"of {adata.n_vars}) -- PCA would have used only those",
            flush=True,
        )
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"    [{tag}] normalised from counts layer", flush=True)
    elif is_log_scale(adata.X):
        print(f"    [{tag}] X already log-normalised -- left as is", flush=True)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"    [{tag}] normalised from raw X", flush=True)
    return adata


def tau_yanai(mat: np.ndarray) -> np.ndarray:
    """Yanai specificity per column. rows = clusters, cols = genes.

    tau = sum_i (1 - x_i/x_max) / (n-1); 0 = uniform, 1 = one-cluster specific.
    """
    x = np.asarray(mat, float)
    xmax = x.max(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = xmax > 0
    xhat = x[:, ok] / xmax[ok]
    out[ok] = (1.0 - xhat).sum(axis=0) / (x.shape[0] - 1)
    return out


def group_means(X: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Fast per-group column means via sort + reduceat. (n_groups, n_genes)."""
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    starts = np.searchsorted(cs, np.arange(n_groups), side="left")
    stops = np.searchsorted(cs, np.arange(n_groups), side="right")
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        if stops[g] > starts[g]:
            out[g] = X[order[starts[g] : stops[g]]].mean(axis=0)
    return out


def size_matched_partition(codes: np.ndarray, rng) -> np.ndarray:
    """Random relabelling preserving the cluster-size distribution exactly."""
    return rng.permutation(codes)


def perm_null_spearman(a, b, nperm, rng):
    """Empirical two-sided p for a Spearman correlation by shuffling b."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[m], np.asarray(b)[m]
    obs = spearmanr(a, b).statistic
    nulls = np.empty(nperm)
    bb = b.copy()
    for i in range(nperm):
        rng.shuffle(bb)
        nulls[i] = spearmanr(a, bb).statistic
    p = (np.sum(np.abs(nulls) >= abs(obs)) + 1) / (nperm + 1)
    return float(obs), float(p), nulls


# ══════════════════════════════════════════════════════════ substrate ═══
def build_substrate():
    print("\n" + "=" * 78, flush=True)
    print("SUBSTRATE", flush=True)
    print("=" * 78, flush=True)

    mapping = load_gene_mapping()

    print("loading Xenium combined_clustered.h5ad ...", flush=True)
    xen = ad.read_h5ad(SPATIAL)
    xen.obs_names_make_unique()
    print(f"  {xen.n_obs:,} cells x {xen.n_vars} probes", flush=True)

    # ---- Leiden provenance assertion -----------------------------------
    have_leiden = "leiden" in xen.obs.columns
    have_leiden_orig = "leiden_original" in xen.obs.columns
    assert have_leiden, "obs['leiden'] missing -- Test B/C have no Xenium partition"
    nb_params = dict(xen.uns.get("neighbors", {}).get("params", {}))
    use_rep = str(nb_params.get("use_rep", ""))
    # obsm carries 'spatial' and 'spatial_aligned'; check the neighbour graph
    # was built on neither. Absent use_rep means the scanpy default X_pca, which
    # comes from uns['pca'] below (expression only).
    assert not any(c in use_rep for c in ("spatial", "spatial_aligned")), (
        f"neighbours graph built on a spatial representation: {use_rep!r}"
    )
    print(
        f"  leiden: {xen.obs['leiden'].nunique()} clusters; "
        f"leiden_original present={have_leiden_orig}"
        + (
            f" ({xen.obs['leiden_original'].nunique()} clusters)"
            if have_leiden_orig
            else ""
        ),
        flush=True,
    )
    print(f"  STORED PROVENANCE uns['neighbors']['params'] = {nb_params}", flush=True)
    print(
        f"  STORED PROVENANCE uns['pca']['params'] = "
        f"{dict(xen.uns.get('pca', {}).get('params', {}))}",
        flush=True,
    )
    print(
        f"  obsm keys = {list(xen.obsm)} (spatial coords are PRESENT in the "
        f"file but absent from the neighbour graph's representation)",
        flush=True,
    )
    print(
        "  VERIFIED (source read): combined_analysis.py::cluster uses "
        "normalize->log1p->HVG->PCA->neighbors->leiden; no spatial coords, "
        "no anatomy, no cell-type label. scripts/atlas/spatial_leiden_resolution.py, "
        "which wrote the uns['leiden_r*'] keys, only re-runs sc.tl.leiden on "
        "that SAME stored expression graph ('spatial' in its name means the "
        "spatial dataset, not coordinates) and writes nothing back. Leiden "
        "is admissible.",
        flush=True,
    )

    # ---- gene bridge ----------------------------------------------------
    sp_v43 = map_spatial_genes_to_v43(list(xen.var_names), mapping)
    gm = pd.DataFrame({"spatial_var": list(xen.var_names), "v43": sp_v43})

    print("loading snRNA CentralBrain reference ...", flush=True)
    sn = ad.read_h5ad(SN_CB)
    sn_v43 = set(map(str, sn.var_names))
    gm["in_snuc"] = gm.v43.isin(sn_v43)
    keep = gm[gm.in_snuc].reset_index(drop=True)
    # a v43 id can be hit by two probes (two probes share the gene "fatty acid
    # synthase"); the shared-gene space is the unique v43 set
    dup = keep.v43.duplicated(keep=False).sum()
    keep = keep.drop_duplicates(subset="v43", keep="first").reset_index(drop=True)
    genes_v43 = keep.v43.tolist()
    sp_cols = keep.spatial_var.tolist()
    print(
        f"  panel probes {len(gm)}; mapped into reference {int(gm.in_snuc.sum())}; "
        f"duplicate v43 probes {dup}; SHARED GENE SPACE = {len(genes_v43)}",
        flush=True,
    )
    gm.to_csv(OUT / "gene_bridge.csv", index=False)

    # ---- Xenium expression: area-normalised raw counts -------------------
    raw = xen.raw.to_adata() if xen.raw is not None else xen
    raw = raw[:, xen.var_names]
    Xr = dense(raw[:, sp_cols].X).astype(np.float32)
    assert not is_log_scale(Xr), "Xenium raw.X is not integer counts"
    area = xen.obs["cell_area"].to_numpy(np.float32)
    med_area = float(np.median(area[area > 0]))
    Xarea = (Xr / np.clip(area[:, None], 1e-6, None) * med_area).astype(np.float32)
    del Xr, raw
    print(f"  Xenium area-normalised (median cell_area {med_area:.1f} um2)", flush=True)

    # Xenium log-normalised X (already log1p -- trap (c))
    assert is_log_scale(xen.X), "combined_clustered X is not on a log scale"
    Xlog = dense(xen[:, sp_cols].X).astype(np.float32)
    print("  Xenium X confirmed already log1p -- NOT renormalised", flush=True)

    calyx = np.zeros(xen.n_obs, bool)
    for f in CALYX_FLAGS:
        calyx |= xen.obs[f].to_numpy(bool)
    optic = np.zeros(xen.n_obs, bool)
    for f in OPTIC_FLAGS:
        optic |= xen.obs[f].to_numpy(bool)

    xmeta = pd.DataFrame(
        {
            "brain": xen.obs["brain"].astype(str).values,
            "slice_name": xen.obs["slice_name"].astype(str).values,
            "leiden": xen.obs["leiden"].astype(str).values,
            "in_calyx": calyx,
            "in_optic": optic,
            "cell_area": area,
        }
    )
    if have_leiden_orig:
        xmeta["leiden_original"] = xen.obs["leiden_original"].astype(str).values
    # Tangram labels are carried ONLY for Test D, which is declared circular.
    add_canonical(xen, src="tangram_cell_type")
    xmeta["cell_type_canonical_CIRCULAR"] = (
        xen.obs["cell_type_canonical"].astype(str).values
    )
    print(
        f"  anatomy: calyx {calyx.sum():,}, optic {optic.sum():,}, "
        f"both {(calyx & optic).sum():,}",
        flush=True,
    )
    del xen

    # ---- snRNA reference -------------------------------------------------
    # Normalise on the full gene space, then subset; otherwise CP10k becomes a
    # depth normalisation over 100 genes instead of the transcriptome.
    sn = prep(sn, "snRNA CB")
    sn = sn[:, genes_v43].copy()
    for k in list(sn.obsm.keys()):
        del sn.obsm[k]
    smeta = pd.DataFrame(
        {
            "cell_type": sn.obs["cell_type"].astype(str).values,
            "major_cell_type": sn.obs["major_cell_type"].astype(str).values,
            "caste": sn.obs["caste"].astype(str).values,
            "sample": sn.obs["sample"].astype(str).values,
        },
        index=sn.obs_names.astype(str),
    )
    print(
        f"  snRNA CB {sn.n_obs:,} nuclei x {sn.n_vars} shared genes; "
        f"{smeta.major_cell_type.nunique()} major / "
        f"{smeta.cell_type.nunique()} fine types",
        flush=True,
    )

    return dict(
        genes=genes_v43,
        Xarea=Xarea,
        Xlog=Xlog,
        xmeta=xmeta,
        sn=sn,
        smeta=smeta,
        med_area=med_area,
    )


# ══════════════════════════════════════════ TEST A0 -- panel ceiling ═══
def test_A0(S):
    print("\n" + "=" * 78, flush=True)
    print(
        "TEST A0 -- calyx vs optic from Xenium expression alone (CEILING)", flush=True
    )
    print("=" * 78, flush=True)
    print(
        "CIRCULARITY: NOT circular (no reference enters), but also NOT "
        "cross-modal evidence -- it only bounds how much anatomy signal the "
        "panel carries. Read A1 against this ceiling.",
        flush=True,
    )

    m = S["xmeta"]
    sel = m.in_calyx.values ^ m.in_optic.values  # exactly one compartment
    idx = np.flatnonzero(sel)
    X = S["Xlog"][idx]
    y = m.in_calyx.values[idx].astype(int)  # 1 = calyx, 0 = optic
    brain = m.brain.values[idx]
    sect = m.slice_name.values[idx]
    print(
        f"  {len(idx):,} cells in exactly one compartment "
        f"(calyx {y.sum():,}, optic {(1 - y).sum():,})",
        flush=True,
    )

    rows = []

    def run(tr, te, scheme, permute=False):
        yy = y.copy()
        if permute:  # shuffle within section
            for s in np.unique(sect):
                w = np.flatnonzero(sect == s)
                yy[w] = rng.permutation(yy[w])
        sc_ = StandardScaler().fit(X[tr])  # fit on TRAIN ONLY
        clf = LogisticRegression(max_iter=2000, n_jobs=1)
        # one BLAS thread: lbfgs sums in a thread-count-dependent order otherwise
        with threadpool_limits(limits=1):
            clf.fit(sc_.transform(X[tr]), yy[tr])
            score = clf.decision_function(sc_.transform(X[te]))
        a = roc_auc_score(yy[te], score) if len(np.unique(yy[te])) > 1 else np.nan
        rows.append(
            dict(
                scheme=scheme,
                permuted=permute,
                auroc=float(a),
                n_train=len(tr),
                n_test=len(te),
            )
        )
        print(
            f"  {scheme:26s} {'PERMUTED' if permute else 'real':9s} AUROC {a:.4f}",
            flush=True,
        )
        return float(a)

    # blocked sections: contiguous runs of 5 held out together, so a test
    # section's serial neighbours (10 um away) are also unseen
    uniq = np.array(sorted(pd.unique(sect)))
    block = {s: i // 5 for i, s in enumerate(uniq)}
    gb = np.array([block[s] for s in sect])
    nb = gb.max() + 1
    held = rng.choice(nb, max(1, nb // 3), replace=False)
    te = np.flatnonzero(np.isin(gb, held))
    tr = np.flatnonzero(~np.isin(gb, held))
    a_block = run(tr, te, "blocked-section")
    run(tr, te, "blocked-section", permute=True)

    a_cross = {}
    for src in ("queen", "worker"):
        tr = np.flatnonzero(brain == src)
        te = np.flatnonzero(brain != src)
        if len(tr) and len(te):
            tag = "q2w" if src == "queen" else "w2q"
            a_cross[tag] = run(tr, te, f"cross-brain {tag}")
            run(tr, te, f"cross-brain {tag}", permute=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "testA0_supervised.csv", index=False)
    perm_mean = float(df[df.permuted].auroc.mean())
    for k, v in a_cross.items():
        record(
            "A0 ceiling",
            f"calyx-vs-optic AUROC, cross-brain {k}",
            v,
            perm_mean,
            "labels permuted within section (mean AUROC)",
            "NOT circular; NOT cross-modal (ceiling only)",
        )
    record(
        "A0 ceiling",
        "calyx-vs-optic AUROC, blocked-section",
        a_block,
        perm_mean,
        "labels permuted within section (mean AUROC)",
        "NOT circular; NOT cross-modal (ceiling only)",
    )
    return df


# ═══════════════════════ TEST A1 -- snRNA markers -> Xenium anatomy ═══
def test_A1(S):
    print("\n" + "=" * 78, flush=True)
    print(
        "TEST A1 -- snRNA-defined Kenyon-cell markers -> Xenium calyx anatomy",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(
        "CIRCULARITY: NOT circular. The reference contributes a GENE LIST "
        "derived from its own labels; the Xenium side contributes measured "
        "expression and hand-drawn GeoJSON anatomy. No Tangram output is "
        "used anywhere in this test.",
        flush=True,
    )

    sn, smeta, genes = S["sn"], S["smeta"], S["genes"]
    E = dense(sn.X).astype(np.float32)

    # --- markers from the reference's OWN labels -------------------------
    y_kc = (smeta.major_cell_type.values == "Kenyon cells").astype(int)
    assert y_kc.sum() > 100, "no Kenyon-cell nuclei in the reference"
    auc_sn = np.array([roc_auc_score(y_kc, E[:, i]) for i in range(E.shape[1])])
    marker_mask = auc_sn >= 0.70
    if marker_mask.sum() > 15:  # cap; take the strongest
        cut = np.sort(auc_sn)[::-1][15]
        marker_mask = auc_sn > cut
    if marker_mask.sum() < 3:  # fall back to the top 10 by AUROC
        marker_mask = auc_sn >= np.sort(auc_sn)[::-1][9]
        print(
            "  WARNING: fewer than 3 genes clear AUROC 0.70 -- falling back "
            "to the top 10 by snRNA AUROC",
            flush=True,
        )
    markers = [genes[i] for i in np.flatnonzero(marker_mask)]
    print(
        f"  {y_kc.sum():,} KC nuclei / {len(y_kc):,}; "
        f"{len(markers)} KC markers (snRNA one-vs-rest AUROC >= "
        f"{auc_sn[marker_mask].min():.3f})",
        flush=True,
    )

    # --- score Xenium cells ----------------------------------------------
    m = S["xmeta"]
    sel = m.in_calyx.values ^ m.in_optic.values
    idx = np.flatnonzero(sel)
    Xa = S["Xarea"][idx]
    y = m.in_calyx.values[idx].astype(int)
    brain = m.brain.values[idx]
    sect = m.slice_name.values[idx]

    # area-normalisation inflation check (a known confound: the calyx is
    # densely packed, so per-area counts are higher there for ANY gene)
    tot_raw_ratio = float(Xa[y == 1].sum(1).mean() / Xa[y == 0].sum(1).mean())
    print(
        f"  CONFOUND: mean area-normalised total counts calyx/optic = "
        f"{tot_raw_ratio:.2f}x -- a bare AUROC is partly this, which is why "
        f"the null below is abundance-matched, not 0.5",
        flush=True,
    )

    Z = (Xa - Xa.mean(0)) / np.clip(Xa.std(0), 1e-9, None)  # z per gene

    def set_auroc(cols, mask=None):
        s = Z[:, cols].mean(1)
        yy, ss = (y, s) if mask is None else (y[mask], s[mask])
        if len(np.unique(yy)) < 2:
            return np.nan
        return float(roc_auc_score(yy, ss))

    mcols = [genes.index(g) for g in markers]
    auroc_set = set_auroc(mcols)
    auroc_pergene = float(np.mean([set_auroc([c]) for c in mcols]))
    print(
        f"  SET-LEVEL AUROC (mean z of the {len(markers)}-gene score) = "
        f"{auroc_set:.4f}",
        flush=True,
    )
    print(
        f"  PER-MARKER MEAN AUROC (each marker alone, averaged) = {auroc_pergene:.4f}",
        flush=True,
    )

    rows = [
        dict(
            split="all", n=len(y), auroc_set=auroc_set, auroc_pergene_mean=auroc_pergene
        )
    ]
    for b in np.unique(brain):
        mask = brain == b
        rows.append(
            dict(
                split=f"brain:{b}",
                n=int(mask.sum()),
                auroc_set=set_auroc(mcols, mask),
                auroc_pergene_mean=float(
                    np.mean([set_auroc([c], mask) for c in mcols])
                ),
            )
        )
        print(
            f"  brain {b:7s} n={mask.sum():>8,}  set-level AUROC "
            f"{rows[-1]['auroc_set']:.4f}   (biological replication: the two "
            f"brains are independent animals)",
            flush=True,
        )
    per_sec = []
    for s in np.unique(sect):
        mask = sect == s
        if mask.sum() < 200:
            continue
        a = set_auroc(mcols, mask)
        if np.isfinite(a):
            per_sec.append(a)
            rows.append(
                dict(
                    split=f"section:{s}",
                    n=int(mask.sum()),
                    auroc_set=a,
                    auroc_pergene_mean=np.nan,
                )
            )
    if per_sec:
        print(
            f"  per-section (held-out unit) n={len(per_sec)} sections: "
            f"median {np.median(per_sec):.4f}, range "
            f"[{min(per_sec):.4f}, {max(per_sec):.4f}]",
            flush=True,
        )

    # --- abundance-matched random gene-set null ---------------------------
    abund = Xa.mean(0)
    order = np.argsort(abund)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    decile = (rank * 10 // len(order)).clip(0, 9)
    want = pd.Series(decile[mcols]).value_counts().to_dict()
    pool = {
        d: np.flatnonzero((decile == d) & ~np.isin(np.arange(len(genes)), mcols))
        for d in want
    }
    print(
        f"  building {N_NULL_GENESETS} abundance-matched random gene sets "
        f"(decile profile {want}) ...",
        flush=True,
    )
    nulls = np.empty(N_NULL_GENESETS)
    for i in range(N_NULL_GENESETS):
        cols = []
        for d, k in want.items():
            p = pool[d]
            cols.extend(rng.choice(p, size=min(k, len(p)), replace=False))
        nulls[i] = set_auroc(list(cols))
    nulls = nulls[np.isfinite(nulls)]
    null_mean = float(nulls.mean())
    emp_p = float((np.sum(nulls >= auroc_set) + 1) / (len(nulls) + 1))
    print(
        f"  MATCHED-EXPRESSION NULL: mean {null_mean:.4f}, "
        f"95th pct {np.percentile(nulls, 95):.4f}, max {nulls.max():.4f}; "
        f"empirical p = {emp_p:.4f}",
        flush=True,
    )
    print(
        f"  >>> INCREMENT OVER NULL = {auroc_set - null_mean:+.4f} "
        f"(this, not {auroc_set:.2f}, is the interpretable quantity)",
        flush=True,
    )

    pd.DataFrame(rows).to_csv(OUT / "testA1_markerscore.csv", index=False)
    pd.DataFrame({"null_auroc": nulls}).to_csv(
        OUT / "testA1_null_draws.csv", index=False
    )
    pd.DataFrame(
        {"v43": genes, "snuc_KC_auroc": auc_sn, "is_marker": marker_mask}
    ).to_csv(OUT / "testA1_markers.csv", index=False)

    record(
        "A1 KC->calyx",
        "set-level AUROC (all cells)",
        auroc_set,
        null_mean,
        f"{len(nulls)} abundance-matched random {len(markers)}-gene sets "
        f"(mean); empirical p={emp_p:.4f}",
        "NOT circular",
        f"increment {auroc_set - null_mean:+.4f}; "
        f"area-norm calyx/optic count ratio {tot_raw_ratio:.2f}x",
    )
    record(
        "A1 KC->calyx",
        "per-marker mean AUROC",
        auroc_pergene,
        null_mean,
        "same matched gene-set null (set-level scale, approximate)",
        "NOT circular",
        "reported because prior runs quoted this variant",
    )
    for r in rows:
        if r["split"].startswith("brain:"):
            record(
                "A1 KC->calyx",
                f"set-level AUROC, {r['split']}",
                r["auroc_set"],
                null_mean,
                "abundance-matched random gene sets (all-cell null)",
                "NOT circular",
                "independent animal = biological replicate",
            )
    return dict(markers=markers, auroc_set=auroc_set, null_mean=null_mean)


# ═════════════════════ TEST B -- unsupervised structure agreement ═══
def test_B(S):
    print("\n" + "=" * 78, flush=True)
    print("TEST B -- de novo clusters in each modality, matched post hoc", flush=True)
    print("=" * 78, flush=True)
    print(
        "CIRCULARITY: NOT circular. Each modality is clustered on its own "
        "expression; no label crosses. The matching is computed after the "
        "fact and is scored against a null, so the matching itself cannot "
        "manufacture agreement.",
        flush=True,
    )

    sn, _smeta, _genes = S["sn"], S["smeta"], S["genes"]
    partitions = sn_denovo_partitions(S)
    E = dense(sn.X).astype(np.float32)  # unscaled log-norm

    # ---- Xenium de novo = the verified expression-only Leiden ------------
    m = S["xmeta"]
    # dissection-matched primary arm: the CB reference has no optic lobe, so
    # optic Xenium clusters have no possible partner and their presence only
    # adds unmatched rows.
    arms = {
        "dissection-matched (non-optic Xenium)": ~m.in_optic.values,
        "all Xenium cells": np.ones(len(m), bool),
    }

    out_rows = []
    for res, sn_clu in partitions.items():
        n_sn = len(np.unique(sn_clu))
        for arm_name, arm_mask in arms.items():
            out_rows.extend(_test_B_one(S, E, sn_clu, n_sn, res, arm_name, arm_mask))
    pd.DataFrame(out_rows).to_csv(OUT / "testB_matching.csv", index=False)
    return partitions


def sn_denovo_partitions(S) -> dict:
    """De novo Leiden partitions of the snRNA reference on the shared genes.

    Swept over resolution: tau and centroid matching both depend on how finely
    each side is cut, and a single arbitrary resolution would leave that as an
    unreported free parameter.
    """
    sn, smeta, genes = S["sn"], S["smeta"], S["genes"]
    print(
        f"  clustering snRNA de novo on {len(genes)} shared panel genes ...", flush=True
    )
    sn2 = sn.copy()
    # trap (b): nothing may be inherited. Assert the slate is clean BEFORE PCA
    # -- X_pca_harmony survives subsetting and, if present, scanpy helpers can
    # pick it up and make two arms numerically identical.
    assert not sn2.obsm, f"inherited obsm survived into the PCA: {list(sn2.obsm)}"
    assert "highly_variable" not in sn2.var.columns
    sc.pp.scale(sn2, max_value=10)
    n_pc = max(2, min(N_PCS, sn2.n_vars - 1, sn2.n_obs - 1))
    sc.tl.pca(sn2, n_comps=n_pc, svd_solver="arpack", random_state=SEED)
    assert sn2.obsm["X_pca"].shape == (sn2.n_obs, n_pc), "PCA shape wrong"
    assert np.isfinite(sn2.obsm["X_pca"]).all()
    assert sn2.obsm["X_pca"].std(0)[0] > 1e-6, "degenerate first PC"
    vr = np.asarray(sn2.uns["pca"]["variance_ratio"])
    assert len(vr) == n_pc and vr[0] > vr[-1], "PCA variance ratio not decreasing"
    print(
        f"    PCA recomputed from {sn2.n_vars} shared genes "
        f"(PC1 var ratio {vr[0]:.3f}); no inherited embedding",
        flush=True,
    )
    sc.pp.neighbors(sn2, n_neighbors=15, use_rep="X_pca", random_state=SEED)
    # Leiden if a graph backend is installed (matching how the Xenium partition
    # was produced), otherwise K-means on the same PCA embedding. Either way the
    # partition is de novo from expression; the choice of algorithm cannot
    # create agreement, because the nulls below are computed on partitions of
    # the same shape produced the same way.
    partitions, audit = (
        {},
        {
            "nucleus": smeta.index,
            "curated_major": smeta.major_cell_type.values,
            "curated_fine": smeta.cell_type.values,
        },
    )
    for res in SN_LEIDEN_RES_SWEEP:
        method = None
        for flavour in ("igraph", "leidenalg"):
            try:
                kw = dict(n_iterations=2, directed=False) if flavour == "igraph" else {}
                sc.tl.leiden(
                    sn2,
                    resolution=res,
                    flavor=flavour,
                    key_added=f"leiden_{res}",
                    random_state=SEED,
                    **kw,
                )
                method = f"leiden(flavor={flavour}, res={res})"
                break
            except ImportError:
                continue
        if method is None:
            from sklearn.cluster import KMeans

            k = {0.3: 6, 0.5: 10, 1.0: 15}.get(res, 15)
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            sn2.obs[f"leiden_{res}"] = pd.Categorical(
                km.fit_predict(sn2.obsm["X_pca"]).astype(str)
            )
            method = f"KMeans(k={k}) on X_pca -- no igraph/leidenalg available"
            print(
                "    NOTE: no Leiden backend installed; used K-means instead. "
                "The partition is still de novo from expression alone.",
                flush=True,
            )
        lab = sn2.obs[f"leiden_{res}"].astype(str).values
        partitions[res] = lab
        audit[f"sn_denovo_res{res}"] = lab
        print(
            f"    -> res {res}: {len(np.unique(lab))} snRNA de novo clusters "
            f"via {method} ({n_pc} PCs)",
            flush=True,
        )
    pd.DataFrame(audit).to_csv(OUT / "sn_denovo_clusters.csv", index=False)
    return partitions


def _test_B_one(S, E, sn_clu, n_sn, res, arm_name, arm_mask):
    """One (snRNA resolution x Xenium arm) cell of Test B."""
    m = S["xmeta"]
    out_rows = []
    if True:
        xi = np.flatnonzero(arm_mask)
        xclu = pd.Categorical(m.leiden.values[xi])
        n_xen = len(xclu.categories)
        Xa = S["Xarea"][xi]
        print(
            f"\n  ARM: {arm_name}  ({len(xi):,} cells, {n_xen} Leiden "
            f"clusters vs {n_sn} snRNA clusters)",
            flush=True,
        )

        xen_cent = group_means(Xa, xclu.codes.astype(np.int64), n_xen)
        sn_codes = pd.Categorical(sn_clu)
        sn_cent = group_means(E, sn_codes.codes.astype(np.int64), n_sn)
        # trap (b), cross-arm form: the two modalities must not have collapsed
        # onto the same numbers via a shared inherited embedding.
        assert not (
            sn_cent.shape == xen_cent.shape and np.allclose(sn_cent, xen_cent)
        ), "snRNA and Xenium centroids are identical -- arms are not independent"

        def match(A, B):
            """Hungarian on centroid Spearman. A = snRNA (rows), B = Xenium.

            Spearman computed as Pearson on within-row average ranks -- exactly
            scipy's definition, but vectorised so the wrong-partition null can
            afford hundreds of draws.
            """

            def _z(M):
                R = rankdata(M, axis=1).astype(np.float64)
                R -= R.mean(1, keepdims=True)
                n = np.linalg.norm(R, axis=1, keepdims=True)
                return R / np.clip(n, 1e-12, None)

            C = np.nan_to_num(_z(A) @ _z(B).T, nan=0.0)
            r, c = linear_sum_assignment(-C)
            best = C.argmax(1)
            return C, float(C[r, c].mean()), int(len(np.unique(best))), best

        C, mean_matched, n_distinct, best = match(sn_cent, xen_cent)
        pd.DataFrame(
            C,
            index=[f"sn_{c}" for c in sn_codes.categories],
            columns=[f"xen_{c}" for c in xclu.categories],
        ).to_csv(OUT / f"testB_centroid_corr_res{res}_{arm_name.split()[0]}.csv")
        print(f"    Hungarian mean matched Spearman = {mean_matched:.4f}", flush=True)
        print(
            f"    DISTINCT Xenium clusters hit by best-match = "
            f"{n_distinct} of {n_sn} snRNA clusters",
            flush=True,
        )

        # NULL 1 -- gene permutation (weak: preserves nothing about abundance)
        gp = np.empty(min(N_PERM, 300))
        for i in range(len(gp)):
            perm = rng.permutation(sn_cent.shape[1])
            _, gp[i], _, _ = match(sn_cent[:, perm], xen_cent)
        print(
            f"    NULL-1 gene permutation: mean matched rho "
            f"{gp.mean():.4f} "
            f"(p={(np.sum(gp >= mean_matched) + 1) / (len(gp) + 1):.4f})",
            flush=True,
        )

        # NULL 2 -- size-matched wrong partition of the snRNA nuclei. This is the
        # null that matters: wrong partitions can score higher in raw correlation
        # than the true partition, because centroid correlation is dominated by
        # shared abundance structure.
        wp_rho = np.empty(N_WRONGPART)
        wp_dist = np.empty(N_WRONGPART, int)
        codes = sn_codes.codes.astype(np.int64)
        for i in range(N_WRONGPART):
            fake = size_matched_partition(codes, rng)
            fc = group_means(E, fake, n_sn)
            _, wp_rho[i], wp_dist[i], _ = match(fc, xen_cent)
        p_rho = float((np.sum(wp_rho >= mean_matched) + 1) / (N_WRONGPART + 1))
        p_dist = float((np.sum(wp_dist >= n_distinct) + 1) / (N_WRONGPART + 1))
        print(
            f"    NULL-2 wrong partition ({N_WRONGPART}x, sizes preserved):", flush=True
        )
        print(
            f"      matched rho   null mean {wp_rho.mean():.4f}  "
            f"observed {mean_matched:.4f}  p={p_rho:.4f}"
            + (
                "   <-- NULL SCORES HIGHER: rho magnitude is NOT evidence"
                if wp_rho.mean() >= mean_matched
                else ""
            ),
            flush=True,
        )
        print(
            f"      distinct hits null mean {wp_dist.mean():.2f}  "
            f"observed {n_distinct}  p={p_dist:.4f}",
            flush=True,
        )

        out_rows.append(
            dict(
                sn_resolution=res,
                arm=arm_name,
                n_cells=len(xi),
                n_sn_clusters=n_sn,
                n_xen_clusters=n_xen,
                mean_matched_rho=mean_matched,
                n_distinct_xen_hit=n_distinct,
                null_genepermute_rho=float(gp.mean()),
                null_wrongpartition_rho=float(wp_rho.mean()),
                p_rho_vs_wrongpartition=p_rho,
                null_wrongpartition_distinct=float(wp_dist.mean()),
                p_distinct_vs_wrongpartition=p_dist,
            )
        )

        tag = f"res {res}, {arm_name}"
        record(
            "B structure",
            f"distinct Xenium clusters hit [{tag}]",
            n_distinct,
            float(wp_dist.mean()),
            f"{N_WRONGPART} size-matched wrong partitions of the snRNA "
            f"nuclei (mean); p={p_dist:.4f}",
            "NOT circular",
            f"{n_sn} snRNA clusters -> {n_xen} Xenium clusters",
        )
        record(
            "B structure",
            f"Hungarian mean centroid rho [{tag}]",
            mean_matched,
            float(wp_rho.mean()),
            f"{N_WRONGPART} size-matched wrong partitions (mean); p={p_rho:.4f}",
            "NOT circular",
            "ABUNDANCE-CONFOUNDED -- do not headline this magnitude",
        )
    return out_rows


# ═══════════════════════════ TEST C -- independent tau specificity ═══
def test_C(S, partitions):
    print("\n" + "=" * 78, flush=True)
    print(
        "TEST C -- gene specificity (Yanai tau) computed separately per modality",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(
        "CIRCULARITY: NOT circular. Each modality's tau is computed against "
        "that modality's OWN partition; the two vectors only meet at the "
        "final rank correlation. Rank-based, so panel gene selection cannot "
        "produce it.",
        flush=True,
    )

    genes, sn, smeta = S["genes"], S["sn"], S["smeta"]
    m = S["xmeta"]
    E = dense(sn.X).astype(np.float32)
    Xa = S["Xarea"]

    def tau_for(expr, labels):
        cat = pd.Categorical(labels)
        cent = group_means(expr, cat.codes.astype(np.int64), len(cat.categories))
        return tau_yanai(cent), len(cat.categories)

    pairings, sn_labels_for = {}, {}
    t_xe_le, k_xe_le = tau_for(Xa, m.leiden.values)
    # (i) the spec: de novo vs de novo, swept over snRNA cluster resolution
    per_gene = pd.DataFrame({"v43": genes})
    for res, lab in partitions.items():
        t, k = tau_for(E, lab)
        nm = f"de novo x de novo (snRNA leiden res {res})"
        pairings[nm] = (t, t_xe_le, k, k_xe_le)
        sn_labels_for[nm] = lab
        per_gene[f"tau_sn_denovo_res{res}"] = t
    # (ii) the prior pairing, reproduced like-for-like
    t_sn_maj, k_sn_maj = tau_for(E, smeta.major_cell_type.values)
    t_sn_fine, k_sn_fine = tau_for(E, smeta.cell_type.values)
    if "leiden_original" in m.columns:
        t_xe_orig, k_xe_orig = tau_for(Xa, m.leiden_original.values)
        nm = "prior MAJOR (snRNA curated x Xenium leiden_original)"
        pairings[nm] = (t_sn_maj, t_xe_orig, k_sn_maj, k_xe_orig)
        sn_labels_for[nm] = smeta.major_cell_type.values
    nm = "prior FINE (snRNA curated fine x Xenium leiden)"
    pairings[nm] = (t_sn_fine, t_xe_le, k_sn_fine, k_xe_le)
    sn_labels_for[nm] = smeta.cell_type.values

    per_gene["tau_sn_curated_major"] = t_sn_maj
    per_gene["tau_sn_curated_fine"] = t_sn_fine
    per_gene["tau_xen_leiden"] = t_xe_le
    if "leiden_original" in m.columns:
        per_gene["tau_xen_leiden_original"] = t_xe_orig
    per_gene.to_csv(OUT / "testC_tau.csv", index=False)

    rows = []
    for name, (a, b, ka, kb) in pairings.items():
        rho, p_perm, nulls = perm_null_spearman(a, b, N_PERM, rng)
        print(f"\n  {name}", flush=True)
        print(
            f"    partitions: snRNA {ka} clusters, Xenium {kb} clusters "
            f"(tau depends on partition cardinality)",
            flush=True,
        )
        print(f"    Spearman rho = {rho:.4f}", flush=True)
        print(
            f"    NULL-1 gene permutation ({N_PERM}x): mean "
            f"{nulls.mean():+.4f}, 95% band "
            f"[{np.percentile(nulls, 2.5):+.4f}, "
            f"{np.percentile(nulls, 97.5):+.4f}], emp p = {p_perm:.4f}",
            flush=True,
        )

        # NULL-2 wrong partition on the snRNA side (cheap, exact size match)
        wp = np.empty(N_WRONGPART)
        codes = pd.Categorical(sn_labels_for[name]).codes.astype(np.int64)
        k = codes.max() + 1
        for i in range(N_WRONGPART):
            fc = group_means(E, size_matched_partition(codes, rng), k)
            wp[i] = spearmanr(tau_yanai(fc), b, nan_policy="omit").statistic
        wp = wp[np.isfinite(wp)]
        p_wp = float((np.sum(wp >= rho) + 1) / (len(wp) + 1))
        print(
            f"    NULL-2 wrong partition ({len(wp)}x, snRNA sizes preserved): "
            f"mean {wp.mean():+.4f}, emp p = {p_wp:.4f}",
            flush=True,
        )
        frac = 100 * wp.mean() / rho if abs(rho) > 1e-9 else float("nan")
        print(
            f"    >>> DELTA over wrong-partition null = {rho - wp.mean():+.4f} "
            f"({frac:.0f}% of the raw rho is reproducible by a random "
            f"partition of the same shape)",
            flush=True,
        )

        rows.append(
            dict(
                pairing=name,
                n_genes=int(np.isfinite(a * b).sum()),
                k_snrna=ka,
                k_xenium=kb,
                spearman_rho=rho,
                null_genepermute_mean=float(nulls.mean()),
                p_genepermute=p_perm,
                null_wrongpartition_mean=float(wp.mean()),
                p_wrongpartition=p_wp,
                delta_over_wrongpartition=rho - float(wp.mean()),
            )
        )
        record(
            "C tau",
            f"Spearman rho [{name}]",
            rho,
            float(wp.mean()),
            f"{len(wp)} size-matched wrong partitions of the snRNA side "
            f"(mean); gene-permutation null {nulls.mean():+.3f}; "
            f"p_wrongpartition={p_wp:.4f}",
            "NOT circular",
            f"delta over wrong-partition null {rho - float(wp.mean()):+.4f}",
        )

    pd.DataFrame(rows).to_csv(OUT / "testC_summary.csv", index=False)
    return rows


# ══════════════════ TEST D -- composition, CLR vs cosine (CIRCULAR) ═══
def test_D(S):
    print("\n" + "=" * 78, flush=True)
    print("TEST D -- composition agreement: cosine vs Aitchison/CLR", flush=True)
    print("=" * 78, flush=True)
    print(
        "CIRCULARITY: *** CIRCULAR ***. The Xenium side carries "
        "obs['tangram_cell_type'], assigned by Tangram from this same "
        "reference. Nothing here is independent evidence that the atlas is "
        "right. It is run because raw-proportion cosine is the wrong metric "
        "for compositional data and contradicts our own scCODA Methods; the "
        "point is metric consistency, not decircularisation.",
        flush=True,
    )

    spec = importlib.util.spec_from_file_location("_coda", CODA_MODULE)
    coda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coda)
    print(
        f"  CLR imported from {CODA_MODULE.relative_to(REPO)} "
        f"(multiplicative zero replacement, delta = 0.65 x min non-zero; "
        f"Martin-Fernandez et al. 2003)",
        flush=True,
    )

    m = S["xmeta"]
    lab = m["cell_type_canonical_CIRCULAR"].values
    in_optic = m.in_optic.values

    print("  loading snRNA dissection obs ...", flush=True)
    cb = S["smeta"][["cell_type"]].copy()  # already in memory
    ol = ad.read_h5ad(SN_OL, backed="r").obs.copy()
    for o in (cb, ol):
        o["canon"] = [get_canonical(v) for v in o["cell_type"].astype(str)]
    print(f"    CB {len(cb):,} nuclei, OL {len(ol):,} nuclei", flush=True)

    cats = list(CANONICAL_ORDER)

    def props(v):
        vc = pd.Series(np.asarray(v)).value_counts().reindex(cats, fill_value=0)
        return vc / max(vc.sum(), 1)

    cohorts = {
        "snRNA central brain": props(cb["canon"].values),
        "snRNA optic lobe": props(ol["canon"].values),
        "Xenium central brain": props(lab[~in_optic]),
        "Xenium optic regions": props(lab[in_optic]),
        "snRNA all": props(np.concatenate([cb["canon"].values, ol["canon"].values])),
        "Xenium all": props(lab),
    }
    wide = pd.DataFrame(cohorts)
    wide.index.name = "cell_type"
    pooled = wide.copy()
    pooled["group"] = [get_pooled(c) for c in pooled.index]
    pooled = pooled.groupby("group").sum().reindex(POOLED_ORDER).fillna(0.0)
    wide.to_csv(OUT / "testD_composition_wide.csv")
    pooled.to_csv(OUT / "testD_composition_pooled.csv")

    PAIRS = [
        ("snRNA all", "Xenium all", "whole reference vs whole atlas"),
        ("snRNA central brain", "Xenium central brain", "dissection-matched (CB)"),
        ("snRNA optic lobe", "Xenium optic regions", "dissection-matched (OL)"),
        ("snRNA optic lobe", "Xenium central brain", "MISMATCHED"),
        ("snRNA central brain", "Xenium optic regions", "MISMATCHED"),
    ]

    rows = []
    for tbl, gran in ((wide, "fine (17)"), (pooled, "pooled (7)")):
        # CLR jointly on the stacked cohort matrix. clr() is row-wise, so
        # stacking changes nothing except that all cohorts share ONE
        # zero-replacement delta -- required for the cohorts to live in the
        # same Aitchison space and be comparable.
        M = tbl.T.values.astype(float)  # cohorts x categories
        CLR = coda.clr(M)
        clr_tbl = pd.DataFrame(CLR.T, index=tbl.index, columns=tbl.columns)
        clr_tbl.to_csv(OUT / f"testD_clr_{gran.split()[0]}.csv")
        nz = float(np.min(M[M > 0]))
        print(
            f"\n  === {gran} === joint CLR over {M.shape[0]} cohorts x "
            f"{M.shape[1]} categories; zeros {(M == 0).sum()}, "
            f"delta = 0.65 x {nz:.3g} = {0.65 * nz:.3g}",
            flush=True,
        )
        for a_name, b_name, note in PAIRS:
            x, y = tbl[a_name].values, tbl[b_name].values
            cx, cy = clr_tbl[a_name].values, clr_tbl[b_name].values
            r = dict(
                granularity=gran,
                comparison=f"{a_name} vs {b_name}",
                note=note,
                cosine_raw=1 - cosine_dist(x, y),
                pearson_raw=pearsonr(x, y)[0],
                spearman_raw=spearmanr(x, y).statistic,
                l1_raw=float(np.abs(x - y).sum()),
                pearson_clr=pearsonr(cx, cy)[0],
                spearman_clr=spearmanr(cx, cy).statistic,
                aitchison_distance=float(np.sqrt(((cx - cy) ** 2).sum())),
            )
            rows.append(r)
            print(
                f"    {note:28s} cosine {r['cosine_raw']:.3f} | "
                f"CLR r {r['pearson_clr']:+.3f} | Aitchison d "
                f"{r['aitchison_distance']:.3f}",
                flush=True,
            )
        s = pd.DataFrame(rows)
        s = s[s.granularity == gran]
        mat_cos = s.iloc[1:3].cosine_raw.mean()
        mis_cos = s.iloc[3:5].cosine_raw.mean()
        mat_d = s.iloc[1:3].aitchison_distance.mean()
        mis_d = s.iloc[3:5].aitchison_distance.mean()
        print(
            f"    DISCRIMINATION cosine  : matched {mat_cos:.3f} vs "
            f"mismatched {mis_cos:.3f} -> "
            f"{'matched better' if mat_cos > mis_cos else 'FAILS'}",
            flush=True,
        )
        print(
            f"    DISCRIMINATION Aitchison: matched d {mat_d:.3f} vs "
            f"mismatched d {mis_d:.3f} -> "
            f"{'matched closer' if mat_d < mis_d else 'FAILS'}",
            flush=True,
        )
        # The two-pair mean can hide a per-pair reversal, so check the pairs
        # individually.
        d = s.iloc[1:5][
            ["comparison", "note", "aitchison_distance", "cosine_raw", "pearson_clr"]
        ]
        worst_matched_d = d.iloc[0:2].aitchison_distance.max()
        best_mismatched_d = d.iloc[2:4].aitchison_distance.min()
        clean = best_mismatched_d > worst_matched_d
        print(
            f"    PER-PAIR: worst dissection-matched d {worst_matched_d:.3f} "
            f"vs best MISMATCHED d {best_mismatched_d:.3f} -> "
            + (
                "every matched pair beats every mismatched pair"
                if clean
                else "*** a MISMATCHED pair is closer than a matched pair: the "
                "discrimination survives only as a two-pair average ***"
            ),
            flush=True,
        )
        record(
            "D composition (CIRCULAR)",
            f"per-pair separation clean? [{gran}]",
            float(clean),
            np.nan,
            "1 = every dissection-matched pair is closer than every "
            "mismatched pair; 0 = at least one reversal",
            "*** CIRCULAR ***",
            f"worst matched d {worst_matched_d:.3f}, best mismatched d "
            f"{best_mismatched_d:.3f}",
        )
        record(
            "D composition (CIRCULAR)",
            f"Aitchison d, dissection-matched mean [{gran}]",
            mat_d,
            mis_d,
            "same statistic on the two MISMATCHED dissection pairs",
            "*** CIRCULAR *** (Tangram labels on the Xenium side)",
            f"cosine on raw proportions gives {mat_cos:.3f} vs {mis_cos:.3f}",
        )
        record(
            "D composition (CIRCULAR)",
            f"CLR-space Pearson, dissection-matched mean [{gran}]",
            float(s.iloc[1:3].pearson_clr.mean()),
            float(s.iloc[3:5].pearson_clr.mean()),
            "same statistic on the two MISMATCHED dissection pairs",
            "*** CIRCULAR ***",
            f"raw-proportion cosine {mat_cos:.3f} vs {mis_cos:.3f}",
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "testD_clr.csv", index=False)
    if not df.empty:
        fine = df[df.granularity == "fine (17)"]
        print(
            f"\n  METRIC DISAGREEMENT: cosine spread across all 5 pairs = "
            f"{fine.cosine_raw.max() - fine.cosine_raw.min():.3f}; "
            f"CLR Pearson spread = "
            f"{fine.pearson_clr.max() - fine.pearson_clr.min():.3f}. Cosine "
            f"on raw proportions compresses the comparison because one "
            f"category dominates.",
            flush=True,
        )
        print(
            "  CAVEAT: at fine (17) granularity, near-zero categories "
            "dominate the Aitchison distance after zero replacement -- the "
            "pooled (7) numbers are the more stable ones.",
            flush=True,
        )
    return df


# ══════════════════════════════════════════════════════════════ main ═══
def main() -> None:
    print("Tangram-independent validation battery", flush=True)
    print(f"outputs -> {OUT}", flush=True)
    S = build_substrate()

    test_A0(S)
    test_A1(S)
    partitions = test_B(S)
    test_C(S, partitions)
    test_D(S)

    summ = pd.DataFrame(SUMMARY)
    summ.to_csv(OUT / "summary.csv", index=False)
    print("\n" + "=" * 78, flush=True)
    print("SUMMARY -- every number with its null and circularity verdict", flush=True)
    print("=" * 78, flush=True)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(summ.to_string(index=False), flush=True)
    with open(OUT / "summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=2, default=float)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
