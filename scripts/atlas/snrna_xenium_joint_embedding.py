"""snRNA vs Xenium on the same genes: separate UMAPs and a joint embedding.

The point of restricting the snRNA reference to the Xenium probe set is to make
the two modalities directly comparable -- same genes, same pipeline, so any
difference in structure is modality and not gene content. This does that:

  A  snRNA (CB + OL) embedded on the shared panel genes, by canonical cell type
  B  Xenium embedded on the same genes, same pipeline, by canonical cell type
  C  joint embedding of both, coloured by modality -- do they overlap?
  D  the same joint embedding, coloured by cell type

Panel C is the "overlapping of UMAPs" question. If the modalities form separate
islands, the shared gene space is not shared in practice and cross-modal
comparison is doing more work than it looks like.

Mixing is quantified rather than eyeballed: for each cell, the fraction of its k
nearest joint-embedding neighbours drawn from the *other* modality. 0 = fully
separated, ~0.5 = fully mixed at equal representation. Reported per cell type,
because a global number hides which types integrate and which do not.

Xenium is subsampled (cells are 18x more numerous than nuclei) so the joint
embedding is not dominated by one modality; the subsample is stratified by
canonical cell type and the rate is reported.

Outputs to results/reframe/snuc_vs_xenium/:
    snuc_vs_xenium_umap.png
    mixing_by_cell_type.csv
    joint_embedding.h5ad
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import (  # noqa: E402
    add_canonical,
    get_canonical_color,
)
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import PANEL_CROSSWALK  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

REPO = Path(__file__).resolve().parents[2]
SN = REPO / "data" / "sn"
SPATIAL = (
    REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
)
PANEL_CSV = PANEL_CROSSWALK
OUT = REPO / "results" / "reframe" / "snuc_vs_xenium"
OUT.mkdir(parents=True, exist_ok=True)

N_XENIUM = 80_000  # stratified subsample so neither modality dominates
N_PCS = 40
N_NEIGHBORS = 15
SEED = 0
LABEL = "cell_type_canonical"


def prep(adata, tag: str):
    """counts -> CPM -> log1p -> scale, with nothing inherited from the file."""
    for k in ("X_pca", "X_pca_harmony", "X_umap"):
        if k in adata.obsm:
            del adata.obsm[k]
    if "highly_variable" in adata.var.columns:
        del adata.var["highly_variable"]

    # Do not normalise data already on a log scale. The Xenium object has no
    # layers and its X is already log1p-normalised, so a plain "counts if present
    # else X" guard would normalise it twice, which alone drives the modalities
    # apart.
    def _is_log(M):
        d = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        nz = d[d > 0]
        return nz.size > 0 and not np.allclose(nz[:100_000], np.round(nz[:100_000]))

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"    [{tag}] normalised from counts layer", flush=True)
    elif _is_log(adata.X):
        print(f"    [{tag}] X already log-normalised -- left as is", flush=True)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"    [{tag}] normalised from raw X", flush=True)
    print(f"    [{tag}] {adata.n_obs:,} x {adata.n_vars}", flush=True)
    return adata


def embed(adata, tag: str):
    sc.pp.scale(adata, max_value=10)
    n = max(2, min(N_PCS, adata.n_vars - 1, adata.n_obs - 1))
    sc.tl.pca(adata, n_comps=n, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca", random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    print(f"    [{tag}] embedded, {n} PCs", flush=True)
    return adata


def draw(ax, xy, labels, palette, title, legend=False, order=True):
    idx = (
        np.random.default_rng(SEED).permutation(len(xy))
        if order
        else np.arange(len(xy))
    )
    cols = np.array([palette.get(v, "#BBBBBB") for v in labels])
    ax.scatter(
        xy[idx, 0], xy[idx, 1], c=cols[idx], s=1.4, linewidths=0, rasterized=True
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if legend:
        seen = [v for v in palette if v in set(labels)]
        h = [
            plt.Line2D([], [], marker="o", ls="", ms=6, color=palette[v]) for v in seen
        ]
        ax.legend(
            h,
            seen,
            frameon=False,
            fontsize=7.5,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )


def main() -> None:
    panel = read_panel_crosswalk()
    v43_ids = set(panel["v43_id"].dropna())
    np31_to_v43 = dict(zip(panel["np31_id"], panel["v43_id"], strict=True))

    print("loading snRNA CB + OL ...", flush=True)
    cb = sc.read_h5ad(SN / "BIMP_atlas_qw_CentralBrain.h5ad")
    ol = sc.read_h5ad(SN / "BIMP_atlas_qw_OpticLobe.h5ad")
    for a in (cb, ol):
        add_canonical(a, src="cell_type")
    sn_genes = [g for g in cb.var_names if g in v43_ids and g in set(ol.var_names)]
    sn = ad.concat(
        [cb[:, sn_genes], ol[:, sn_genes]],
        label="atlas",
        keys=["CB", "OL"],
        join="outer",
    )
    del cb, ol
    print(f"  snRNA: {sn.n_obs:,} nuclei, {sn.n_vars} panel genes", flush=True)

    print("loading Xenium ...", flush=True)
    sp = sc.read_h5ad(SPATIAL)
    add_canonical(sp, src=("tangram_cell_type", "cell_type"))
    keep, newnames = [], []
    for name in sp.var_names:
        v = np31_to_v43.get(str(name).split("|")[0])
        if v is not None and v in set(sn_genes):
            keep.append(name)
            newnames.append(v)
    sp = sp[:, keep].copy()
    sp.var_names = newnames
    sp = sp[:, ~pd.Index(sp.var_names).duplicated()].copy()

    shared = [g for g in sn_genes if g in set(sp.var_names)]
    sn = sn[:, shared].copy()
    sp = sp[:, shared].copy()
    print(f"  shared gene space: {len(shared)} genes", flush=True)

    # stratified subsample of Xenium so the joint embedding is not swamped
    rng = np.random.default_rng(SEED)
    lab = sp.obs[LABEL].astype(str).values
    take = []
    for ct in np.unique(lab):
        idx = np.flatnonzero(lab == ct)
        n = max(1, round(N_XENIUM * len(idx) / len(lab)))
        take.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    take = np.sort(np.concatenate(take))
    print(
        f"  Xenium subsampled {len(take):,} / {sp.n_obs:,} cells "
        f"({100 * len(take) / sp.n_obs:.1f}%), stratified by cell type",
        flush=True,
    )
    sp_sub = sp[take].copy()
    del sp

    print("normalizing both on the shared genes ...", flush=True)
    sn = prep(sn, "snRNA")
    sp_sub = prep(sp_sub, "Xenium")

    # ── separate embeddings ─────────────────────────────────────────────
    # CB and OL are shown separately as well as jointly. The reference was built
    # as two atlases (fig:atlas_overview panel E shows them that way), so
    # collapsing them into one
    # panel hides whether the two dissections are internally consistent and
    # whether the shared non-neuronal types land together when they are combined.
    print("embedding separately ...", flush=True)
    cb_solo = embed(sn[sn.obs["atlas"] == "CB"].copy(), "snRNA CB solo")
    ol_solo = embed(sn[sn.obs["atlas"] == "OL"].copy(), "snRNA OL solo")
    sn_solo = embed(sn.copy(), "snRNA CB+OL solo")
    sp_solo = embed(sp_sub.copy(), "Xenium solo")

    # ── joint embedding ─────────────────────────────────────────────────
    print("embedding jointly ...", flush=True)
    joint = ad.concat(
        [sn, sp_sub], label="modality", keys=["snRNA", "Xenium"], join="outer"
    )
    # The count layers are ~150k x 100 apiece and nothing below reads them; drop
    # them to keep memory down.
    joint.layers.clear()
    joint = embed(joint, "joint")

    # ── mixing, uncorrected and after integration ───────────────────────
    from sklearn.neighbors import NearestNeighbors

    mod = joint.obs["modality"].astype(str).values
    jl = joint.obs[LABEL].astype(str).values
    frac_sn = (mod == "snRNA").mean()
    ideal = 2 * frac_sn * (1 - frac_sn)  # expected if perfectly mixed

    def mixing(rep: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(rep)
        nbr = nn.kneighbors(rep, return_distance=False)[:, 1:]
        return (mod[nbr] != mod[:, None]).mean(1)

    print("scoring cross-modal mixing (uncorrected) ...", flush=True)
    other = mixing(joint.obsm["X_pca"])
    joint.obs["frac_other_modality"] = other
    print(
        f"  global mixing = {other.mean():.3f} "
        f"(perfect mixing at this ratio = {ideal:.3f})",
        flush=True,
    )

    # Uncorrected separation alone cannot distinguish "these modalities measure
    # different biology" from "these are two platforms with different capture
    # characteristics and nobody corrected for it". Harmony on modality answers
    # the second, so run it and report both numbers side by side.
    other_h = None
    try:
        # Call harmonypy directly rather than sc.external.pp.harmony_integrate.
        # harmonypy 2.0 returns Z_corr already oriented (n_cells, n_pcs), while
        # scanpy's wrapper transposes it expecting the older (n_pcs, n_cells)
        # convention -- it stores a (40,) array into obsm and the assignment
        # fails. Harmony itself converges fine; only the wrapper is wrong.
        import harmonypy

        ho = harmonypy.run_harmony(
            joint.obsm["X_pca"], joint.obs, ["modality"], max_iter_harmony=25
        )
        Z = np.asarray(ho.Z_corr)
        if Z.shape != joint.obsm["X_pca"].shape:  # tolerate either orientation
            Z = Z.T
        if Z.shape != joint.obsm["X_pca"].shape:
            raise RuntimeError(
                f"harmony returned {Z.shape}, expected {joint.obsm['X_pca'].shape}"
            )
        joint.obsm["X_pca_harmony"] = Z

        print("scoring cross-modal mixing (Harmony-integrated) ...", flush=True)
        other_h = mixing(joint.obsm["X_pca_harmony"])
        joint.obs["frac_other_modality_harmony"] = other_h
        print(
            f"  integrated mixing = {other_h.mean():.3f} (perfect = {ideal:.3f})",
            flush=True,
        )
        sc.pp.neighbors(
            joint,
            n_neighbors=N_NEIGHBORS,
            use_rep="X_pca_harmony",
            key_added="harmony",
            random_state=SEED,
        )
        sc.tl.umap(joint, neighbors_key="harmony", random_state=SEED)
        joint.obsm["X_umap_harmony"] = joint.obsm["X_umap"].copy()
    except Exception as exc:
        # Report the failure loudly, so a missing integrated column is not mistaken
        # for a choice not to run Harmony.
        print(f"  !! HARMONY FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(
            "  !! reporting the UNCORRECTED comparison only -- the integrated "
            "arm is MISSING from the outputs, do not read this as 'no "
            "integration was needed'",
            flush=True,
        )

    rows = []
    for ct in np.unique(jl):
        m = jl == ct
        if m.sum() < 100:
            continue
        r = dict(
            cell_type=ct,
            n=int(m.sum()),
            n_snrna=int((m & (mod == "snRNA")).sum()),
            n_xenium=int((m & (mod == "Xenium")).sum()),
            mixing_uncorrected=float(other[m].mean()),
        )
        if other_h is not None:
            r["mixing_harmony"] = float(other_h[m].mean())
        rows.append(r)
    mix = pd.DataFrame(rows).sort_values(
        "mixing_harmony" if other_h is not None else "mixing_uncorrected",
        ascending=False,
    )
    # `ideal` (2*f*(1-f)) is the global ceiling at the overall snRNA:Xenium
    # ratio. Within a cell type the ratio can differ greatly (Other Kenyon cells:
    # 1441 snRNA nuclei vs 24 Xenium cells, so at most ~6% cross-modal neighbours),
    # so each row uses its own snRNA:Xenium ratio and mixing is reported relative
    # to that row's ceiling.
    f_ct = mix["n_snrna"] / mix["n"]
    mix["perfect_mixing"] = 2 * f_ct * (1 - f_ct)
    primary = "mixing_harmony" if other_h is not None else "mixing_uncorrected"
    mix["frac_of_own_ceiling"] = (mix[primary] / mix["perfect_mixing"]).where(
        mix["perfect_mixing"] > 0
    )
    mix.to_csv(OUT / "mixing_by_cell_type.csv", index=False)
    print(mix.to_string(index=False), flush=True)

    # ── figure ──────────────────────────────────────────────────────────
    cats = [
        c for c in joint.obs[LABEL].cat.categories if (joint.obs[LABEL] == c).sum() > 0
    ]
    pal = {c: get_canonical_color(c) for c in cats}
    mod_pal = {"snRNA": "#5B4C7C", "Xenium": "#B06715"}

    mix_txt = f"mixing = {other.mean():.3f} (perfect = {ideal:.3f})"
    if other_h is not None:
        mix_txt += f"\nafter Harmony = {other_h.mean():.3f}"

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12))
    draw(
        axes[0, 0],
        cb_solo.obsm["X_umap"],
        cb_solo.obs[LABEL].astype(str).values,
        pal,
        f"A  snRNA central brain ({cb_solo.n_obs:,} nuclei)",
    )
    draw(
        axes[0, 1],
        ol_solo.obsm["X_umap"],
        ol_solo.obs[LABEL].astype(str).values,
        pal,
        f"B  snRNA optic lobe ({ol_solo.n_obs:,} nuclei)",
    )
    draw(
        axes[0, 2],
        sn_solo.obsm["X_umap"],
        sn_solo.obs[LABEL].astype(str).values,
        pal,
        f"C  snRNA, both atlases together ({sn_solo.n_obs:,})",
        legend=True,
    )
    draw(
        axes[1, 0],
        sp_solo.obsm["X_umap"],
        sp_solo.obs[LABEL].astype(str).values,
        pal,
        f"D  Xenium, same {len(shared)} genes ({sp_solo.n_obs:,} cells)",
    )
    draw(
        axes[1, 1],
        joint.obsm["X_umap"],
        mod,
        mod_pal,
        f"E  snRNA + Xenium jointly, by modality\n{mix_txt}",
        legend=True,
    )
    draw(
        axes[1, 2],
        joint.obsm["X_umap"],
        jl,
        pal,
        "F  The same joint embedding, by cell type",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"snuc_vs_xenium_umap.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Persist only what downstream work needs (two embeddings and two labels),
    # not the full joint object with both modalities' count layers (~270 MB).
    slim = pd.DataFrame(
        {
            "modality": mod,
            LABEL: jl,
            "frac_other_modality": other,
            "umap_1": joint.obsm["X_umap"][:, 0],
            "umap_2": joint.obsm["X_umap"][:, 1],
        }
    )
    if other_h is not None:
        slim["frac_other_modality_harmony"] = other_h
    slim.to_csv(OUT / "joint_embedding_cells.csv.gz", index=False, compression="gzip")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
