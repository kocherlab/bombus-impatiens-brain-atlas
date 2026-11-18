"""snRNA UMAPs (central brain + optic lobe) recomputed on the Xenium probe set only.

Asks the question the atlas framing needs answered: does the 101-gene panel
recover the structure the full transcriptome resolves, or does restricting to
the panel collapse the cell types together? Everything downstream in the
spatial data sees only these genes, so this is an upper bound on what the
spatial atlas can resolve.

For each reference atlas we recompute the embedding twice from raw counts --
once on all genes, once on the panel subset -- with an otherwise identical
pipeline, then score how well the panel-only embedding preserves the full
one (neighbourhood overlap + per-cell-type silhouette).

Outputs to results/reframe/sn_panel_umap/:
    {cb,ol}_panel.h5ad          embedding + labels (panel genes only)
    {cb,ol}_plotdata.npz        UMAP coords + labels for BOTH arms (replot cache)
    results/figures/si/figS_sn_panel_umap.png  both atlases, four panels, one legend
    panel_vs_full_metrics.csv   per-atlas / per-cell-type preservation scores
    provenance.json             commit, seed, numeric env, package versions and
                                input/output sha256 of the run that built the cache

Deterministic: seed 0 for PCA, kNN (pynndescent) and UMAP, every BLAS/numba
pool forced to one thread, and numpy/OpenBLAS pinned to their AVX2/Haswell
kernels (see below), so runs on any node give byte-identical outputs: checked
with an AMD 4-CPU job and an Intel 6-CPU job, both of which reproduce
the cache the registered figure was drawn from.

``--replot`` redraws the figure from the npz caches without recomputing any
embedding. Both arms' coordinates are written by the same run that computes the
metrics, so a replot can never pair arms from two different runs.
"""

from __future__ import annotations

import os

# Forced numeric environment (not defaults the caller can override); must be
# set before numpy/scipy/numba import. The all-genes PCA is a BLAS reduction
# over 10,661 scaled genes, and the last bits of its result move the kNN graph
# and so the whole UMAP layout. Measured with the same seed throughout:
#   - threads: one BLAS thread (which built the stored cache) and 8 or 32
#     threads give different layouts, so the count is forced rather than left
#     to the launch environment. One thread also avoids the segfault scipy's
#     OpenBLAS hits on this host.
#   - CPU: on an Intel (AVX-512) node numpy's log1p and OpenBLAS's kernels round
#     differently from the AMD main partition. Pin numpy to its AVX2 kernels
#     and OpenBLAS to core type ZEN (the Haswell kernels the AMD nodes run), so
#     any node reproduces the AMD-built cache.
_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
for _v in _THREAD_VARS:
    os.environ[_v] = "1"
os.environ["NPY_DISABLE_CPU_FEATURES"] = (
    "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL "
    "AVX512_ICL AVX512_SPR"
)
os.environ["OPENBLAS_CORETYPE"] = "ZEN"

import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Text is normalised to the house printed sizes at save time; this script
# reaches neither bimp_st.style.set_pub_style() nor scripts/_si_common,
# so the hook is installed here.
from bimp_st.style import set_pub_style  # noqa: E402

# House figure style (bold titles, no top/right spines, shared base fonts),
# matching the neighbouring figures.
set_pub_style()
from bimp_st.print_scale import install as _install_print_scale  # noqa: E402

_install_print_scale()
from bimp_st.cluster_labels import (  # noqa: E402
    CANONICAL_ORDER,
    add_canonical,
    get_canonical_color,
)
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import PANEL_CROSSWALK  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

REPO = Path(__file__).resolve().parents[2]
SN = REPO / "data" / "sn"
PANEL_CSV = PANEL_CROSSWALK
OUT = REPO / "results" / "reframe" / "sn_panel_umap"
OUT.mkdir(parents=True, exist_ok=True)

ATLASES = {
    "cb": ("BIMP_atlas_qw_CentralBrain.h5ad", "Central brain"),
    "ol": ("BIMP_atlas_qw_OpticLobe.h5ad", "Optic lobe"),
}

N_PCS_FULL = 50
N_NEIGHBORS = 15
SEED = 0


def embed(adata, n_pcs: int, tag: str):
    """Normalize -> log1p -> scale -> PCA -> neighbours -> UMAP, in
    place."""
    # Drop everything inherited from the source file, or the "full vs panel"
    # comparison silently becomes "the file's embedding vs itself".
    for k in ("X_pca", "X_pca_harmony", "X_umap"):
        if k in adata.obsm:
            del adata.obsm[k]

    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    # The reference h5ads carry a `highly_variable` column from their own
    # processing, and scanpy's pca() silently honours it -- on the panel subset
    # that leaves only 45 of the 101 genes and the comparison stops being
    # "all genes vs the panel". Drop the mask so PCA sees every gene given.
    if "highly_variable" in adata.var.columns:
        del adata.var["highly_variable"]

    n_pcs = max(2, min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))
    solver = "arpack" if n_pcs < min(adata.n_obs, adata.n_vars) - 1 else "full"
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver=solver, random_state=SEED)

    # Uncorrected PCA, by design: the figure shows raw PCA (no Harmony on
    # `sample`), and harmonypy is not a dependency. (scanpy's harmony_integrate
    # also mis-transposes the matrix for harmonypy 2.0.)
    rep = "X_pca"
    adata.uns["reframe_rep"] = rep
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep=rep, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    print(
        f"    [{tag}] {adata.n_obs:,} x {adata.n_vars} genes, {n_pcs} PCs, rep={rep}",
        flush=True,
    )
    return rep


def _rep(ad) -> np.ndarray:
    """The embedding this script computed, never one inherited from the file.

    The reference h5ads ship with their own X_pca / X_pca_harmony / X_umap.
    Subsetting to the panel copies them across, so a naive
    "use X_pca_harmony if present" rule silently compares the file's original
    embedding against itself (kNN overlap 1.000, identical silhouettes).
    embed() records which key it wrote in uns["reframe_rep"]; trust only that.
    """
    key = ad.uns.get("reframe_rep")
    if key is None:
        raise KeyError(
            "embed() has not run on this object; refusing to fall back "
            "to an inherited embedding"
        )
    return ad.obsm[key]


def _blas_limit(n: int = 16):
    """Hold BLAS to n threads for a block, if threadpoolctl is installed.

    Belt to the n_jobs=1 braces below: threadpoolctl clamps the already-loaded
    OpenBLAS at runtime, which the environment variables cannot do once the
    library is mapped. A no-op when threadpoolctl is absent.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        from contextlib import nullcontext

        return nullcontext()
    return threadpool_limits(limits=n, user_api="blas")


def knn_overlap_arr(A: np.ndarray, B: np.ndarray, k: int = N_NEIGHBORS) -> float:
    """Mean fraction of each row's kNN in A that is retained in B (1 = identical).

    Takes embeddings rather than AnnData: subsetting an AnnData to a cell type
    copies the whole expression matrix (53k x 10.6k floats per call), while
    only obsm is ever read.
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(A) - 1)
    if k < 2:
        return np.nan

    def nn(X):
        # n_jobs=1: this is what segfaults otherwise. scipy's bundled OpenBLAS
        # is compiled for fewer threads than this host has cores, and joblib
        # fanning kneighbors out over every core makes it exceed that limit --
        # it warns ("precompiled NUM_THREADS exceeded") and then dies in
        # libscipy_openblas rather than raising. Capping OPENBLAS_NUM_THREADS
        # alone does NOT prevent it: the limit that matters is the number of
        # concurrent callers into BLAS, not BLAS's own internal thread count.
        m = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(X)
        return m.kneighbors(X, return_distance=False)[:, 1:]

    with _blas_limit():
        ia, ib = nn(A), nn(B)
    return float(
        np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib, strict=False)])
    )


def knn_overlap(a, b, k: int = N_NEIGHBORS) -> float:
    return knn_overlap_arr(_rep(a), _rep(b), k)


def silhouette_by_type(adata, label: str) -> float:
    from sklearn.metrics import silhouette_score

    rep = adata.uns["reframe_rep"]
    y = adata.obs[label].astype(str).values
    if len(np.unique(y)) < 2:
        return np.nan
    idx = np.random.default_rng(SEED).choice(
        adata.n_obs, size=min(10_000, adata.n_obs), replace=False
    )
    return float(silhouette_score(adata.obsm[rep][idx], y[idx]))


def panel_of(adata, panel_ids: set[str]) -> list[str]:
    return [g for g in adata.var_names if g in panel_ids]


def plotdata(full, panel, nice: str, label: str = "cell_type_canonical") -> dict:
    """The minimum a redraw needs, so the two big AnnDatas can be freed after."""
    return dict(
        name=nice,
        umap_full=full.obsm["X_umap"].astype(np.float32),
        umap_panel=panel.obsm["X_umap"].astype(np.float32),
        labels=full.obs[label].astype(str).values.astype("U40"),
        n_genes_full=full.n_vars,
        n_genes_panel=panel.n_vars,
        n_obs=full.n_obs,
    )


def draw_combined(data: dict[str, dict], path: Path) -> None:
    """Both atlases as one four-panel strip under a single shared legend.

    One question asked of two atlases, so one figure (two stacked full-width
    blocks would take ~310 pt of a ~500 pt SI page).

    The atlases' label sets differ (central brain 15 fine types, optic lobe 8)
    but their union is exactly the canonical 17, so one legend in
    CANONICAL_ORDER covers both without inventing a vocabulary. Per-atlas type
    counts go in the group titles, so a reader does not hunt the optic-lobe
    panels for the nine entries that exist only in the central brain.
    """
    keys = [k for k in ("cb", "ol") if k in data]
    # One pass over CANONICAL_ORDER, not one per atlas: iterating per atlas would
    # append the optic lobe's generic "Kenyon cells" after every central-brain
    # entry, so the legend would not be the canonical 17 in order.
    present = {
        c for k in keys for c in CANONICAL_ORDER if (data[k]["labels"] == c).sum()
    }
    cats = [c for c in CANONICAL_ORDER if c in present]
    colors = {c: get_canonical_color(c) for c in cats}

    # Point size per atlas: at a quarter-width panel a single fixed size
    # over-inks the 53k-nucleus central brain and under-inks the 20k-nucleus optic
    # lobe.
    psize = {"cb": 0.5, "ol": 0.85}

    # Sized a little larger than the 6.69in it prints at, keeping the 9pt
    # legend legible after scaling (every font here divides by 1.5).
    panel_w, panel_h, gap_w, legend_h = 2.4, 2.0, 0.35, 0.62
    fig = plt.figure(figsize=(4 * panel_w + gap_w, panel_h + legend_h))
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[panel_w, panel_w, gap_w, panel_w, panel_w],
        height_ratios=[panel_h, legend_h],
        wspace=0.06,
        hspace=0.0,
        left=0.005,
        right=0.995,
        bottom=0.01,
        top=0.83,
    )
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis("off")
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])

    group_axes = {}
    for k, cols in zip(keys, ((0, 1), (3, 4)), strict=False):
        d = data[k]
        for col, (arm, sub) in zip(
            cols,
            (("umap_full", "All genes"), ("umap_panel", "Panel genes only")),
            strict=False,
        ):
            ax = fig.add_subplot(gs[0, col])
            xy, lab = d[arm], d["labels"]
            for c in cats:
                m = lab == c
                if m.sum():
                    ax.scatter(
                        xy[m, 0],
                        xy[m, 1],
                        s=psize.get(k, 1.2),
                        c=colors[c],
                        linewidths=0,
                        rasterized=True,
                    )
            ax.set_title(sub, fontsize=8, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            # No equal aspect: UMAP axes carry no units, and at four panels across,
            # letterboxing the wide optic-lobe embeddings to match the taller
            # central-brain ones would waste height.
            for s in ax.spines.values():
                s.set_visible(False)
            group_axes.setdefault(k, []).append(ax)

    # No group titles: the caption states each atlas's nuclei, cell-type and
    # gene counts (house rule: in-figure titles are short identity labels only).
    fig.canvas.draw()

    # Panel letters D (central brain) and E (optic lobe), matching the
    # caption; each sits at the left of that atlas's first embedding.
    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [
            (group_axes[k][0], letter, tag)
            for k, letter, tag in zip(
                keys, "DE", ("Central brain", "Optic lobe"), strict=False
            )
        ],
        width_frac=1.0,
        dy_pts=17,
    )

    h = [plt.Line2D([], [], marker="o", ls="", ms=5, color=colors[c]) for c in cats]
    ax_legend.legend(
        h,
        cats,
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=9,
        handletextpad=0.35,
        columnspacing=1.4,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    # No on-figure suptitle: the LaTeX caption already carries the figure
    # title (house rule -- captions own titles, the artwork does not).
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}", flush=True)


FIG = REPO / "results" / "figures" / "si" / "figS_sn_panel_umap.png"


def write_provenance() -> None:
    """provenance.json beside the cache: what built it, from what, how seeded.

    The cache is the embedding the registered figure draws; without this
    record a recomputed cache is indistinguishable from the stored one.
    """
    import hashlib
    import importlib.metadata as md
    import json
    import subprocess

    def sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    rec = {
        "script": "scripts/atlas/si_snrna_panel_umap.py",
        "commit": commit,
        "seed": SEED,
        "numeric_env": {
            v: os.environ.get(v)
            for v in (*_THREAD_VARS, "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE")
        },
        "versions": {
            p: md.version(p)
            for p in ("scanpy", "anndata", "umap-learn", "pynndescent", "numba")
            + ("numpy", "scipy", "scikit-learn")
        },
        "inputs": {
            str((SN / f).relative_to(REPO)): sha256(SN / f)
            for f, _ in ATLASES.values()
        },
        "outputs": {
            p.name: sha256(p)
            for p in sorted(OUT.glob("*_plotdata.npz"))
            + [OUT / "panel_vs_full_metrics.csv"]
            if p.exists()
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {OUT / 'provenance.json'}", flush=True)


def _load_cache(key: str) -> dict:
    z = np.load(OUT / f"{key}_plotdata.npz", allow_pickle=False)
    return dict(
        name=str(z["name"]),
        umap_full=z["umap_full"],
        umap_panel=z["umap_panel"],
        labels=z["labels"],
        n_genes_full=int(z["n_genes_full"]),
        n_genes_panel=int(z["n_genes_panel"]),
        n_obs=int(z["n_obs"]),
    )


def replot() -> None:
    data = {}
    for key in ATLASES:
        if not (OUT / f"{key}_plotdata.npz").exists():
            raise SystemExit(f"no replot cache for {key}; run without --replot first")
        data[key] = _load_cache(key)
        print(f"  loaded {key}_plotdata.npz: {data[key]['n_obs']:,} nuclei", flush=True)
    draw_combined(data, FIG)


def main(embed_only: bool = False, resume: bool = False) -> None:
    panel_ids = set(read_panel_crosswalk()["v43_id"].dropna())
    print(f"panel: {len(panel_ids)} v4.3 gene ids", flush=True)

    rows = []
    data: dict[str, dict] = {}
    for key, (fname, nice) in ATLASES.items():
        print(f"\n=== {nice} ({fname}) ===", flush=True)
        if resume and (OUT / f"{key}_plotdata.npz").exists():
            data[key] = _load_cache(key)
            print(
                f"  resuming from {key}_plotdata.npz "
                f"({data[key]['n_obs']:,} nuclei); not re-embedding",
                flush=True,
            )
            continue
        full = sc.read_h5ad(SN / fname)
        add_canonical(full, src="cell_type")

        keep = panel_of(full, panel_ids)
        print(f"  panel genes present: {len(keep)} / {len(panel_ids)}", flush=True)

        panel = full[:, keep].copy()

        print("  embedding on all genes...", flush=True)
        embed(full, N_PCS_FULL, f"{key}/full")
        print("  embedding on panel genes only...", flush=True)
        embed(panel, min(N_PCS_FULL, len(keep) - 1), f"{key}/panel")

        # Guard: if the two arms produced the same embedding, something has
        # leaked between them and every metric below is meaningless.
        Rf0, Rp0 = _rep(full), _rep(panel)
        if Rf0.shape == Rp0.shape and np.allclose(Rf0, Rp0):
            raise SystemExit(
                f"[{key}] the full and panel embeddings are identical -- the arms "
                "have leaked into each other; refusing to report a fake result"
            )

        # Cache the coordinates BEFORE the metrics. The embeddings are the
        # expensive half (tens of minutes); the kNN sweep below is the half that
        # crashes. Writing here means a crash costs the metrics, not the run.
        data[key] = plotdata(full, panel, nice)
        np.savez_compressed(OUT / f"{key}_plotdata.npz", **data[key])

        if embed_only:
            print("  --embed-only: skipping the metric sweep", flush=True)
            panel.write_h5ad(OUT / f"{key}_panel.h5ad")
            del full, panel
            continue

        ov = knn_overlap(full, panel)
        s_full = silhouette_by_type(full, "cell_type_canonical")
        s_panel = silhouette_by_type(panel, "cell_type_canonical")
        print(
            f"  kNN overlap={ov:.3f}  silhouette full={s_full:.3f} panel={s_panel:.3f}",
            flush=True,
        )

        rows.append(
            dict(
                atlas=key,
                name=nice,
                n_cells=full.n_obs,
                n_genes_full=full.n_vars,
                n_genes_panel=panel.n_vars,
                knn_overlap=ov,
                silhouette_full=s_full,
                silhouette_panel=s_panel,
            )
        )

        # per-cell-type: is this type still separable on the panel alone?
        Rf, Rp = _rep(full), _rep(panel)
        for ct in panel.obs["cell_type_canonical"].astype(str).unique():
            m = (panel.obs["cell_type_canonical"].astype(str) == ct).values
            if m.sum() < 50:
                continue
            rows.append(
                dict(
                    atlas=key,
                    name=nice,
                    cell_type=ct,
                    n_cells=int(m.sum()),
                    knn_overlap=knn_overlap_arr(Rf[m], Rp[m]),
                )
            )
            print(
                f"    {ct:34s} n={int(m.sum()):6d}  "
                f"overlap={rows[-1]['knn_overlap']:.3f}",
                flush=True,
            )

        panel.write_h5ad(OUT / f"{key}_panel.h5ad")
        del full, panel

    draw_combined(data, FIG)

    # Only rewrite the metrics CSV when this run actually measured every atlas.
    # A partial run (--embed-only, or --resume skipping an atlas) would
    # otherwise replace a complete table with a fragment of one.
    if len(rows) and not embed_only and not resume:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "panel_vs_full_metrics.csv", index=False)
        print(f"\nwrote {OUT / 'panel_vs_full_metrics.csv'}", flush=True)
        print(df.to_string(index=False), flush=True)
    else:
        print("\npartial run: leaving panel_vs_full_metrics.csv untouched", flush=True)
    # Same guard: a --resume run did not build the caches it would be stamping.
    if not resume:
        write_provenance()


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        if "--replot" in sys.argv:
            replot()
        else:
            main(embed_only="--embed-only" in sys.argv, resume="--resume" in sys.argv)
