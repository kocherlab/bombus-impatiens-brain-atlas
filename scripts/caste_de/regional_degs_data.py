"""Data for si_regional_degs.py:
loading, computation and the constants they use.

See si_regional_degs.py (the composer) for the figure and how to render it.
"""
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import (
    chi2_contingency,
    false_discovery_control,
    kruskal,
    mannwhitneyu,
)
from sklearn.decomposition import PCA

from bimp_st.cluster_labels import (
    POOLED_ORDER,
    get_pooled,
    hierarchical_vote,
)
from bimp_st.nine_region_common import (
    STATISTICAL_REGIONS,
)
from bimp_st.paths import ALIGNED_QUEEN, ALIGNED_WORKER, CANONICAL_TANGRAM


def _composition_labels(adata) -> np.ndarray:
    """Per-cell pooled cell-type labels for the region-conditioned composition panel.

    (1) The raw per-cell ``tangram_cell_type`` argmax is panel-unresolvable
        noise and must
        never be used as a final label -- read directly, it puts ``OL_8``/
        ``OL_9`` and ``Hem`` into these composition bars. Labels must come
        from a per-cluster vote over the combined Tangram run.

    (2) The vote is the canonical one, ``hierarchical_vote`` with the same
        arguments as ``caste_comparison_sccoda.py::_display_ct_series`` and
        `fig:atlas_overview`, so every
        figure recomputes its spatial labels with one voting function. It
        reads expression only, never the ``in_*`` flags this panel conditions
        on (``anatomy_guided_vote`` would make "Kenyon cells dominate the KC
        regions" true by construction). Its labels are then mapped to the
        pooled groups (``get_pooled``) because this panel and its table are
        drawn in the coarse POOLED_ORDER vocabulary: the glial subtypes
        ``hierarchical_vote`` names separately are counted as "Glia" here.

    Recomputed on every call (cheap groupby) so a stale persisted vote can
    never leak through. Clusters the vote leaves undecided read as "Mixed" and
    are kept visible -- dropping them would silently renormalize every other
    proportion in the panel.
    """
    hierarchical_vote(
        adata,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        dst="cluster_vote_hier",
    )
    pooled = np.array(
        [get_pooled(v) for v in adata.obs["cluster_vote_hier"].astype(str)],
        dtype=object,
    )
    unexpected = set(pooled) - set(POOLED_ORDER) - {"Mixed"}
    assert not unexpected, f"labels outside the pooled vocabulary: {unexpected}"
    return pooled


def _pooled_label_order(labels) -> list[str]:
    """Pooled groups present in *labels*, in canonical POOLED_ORDER (+ Mixed)."""
    present = {str(x) for x in labels}
    ordered = [g for g in POOLED_ORDER if g in present]
    return ordered + sorted(present - set(ordered))


# ── Configuration ──────────────────────────────────────────────────────────
BRAINS = {
    "queen": str(ALIGNED_QUEEN),
    "worker": str(ALIGNED_WORKER),
}
# Combined h5ad with unified Tangram cell type assignments. The per-brain
# h5ads each have their own independent Leiden numbering — those IDs are not
# comparable across brains and must never be used as cell type labels. Cell
# types are always derived from this file via ``_composition_labels`` (combined
# Tangram + the canonical hierarchical_vote, pooled), never from raw per-cell
# argmax.
## Must be the column-corrected Tangram object. Only ``composition_by_region``
## reads it, so only the overview figure's composition panels (B and its caste
## contrast C) depend on the Tangram names; the volcano/lollipop/interaction
## figures are per-brain pseudobulk expression. The per-cluster vote is
## recomputed from this object's ``tangram_cell_type`` on every call.
COMBINED_TANGRAM_H5AD = str(CANONICAL_TANGRAM)
# BIMP_REGIONAL_OUT redirects every table and the analysis figures, and
# BIMP_REGIONAL_FIG_OUT the registered volcano figure, so a check render
# never overwrites the registered files.
OUTPUT_DIR = Path(os.environ.get("BIMP_REGIONAL_OUT", "results/regional_degs"))
FIG_DIR = Path(os.environ.get("BIMP_REGIONAL_FIG_OUT", "results/figures/si"))
#: The per-region caste effect tables: log2FC and the slice consistency
#: (rank-biserial r of the queen vs worker section means; no P value, one
#: brain per caste). Written by ``--only degs`` or a full run.
EFFECTS_DIR = OUTPUT_DIR / "caste_effects"
CLUSTER_COL = "leiden"
MIN_SLICES = 3  # min slices per brain per region for pseudobulk
MIN_CELLS_PER_SLICE = 5  # min cells to form a pseudobulk sample

# Regions present in both brains (for queen-vs-worker DEG comparison).
# Seven regions: the original six anatomical regions plus the antennal lobe
# (comparable in size to the lobula), so we get a full
# per-region volcano grid (3 optic-lobe + 3 calyx + AL). The two other new
# regions -- ellipsoid_body and fan_shaped_body, the central-complex
# substructures -- are deliberately excluded from every fit in this script:
# at their very low prevalence they would
# degrade the estimates for the other seven regions in any pseudobulk/
# multinomial fit, not merely be unreliable themselves. They are
# nine_region_common.DESCRIPTIVE_ONLY_REGIONS; see si_percell_vs_vote.py /
# si_tangram_confidence.py for their descriptive-only treatment. Kept as
# an explicit local list (not STATISTICAL_REGIONS's order) so the KC-IC/
# KC-NC/KC-OC ordering this file's figures document is preserved.
COMMON_REGIONS = [
    "lamina",
    "medulla",
    "lobula",
    "kc_ic",
    "kc_nc",
    "kc_oc",
    "al",
]
assert set(COMMON_REGIONS) == set(STATISTICAL_REGIONS), (
    "COMMON_REGIONS has drifted from nine_region_common.STATISTICAL_REGIONS"
)
# All hand-annotated regions this script fits. An alias of COMMON_REGIONS
# (same list, same order) rather than a second copy, so the two cannot end up
# in different orders (STATISTICAL_REGIONS orders kc_oc/kc_ic/kc_nc; this
# file's figures use COMMON_REGIONS's order).
ALL_REGIONS = list(COMMON_REGIONS)

# Regions a queen-versus-worker significance test may be run on. Every caste
# contrast in the spatial arm is pseudoreplicated (one brain per caste). The
# antennal lobe is additionally confounded: it is traced on every queen section
# but not on every worker section, and unevenly between the worker's two
# slides (0069041 contributes a few small sections, 0068401 most of the
# worker's antennal-lobe cells). Whether
# that slide was under-drawn or cut through the edge of the structure is not
# established (see nine_region_common, above ``display``), and it does not
# matter here -- either way the sampling differs between the brains. A
# caste test on it therefore compares how much of the structure each brain's
# sections sampled as much as it compares castes. AL is not fitted here at all:
# attach_region_flags below is called with BASE_REGIONS (six), so no in_al
# column reaches any fit in this script.
CASTE_TEST_REGIONS = [r for r in COMMON_REGIONS if r != "al"]


# ── Helpers ────────────────────────────────────────────────────────────────


def get_region_cols(adata):
    """Return in_* boolean columns for the seven statistical regions.

    Restricted to COMMON_REGIONS rather than every in_* column the object
    happens to carry: ellipsoid_body/fan_shaped_body may well be present
    (another pipeline stage adds them for the descriptive-only figures) but
    must never enter a fit in this script -- see the COMMON_REGIONS comment.
    Excluding them here, by name, is what makes that exclusion explicit
    rather than an accident of which columns a given object has.
    """
    return [f"in_{r}" for r in COMMON_REGIONS if f"in_{r}" in adata.obs.columns]


def assign_regions(adata):
    """Assign each cell to a single region (first True in_* col wins).

    ``in_al`` is a nullable boolean -- pd.NA means "this section has no
    antennal-lobe tracing yet", not "outside the antennal lobe" (see
    nine_region_common.py). ``na_value=False`` below only controls
    assignment priority: an untraced cell simply isn't claimed by "al" and
    falls through to the next column, exactly as it would if it were
    genuinely outside every drawn region, ending up "unassigned". It does
    not fill missing tracing into a caste-comparison denominator, because
    every consumer of this function's output drops "unassigned" cells
    before computing a proportion or running a test. Same idiom as
    fig_atlas_overview.get_region_labels.
    """
    bool_cols = get_region_cols(adata)
    if not bool_cols:
        raise ValueError("No in_* region columns found")

    region_arr = np.full(adata.n_obs, "unassigned", dtype=object)
    for col in bool_cols:
        mask = adata.obs[col].to_numpy(dtype=bool, na_value=False)
        region_name = col.replace("in_", "")
        new_assign = mask & (region_arr == "unassigned")
        region_arr[new_assign] = region_name
    return pd.Series(region_arr, index=adata.obs.index)


def convex_hull_area(coords):
    """Compute area of 2D convex hull. Returns 0 if too few points."""
    if len(coords) < 3:
        return 0.0
    try:
        hull = ConvexHull(coords[:, :2])
        return hull.volume  # 2D: volume = area
    except Exception:
        return 0.0


def _build_gene_name_map():
    """Build BIMP gene ID → readable protein name mapping from panel CSV."""
    import re

    panel_path = Path("data/panel/XeniumBrainPanel_BIMP+MGEN_vers2.csv")
    if not panel_path.exists():
        return {}
    panel = pd.read_csv(panel_path, encoding="utf-8-sig")
    panel = panel.dropna(subset=["BIMP_NP_Symbol"])

    def _clean(prot: str) -> str:
        """Strip the panel CSV's punctuation-substitution artefacts.

        The vendor sheet replaced every comma, colon and parenthesis inside a
        [protein=...] field with an underscore, so NCBI's "protein tramtrack,
        beta isoform" arrives as "protein tramtrack_ beta isoform". Others in
        the same panel:
        "glutamate receptor ionotropic_ kainate 2" (comma), "DNA
        _cytosine-5_-methyltransferase 1" (parentheses), "LOW QUALITY PROTEIN_
        ..." (colon). Drop the underscores rather than guess which punctuation
        mark each one was, and collapse the space that leaves behind.
        """
        # Delete the underscore where a space or hyphen already separates the
        # words ("tramtrack_ beta", "cytosine-5_-methyltransferase"); only
        # substitute a space where dropping it would fuse two words.
        prot = re.sub(r"_(?=[\s-])|(?<=[\s-])_", "", prot)
        prot = prot.replace("_", " ")
        return re.sub(r"\s{2,}", " ", prot).strip()

    mapping = {}
    for _, row in panel.iterrows():
        bimp_id = str(row["BIMP_NP_Symbol"])
        desc = str(row.get("BIMP_NP_Description", ""))
        m = re.search(r"\[protein=(.+?)\]", desc)
        if m:
            prot = _clean(m.group(1))
            prot = re.sub(r"\s*isoform.*", "", prot)
            prot = re.sub(r"-like$", "", prot)
            if len(prot) < 35:
                mapping[bimp_id] = prot
                continue
        mapping[bimp_id] = str(row.get("BIMP2.2_Symbol", bimp_id))
    return mapping


_GENE_NAME_MAP = _build_gene_name_map()

# Panel annotations that assert an identity the probe does not have. The
# [protein=...] field for the vitellogenin-family probe reads plain
# "vitellogenin", which would display the gene as canonical Vg.
# It is BIMP_v4.3_001050, a 319-residue paralog keeping only the N-terminal
# lipid-binding domain and distinct from Vgn (BIMP_v4.3_005500, 1772 aa),
# which the panel does not target at all. Matches _si_common.FOCAL_GENES["Vg"].
_GENE_NAME_OVERRIDES = {
    "BIMP_vNP3.1_05833": "Vg-family paralog",
}


def gene_short_name(gene_id):
    """Convert BIMP gene ID to readable protein/gene name."""
    # Try full ID first (with |MGEN...)
    bimp_part = gene_id.split("|")[0] if "|" in gene_id else gene_id
    bimp_part = bimp_part.replace("BIMP_v4.3_", "BIMP_vNP3.1_")
    if bimp_part in _GENE_NAME_OVERRIDES:
        return _GENE_NAME_OVERRIDES[bimp_part]
    if bimp_part in _GENE_NAME_MAP:
        return _GENE_NAME_MAP[bimp_part]
    # Fallback to numeric ID
    return bimp_part.replace("BIMP_vNP3.1_", "").replace("BIMP_v4.3_", "")


# ── 1. Cell density by region ─────────────────────────────────────────────


def compute_density(adatas):
    """Compute cell density (cells / convex hull area) per region.

    Returns DataFrame with counts, areas, densities, and proportions.
    Also runs a chi-square test on region composition between brains.
    """
    print("\n=== Cell Density by Region ===", flush=True)
    records = []

    for brain, adata in adatas.items():
        regions = assign_regions(adata)
        coords = adata.obsm["spatial_aligned"]
        total_cells = adata.n_obs

        for region in sorted(regions.unique()):
            mask = (regions == region).values
            n_cells = int(mask.sum())
            region_coords = coords[mask]
            area = convex_hull_area(region_coords)
            density = n_cells / area if area > 0 else 0.0
            proportion = n_cells / total_cells

            records.append(
                {
                    "brain": brain,
                    "region": region,
                    "n_cells": n_cells,
                    "hull_area": round(area, 2),
                    "density": round(density, 6),
                    "proportion": round(proportion, 4),
                    "total_cells": total_cells,
                }
            )

        print(
            f"  [{brain}] {total_cells:,} cells, {len(regions.unique())} regions",
            flush=True,
        )

    df = pd.DataFrame(records)

    # Chi-square on common region composition
    common = df[df["region"].isin(COMMON_REGIONS)]
    if len(common) > 0:
        ct = common.pivot(index="brain", columns="region", values="n_cells").fillna(0)
        chi2, p, dof, _ = chi2_contingency(ct.values)
        print(
            f"  Chi-square (common regions): chi2={chi2:.1f}, p={p:.2e}, dof={dof}",
            flush=True,
        )
        df.attrs["chi2"] = chi2
        df.attrs["chi2_p"] = p

    return df


# ── 2. Cell type composition by region ────────────────────────────────────


def composition_by_region(adatas):
    """Cell type proportions per region per brain.

    Loads the combined Tangram h5ad (the per-brain objects carry independent
    Leiden numbering). Regions come from ``bimp_st.composition.atlas_regions``:
    fresh nine-region flags, and a cell counts only if it lies inside exactly
    one of the seven statistical regions, the same cell set as the
    caste-composition figure and its counts
    (scripts/regions/region_composition.py).

    Cell types are the expression-only per-cluster vote (see
    ``_composition_labels``), pooled -- not the raw per-cell argmax, and
    deliberately not the anatomy-guided vote, which would be circular here.

    Descriptive: one brain per caste, so no queen-versus-worker test is run.
    Returns (composition table, empty test table).
    """
    from bimp_st.composition import atlas_regions, composition_table

    print("\n=== Cell Type Composition by Region ===", flush=True)
    print(f"  Loading {COMBINED_TANGRAM_H5AD}...", flush=True)
    c = ad.read_h5ad(COMBINED_TANGRAM_H5AD)
    print(
        f"  {c.n_obs:,} cells "
        f"(queen {(c.obs['brain'] == 'queen').sum():,}, "
        f"worker {(c.obs['brain'] == 'worker').sum():,})",
        flush=True,
    )
    region_arr = atlas_regions(c, regions=COMMON_REGIONS)
    cell_types = _composition_labels(c)
    print(
        "  Cell-type labels (hierarchical_vote, pooled): "
        + ", ".join(_pooled_label_order(cell_types)),
        flush=True,
    )
    comp_df = composition_table(
        c.obs["brain"].astype(str).to_numpy(),
        region_arr,
        cell_types,
        regions=CASTE_TEST_REGIONS,  # the six panel regions, as before
    )
    print(
        f"  Built composition: {comp_df['cluster'].nunique()} cell types "
        f"x {comp_df['region'].nunique()} regions x 2 brains",
        flush=True,
    )
    return comp_df, pd.DataFrame()


# ── 3. Per-region DEGs (pseudobulk) ───────────────────────────────────────


def regional_degs(adatas):
    """Pseudobulk DEGs per common region (queen vs worker).

    Aggregates expression by slice_name within each brain, then
    runs Mann-Whitney U between queen and worker slice means.
    """
    print("\n=== Per-Region Pseudobulk DEGs ===", flush=True)
    all_deg_dfs = {}

    for region in COMMON_REGIONS:
        print(f"\n  Region: {region}", flush=True)
        region_col = f"in_{region}"

        # Collect pseudobulk samples per brain
        pb_records = []
        brain_slice_counts = {}

        for brain, adata in adatas.items():
            if region_col not in adata.obs.columns:
                print(
                    f"    [{brain}] No '{region_col}' column — skipping",
                    flush=True,
                )
                continue

            # in_al is a nullable boolean; na_value=False excludes untraced
            # cells from this region's pseudobulk rather than raising or
            # miscounting them as a real absence (see assign_regions()).
            mask = adata.obs[region_col].to_numpy(dtype=bool, na_value=False)
            n_region = mask.sum()
            if n_region == 0:
                print(
                    f"    [{brain}] 0 cells in {region}",
                    flush=True,
                )
                continue

            sub = adata[mask]
            slices = sub.obs["slice_name"].unique()
            n_valid = 0
            genes = list(adata.var_names)

            for sl in slices:
                sl_mask = sub.obs["slice_name"] == sl
                n_cells = sl_mask.sum()
                if n_cells < MIN_CELLS_PER_SLICE:
                    continue
                X_sl = sub[sl_mask]
                means = (
                    np.asarray(X_sl.X.mean(axis=0)).ravel()
                    if hasattr(X_sl.X, "mean")
                    else X_sl.X.mean(axis=0)
                )
                row = {
                    "brain": brain,
                    "slice": sl,
                    "n_cells": int(n_cells),
                }
                for gi, g in enumerate(genes):
                    row[g] = means[gi]
                pb_records.append(row)
                n_valid += 1

            brain_slice_counts[brain] = n_valid
            print(
                f"    [{brain}] {n_region:,} cells, "
                f"{n_valid}/{len(slices)} valid slices",
                flush=True,
            )

        if not pb_records:
            print(f"    No data for {region}", flush=True)
            continue

        pb = pd.DataFrame(pb_records)
        pb_q = pb[pb["brain"] == "queen"]
        pb_w = pb[pb["brain"] == "worker"]

        n_q = len(pb_q)
        n_w = len(pb_w)
        if n_q < MIN_SLICES or n_w < MIN_SLICES:
            print(
                f"    Skipping {region}: queen={n_q}, "
                f"worker={n_w} slices (need {MIN_SLICES})",
                flush=True,
            )
            continue

        # Effect sizes per gene. One brain per caste: the sections are
        # pseudoreplicates of one animal, so no P value is computed. The slice
        # consistency is the rank-biserial correlation of the two brains'
        # section means, 1 - 2 U_queen / (n_q n_w): +1 when every worker section
        # is above every queen section, -1 for the reverse (same sign as log2fc).
        genes = list(adatas["queen"].var_names)
        deg_records = []
        for g in genes:
            v_q = pb_q[g].values
            v_w = pb_w[g].values
            if v_q.std() == 0 and v_w.std() == 0:
                continue
            u_q = mannwhitneyu(v_q, v_w, alternative="two-sided").statistic
            m_q, m_w = v_q.mean(), v_w.mean()
            log2fc = np.log2((m_w + 0.01) / (m_q + 0.01))
            deg_records.append(
                {
                    "gene": g,
                    "gene_short": gene_short_name(g),
                    "mean_queen": round(m_q, 4),
                    "mean_worker": round(m_w, 4),
                    "log2fc": round(log2fc, 4),
                    "slice_consistency": round(1 - 2 * u_q / (n_q * n_w), 4),
                    "n_slices_queen": n_q,
                    "n_slices_worker": n_w,
                }
            )

        deg_df = pd.DataFrame(deg_records)
        if len(deg_df):
            if region in CASTE_TEST_REGIONS:
                deg_df = deg_df.assign(
                    _c=deg_df.slice_consistency.abs(), _f=deg_df.log2fc.abs()
                )
                deg_df = deg_df.sort_values(
                    ["_c", "_f"], ascending=False, kind="mergesort"
                ).drop(columns=["_c", "_f"])
                print(
                    f"    {len(deg_df)} genes, effect sizes (no P value)",
                    flush=True,
                )
            else:
                # The antennal lobe is present on all sixteen queen sections
                # and eleven of the worker's sixteen, unevenly across its two
                # slides, so a queen-versus-worker test here compares how much
                # of the structure each brain's sections sampled as much as
                # it compares castes.
                # Effect sizes are kept, but the region is not drawn or
                # tabulated, so it cannot be read as a biological claim.
                deg_df["caste_test_valid"] = False
                deg_df = deg_df.sort_values("log2fc", key=np.abs, ascending=False)
                print(
                    f"    {len(deg_df)} genes, effect sizes only -- no caste "
                    f"test for {region} (annotation coverage differs by caste)",
                    flush=True,
                )
        all_deg_dfs[region] = deg_df

    return all_deg_dfs


# ── 4. Cross-region comparison (region × brain interaction) ───────────────


def cross_region_compare(adatas):
    """Identify genes with region x brain interaction effects.

    For each gene, computes mean expression per (region, brain) group,
    then runs Kruskal-Wallis on the four groups to detect interaction.
    """
    print("\n=== Cross-Region Comparison ===", flush=True)

    # Build per-slice expression for each (brain, region)
    slice_records = []
    for brain, adata in adatas.items():
        regions = assign_regions(adata)
        genes = list(adata.var_names)

        for region in COMMON_REGIONS:
            region_mask = (regions == region).values
            if region_mask.sum() == 0:
                continue
            sub = adata[region_mask]
            slices = sub.obs["slice_name"].unique()

            for sl in slices:
                sl_mask = sub.obs["slice_name"] == sl
                n_cells = sl_mask.sum()
                if n_cells < MIN_CELLS_PER_SLICE:
                    continue
                X_sl = sub[sl_mask]
                means = (
                    np.asarray(X_sl.X.mean(axis=0)).ravel()
                    if hasattr(X_sl.X, "mean")
                    else X_sl.X.mean(axis=0)
                )
                row = {
                    "brain": brain,
                    "region": region,
                    "slice": sl,
                    "n_cells": int(n_cells),
                }
                for gi, g in enumerate(genes):
                    row[g] = means[gi]
                slice_records.append(row)

    if not slice_records:
        print("  No data for interaction test", flush=True)
        return pd.DataFrame()

    pb = pd.DataFrame(slice_records)
    pb["group"] = pb["brain"] + "_" + pb["region"]
    genes = list(adatas["queen"].var_names)

    records = []
    for g in genes:
        groups = []
        group_labels = []
        for grp_name in sorted(pb["group"].unique()):
            vals = pb.loc[pb["group"] == grp_name, g].values
            if len(vals) >= 2:
                groups.append(vals)
                group_labels.append(grp_name)

        if len(groups) < 4:
            continue

        try:
            stat, pval = kruskal(*groups)
        except ValueError:
            continue

        # Effect size: compute eta-squared approximation
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else 0

        # Per-group means for the record
        means = {}
        for grp_name, vals in zip(group_labels, groups, strict=False):
            means[f"mean_{grp_name}"] = round(vals.mean(), 4)

        row = {
            "gene": g,
            "gene_short": gene_short_name(g),
            "kw_stat": round(stat, 3),
            "p_value": pval,
            "eta_squared": round(eta2, 4),
            **means,
        }
        records.append(row)

    df = pd.DataFrame(records)
    if len(df):
        df["p_adj"] = false_discovery_control(df["p_value"].values, method="bh")
        df = df.sort_values("p_adj")
        n_sig = (df["p_adj"] < 0.05).sum()
        print(
            f"  {n_sig}/{len(df)} genes with significant "
            f"region x brain interaction (FDR<0.05)",
            flush=True,
        )

    return df


# ── 5. One-vs-rest domain marker genes ────────────────────────────────────


def one_vs_rest_degs(adatas):
    """Identify genes enriched in each anatomical region (1-vs-rest).

    Uses pseudobulk (per-slice means) + Mann-Whitney U to get realistic
    p-values with proper biological replicates, avoiding the p ≈ 0
    problem of cell-level tests on 40K+ cells.
    """
    print("\n=== One-vs-Rest Domain Markers (pseudobulk) ===", flush=True)
    results = {}

    for brain, adata in adatas.items():
        regions = assign_regions(adata)
        mask = regions != "unassigned"
        region_vals = regions[mask]
        sub = adata[mask.values]
        genes = list(adata.var_names)

        # Build slice-level pseudobulk per region
        valid_regions = []
        pb_data = {}  # region -> DataFrame (slices x genes)
        for region in sorted(region_vals.unique()):
            r_mask = (region_vals == region).values
            r_sub = sub[r_mask]
            slices = r_sub.obs["slice_name"].unique()
            slice_records = []
            for sl in slices:
                sl_mask = r_sub.obs["slice_name"] == sl
                n_cells = sl_mask.sum()
                if n_cells < MIN_CELLS_PER_SLICE:
                    continue
                X_sl = r_sub[sl_mask]
                means = (
                    np.asarray(X_sl.X.mean(axis=0)).ravel()
                    if hasattr(X_sl.X, "mean")
                    else X_sl.X.mean(axis=0)
                )
                slice_records.append(means)
            if len(slice_records) >= 3:
                pb_data[region] = np.array(slice_records)
                valid_regions.append(region)

        print(
            f"  [{brain}] {len(valid_regions)} regions with "
            f"≥3 pseudobulk samples: {valid_regions}",
            flush=True,
        )

        # 1-vs-rest: for each region, MWU against pooled rest
        brain_results = {}
        for region in valid_regions:
            region_pb = pb_data[region]  # (n_slices_region, n_genes)
            rest_pb = np.vstack(
                [pb_data[r] for r in valid_regions if r != region]
            )  # (n_slices_rest, n_genes)

            deg_records = []
            for gi, g in enumerate(genes):
                v_region = region_pb[:, gi]
                v_rest = rest_pb[:, gi]
                if v_region.std() == 0 and v_rest.std() == 0:
                    continue
                _, pval = mannwhitneyu(
                    v_region,
                    v_rest,
                    alternative="two-sided",
                )
                m_region = v_region.mean()
                m_rest = v_rest.mean()
                log2fc = np.log2((m_region + 0.01) / (m_rest + 0.01))
                # Fraction of cells expressing (for reference)
                deg_records.append(
                    {
                        "names": g,
                        "gene_short": gene_short_name(g),
                        "logfoldchanges": round(log2fc, 4),
                        "pvals": pval,
                        "mean_region": round(m_region, 4),
                        "mean_rest": round(m_rest, 4),
                        "n_slices_region": len(region_pb),
                        "n_slices_rest": len(rest_pb),
                    }
                )

            df = pd.DataFrame(deg_records)
            if len(df):
                df["pvals_adj"] = false_discovery_control(
                    df["pvals"].values,
                    method="bh",
                )
                df = df.sort_values("pvals_adj")
            df["brain"] = brain
            brain_results[region] = df

            n_sig = (df["pvals_adj"] < 0.05).sum()
            n_up = ((df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0)).sum()
            print(
                f"    {region}: {n_sig} sig ({n_up} up), "
                f"{len(region_pb)} vs {len(rest_pb)} slices",
                flush=True,
            )

        results[brain] = brain_results

    return results


def pseudobulk_by_domain(adatas):
    """Compute pseudobulk mean expression per region per brain.

    Returns a DataFrame with columns: brain, region, gene1, gene2, ...
    """
    print("\n=== Pseudobulk by Domain ===", flush=True)
    records = []

    for brain, adata in adatas.items():
        regions = assign_regions(adata)
        genes = list(adata.var_names)

        for region in sorted(regions.unique()):
            if region == "unassigned":
                continue
            mask = (regions == region).values
            n_cells = mask.sum()
            if n_cells < 10:
                continue

            X_region = adata[mask]
            means = (
                np.asarray(X_region.X.mean(axis=0)).ravel()
                if hasattr(X_region.X, "mean")
                else X_region.X.mean(axis=0)
            )
            row = {"brain": brain, "region": region, "n_cells": int(n_cells)}
            for gi, g in enumerate(genes):
                row[g] = means[gi]
            records.append(row)

    df = pd.DataFrame(records)
    print(f"  {len(df)} pseudobulk profiles", flush=True)
    return df


def pseudobulk_pca(adatas):
    """PCA of slice-level pseudobulk profiles colored by region.

    Returns (pca_df, pca_model) where pca_df has columns:
    brain, region, slice, PC1, PC2, ...
    """
    print("\n=== Pseudobulk PCA ===", flush=True)
    records = []

    for brain, adata in adatas.items():
        regions = assign_regions(adata)
        genes = list(adata.var_names)

        for region in sorted(regions.unique()):
            if region == "unassigned":
                continue
            mask = (regions == region).values
            if mask.sum() == 0:
                continue

            sub = adata[mask]
            slices = sub.obs["slice_name"].unique()
            for sl in slices:
                sl_mask = sub.obs["slice_name"] == sl
                n_cells = sl_mask.sum()
                if n_cells < MIN_CELLS_PER_SLICE:
                    continue
                X_sl = sub[sl_mask]
                means = (
                    np.asarray(X_sl.X.mean(axis=0)).ravel()
                    if hasattr(X_sl.X, "mean")
                    else X_sl.X.mean(axis=0)
                )
                row = {
                    "brain": brain,
                    "region": region,
                    "slice": sl,
                    "n_cells": int(n_cells),
                }
                for gi, g in enumerate(genes):
                    row[g] = means[gi]
                records.append(row)

    if not records:
        print("  No data for PCA", flush=True)
        return pd.DataFrame(), None

    pb = pd.DataFrame(records)
    genes = list(adatas["queen"].var_names)
    X = pb[genes].values

    # Log-transform + z-score
    X_log = np.log1p(X)
    X_z = (X_log - X_log.mean(axis=0)) / (X_log.std(axis=0) + 1e-8)

    pca = PCA(n_components=min(5, X_z.shape[0], X_z.shape[1]))
    coords = pca.fit_transform(X_z)
    for i in range(coords.shape[1]):
        pb[f"PC{i + 1}"] = coords[:, i]

    print(
        f"  {len(pb)} samples, "
        f"PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}",
        flush=True,
    )
    return pb, pca


# ── Figure ─────────────────────────────────────────────────────────────────


# ── Three SI figures ───────────────────────────────────────────────────────

# Small-type / Large-type / Class II are the display names for these
# three Kenyon compartments in every figure (matches fig_s_regional_diversity,
# figS_tangram_confidence, si_exploded_regions, fig_atlas_overview, and the
# identical dict in pipelines/subcellular/bento_differential.py).
REGION_DISPLAY = {
    "lamina": "Lamina",
    "medulla": "Medulla",
    "lobula": "Lobula",
    "kc_ic": "Small-type",
    "kc_nc": "Large-type",
    "kc_oc": "Class II",
    "al": "Antennal lobe",
}


def _pretty_region(name: str) -> str:
    return REGION_DISPLAY.get(name, name.replace("_", " ").title())
