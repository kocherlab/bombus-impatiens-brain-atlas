"""Canonical data paths, and a loader that guards against known data traps.

Import the constants here instead of hardcoding ``results/...`` strings, so
every script reads the same object.

Guarded invariants:

1. Cell-type column names must be keyed by name, not by position. Tangram
   objects whose column names were assigned by position live under
   ``results/tangram/_superseded/``, so a hardcoded path to one raises
   ``FileNotFoundError``; :func:`load_tangram` refuses them unless
   ``allow_superseded=True``.

2. Columns listed in :data:`POISONED_COLUMNS` are not read from the canonical
   object; :func:`load_tangram` renames them out of the way.

3. :data:`SPATIAL_COMBINED` carries its own ``tangram_cell_type`` column, which
   is not the canonical labelling. Use that object for coordinates and
   expression, never for cell-type identity.

4. ``combined_clustered.h5ad`` has 562,028 duplicate ``obs_names``, so
   ``obs.loc[subset.index]`` silently pulls in rows outside the subset. Index
   positionally; :func:`load_tangram` can raise on duplicates
   (``allow_duplicate_obs_names=False``).

A missing input raises :class:`MissingInputError` (a ``FileNotFoundError``)
naming the file, where it goes and the README "Data" section that lists it
(:func:`require_input`, :func:`explain_missing_inputs`).

Run ``scripts/paper/check_data_provenance.py`` to lint for regressions.
"""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

#: The LaTeX-sources checkout scripts/paper reads and writes. Defaults to the
#: submodule; set ``BIMP_MANUSCRIPT`` to check a separate clone (a worktree
#: whose submodule is not initialised, or a feature branch of the sources).
MANUSCRIPT = Path(
    os.environ.get(
        "BIMP_MANUSCRIPT", REPO / "submodules" / "bombus_impatiens_atlas_manuscript"
    )
)
TANGRAM_DIR = RESULTS / "tangram"
SUPERSEDED_DIR = TANGRAM_DIR / "_superseded"

#: The canonical combined run, chosen by a criterion fixed before mapping
#: (4 configurations x seeds 42/43/44, gated on stability, a held-out-gene floor
#: and Kenyon/optic placement floors); it is the only configuration that passes.
#:
#: Configuration (``pipelines/combined_atlas/tangram_mapping.py
#: train``): mode=clusters, full 73,463-nucleus reference, density term on with
#: ``density_prior='rna_count_based'``, ``lambda_d=1``, reference-side
#: ``d_source`` = within-dissection cluster density
#: (``--source-density cluster_density_within``), seed 42.
#:
#: Caches are keyed on the file name; do not rename it.
CANONICAL_TANGRAM = TANGRAM_DIR / "combined_clustered_g2_R2_s42_tangram.h5ad"

CANONICAL_CENTRAL = TANGRAM_DIR / "combined_clustered_central_colfix_tangram.h5ad"
CANONICAL_OPTIC = TANGRAM_DIR / "combined_clustered_optic_colfix_tangram.h5ad"

#: Stitched labels composed from the corrected central+optic runs by
#: scripts/atlas/build_region_stitched_labels.py.
CANONICAL_STITCHED = TANGRAM_DIR / "combined_clustered_stitched_colfix_tangram.h5ad"

#: The 101 Xenium panel probes: NP_31 symbol <-> BIMP v4.3 id <-> LOC id, with the
#: NCBI description (read with :func:`bimp_st.genes.read_panel_crosswalk`).
PANEL_CROSSWALK = REPO / "data" / "panel" / "np31_to_v43_mapping.csv"

#: The snRNA-seq reference atlas. The combined CB+OL file is the mapping
#: reference; the region-specific files feed the separate CB/OL Tangram runs.
SN_DIR = REPO / "data" / "sn"
SN_COMBINED = SN_DIR / "BIMP_atlas_qw_combined.h5ad"
SN_CB = SN_DIR / "BIMP_atlas_qw_CentralBrain.h5ad"
SN_OL = SN_DIR / "BIMP_atlas_qw_OpticLobe.h5ad"

#: Per-brain aligned+clustered wholebrain objects (queen/worker Snakemake
#: pipelines' `cluster` rule output). Own per-brain `leiden` numbering, the
#: `in_*` region flags, `spatial_aligned`.
ALIGNED_QUEEN = REPO / "pipelines/wholebrain/queen/data/aligned_clustered.h5ad"
ALIGNED_WORKER = REPO / "pipelines/wholebrain/worker/data/aligned_clustered.h5ad"

#: The pipeline's spatial object. Its own ``tangram_cell_type`` column is not
#: the canonical labelling. Use it for coordinates and expression; take labels
#: from :data:`CANONICAL_TANGRAM`.
SPATIAL_COMBINED = (
    REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
)

#: Tk 3D spillover tables (scripts/neuropeptides/tk_3d_spillover.py): per-section
#: Kenyon-vs-rest fold, 3D distance medians and rank-biserial effect sizes (no P
#: values), on the nullable nine-region Kenyon mask.
TK_3D_SPILLOVER = RESULTS / "reframe" / "tk_3d_spillover_v2"

#: Columns that are wrong even in the canonical object, and why.
POISONED_COLUMNS = {
    "tangram_cell_type_stitched": (
        "the copy stored inside the canonical object is not composed from "
        "CANONICAL_CENTRAL and CANONICAL_OPTIC. Read CANONICAL_STITCHED "
        "(combined_clustered_stitched_colfix_tangram.h5ad), or compose it from "
        "CANONICAL_CENTRAL and CANONICAL_OPTIC as "
        "scripts/identity/si_label_sensitivity.py does."
    ),
}

SUPERSEDED_SUFFIX = "__SUPERSEDED_DO_NOT_USE"


def tangram_object(name: str) -> Path:
    """Resolve a Tangram object by basename, looking in ``_superseded/`` too,
    so a script can name an object without hardcoding that location."""
    live = TANGRAM_DIR / name
    if live.exists():
        return live
    old = SUPERSEDED_DIR / name
    if old.exists():
        return old
    raise FileNotFoundError(
        f"No Tangram object named {name!r} in {TANGRAM_DIR} or {SUPERSEDED_DIR}"
    )


def assert_unique_obs_names(adata, where: str = "") -> None:
    """Raise if ``obs_names`` are duplicated, naming the safe alternative.

    A duplicated index makes every ``.loc[labels]`` lookup silently wrong. Index
    positionally (boolean masks, ``.values``) instead of by label.
    """
    dup = int(adata.obs_names.duplicated().sum())
    if dup:
        raise ValueError(
            f"{where or 'object'} has {dup:,} duplicated obs_names. Label-based "
            f".loc lookups will silently expand and fold in unrelated rows -- "
            f"index positionally instead. Pass allow_duplicate_obs_names=True "
            f"only if every access in this script is positional."
        )


def keyed_obs_column(
    target,
    column: str = "tangram_cell_type",
    path: Path | str | None = None,
):
    """``column`` of ``path``'s obs (default :data:`CANONICAL_TANGRAM`), aligned to
    ``target`` row by row on the cell key ``(slice_name, cell_id)``.

    ``target`` is an AnnData or an obs DataFrame with ``slice_name`` and
    ``cell_id`` columns. obs_names cannot be the key (562,028 duplicates), and a
    length check does not prove two objects list cells in the same order; this
    joins on the key and raises if either side repeats a key or if any target
    cell is missing from the source. Returns a numpy array in ``target`` order.
    """
    import h5py
    import numpy as np
    import pandas as pd
    from anndata.io import read_elem

    path = Path(path) if path is not None else CANONICAL_TANGRAM
    if path.resolve() == CANONICAL_TANGRAM.resolve() and column in POISONED_COLUMNS:
        raise KeyError(
            f"{column} is poisoned in {path.name}: {POISONED_COLUMNS[column]}"
        )
    obs = getattr(target, "obs", target)

    def _key(slice_name, cell_id) -> pd.Index:
        return pd.Index(
            pd.Series(np.asarray(slice_name)).astype(str).to_numpy()
            + "||"
            + pd.Series(np.asarray(cell_id)).astype(str).to_numpy()
        )

    with h5py.File(path, "r") as f:
        g = f["obs"]
        src_key = _key(read_elem(g["slice_name"]), read_elem(g["cell_id"]))
        values = np.asarray(read_elem(g[column]))
    tgt_key = _key(obs["slice_name"], obs["cell_id"])
    for name, k in (("source " + path.name, src_key), ("target", tgt_key)):
        if k.has_duplicates:
            raise ValueError(f"(slice_name, cell_id) repeats in the {name}")
    pos = src_key.get_indexer(tgt_key)
    if (pos < 0).any():
        raise ValueError(
            f"{int((pos < 0).sum()):,} target cells have no (slice_name, cell_id) "
            f"match in {path.name}"
        )
    return values[pos]


def load_tangram(
    path: Path | str | None = None,
    *,
    allow_duplicate_obs_names: bool = True,
    keep_poisoned_columns: bool = False,
    allow_superseded: bool = False,
    **kwargs,
):
    """Read a Tangram object with the known traps disarmed.

    Parameters
    ----------
    path
        Defaults to :data:`CANONICAL_TANGRAM`.
    allow_duplicate_obs_names
        Default ``True`` because the canonical object legitimately has them and
        most callers index positionally. Set ``False`` in any script that uses
        ``.loc`` on ``obs``, to turn a silent corruption into an exception.
    keep_poisoned_columns
        Default ``False``: in :data:`CANONICAL_TANGRAM` only, columns in
        :data:`POISONED_COLUMNS` are renamed with :data:`SUPERSEDED_SUFFIX` so
        that reading them raises ``KeyError`` instead of returning stale labels.
        (Other objects, e.g. :data:`CANONICAL_STITCHED`, carry a correct column
        of the same name.)
    allow_superseded
        Default ``False``: refuse any path that resolves into
        :data:`SUPERSEDED_DIR`, including through a symlink.
    """
    import anndata as ad

    path = Path(path) if path is not None else CANONICAL_TANGRAM
    if not path.exists():
        if (SUPERSEDED_DIR / path.name).exists():
            raise FileNotFoundError(
                f"{path} does not exist.\n  {SUPERSEDED_DIR / path.name} carries "
                f"cell-type column names assigned by position. Use "
                f"paths.CANONICAL_TANGRAM, or pass that path with "
                f"allow_superseded=True."
            )
        raise missing_input(path)

    resolved = path.resolve()
    if not allow_superseded and resolved.is_relative_to(SUPERSEDED_DIR.resolve()):
        raise ValueError(
            f"{path} resolves into {SUPERSEDED_DIR}: an object whose cell-type "
            f"column names were assigned by position. Pass allow_superseded=True "
            f"only to compare against it."
        )

    adata = ad.read_h5ad(path, **kwargs)
    if not allow_duplicate_obs_names:
        assert_unique_obs_names(adata, where=str(path))
    if not keep_poisoned_columns and resolved == CANONICAL_TANGRAM.resolve():
        rename = {c: c + SUPERSEDED_SUFFIX for c in POISONED_COLUMNS if c in adata.obs}
        if rename:
            adata.obs.rename(columns=rename, inplace=True)
    return adata


# ── snRNA caste DE (the ~ trial + caste fit; bimp_st.caste_de) ──────────────────
#: Fine-type pseudobulk builder outputs (and an unblocked fit no figure or
#: table uses).
CASTE_DE_PSEUDOBULK_DIR = RESULTS / "caste_de_pseudobulk"
#: Directory names of the snRNA-seq caste-DE results that caches, md5 pins and
#: the figure registry key on. They are spelled here only; use the constants.
_SAVED_DE_DIRNAME = "caste_de_ian_port"
_SAVED_INPUTS_DIRNAME = "ian_inputs"
_MB_OL_CALLS_DIRNAME = "ian_round2"
#: Repository port of the caste-DE analysis (fine-type DE, panel tables) and the
#: frozen copies of the analysis's saved outputs it is compared with.
CASTE_DE_SAVED_DIR = RESULTS / _SAVED_DE_DIRNAME
#: Frozen saved outputs of the snRNA-seq caste-DE / cNMF / GSEA analysis
#: (de_all.csv.gz, joint cNMF usages and program labels, BIMP_lookup.csv).
CASTE_DE_SAVED_INPUTS = CASTE_DE_SAVED_DIR / _SAVED_INPUTS_DIRNAME
#: Joint cNMF / GSEA pipeline (scripts/caste_de/cnmf_gsea/run_pipeline.sh).
CASTE_PROGRAMS_DIR = RESULTS / "caste_programs"
#: Pseudobulk counts per fine type (``counts_<fn(type)>.csv``, ``groups.json``)
#: and nuclei per sample x fine type; written by
#: ``scripts/caste_de/fine_type_deseq2.py`` (pseudobulk builder).
CASTE_DE_PSEUDOBULK_FINE = CASTE_DE_PSEUDOBULK_DIR / "fine_de" / "pseudobulk"
CASTE_DE_NUCLEI = CASTE_DE_PSEUDOBULK_DIR / "fine_de" / "nuclei_per_sample.csv"
#: Pseudobulk counts of the whole brain (``all``) and the pooled classes, and the
#: gene descriptions; written by ``scripts/caste_de/pseudobulk_deseq2.py``.
CASTE_DE_PSEUDOBULK_COARSE = RESULTS / "figures" / "fig3" / "pseudobulk"
#: Folder of the caste-DE analyses (benchmarks, concordance, controls).
CASTE_DE_DIR = RESULTS / "caste_de"
#: The fit itself (``scripts/caste_de/caste_de_blocked_all.py``): samples kept
#: only as whole queen + worker trial pairs, and a type tested only with all
#: three pairs (bimp_st.caste_de.MIN_PER_CASTE), so Other glia is untested.
CASTE_DE_FIT_DIR = CASTE_DE_DIR / "three_trial_pairs"
#: The same fit requiring only two trial pairs (Other glia fit 2 vs 2 with one
#: residual df); every other type is identical (parity_previous.csv).
CASTE_DE_FIT_DIR_TWO_PAIRS = CASTE_DE_DIR / "trial_pairs"
CASTE_DE_FINE = CASTE_DE_FIT_DIR / "de_fine_blocked.csv"
CASTE_DE_COARSE = CASTE_DE_FIT_DIR / "de_coarse_blocked.csv"
CASTE_DE_COUNTS = CASTE_DE_FIT_DIR / "deg_counts.csv"
CASTE_DE_FOCAL = CASTE_DE_FIT_DIR / "focal_genes.csv"
#: Per-trial queen/worker log ratio sign of every DEG row (per_trial_consistency).
CASTE_DE_TRIAL_SIGNS = CASTE_DE_FIT_DIR / "per_trial_signs.csv"
#: Agreement of the repository fit with the saved DE table
#: (``scripts/caste_de/parity_saved_de.py``).
CASTE_DE_PARITY_SAVED = CASTE_DE_FIT_DIR / "parity_saved.csv"
#: CellBender ambient profiles of the snRNA-seq libraries, frozen with md5s
#: (``scripts/caste_de/freeze_cellbender_ambient.py``).
CASTE_DE_AMBIENT_DIR = CASTE_DE_DIR / _SAVED_INPUTS_DIRNAME
#: BIMP v4.3 gene annotation (frozen copy; symbol, description, biotype per
#: gene_id; tab-separated).
BIMP_GENE_LOOKUP = CASTE_PROGRAMS_DIR / "inputs" / "BIMP_lookup.csv"
#: Recurrence of caste DEGs across cell-type lists (``si_deg_recurrence.py``,
#: read by ``snrna_controls.py``); no registered figure or table.
CASTE_DE_RECURRENCE = CASTE_DE_DIR / "cross_list"

#: Mushroom-body / optic-lobe Kenyon-cell calls and the compartment ceiling
#: (``scripts/kenyon/project_kc_populations_with_optic.py``,
#: ``kc_compartment_ceiling.py``; read by ``table_mb_ol_kc_labels.py``).
MB_OL_CALLS_DIR = RESULTS / "round4" / _MB_OL_CALLS_DIRNAME


# ── missing inputs ──────────────────────────────────────────────────────────────
#: README section that lists every input, its size and where it goes.
DATA_README = "README.md, section 'Data'"

#: Repository-relative prefix -> (what it is, site key or None when the file
#: goes at that path in the checkout). Longest match wins; it names the input in
#: :class:`MissingInputError`.
DATA_MAP: dict[str, tuple[str, str | None]] = {
    "results/tangram/": ("Tangram cell-type mapping of the atlas", None),
    "results/": (
        "an analysis output: run the analysis step that writes it (the paper "
        "Snakefile, scripts/paper/Snakefile, runs them before the generators)",
        None,
    ),
    "data/sn/": ("snRNA-seq reference AnnData", None),
    "pipelines/combined_atlas/data/": (
        "combined two-brain atlas object (pipelines/combined_atlas)",
        None,
    ),
    "pipelines/wholebrain/": (
        "per-brain Xenium object (pipelines/wholebrain/{queen,worker})",
        None,
    ),
    "pipelines/maldi/": ("MALDI-MSI input (pipelines/maldi)", None),
    "pipelines/subcellular/": ("subcellular localisation output", None),
    "pipelines/snrna_reference/": ("snRNA-seq reference pipeline output", None),
}


class MissingInputError(FileNotFoundError):
    """An input file is absent; the message says what it is and where it goes."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def missing_input(
    path: Path | str, *, what: str | None = None, site_key: str | None = None
) -> MissingInputError:
    """A :class:`MissingInputError` for ``path`` (returned, not raised)."""
    p = Path(path)
    try:
        full = Path(os.path.abspath(p if p.is_absolute() else REPO / p))
        rel = str(full.relative_to(REPO))
    except ValueError:
        rel = str(p)
    hit = max((k for k in DATA_MAP if rel.startswith(k)), key=len, default=None)
    if hit is not None:
        what = what or DATA_MAP[hit][0]
        site_key = site_key or DATA_MAP[hit][1]
    lines = [f"missing input: {rel}"]
    if what:
        lines.append(f"  what: {what}")
    if site_key:
        lines.append(
            f"  where: the path set by site key {site_key!r} "
            f"(${'BIMP_' + site_key.upper()}; template config/site.example.toml)"
        )
    else:
        lines.append(f"  where: {rel} in this checkout (a symlink is fine)")
    lines.append(f"  see: {DATA_README}")
    err = MissingInputError("\n".join(lines))
    err.filename = str(p)
    return err


def require_input(
    path: Path | str, *, what: str | None = None, site_key: str | None = None
) -> Path:
    """Return ``path`` if it exists, else raise :class:`MissingInputError`."""
    if not Path(path).exists():
        raise missing_input(path, what=what, site_key=site_key)
    return Path(path)


_H5_NAME = re.compile(r"name = '([^']+)'")


@contextlib.contextmanager
def explain_missing_inputs():
    """Re-raise a bare ``FileNotFoundError`` (pandas, h5py, open) inside the block
    as a :class:`MissingInputError` naming the file. Wrap a generator's main."""
    try:
        yield
    except MissingInputError:
        raise
    except FileNotFoundError as e:
        name = e.filename
        if name is None:
            m = _H5_NAME.search(str(e))
            name = m.group(1) if m else None
        if name is None:
            raise
        raise missing_input(name) from e
