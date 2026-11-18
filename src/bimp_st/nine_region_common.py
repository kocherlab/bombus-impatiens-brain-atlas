"""Shared vocabulary and annotation-source resolution for the nine-region atlas.

The six original hand-drawn regions are extended with the antennal lobe and
the central complex (ellipsoid and fan-shaped body; see POOL_CENTRAL_COMPLEX).

1.  The crop mapping is not redeclared here: it is imported from
    ``ANNOTATION_MAPPING`` in each pipeline's ``labeling.py`` and re-keyed by
    (slide, section), so there is one copy of that state.

2.  "Not annotated" is not False. A region never traced on a slide is null
    there, not False (today: the central complex on queen slide 0040240; the
    antennal lobe has no nulls). Recording these as False
    would create a caste difference from missing data. Never ``fillna(False)``
    on the new regions before restricting to sections where they were traced.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Literal, TypedDict


class _BrainInfo(TypedDict):
    dir: Path  # the brain's whole-brain pipeline directory (labeling.py, data/)
    annotations: Path  # its region annotations: v1/ (original) ... v4/
    slides: tuple[str, str]


REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# The live vocabulary is discovered from the GeoJSON files (see
# ``discover_vocabulary``); the tables below only supply normalisation, display
# names and ordering. An unlisted region still flows through the pipeline with a
# title-cased name, sorted after the known ones.

# The six regions of the first annotation round.
BASE_REGIONS = ["lamina", "medulla", "lobula", "kc_oc", "kc_ic", "kc_nc"]

# Raw annotator strings -> canonical region name.  Add a line here only when an
# annotator spells something a new way; a genuinely new region needs nothing.
ALIASES = {
    "lobulla": "lobula",  # long-standing typo in the exports
    "antennal lobe": "al",
    "antennal_lobe": "al",
    "eb": "ellipsoid_body",
    "ellipsoid body": "ellipsoid_body",
    "fb": "fan_shaped_body",
    "fan-shaped body": "fan_shaped_body",
    "fan shaped body": "fan_shaped_body",
    "fsb": "fan_shaped_body",
    "central complex": "central_complex",
    "cx": "central_complex",
}

# Regions whose polygons carry their real identity in a `subregion` property
# (or a `b#_xx` name) rather than in the classification.  The central complex
# is drawn as one class but must be split into ellipsoid and fan-shaped body.
SPLIT_CLASSES = {
    "central complex": {"eb": "ellipsoid_body", "fb": "fan_shaped_body"},
    "central_complex": {"eb": "ellipsoid_body", "fb": "fan_shaped_body"},
    "cc": {"eb": "ellipsoid_body", "fb": "fan_shaped_body"},
}

# Report the central complex as one region rather than its two substructures,
# which are small. Polygons keep their
# `subregion` property, so setting this to False restores the split.
POOL_CENTRAL_COMPLEX = True
CENTRAL_COMPLEX = "central_complex"

# Labels that are annotation-tool residue, not anatomy.  QuPath leaves a stray
# "Selection N" object behind if a shape is drawn and never given a class;
# there is exactly one in each brain's current drop.  Excluded by name so that
# a genuinely new region is never caught by this filter.
EXCLUDE = {"unclassified", "unknown", "", "none"}

# A stray QuPath object with no class falls through to its `name`, which is
# literally "Selection 3". Matched by pattern because the number varies, so a
# plain name set would not catch it.
EXCLUDE_PATTERNS = (re.compile(r"^selection\s*\d*$"), re.compile(r"^annotation\s*\d*$"))

DISPLAY = {
    "lamina": "Lamina",
    "medulla": "Medulla",
    "lobula": "Lobula",
    "kc_oc": "Class II",
    "kc_ic": "Small-type",
    "kc_nc": "Large-type",
    "al": "Antennal lobe",
    "ellipsoid_body": "Ellipsoid body",
    "fan_shaped_body": "Fan-shaped body",
    "central_complex": "Central complex",
}

# Preferred display order; discovered regions not listed sort alphabetically
# after these.
PREFERRED_ORDER = [
    "lamina",
    "medulla",
    "lobula",
    "kc_oc",
    "kc_ic",
    "kc_nc",
    "al",
    "central_complex",
    "ellipsoid_body",
    "fan_shaped_body",
]

# Region palette shared by fig_atlas_overview.py, si_exploded_regions.py and
# fig_atlas_structure.py. The new regions use the remaining Tableau-10 swatches.
REGION_COLORS = {
    "lamina": "#59A14F",
    "medulla": "#4E79A7",
    "lobula": "#E15759",
    "kc_oc": "#EDC948",
    "kc_ic": "#B07AA1",
    "kc_nc": "#F28E2B",
    "al": "#76B7B2",
    "ellipsoid_body": "#FF9DA7",
    "fan_shaped_body": "#9C755F",
    # deep indigo: pink #FF9DA7 is hard to tell from lobula red, and the
    # spare Tableau brown is Middle-type Kenyon cells' colour
    "central_complex": "#3D2C8D",
}

# Kenyon cell POPULATION palette (snRNA-seq reference types), shared by
# fig:atlas_overview panel E (fig_atlas_overview.py), fig:kc_subtypes
# (fig_kc_subtypes.py, as TYPE_COLORS) and
# Supplementary figS_xenium_kc_populations (fig_xenium_celltype_overlay.py,
# via fig_kc_subtypes).  Each named population takes the colour of the
# compartment named after it, so a colour means one thing across figures:
# Large-type orange, Small-type mauve, Class II yellow.  Middle-type and Other
# Kenyon cells have no compartment and get neutral tones.  The fine-type
# palette in bimp_st.cluster_labels (_CURATED, get_color, get_canonical_color)
# takes these five colours from here too, so every figure that draws the
# populations agrees.  Keep this module free of a module-level
# cluster_labels import: cluster_labels imports this dict.
KC_POPULATION_COLORS = {
    "Large-type Kenyon cells": REGION_COLORS["kc_nc"],
    "Small-type Kenyon cells": REGION_COLORS["kc_ic"],
    "Class II Kenyon cells": REGION_COLORS["kc_oc"],
    "Middle-type Kenyon cells": "#9C755F",  # warm brown
    "Other Kenyon cells": "#8C8C8C",  # mid grey: the residual
}


# ---------------------------------------------------------------------------
# Which regions may carry which claims
# ---------------------------------------------------------------------------
# Cell counts across both brains, measured (results/nine_region/coverage.json):
#
#     KC-NC 145,340   KC-IC 106,281   KC-OC 81,061   Medulla 62,672
#     Lamina 47,254   Antennal lobe 10,676   Lobula 5,108
#     Central complex 2,080 (queen 1,046 + worker 1,034)
#
# The antennal lobe is twice the lobula, so it joins the statistical analyses.
# The central complex is too small: rare classes degrade the
# other estimates in a multinomial fit.
#
# Scripts should import these rather than deciding for themselves.

# Descriptive, spatial and coverage work: everything drawn.
DESCRIPTIVE_REGIONS = (
    [r for r in PREFERRED_ORDER if r not in ("ellipsoid_body", "fan_shaped_body")]
    if POOL_CENTRAL_COMPLEX
    else [r for r in PREFERRED_ORDER if r != CENTRAL_COMPLEX]
)

# Region prediction, diversity, PERMANOVA, regional DE: six + antennal lobe.
# The balanced null for these is 1/7, not 1/9.
STATISTICAL_REGIONS = BASE_REGIONS + ["al"]

# Shown and counted, never fitted.
DESCRIPTIVE_ONLY_REGIONS = (
    [CENTRAL_COMPLEX] if POOL_CENTRAL_COMPLEX else ["ellipsoid_body", "fan_shaped_body"]
)

# Every caste contrast involving a new region is descriptive and carries no
# p-value: the spatial arm is one brain per caste, so a per-region caste test
# would be pseudoreplicated (as for the optic lobe).
NO_PVALUE_REGIONS = ["al"] + DESCRIPTIVE_ONLY_REGIONS
NEW_REGIONS_STATIC = list(NO_PVALUE_REGIONS)  # separate object: callers mutate

# Sections where a region was not traced although it is traced elsewhere on the
# same slide; the slide-level rule in ``relabel_nine_regions.slide_region_drawn``
# would otherwise record their cells as negatives. Entries are listed with
# evidence, not inferred; currently there are none. The partial central-complex
# coverage on worker 0068401 is disclosed in the coverage table instead, and a
# mask must never name a section that has been traced. relabel_nine_regions.stage_crops
# asserts each listed region is absent from its section, so a stale entry fails.
UNTRACED_SECTIONS: dict[tuple[str, str, int], set[str]] = {}


# Open question, not implemented: worker slide 0069041, antennal lobe. The
# slide is not excluded from the antennal-lobe denominator. Its antennal-lobe
# polygons are much smaller than on the other slides, and so is its
# antennal-lobe share of cells. This is either under-drawing (biases worker
# prevalence down) or an oblique cut (real anatomy); checking the polygons
# against DAPI would decide. The queen/worker prevalence ratio depends on
# whether the slide is kept. No caste test reads it (`al` is in
# NO_PVALUE_REGIONS and excluded from si_regional_degs' CASTE_TEST_REGIONS).
def display(region: str) -> str:
    """Human-readable name, falling back sensibly for a newly drawn region."""
    return DISPLAY.get(region, region.replace("_", " ").capitalize())


# Hyphenated compartment names (stored as values in some results/ CSVs) ->
# reader-facing display names.  Keys of DISPLAY are the code identifiers;
# these are *string values*.
LEGACY_REGION_DISPLAY = {
    "KC-NC": DISPLAY["kc_nc"],
    "KC-IC": DISPLAY["kc_ic"],
    "KC-OC": DISPLAY["kc_oc"],
}


def display_region(name: str) -> str:
    """Display name for an identifier ('kc_nc') or a hyphenated name ('KC-NC')."""
    if name in LEGACY_REGION_DISPLAY:
        return LEGACY_REGION_DISPLAY[name]
    if name in DISPLAY:
        return DISPLAY[name]
    return name  # already a display name (e.g. 'Class II', 'Lamina') or unknown


def _order_key(region: str):
    return (
        PREFERRED_ORDER.index(region)
        if region in PREFERRED_ORDER
        else len(PREFERRED_ORDER),
        region,
    )


def discover_vocabulary(brains=None, source: str = "v2") -> list[str]:
    """Every region actually drawn in the annotation files, in display order.

    This is the live vocabulary.  Draw a new region in QuPath, re-run, and it
    appears here -- no code change.  ``source`` picks which directory defines
    the vocabulary ('v2' = the current drop, 'old' = the original six).
    """
    brains = brains or list(BRAINS)
    found: set[str] = set()
    for brain in brains:
        for _key, s in annotation_sources(brain).items():
            path = s.get(source) or (s.get("old") if source == "v2" else None)
            if path is None:
                continue
            found.update(load_regions(path, vocabulary=None))
    return sorted(found, key=_order_key)


# Discovered lazily on first use so that importing this module stays cheap and
# never fails when a directory is missing.
_VOCAB_CACHE: dict = {}


def all_regions(source: str = "v2") -> list[str]:
    if source not in _VOCAB_CACHE:
        _VOCAB_CACHE[source] = discover_vocabulary(source=source)
    return list(_VOCAB_CACHE[source])


def new_regions(source: str = "v2") -> list[str]:
    """Regions present in the current drop that the six-region atlas lacked."""
    return [r for r in all_regions(source) if r not in BASE_REGIONS]


BRAINS: dict[str, _BrainInfo] = {
    "queen": {
        "dir": REPO / "pipelines/wholebrain/queen",
        "annotations": REPO / "data/annotations/queen",
        "slides": ("0040240", "0040243"),
    },
    "worker": {
        "dir": REPO / "pipelines/wholebrain/worker",
        "annotations": REPO / "data/annotations/worker",
        "slides": ("0068401", "0069041"),
    },
}

PIXEL_SCALE = 0.2125


RESULTS = REPO / "results/nine_region"


def flags_table():
    """One table of region flags for every atlas cell, keyed by slice + cell_id.

    Built from the per-crop sidecars, so every consumer attaches the region
    labels the same way.
    """
    import pandas as pd

    cache = RESULTS / "region_flags.parquet"
    sidecars = sorted((RESULTS / "crops").glob("*/*.parquet"))
    if cache.exists():
        # Rebuild if any sidecar is newer than the cache. results/ is a shared
        # symlink with no git history, so a stale cache would go unnoticed.
        newest = max((p.stat().st_mtime for p in sidecars), default=0)
        if sidecars and newest <= cache.stat().st_mtime:
            return pd.read_parquet(cache)
        print(
            f"region_flags.parquet is older than the crop sidecars; "
            f"rebuilding from {len(sidecars)} files",
            flush=True,
        )

    frames = []
    for brain in BRAINS:
        d = RESULTS / "crops" / brain
        if not d.exists():
            continue
        for p in sorted(d.glob("*.parquet")):
            f = pd.read_parquet(p)
            f["brain"] = brain
            frames.append(f)
    if not frames:
        raise SystemExit("no crop sidecars; run the nine_region pipeline first")
    df = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def attach_region_flags(adata, regions=None, strict: bool = True):
    """Add in_<region> columns to `adata` for the nine-region vocabulary.

    Joins on (slice_name, cell_id), never on position: the crops carry cells
    that QC later drops, so a positional join silently shifts labels within a
    slice.  Values are nullable booleans -- pd.NA means "this section has no
    tracing for that region yet", which is not the same as absent, and must not
    be filled with False (doing so manufactures a caste difference out of
    missing data).

    Returns the list of columns added.
    """
    import numpy as np
    import pandas as pd

    regions = list(regions or all_regions("v2"))
    tab = flags_table()

    key = (
        pd.Series(adata.obs["slice_name"].astype(str)).to_numpy()
        + "||"
        + pd.Series(adata.obs["cell_id"].astype(str)).to_numpy()
    )
    tab = tab.assign(
        _k=tab["slice_name"].astype(str) + "||" + tab["cell_id"].astype(str)
    ).drop_duplicates("_k")
    idx = tab.set_index("_k").index.get_indexer(key)

    missing = int((idx < 0).sum())
    if missing and strict:
        raise SystemExit(
            f"{missing:,} of {len(key):,} cells have no region flags -- refusing "
            "to attach partial labels. Pass strict=False only if you have "
            "checked that the unmatched cells are genuinely out of scope."
        )

    tab = tab.set_index("_k")
    added = []
    for r in regions:
        col = f"in_{r}"
        if col not in tab.columns:
            continue
        vals = tab[col].to_numpy()[idx]
        arr = pd.array(vals, dtype="boolean")
        if missing:
            arr[np.asarray(idx) < 0] = pd.NA
        adata.obs[col] = arr
        added.append(col)
    return added


KC_REGIONS = ("kc_oc", "kc_ic", "kc_nc")
OPTIC_REGIONS = ("lamina", "medulla", "lobula")

#: region_of() labels for cells in no single region.
OUTSIDE = "outside"  # traced for every region asked about, inside none
OVERLAP = "overlap"  # inside two or more (overlap="drop")


def region_mask(obs, regions):
    """Cells inside any of ``regions``, from the ``in_<region>`` columns.

    Kleene logic on the nullable flags: True if any flag is True, NA if none is
    True and at least one is NA (an untraced section), else False. Returns a
    pandas BooleanArray; restrict to ``mask.notna()`` (traced) before treating
    False as "outside", and never fill the NA with False.
    """
    import pandas as pd

    out = pd.array([False] * len(obs), dtype="boolean")
    for r in regions:
        out = out | pd.array(obs[f"in_{r}"], dtype="boolean")
    return out


def kc_mask(obs):
    """Cells in any Kenyon cell compartment (see :func:`region_mask`)."""
    return region_mask(obs, KC_REGIONS)


def optic_mask(obs):
    """Cells in any optic neuropil (see :func:`region_mask`)."""
    return region_mask(obs, OPTIC_REGIONS)


def traced_mask(mask):
    """A :func:`region_mask` result as a plain bool array.

    Raises if any cell is NA (its section is untraced for one of the regions):
    restrict to traced sections first rather than filling NA with False.
    """
    import numpy as np
    import pandas as pd

    m = pd.array(mask, dtype="boolean")
    n_na = int(m.isna().sum())
    if n_na:
        raise ValueError(
            f"{n_na:,} cells are untraced (NA) for these regions; restrict to "
            "traced sections before treating the mask as bool"
        )
    return np.asarray(m.to_numpy(dtype=bool))


def region_of(obs, regions=None, overlap: str = "drop"):
    """One exclusive region per cell, from the ``in_<region>`` columns.

    Returns an object array holding the region name for a cell inside exactly
    one of ``regions`` (default: :data:`DESCRIPTIVE_REGIONS` present in
    ``obs``); :data:`OUTSIDE` for a cell traced as inside none; ``None`` for a
    cell inside none whose section lacks a tracing for any of them (untraced);
    and for a cell inside two or more, :data:`OVERLAP` (``overlap="drop"``) or
    the first in ``regions`` order (``overlap="first"``).

    Every producer of per-region counts, compositions or classifier labels goes
    through this, so a figure, its counts and its ratios use one cell set.
    """
    import numpy as np
    import pandas as pd

    if overlap not in ("drop", "first"):
        raise ValueError(f"overlap must be 'drop' or 'first', not {overlap!r}")
    if regions is None:
        regions = [r for r in DESCRIPTIVE_REGIONS if f"in_{r}" in obs]
    regions = list(regions)
    flags = np.column_stack(
        [
            pd.array(obs[f"in_{r}"], dtype="boolean").to_numpy(
                dtype=float, na_value=np.nan
            )
            for r in regions
        ]
    )
    inside = flags == 1
    n_in = inside.sum(axis=1)
    any_na = np.isnan(flags).any(axis=1)
    out = np.full(len(flags), None, dtype=object)
    out[(n_in == 0) & ~any_na] = OUTSIDE
    one = n_in == 1
    out[one] = np.asarray(regions, dtype=object)[inside[one].argmax(axis=1)]
    many = n_in > 1
    if overlap == "drop":
        out[many] = OVERLAP
    else:
        out[many] = np.asarray(regions, dtype=object)[inside[many].argmax(axis=1)]
    return out


def _load_labeling(brain: str):
    """Import a pipeline's labeling.py without importing its whole package."""
    path = BRAINS[brain]["dir"] / "labeling.py"
    spec = importlib.util.spec_from_file_location(f"_labeling_{brain}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load labeling.py for {brain} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_slide_section(name: str):
    m = re.search(r"__(\d{7})__Region_(\d+)__", name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def crop_map(brain: str) -> dict:
    """(slide, section) -> list of crop suffixes, re-keyed from ANNOTATION_MAPPING."""
    mapping = _load_labeling(brain).ANNOTATION_MAPPING
    out: dict = {}
    for fname, crops in mapping.items():
        key = parse_slide_section(fname)
        if key[0] is None:
            raise ValueError(f"unparseable ANNOTATION_MAPPING key: {fname}")
        if key in out:
            raise ValueError(f"duplicate section in ANNOTATION_MAPPING: {key}")
        out[key] = list(crops)
    return out


def annotation_sources(brain: str, newest_dir: str | None = None) -> dict:
    """(slide, section) -> {'v2': path|None, 'old': path|None}.

    Sections present in the newest set use it.  Sections absent from it fall
    back to the original set (v1/) so their six existing regions
    are preserved.

    ``newest_dir`` names the newest set explicitly. Any script that writes one
    of these directories must pass it: the default picks the newest directory
    that exists, even one the caller has just created empty, and would then
    read its own empty output as the source.
    """
    base = BRAINS[brain]["annotations"]
    src: dict = {}
    # v3 is the working set: the original labels with the three added
    # regions appended, nothing else changed. v2 (the export the added regions
    # were taken from) holds a doubled file and two stray objects; it is used
    # only if v3 has not been built.
    if newest_dir is not None:
        newest = base / newest_dir
    else:
        # Newest first. v4 = v3 plus the nine-region relabel, deduplicated and
        # with every polygon name standardised; v3 = the original labels with
        # the three new regions appended and one section reclassified.
        newest = base / "v2"
        for cand in ("v3", "v4"):
            if (base / cand).exists():
                newest = base / cand
        # An existing but empty newest set is a build in progress (the builders
        # rmtree + mkdir it). Falling back to the original six regions here
        # would silently shrink the region vocabulary for the whole process.
        if newest.exists() and not any(newest.glob("*.geojson")):
            raise RuntimeError(
                f"{newest} exists but holds no GeoJSON (a rebuild in progress?); "
                "pass newest_dir= explicitly"
            )
    for tag, d in (("old", base / "v1"), ("v2", newest)):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.geojson")):
            key = parse_slide_section(f.name)
            if key[0] is None:
                continue
            src.setdefault(key, {}).setdefault(tag, f)
    for key in src:
        src[key].setdefault("v2", None)
        src[key].setdefault("old", None)
    return src


def canonical_name(props: dict) -> str | None:
    """Annotator's raw labelling -> canonical region name, or None to skip."""
    cl = props.get("classification")
    raw = (cl.get("name") if isinstance(cl, dict) else cl) or props.get("name", "")
    name = re.sub(r"\s+", " ", str(raw)).strip().lower()

    if name in SPLIT_CLASSES:
        sub = props.get("subregion")
        if not sub:
            m = re.search(r"_(eb|fb)(?![a-z])", str(props.get("name", "")))
            if not m:
                return None  # unlabelled substructure: cannot assign
            sub = SPLIT_CLASSES[name][m.group(1)]
        sub = str(sub).strip().lower()
        sub = ALIASES.get(sub, sub) or None
        # Pool after resolving which substructure it is, so `subregion` still
        # decides and switching POOL_CENTRAL_COMPLEX off needs nothing else.
        if sub in ("ellipsoid_body", "fan_shaped_body") and POOL_CENTRAL_COMPLEX:
            return CENTRAL_COMPLEX
        return sub

    name = ALIASES.get(name, name)
    if name in ("ellipsoid_body", "fan_shaped_body") and POOL_CENTRAL_COMPLEX:
        return CENTRAL_COMPLEX
    if not name or name in EXCLUDE:
        return None
    if any(p.match(name) for p in EXCLUDE_PATTERNS):
        return None
    return name


def load_regions(
    path: Path,
    pixel_scale: float = PIXEL_SCALE,
    vocabulary: list[str] | Literal["auto"] | None = "auto",
    repair_invalid: bool = True,
    report: list | None = None,
) -> dict:
    """GeoJSON -> {region_name: shapely geometry}.

    Differs from ``labeling.load_geojson_regions`` in three ways:

    * central-complex substructures become regions in their own right;
    * region names are normalised through ``ALIASES`` and whitespace-collapsed,
      so a stray trailing space cannot silently create a second column;
    * self-intersecting polygons are repaired rather than silently dropped
      (the original drops them; e.g. a 439-cell KC-NC polygon on queen slide
      0040240 section 6). Pass ``repair_invalid=False`` to reproduce the
      original behaviour.

    ``vocabulary=None`` accepts every region drawn (used for discovery);
    ``"auto"`` accepts the discovered vocabulary; a list restricts to it.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    with open(path) as fh:
        obj = json.load(fh)
    feats = obj["features"] if isinstance(obj, dict) else obj

    allowed = None
    if vocabulary == "auto":
        # Filter to the discovered vocabulary. Discovery itself calls in with
        # vocabulary=None, so this cannot recurse; a discovery failure raises.
        allowed = set(all_regions("v2"))
    elif vocabulary is not None:
        allowed = set(vocabulary)

    by_name: dict[str, list] = {}
    for ft in feats:
        props = ft.get("properties", {}) or {}
        name = canonical_name(props)
        if not name:
            continue
        if allowed is not None and name not in allowed:
            continue

        geom = ft.get("geometry", {}) or {}
        gtype = geom.get("type")
        if gtype not in ("Polygon", "MultiPolygon"):
            if gtype and report is not None:
                report.append(dict(file=path.name, region=name, skipped_geometry=gtype))
            continue

        # A region drawn as two disjoint blobs on one section arrives as a
        # MultiPolygon. Silently skipping it would mark every cell in both
        # blobs as outside the region, with no diagnostic trail.
        parts = (
            (geom.get("coordinates") or [])
            if gtype == "MultiPolygon"
            else [geom.get("coordinates") or []]
        )

        for coords in parts:
            if not coords:
                continue
            rings = coords
            if pixel_scale is not None:
                rings = [
                    [[x * pixel_scale, y * pixel_scale] for x, y in r] for r in rings
                ]
            ring, holes = rings[0], rings[1:]
            if len(ring) < 3:
                continue
            # Interior rings are holes the annotator cut out of the region;
            # ignoring them would silently count cells inside the hole as
            # belonging to the region.
            poly = Polygon(ring, holes=[h for h in holes if len(h) >= 3])
            if not poly.is_valid:
                if report is not None:
                    report.append(
                        dict(
                            file=path.name,
                            region=name,
                            n_vertices=len(ring),
                            area=float(poly.buffer(0).area),
                        )
                    )
                if not repair_invalid:
                    continue  # the original pipeline's silent drop
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty:
                by_name.setdefault(name, []).append(poly)

    return {
        n: (polys[0] if len(polys) == 1 else unary_union(polys))
        for n, polys in by_name.items()
    }
