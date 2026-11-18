"""Headline coverage and per-region counts for the nine-region atlas.

Counts the cells that fall within exactly one annotated region of the
nine-region atlas, and computes the majority-class null that the
region-prediction analysis is scored against (balanced baseline 1/9).

Honest treatment of the two pending sections
--------------------------------------------
Worker slide 0068401 sections 1 and 4 have no antennal-lobe or central-complex
tracing.  A cell there that falls in none of the six original regions cannot be
called "outside every annotated region" -- it might sit in an untraced antennal
lobe.  Counting it as outside would inflate coverage's complement and bias the
worker's antennal lobe toward zero.

So coverage is reported on the FULLY ANNOTATED subset, with the pending cells
reported separately as a stated exclusion rather than folded in either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (  # noqa: E402
    BASE_REGIONS,
    REPO,
    STATISTICAL_REGIONS,
    all_regions,
    display,
    new_regions,
    region_mask,
)

ALL_REGIONS = all_regions("v2")
NEW_REGIONS = new_regions("v2")
DISPLAY = {r: display(r) for r in ALL_REGIONS}

OUT = REPO / "results/nine_region"


def load_side(brain: str) -> pd.DataFrame:
    """Region flags for the ATLAS cells of one brain.

    Deliberately reads the joined atlas object, not the crop sidecars.  The
    crops still hold the cells that QC drops during alignment, and the
    coverage denominator is the atlas.  Summing sidecars would
    quietly inflate every count by those dropped cells.
    """
    import h5py

    path = OUT / "atlas" / f"{brain}_aligned_clustered_9region.h5ad"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --stage join first")

    out = {}
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        for r in ALL_REGIONS:
            key = f"in_{r}"
            if key not in obs:
                continue
            node = obs[key]
            # nullable boolean round-trips as a group with values + mask
            if isinstance(node, h5py.Group):
                vals = node["values"][:].astype(bool)
                mask = node["mask"][:].astype(bool) if "mask" in node else None
            else:
                vals, mask = node[:].astype(bool), None
            s = pd.array(vals, dtype="boolean")
            if mask is not None:
                s[mask] = pd.NA
            out[key] = s
    df = pd.DataFrame(out)
    df["brain"] = brain
    return df


def summarise(df: pd.DataFrame, regions: list[str], label: str) -> dict:
    """Exclusive-category coverage over cells where every region is annotated."""
    cols = [f"in_{r}" for r in regions]
    sub = df[cols]

    annotated_everywhere = ~sub.isna().any(axis=1)
    n_total = len(df)
    n_pending = int((~annotated_everywhere).sum())
    d = sub[annotated_everywhere].astype(bool)

    k = d.sum(axis=1)
    n_eval = len(d)
    n_one = int((k == 1).sum())
    n_multi = int((k > 1).sum())
    n_none = int((k == 0).sum())

    per_region = {r: int(d[f"in_{r}"].sum()) for r in regions}
    # exclusive counts: cells in exactly this one region
    excl = {r: int((d[f"in_{r}"] & (k == 1)).sum()) for r in regions}
    maj = max(excl.values()) / n_one if n_one else float("nan")

    return dict(
        label=label,
        n_regions=len(regions),
        n_total_cells=n_total,
        n_pending_excluded=n_pending,
        n_evaluated=n_eval,
        n_in_exactly_one=n_one,
        pct_in_exactly_one=100 * n_one / n_eval if n_eval else float("nan"),
        n_in_multiple=n_multi,
        n_in_none=n_none,
        pct_in_none=100 * n_none / n_eval if n_eval else float("nan"),
        majority_class_null=maj,
        balanced_null=1 / len(regions),
        per_region_any=per_region,
        per_region_exclusive=excl,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dfs = {b: load_side(b) for b in ("queen", "worker")}
    both = pd.concat(dfs.values(), ignore_index=True)

    report = {}
    report["six_region_atlas"] = summarise(
        both, BASE_REGIONS, "six regions (as published)"
    )

    # Each region has its OWN denominator, and they differ a lot: the antennal
    # lobe was never traced on worker slide 0069041, and the central complex
    # never on queen slide 0040240.  A single "nine-region coverage" figure
    # would require every region to be annotated for a cell to count, which
    # discards most cells and drags each region's number down for reasons
    # that have nothing to do with that region.  Report per region instead.
    per_region = {}
    for r in ALL_REGIONS:
        col = f"in_{r}"
        for scope, d in (
            ("both", both),
            ("queen", dfs["queen"]),
            ("worker", dfs["worker"]),
        ):
            s = d[col]
            denom = int(s.notna().sum())
            per_region.setdefault(r, {})[scope] = dict(
                annotated_cells=denom,
                not_annotated_cells=int(s.isna().sum()),
                in_region=int(s.sum(skipna=True)),
                pct_of_annotated=(
                    100 * float(s.sum(skipna=True)) / denom if denom else float("nan")
                ),
            )
    report["per_region"] = per_region

    # The set the fitted analyses use: six + antennal lobe.  Its evaluable
    # subset excludes only the sections with no AL tracing.
    report["seven_region_statistical"] = summarise(
        both, STATISTICAL_REGIONS, "seven regions (six + antennal lobe), fitted set"
    )
    sevencols = [f"in_{r}" for r in STATISTICAL_REGIONS]
    matched = both[~both[sevencols].isna().any(axis=1)]
    report["six_region_matched_to_seven"] = summarise(
        matched, BASE_REGIONS, "six regions, on the seven-region evaluable cells"
    )
    for b, d in dfs.items():
        report[f"nine_region_{b}"] = summarise(d, ALL_REGIONS, f"nine regions, {b}")
        report[f"six_region_{b}"] = summarise(d, BASE_REGIONS, f"six regions, {b}")

    # What the three new regions add, per brain, on fully annotated cells only.
    # Each new region judged on the cells where THAT region was annotated.
    # Requiring all three jointly would judge the central complex on the slide
    # where the antennal lobe is missing, which drove worker ellipsoid body
    # down to 2 cells for reasons that have nothing to do with the ellipsoid
    # body.
    add = {}
    for b, d in dfs.items():
        base_any = region_mask(d, BASE_REGIONS)
        rows = {}
        for r in NEW_REGIONS:
            s = d[f"in_{r}"]
            ann = s.notna()
            # cells annotated for r and for every base region; an untraced cell
            # is in neither count
            ok = ann.to_numpy(dtype=bool) & ~base_any.isna()
            inr = s[ok].astype(bool).to_numpy()
            base = base_any[ok].astype(bool)
            rows[r] = dict(
                annotated_cells=int(ann.sum()),
                not_annotated_cells=int((~ann).sum()),
                in_region=int(s.sum(skipna=True)),
                newly_covered=int((inr & ~base).sum()),
                overlap_with_existing=int((inr & base).sum()),
            )
        add[b] = rows
    report["new_region_gain"] = add

    (OUT / "coverage.json").write_text(json.dumps(report, indent=2, default=str))

    def line(k):
        r = report[k]
        print(f"\n--- {r['label']} ---")
        print(
            f"  evaluated cells      : {r['n_evaluated']:,}"
            f"  (excluded, not yet annotated: {r['n_pending_excluded']:,})"
        )
        print(
            f"  in exactly one region: {r['n_in_exactly_one']:,} "
            f"({r['pct_in_exactly_one']:.1f}%)"
        )
        print(f"  in more than one     : {r['n_in_multiple']:,}")
        print(f"  in none              : {r['n_in_none']:,} ({r['pct_in_none']:.1f}%)")
        print(
            f"  majority-class null  : {r['majority_class_null']:.3f}"
            f"   balanced null: 1/{r['n_regions']} = {r['balanced_null']:.3f}"
        )
        for reg, n in sorted(r["per_region_exclusive"].items(), key=lambda t: -t[1]):
            print(f"     {DISPLAY[reg]:<16} {n:>9,}")

    print("=" * 64)
    line("six_region_atlas")
    line("six_region_matched_to_seven")
    line("seven_region_statistical")

    print("\n--- per region, each against its OWN denominator ---")
    print(
        f"  {'region':<17}{'annotated':>12}{'in region':>11}{'% of ann.':>10}"
        f"   {'not annotated':>13}"
    )
    for r in ALL_REGIONS:
        s = per_region[r]["both"]
        print(
            f"  {DISPLAY[r]:<17}{s['annotated_cells']:>12,}{s['in_region']:>11,}"
            f"{s['pct_of_annotated']:>9.1f}%   {s['not_annotated_cells']:>13,}"
        )

    a = report["six_region_matched_to_seven"]
    b = report["seven_region_statistical"]
    print("\n--- like-for-like gain (same cells, six regions -> nine) ---")
    print(
        f"  in exactly one region: {a['n_in_exactly_one']:,} "
        f"({a['pct_in_exactly_one']:.1f}%)  ->  {b['n_in_exactly_one']:,} "
        f"({b['pct_in_exactly_one']:.1f}%)"
    )
    print(
        f"  gain: {b['n_in_exactly_one'] - a['n_in_exactly_one']:,} cells, "
        f"{b['pct_in_exactly_one'] - a['pct_in_exactly_one']:+.1f} percentage points"
    )
    print("\n--- what the three new regions add, per brain ---")
    for b, rows in add.items():
        print(f"  {b}:")
        for r, s in rows.items():
            print(
                f"     {DISPLAY[r]:<16} {s['in_region']:>7,} cells "
                f"({s['newly_covered']:,} newly covered, "
                f"{s['overlap_with_existing']:,} overlapping an existing region) "
                f"| annotated on {s['annotated_cells']:,} cells, "
                f"{s['not_annotated_cells']:,} not annotated"
            )
    print(f"\nwrote {OUT / 'coverage.json'}")


if __name__ == "__main__":
    main()
