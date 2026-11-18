"""Depth control for the region comparison: KC cells carry more transcripts per cell
than optic/antennal-lobe cells. Compare regions WITHIN bins of LP calls per cell, so the
comparison is at matched depth.

Input : results/bento/by_section/region_cells_{queen,worker}.parquet (per cell)
Output: results/bento/regions_summary/depth_matched_{fractions,contrast}.csv, depth_matched.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bimp_st.si_common import QUEEN_COLOR, WORKER_COLOR, set_style

R = Path(__file__).resolve().parents[2] / "results/bento"
OUT = R / "regions_summary"
CAST = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}
KC = ["KC-IC", "KC-NC", "KC-OC"]
OPT = ["lamina", "medulla", "lobula", "al"]
EDGES = [20, 30, 40, 50, 60, 80, 120]
lines = []
fr = []
for caste in CAST:
    c = pd.read_parquet(R / f"by_section/region_cells_{caste}.parquet")
    c = c[c.comp.isin(KC + OPT)]
    c["bin"] = pd.cut(c.n_calls, EDGES, right=False)
    c = c.dropna(subset=["bin"])
    g = c.groupby(["comp", "bin"], observed=True)[
        ["nuclear", "none_only", "n_calls", "n_classified"]
    ].sum()
    g["n_cells"] = c.groupby(["comp", "bin"], observed=True).size()
    # bins are on all calls per cell (depth); fractions are among classified calls
    g["frac_nuclear"] = g.nuclear / g.n_classified
    g["frac_none"] = g["none_only"] / g.n_classified
    g["caste"] = caste
    fr.append(g.reset_index())
fr = pd.concat(fr)
fr.to_csv(OUT / "depth_matched_fractions.csv", index=False)

rows = []
for caste, g in fr.groupby("caste"):
    for b, d in g.groupby("bin", observed=True):
        k = d[d.comp.isin(KC)]
        o = d[d.comp.isin(OPT)]
        if k.n_cells.sum() >= 200 and o.n_cells.sum() >= 200:
            kn = k.nuclear.sum() / k.n_classified.sum()
            on = o.nuclear.sum() / o.n_classified.sum()
            rows.append(dict(caste=caste, depth_bin=str(b), kc_cells=int(k.n_cells.sum()),
                             optic_cells=int(o.n_cells.sum()), kc_nuclear=kn, optic_nuclear=on,
                             ratio=on / kn))
con = pd.DataFrame(rows)
con.to_csv(OUT / "depth_matched_contrast.csv", index=False)
lines.append("Optic+AL vs KC nuclear fraction within calls-per-cell bins (>=200 cells each side)")
lines.append(con.round(3).to_string(index=False))
lines.append(f"\nratio range across matched bins: {con.ratio.min():.2f}-{con.ratio.max():.2f}; "
             f"optic > KC in {(con.ratio > 1).sum()}/{len(con)} bins")
print("\n".join(lines), flush=True)
(OUT / "depth_matched_summary.txt").write_text("\n".join(lines))

set_style()
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, caste in zip(ax, CAST, strict=False):
    g = fr[fr.caste == caste]
    for comp in KC + OPT:
        d = g[(g.comp == comp) & (g.n_cells >= 100)]
        x = [b.mid for b in d.bin]
        a.plot(x, d.frac_nuclear, "-o" if comp in OPT else "--s", ms=3, label=comp)
    a.set_title(caste, color=CAST[caste])
    a.set_xlabel("LP calls per cell"); a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("nuclear fraction"); ax[1].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "depth_matched.png", dpi=300)
