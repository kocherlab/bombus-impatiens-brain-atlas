#!/usr/bin/env python
"""Top-500 genes per anatomical region of the Xenium atlas, measured vs inferred.

The spatial counterpart of snrna_markers_top500.py: the groups are the hand-drawn
brain regions instead of snRNA cell types. Each region is compared with the cells
of the same atlas that were traced and lie outside it.

Expression is the Tangram projection of the snRNA reference onto every Xenium cell
(combined corrected run, every reference gene and atlas cell; read from disk in
row blocks).
Only the 100 genes on the Xenium panel are MEASURED; every other gene is INFERRED
from the snRNA reference through each cell's cell-type mapping. An inferred gene
therefore says what the region's cell types express in snRNA, not an independent
spatial observation, and each row is labelled Measured or Inferred.

Regions (bimp_st.nine_region_common): the seven statistical regions plus the
central complex, which is descriptive only. A cell that was never traced for a
region (nullable flag) is left out of that region's comparison, never counted as
outside it.

Score: Cohen's d (standardised mean difference, pooled SD) of region versus
elsewhere, used to rank. Unlike a t statistic it does not grow with the number of
cells, so it is comparable across regions of very different size. Cells are not
independent and there is one brain per caste, so no p-values are given.

Outputs (results/handoff/D_tables/):
    spatial_markers_top500.csv      long format, one row per (region, rank)
    spatial_markers_top500.xlsx     README, one tab per region, a Measured-genes tab
    spatial_markers_panel_scores.csv  the 100 measured genes x regions (score)

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/gene_annotation/spatial_markers_top500.py
"""

# ruff: noqa: E501  (README text strings)
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tables_s1_s2 as t  # noqa: E402

from bimp_st import nine_region_common as nr  # noqa: E402

N = 500
BLOCK = 25_000
PSEUDO = 1e-3
ATLAS = t.ROOT / "results/tangram/combined_clustered_g2_R2_s42_imputed_atlas_v2.h5ad"
REGIONS = list(nr.all_regions("v2"))  # statistical seven + central_complex
NAMES = {**nr.DISPLAY, "central_complex": "Central complex"}

f = h5py.File(ATLAS)
var = read_elem(f["var"])
obs = read_elem(f["obs"])
gene_ids = np.array([g.replace("bimp_v4.3", "BIMP_v4.3") for g in var.index])
n_cells, n_genes = f["X"].shape

# region flags: nullable, joined on (slice_name, cell_id)
tmp = ad.AnnData(obs=obs[["slice_name", "cell_id"]].copy())
nr.attach_region_flags(tmp, regions=REGIONS)
inn = np.zeros((len(REGIONS), n_cells), dtype=np.float32)
known = np.zeros_like(inn)
for i, r in enumerate(REGIONS):
    s = tmp.obs[f"in_{r}"]
    known[i] = s.notna().to_numpy()
    inn[i] = s.fillna(False).astype(bool).to_numpy()
print({r: int(inn[i].sum()) for i, r in enumerate(REGIONS)}, flush=True)

# stream X once: per-region sums and sums of squares, inside and among traced cells
R = len(REGIONS)
S = np.zeros((2 * R, n_genes))
Q = np.zeros((2 * R, n_genes))
M = np.vstack([inn, known])
X = f["X"]
for a in range(0, n_cells, BLOCK):
    b = min(a + BLOCK, n_cells)
    x = X[a:b].astype(np.float32)
    m = M[:, a:b]
    S += m @ x
    Q += m @ (x * x)
    print(f"  {b:,}/{n_cells:,}", flush=True)
n1, nk = M[:R].sum(1), M[R:].sum(1)
n0 = nk - n1
m1, m0 = S[:R] / n1[:, None], (S[R:] - S[:R]) / n0[:, None]
v1 = (Q[:R] - n1[:, None] * m1**2) / (n1[:, None] - 1)
v0 = (Q[R:] - Q[:R] - n0[:, None] * m0**2) / (n0[:, None] - 1)
pooled = np.sqrt(
    (
        (n1[:, None] - 1) * np.clip(v1, 0, None)
        + (n0[:, None] - 1) * np.clip(v0, 0, None)
    )
    / (n1[:, None] + n0[:, None] - 2)
    + 1e-30
)
tscore = (m1 - m0) / pooled  # Cohen's d
lfc = np.log2((m1 + PSEUDO) / (m0 + PSEUDO))

s1 = t.build_s1(set(gene_ids))
panel_ids = set(s1.v43_id)
sym = {r.v43_id: r.symbol for r in s1.itertuples()}
lk = pd.read_csv(t.LOOKUP, sep="\t").set_index("gene_id")
measured = np.array([g in panel_ids for g in gene_ids])


def annot(g):
    s = sym.get(g) or lk["gene"].get(g)
    d = lk["description"].get(g)
    return (s if isinstance(s, str) else ""), (d if isinstance(d, str) else "")


rows = []
for i, r in enumerate(REGIONS):
    for rank, j in enumerate(np.argsort(-tscore[i])[:N], 1):
        s, d = annot(gene_ids[j])
        rows.append(
            dict(
                region=NAMES[r],
                rank=rank,
                gene_id=gene_ids[j],
                symbol=s,
                description=d,
                basis="Measured" if measured[j] else "Inferred",
                score=tscore[i, j],
                log2fc=lfc[i, j],
                mean_in=m1[i, j],
                mean_out=m0[i, j],
            )
        )
long = pd.DataFrame(rows)
long.to_csv(t.OUT / "spatial_markers_top500.csv", index=False)

pj = np.where(measured)[0]
panel = pd.DataFrame(
    {NAMES[r]: tscore[i, pj] for i, r in enumerate(REGIONS)},
    index=pd.Index(gene_ids[pj], name="gene_id"),
)
panel.insert(0, "symbol", [annot(g)[0] for g in panel.index])
panel.insert(1, "description", [annot(g)[1] for g in panel.index])
panel.to_csv(t.OUT / "spatial_markers_panel_scores.csv")

counts = long.groupby("region", sort=False).basis.apply(
    lambda s: (s == "Measured").sum()
)
readme = [
    ("Top 500 genes per brain region, B. impatiens Xenium atlas", ""),
    ("", ""),
    (
        "What this is",
        "For each hand-drawn region, the 500 genes that best separate its cells "
        "from the other traced cells of the atlas (queen and worker pooled, 1.3M cells). "
        "Regions: lamina, medulla, lobula, the three Kenyon-cell compartments (Class II, Small-type, "
        "Large-type), antennal lobe, and central complex.",
    ),
    (
        "Measured vs inferred",
        "Only 100 genes are on the Xenium panel and were measured. "
        "Every other gene is INFERRED: projected onto the spatial cells from the snRNA reference "
        "(Tangram), so it reflects what each region's cell types express in snRNA, not a direct spatial "
        "measurement. The Measured column is TRUE for measured genes and FALSE for inferred ones. The last tab ranks only the measured genes.",
    ),
    (
        "Counts",
        "Measured genes in each region's top 500: "
        + "; ".join(f"{k} {v}" for k, v in counts.items())
        + ".",
    ),
    (
        "Score",
        "Cohen's d: the difference in mean expression between the region and elsewhere, in units of "
        "the pooled standard deviation. It does not grow with the number of cells, so it is comparable "
        "across regions of different size. No p-values: cells are not independent and there is one "
        "brain per caste.",
    ),
    (
        "Central complex",
        "Descriptive only. The traced central-complex cells are almost all glia "
        "(neuropil), so its list is largely glial genes.",
    ),
    (
        "Caveat",
        "Inferred genes mostly track cell-type composition, so they rank cell types that are "
        "enriched in a region. Treat them as candidates to check, not as spatial results.",
    ),
    (
        "Source",
        "scripts/gene_annotation/spatial_markers_top500.py in the bimp-st repository.",
    ),
]
out_x = t.OUT / "spatial_markers_top500.xlsx"
COLS = {
    "rank": "Rank",
    "gene_id": "Gene id",
    "symbol": "Symbol",
    "description": "Description",
    "basis": "Basis",
    "score": "Effect size (Cohen's d)",
    "log2fc": "Log2 fold change",
    "mean_in": "Mean in region",
    "mean_out": "Mean elsewhere",
}
WIDTH = [7, 18, 16, 60, 11, 15, 16, 15, 15]
with pd.ExcelWriter(out_x, engine="openpyxl") as xw:
    pd.DataFrame(readme).to_excel(xw, sheet_name="README", index=False, header=False)
    for r in REGIONS:
        sub = long[long.region == NAMES[r]][list(COLS)].rename(columns=COLS)
        assert len(sub) == N
        sub["Basis"] = sub["Basis"] == "Measured"
        sub = sub.rename(columns={"Basis": "Measured (TRUE = on Xenium panel, FALSE = inferred)"})
        sub.to_excel(xw, sheet_name=NAMES[r][:31], index=False)
    panel.reset_index().to_excel(xw, sheet_name="Measured genes only", index=False)
wb = load_workbook(out_x)
plain, bold = Font(bold=False), Font(bold=True)
for ws in wb.worksheets:
    for row in ws.iter_rows():
        for c in row:
            c.font, c.alignment = plain, Alignment(vertical="top", wrap_text=False)
ws = wb["README"]
ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 20, 110
ws["A1"].font = Font(bold=True, size=13)
for row in ws.iter_rows(min_row=3, max_col=1):
    row[0].font = bold
for row in ws.iter_rows(min_row=3, min_col=2, max_col=2):
    row[0].alignment = Alignment(wrap_text=True, vertical="top")
for ws in wb.worksheets[1:]:
    for c in ws[1]:
        c.font = bold
    widths = WIDTH if ws.title != "Measured genes only" else [18, 16, 60] + [14] * 8
    for col, w in zip("ABCDEFGHIJK", widths, strict=False):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
wb.save(out_x)
print("wrote", out_x.name, counts.to_dict())
