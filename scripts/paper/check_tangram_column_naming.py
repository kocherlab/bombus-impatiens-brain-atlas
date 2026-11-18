"""Check that Tangram posterior columns are named by identity, not by position.

Tiny synthetic clusters-mode run whose cluster sizes are deliberately not in
alphabetical order, so the two candidate orderings disagree on 4 of 5 columns.
Asserts that

  1. ``ad_map.obs[cluster_label]`` is in ``value_counts()`` (frequency) order,
     while ``ad_map.obs_names`` are bare positional integers, so names taken by
     position would land on the wrong columns;
  2. ``tg.project_cell_annotations`` yields a pure column permutation of
     ``ad_map.X.T`` (values untouched) with names on the right columns;
  3. ``tangram_mapping._check_column_identity`` passes the correct naming and
     fails the alphabetical naming.

Run with the .venv interpreter (~4 min, CPU):
    PYTHONPATH=src:scripts .venv/bin/python scripts/paper/check_tangram_column_naming.py
"""

import anndata as ad
import numpy as np
import pandas as pd
import torch

# login node has a CUDA-built torch but no device; the Adam health check calls
# into CUDA regardless of device="cpu". Test-harness only.
torch.backends.cuda.is_built = lambda: False

import tangram2.mapping as tg  # noqa: E402

rng = np.random.default_rng(0)
# 5 types whose frequency order is not alphabetical
names = ["Aaa", "Bbb", "Ccc", "Ddd", "Eee"]
counts = [40, 120, 20, 200, 60]  # freq order: Ddd,Bbb,Eee,Aaa,Ccc
G = 60
sig = rng.random((5, G)) * 5
rows, lab = [], []
for i, (n, c) in enumerate(zip(names, counts, strict=False)):
    rows.append(rng.poisson(sig[i], size=(c, G)))
    lab += [n] * c
Xr = np.vstack(rows).astype(np.float32)
ref = ad.AnnData(
    X=Xr,
    obs=pd.DataFrame({"cell_type": pd.Categorical(lab)}),
    var=pd.DataFrame(index=[f"g{j}" for j in range(G)]),
)
ref.obs_names = [f"c{i}" for i in range(ref.n_obs)]

pick = rng.integers(0, 5, 300)
Xs = rng.poisson(sig[pick] * 3).astype(np.float32)
sp = ad.AnnData(X=Xs, var=pd.DataFrame(index=[f"g{j}" for j in range(G)]))
sp.obs_names = [f"s{i}" for i in range(sp.n_obs)]

tg.pp_adatas(ref, sp, genes=[f"g{j}" for j in range(G)])
m = tg.map_cells_to_space(
    ref,
    sp,
    device="cpu",
    num_epochs=30,
    density_prior="rna_count_based",
    mode="clusters",
    cluster_label="cell_type",
)

print("ad_map.obs_names      :", list(m.obs_names))
print("ad_map.obs.cell_type  :", list(m.obs["cell_type"]))
alpha = dict(enumerate(sorted(names)))
old = [alpha[int(c)] for c in m.obs_names]
print("OLD (alphabetical)    :", old)

tg.project_cell_annotations(m, sp, annotation="cell_type")
new = list(sp.obsm["tangram_ct_pred"].columns)
print("NEW (project_cell_ann):", new)

true_order = list(ref.obs["cell_type"].value_counts().index)
print("TRUE row order        :", true_order)
assert list(m.obs["cell_type"]) == true_order
assert old != true_order, "test is not exercising the bug"

M = np.asarray(m.X).T
P = sp.obsm["tangram_ct_pred"].to_numpy()
for j, nm in enumerate(new):
    k = true_order.index(nm)
    assert np.allclose(P[:, j], M[:, k]), nm
print("\nOK: values are a pure column permutation; names now match row order")
print(
    "columns positional naming would misname:",
    sum(a != b for a, b in zip(old, true_order, strict=False)),
    "/",
    len(names),
)

# --- self-check function on this toy run -------------------------------------
import sys

sys.path.insert(0, "pipelines/combined_atlas")
from tangram_mapping import _check_column_identity  # noqa: E402

ctp = sp.obsm["tangram_ct_pred"]
ctp = ctp / ctp.to_numpy().sum(1, keepdims=True)
ok, df, summ = _check_column_identity(ctp, sp, ref, "cell_type")
print("\nself-check on CORRECT naming:", ok, summ)
print(df.to_string(index=False))

bad = ctp.copy()
bad.columns = old  # alphabetical (incorrect) naming
ok2, df2, summ2 = _check_column_identity(bad, sp, ref, "cell_type")
print("\nself-check on MISNAMED columns:", ok2, summ2)
assert ok, "self-check should PASS on correct naming"
assert not ok2, "self-check should FAIL on the historical mis-naming"
print("\nSELF-CHECK BEHAVES: passes correct naming, fails the known bug")
