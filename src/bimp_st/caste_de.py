"""The snRNA caste-DE recipe (design: I. Traniello), shared by every script.

- one pseudobulk sample per caste x trial (``BIMP_{Queen,Worker}_t{1,2,3}``),
  ambient-corrected (CellBender) counts summed over the nuclei of the cell type;
- a SAMPLE is dropped when it has fewer than ``MIN_NUCLEI`` nuclei of the type,
  and a trial is kept only as a whole queen + worker pair, so its partner is
  dropped too (:func:`retained_samples`); the type is tested only when all
  ``MIN_PER_CASTE`` (three) trial pairs remain, because ``~ trial + caste`` on
  k pairs leaves k - 1 residual degrees of freedom and two pairs leave one
  (Other glia, whose trial-1 queen sample has 8 nuclei, is therefore untested);
- only genes with zero counts over the retained samples are removed (no summed
  count threshold);
- PyDESeq2 with ``DESIGN`` (``~ trial + caste``), Wald ``CONTRAST`` queen vs
  worker (positive log2FC = queen-up), BH padj after PyDESeq2's independent
  filtering, ``FDR`` 0.05, no fold-change threshold; ``tested`` = padj not NA.

Each trial put one queen and one worker library in one GEM well, so trial is
blocked. Everything else is PyDESeq2's default, as in the original analysis
(ratio size factors, parametric dispersion trend, independent filtering at alpha
0.05, no LFC shrinkage). Cook's-distance handling never acts in this design: PyDESeq2
refits outliers only among samples sharing a design row with at least 7 others
and flags them only with at least 3, and here every sample has its own design
row (one queen and one worker per trial). The outlier check actually used is
:func:`per_trial_consistency`: the sign of each DEG's queen/worker log ratio in
every trial pair.

The fit needs PyDESeq2 (``.venv-deseq2``); everything else here needs only
pandas and numpy, so the figure scripts import this module from ``.venv``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_NUCLEI = 10  # per sample (MIN_CELLS of the original analysis)
#: Samples (= trial pairs) per caste a tested type needs: all three, so every
#: fit has two residual degrees of freedom (two pairs would leave one).
MIN_PER_CASTE = 3
FDR = 0.05
DESIGN = "~ trial + caste"
CONTRAST = ["caste", "Queen", "Worker"]
CASTES = ("Queen", "Worker")
TRIALS = ("t1", "t2", "t3")

#: Focal genes on the Xenium panel, by the NCBI protein name the panel crosswalk
#: carries (``paths.PANEL_CROSSWALK``); IDs are looked up, never typed.
PANEL_FOCAL = {
    "Kr-h1": "Krueppel homolog 1",
    "sNPF": "short neuropeptide F",
    "Tk": "tachykinins",
    "Vg-like": "vitellogenin",
}
#: Focal genes off the panel, by their BIMP v4.3 symbol (``paths.BIMP_GENE_LOOKUP``).
LOOKUP_FOCAL = {"Vgn": "Vgn"}


def fn(ct: str) -> str:
    """File-name form of a cell type (``counts_<fn(ct)>.csv``)."""
    return ct.replace(" ", "_").replace("/", "_").replace("-", "_")


def sample_name(caste: str, trial: str) -> str:
    return f"BIMP_{caste}_{trial}"


def sample_meta(samples) -> pd.DataFrame:
    """caste and trial from ``BIMP_<Caste>_t<k>`` sample names."""
    samples = list(samples)
    return pd.DataFrame(
        {
            "caste": [s.split("_")[1] for s in samples],
            "trial": [s.split("_")[2] for s in samples],
        },
        index=samples,
    )


def retained_samples(nuclei: pd.Series) -> list[str] | None:
    """Samples kept for one cell type, or None when the type is not tested.

    ``nuclei`` maps sample -> nuclei of the type. A sample needs at least
    ``MIN_NUCLEI``; a trial is kept only when both its queen and its worker
    sample pass, so dropping one sample drops its trial partner too. The type is
    tested when at least ``MIN_PER_CASTE`` trials remain (all three).
    """
    ok = {s for s, n in nuclei.items() if n >= MIN_NUCLEI}
    trials = [t for t in TRIALS if all(sample_name(c, t) in ok for c in CASTES)]
    if len(trials) < MIN_PER_CASTE:
        return None
    return [s for s in nuclei.index if s.split("_")[2] in trials]


def fit(
    counts: pd.DataFrame,
    nuclei: pd.Series | None,
    design: str = DESIGN,
    metadata: pd.DataFrame | None = None,
):
    """Fit one cell type. ``counts`` is samples x genes (counts).

    ``nuclei`` maps sample -> nuclei of the type (None: every sample kept).
    ``metadata`` adds covariates (indexed by sample; must hold caste and trial);
    by default caste and trial are parsed from the sample names.
    Returns ``(results_df, dds)``, or ``(None, None)`` when the type is skipped.
    ``results_df`` is indexed by gene with PyDESeq2's columns plus ``tested``.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    keep = list(counts.index) if nuclei is None else retained_samples(nuclei)
    if keep is None:
        return None, None
    keep = [s for s in counts.index if s in set(keep)]  # counts' sample order
    c = counts.loc[keep]
    c = c.loc[:, c.sum(axis=0) > 0].astype(int)
    dds = DeseqDataSet(
        counts=c,
        metadata=sample_meta(keep) if metadata is None else metadata.loc[keep],
        design=design,
        refit_cooks=True,  # PyDESeq2's default; never acts here (module doc)
        n_cpus=1,
        quiet=True,
    )
    dds.deseq2()
    st = DeseqStats(dds, contrast=CONTRAST, n_cpus=1, quiet=True)
    st.summary()
    r = st.results_df.copy()
    r["tested"] = r.padj.notna()
    return r, dds


def n_sig(r: pd.DataFrame) -> tuple[int, int]:
    """(queen-up, worker-up) at FDR."""
    s = r.padj < FDR
    return int((s & (r.log2FoldChange > 0)).sum()), int(
        (s & (r.log2FoldChange < 0)).sum()
    )


def per_trial_consistency(
    counts: pd.DataFrame, de: pd.DataFrame, samples: list[str]
) -> pd.DataFrame:
    """Per-trial direction of a cell type's DEGs.

    ``counts`` is samples x genes; ``de`` holds the type's rows (``gene``,
    ``log2FoldChange``, ``padj``); ``samples`` the retained samples. For every
    DEG (padj < FDR) and every retained trial pair, the queen/worker log2 ratio
    of CPM + 1 (library size over all genes). ``all_agree`` is True when every
    trial's ratio has the sign of the fitted log2FC.
    """
    cpm = counts.div(counts.sum(axis=1), axis=0) * 1e6
    sig = de[de.padj < FDR].set_index("gene")
    trials = sorted(set(sample_meta(samples).trial))
    lr = pd.DataFrame(
        {
            f"log2ratio_{t}": np.log2(
                (cpm.loc[sample_name("Queen", t)] + 1)
                / (cpm.loc[sample_name("Worker", t)] + 1)
            )
            for t in trials
        }
    ).reindex(sig.index)
    agree = np.sign(lr).eq(np.sign(sig.log2FoldChange), axis=0)
    out = lr.copy()
    out.insert(0, "log2FoldChange", sig.log2FoldChange)
    out["n_trials"] = len(trials)
    out["n_agree"] = agree.sum(axis=1).astype(int)
    out["all_agree"] = out.n_agree == len(trials)
    out.index.name = "gene"
    return out.reset_index()


def _protein_name(description: str) -> str:
    """``LOC... [protein=X isoform X1]`` -> ``X``."""
    m = re.search(r"\[protein=(.*?)\]", str(description))
    name = m.group(1) if m else str(description)
    return re.sub(r" isoform \S+$", "", name).strip()


def focal_gene_ids(crosswalk=None, lookup=None) -> dict[str, str]:
    """Symbol -> BIMP v4.3 id of the focal genes (Kr-h1, sNPF, Tk, Vg-like, Vgn).

    Panel genes are found in the panel crosswalk by NCBI protein name, Vgn in
    the BIMP v4.3 annotation by symbol; each must match exactly one gene.
    """
    if crosswalk is None:
        from bimp_st.genes import read_panel_crosswalk

        crosswalk = read_panel_crosswalk()
    if lookup is None:
        from bimp_st.paths import BIMP_GENE_LOOKUP

        lookup = pd.read_csv(BIMP_GENE_LOOKUP, sep="\t")
    names = crosswalk.description.map(_protein_name)
    out = {}
    for sym, protein in PANEL_FOCAL.items():
        hit = crosswalk.v43_id[names == protein].unique()
        if len(hit) != 1:
            raise ValueError(f"{sym}: {len(hit)} panel genes named {protein!r}")
        out[sym] = hit[0]
    for sym, gene in LOOKUP_FOCAL.items():
        hit = lookup.gene_id[lookup.gene == gene].unique()
        if len(hit) != 1:
            raise ValueError(f"{sym}: {len(hit)} genes with symbol {gene!r}")
        out[sym] = hit[0]
    return out
