"""bimp_st.caste_de: sample retention, per-trial signs, focal gene lookup."""

import numpy as np
import pandas as pd
import pytest

from bimp_st import caste_de as cd

SAMPLES = [f"BIMP_{c}_t{t}" for c in ("Queen", "Worker") for t in (1, 2, 3)]


def test_retained_samples_needs_all_three_trial_pairs():
    # Other glia: the trial-1 queen has 8 nuclei, so two pairs remain (one
    # residual df under ~ trial + caste) and the type is not tested.
    n = pd.Series([8, 52, 28, 62, 29, 24], index=SAMPLES)
    assert cd.MIN_PER_CASTE == 3
    assert cd.retained_samples(n) is None
    assert cd.retained_samples(pd.Series(100, index=SAMPLES)) == SAMPLES


def test_retained_samples_untestable_below_two_pairs():
    n = pd.Series([5, 50, 50, 50, 5, 50], index=SAMPLES)  # t1 queen, t2 worker low
    assert cd.retained_samples(n) is None


def test_sample_meta():
    m = cd.sample_meta(SAMPLES[:2])
    assert list(m.caste) == ["Queen", "Queen"]
    assert list(m.trial) == ["t1", "t2"]


def test_fn():
    assert cd.fn("Large-type Kenyon cells") == "Large_type_Kenyon_cells"


def test_per_trial_consistency():
    counts = pd.DataFrame(
        {
            "g1": [20, 20, 20, 10, 10, 10],
            "g2": [10, 30, 10, 20, 20, 20],
            "g3": [1000] * 6,  # sets the library size
        },
        index=SAMPLES,
    ).astype(float)
    de = pd.DataFrame(
        {"gene": ["g1", "g2"], "log2FoldChange": [1.0, -0.5], "padj": [0.01, 0.01]}
    )
    s = cd.per_trial_consistency(counts, de, SAMPLES).set_index("gene")
    assert bool(s.at["g1", "all_agree"])
    assert not bool(s.at["g2", "all_agree"])  # trial 2 goes the other way
    assert int(s.at["g2", "n_agree"]) == 2
    cpm = counts.div(counts.sum(axis=1), axis=0) * 1e6
    want = np.log2((cpm.at[SAMPLES[0], "g1"] + 1) / (cpm.at[SAMPLES[3], "g1"] + 1))
    assert np.isclose(s.at["g1", "log2ratio_t1"], want)


def test_focal_gene_ids_from_tables():
    cw = pd.DataFrame(
        {
            "v43_id": ["A", "B", "C", "D"],
            "description": [
                "LOC1 [protein=Krueppel homolog 1]",
                "LOC2 [protein=short neuropeptide F]",
                "LOC3 [protein=tachykinins]",
                "LOC4 [protein=vitellogenin isoform X1]",
            ],
        }
    )
    lk = pd.DataFrame({"gene_id": ["E", "F"], "gene": ["Vgn", "Tk"]})
    ids = cd.focal_gene_ids(cw, lk)
    assert ids == {"Kr-h1": "A", "sNPF": "B", "Tk": "C", "Vg-like": "D", "Vgn": "E"}


def test_focal_gene_ids_real_data():
    from bimp_st.paths import BIMP_GENE_LOOKUP, PANEL_CROSSWALK

    if not (PANEL_CROSSWALK.exists() and BIMP_GENE_LOOKUP.exists()):
        pytest.skip("panel crosswalk / gene lookup not present")
    ids = cd.focal_gene_ids()
    assert ids["Tk"] == "BIMP_v4.3_009257"
    assert ids["Vgn"] == "BIMP_v4.3_005500"
