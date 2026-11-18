"""Cell-type vote rules (bimp_st.cluster_labels)."""

import numpy as np
import pandas as pd
import pytest

from bimp_st import cluster_labels as cl


class Obs:
    """Minimal AnnData stand-in: the votes only touch .obs and .n_obs."""

    def __init__(self, obs: pd.DataFrame):
        self.obs = obs.copy()
        self.n_obs = len(obs)


def one_cluster(labels, **cols):
    return Obs(
        pd.DataFrame(
            {"leiden": ["0"] * len(labels), "tangram_cell_type": labels, **cols}
        )
    )


def test_hierarchical_vote_names_a_clear_fine_winner():
    a = one_cluster(["Perineurial glia"] * 7 + ["Hemocytes"] * 3)
    cl.hierarchical_vote(a, dst="v")
    assert a.obs.v.iloc[0] == "Perineurial glia"


def test_exact_tie_is_undecided_at_fine_level_and_order_invariant():
    labels = ["Perineurial glia", "Perineurial glia", "Hemocytes", "Hemocytes"]
    out = []
    for order in (labels, labels[::-1]):
        a = one_cluster(order)
        cl.hierarchical_vote(a, dst="v")
        out.append(a.obs.v.iloc[0])
    assert out[0] == out[1]
    assert out[0] not in ("Perineurial glia", "Hemocytes")


def test_kenyon_fine_types_are_capped_to_the_pooled_name():
    a = one_cluster(["Large-type Kenyon cells"] * 9 + ["Hemocytes"])
    cl.hierarchical_vote(a, dst="v")
    assert a.obs.v.iloc[0] == "Kenyon cells"


def test_missing_labels_never_win_a_vote():
    a = one_cluster([np.nan, np.nan, np.nan, "Hemocytes"])
    cl.cluster_vote(a, dst="v")
    assert a.obs.v.iloc[0] == "Hemocytes"


def test_cluster_vote_below_purity_is_mixed():
    a = one_cluster(["Hemocytes", "Photoreceptors", "Perineurial glia", "Optic lobe"])
    cl.cluster_vote(a, min_purity=0.5, dst="v")
    assert a.obs.v.iloc[0] == "Mixed"


def test_anatomy_vote_treats_untraced_flags_as_outside():
    flags = pd.array([True, None, None, False], dtype="boolean")
    a = one_cluster(["Hemocytes"] * 4, in_lamina=flags)
    cl.anatomy_guided_vote(a, dst="v")
    assert a.obs.v.iloc[0] != "Optic lobe"


@pytest.mark.parametrize("base", ["flat", "hierarchical"])
def test_anatomy_vote_overrides_inside_optic_anatomy(base):
    a = one_cluster(["Hemocytes"] * 4, in_lamina=pd.array([True] * 4, dtype="boolean"))
    cl.anatomy_guided_vote(a, dst="v", base=base)
    assert a.obs.v.iloc[0] == "Optic lobe"


def test_spatial_helpers_are_total():
    assert cl.get_spatial_short(3) == "S3"
    assert cl.get_spatial_short("Medulla") == "Medulla"
    assert cl.get_spatial_color("not a cluster") == "#999999"
    assert cl.get_spatial_color(0) == cl.get_spatial_color(cl.SPATIAL_N_CLUSTERS)
