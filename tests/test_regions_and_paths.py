"""Region tiers, annotation resolution and the Tangram loader's guards."""

import pytest

from bimp_st import nine_region_common as nrc
from bimp_st import paths


def test_region_tiers():
    assert len(nrc.DESCRIPTIVE_REGIONS) == 8
    assert len(nrc.STATISTICAL_REGIONS) == 7
    assert set(nrc.NO_PVALUE_REGIONS) == {"al", "central_complex"}
    assert set(nrc.STATISTICAL_REGIONS) <= set(nrc.DESCRIPTIVE_REGIONS)
    assert nrc.NO_PVALUE_REGIONS is not nrc.NEW_REGIONS_STATIC


def test_empty_newest_annotation_set_raises(tmp_path, monkeypatch):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v4").mkdir()  # exists but empty: a rebuild in progress
    monkeypatch.setitem(
        nrc.BRAINS, "queen", {**nrc.BRAINS["queen"], "annotations": tmp_path}
    )
    with pytest.raises(RuntimeError, match="no GeoJSON"):
        nrc.annotation_sources("queen")


def test_tracked_annotations_give_the_eight_regions():
    pytest.importorskip("shapely")
    nrc._VOCAB_CACHE.clear()
    assert set(nrc.all_regions("v2")) == set(nrc.DESCRIPTIVE_REGIONS)


def test_load_tangram_refuses_superseded_objects(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    sup = tmp_path / "_superseded"
    sup.mkdir()
    (sup / "old.h5ad").write_bytes(b"")
    link = tmp_path / "looks_current.h5ad"
    link.symlink_to(sup / "old.h5ad")
    monkeypatch.setattr(paths, "SUPERSEDED_DIR", sup)
    with pytest.raises(ValueError, match="superseded"):
        paths.load_tangram(link)


def test_region_of_and_masks_are_nullable_aware():
    pd = pytest.importorskip("pandas")

    def flags(*v):
        return pd.array(list(v), dtype="boolean")

    obs = pd.DataFrame(
        {
            "in_kc_oc": flags(True, False, False, True, None),
            "in_kc_ic": flags(False, False, False, True, False),
            "in_lamina": flags(False, False, None, False, False),
        }
    )
    regs = ["kc_oc", "kc_ic", "lamina"]
    assert list(nrc.region_of(obs, regs)) == [
        "kc_oc",
        nrc.OUTSIDE,
        None,
        nrc.OVERLAP,
        None,
    ]
    assert nrc.region_of(obs, regs, overlap="first")[3] == "kc_oc"
    kc = nrc.region_mask(obs, ["kc_oc", "kc_ic"])
    assert kc.isna().tolist() == [False, False, False, False, True]
    assert kc.fillna(False).tolist() == [True, False, False, True, False]
    # an untraced lamina stays NA: a tracing gap is not "outside the lamina"
    assert nrc.region_mask(obs, ["lamina"]).isna().tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]


def test_traced_mask_refuses_untraced_cells():
    pd = pytest.importorskip("pandas")
    ok = pd.array([True, False], dtype="boolean")
    assert nrc.traced_mask(ok).tolist() == [True, False]
    with pytest.raises(ValueError):
        nrc.traced_mask(pd.array([True, None], dtype="boolean"))


def test_composition_contrast_symmetric_floor():
    pd = pytest.importorskip("pandas")
    from bimp_st.composition import composition_contrast, composition_table

    brain = ["queen"] * 200 + ["worker"] * 200
    region = (["a"] * 100 + ["b"] * 100) * 2
    # region a: 90/10 in both brains except worker 80/20; region b: 10/90.
    lab = (
        ["x"] * 90
        + ["y"] * 10
        + ["x"] * 10
        + ["y"] * 90
        + ["x"] * 80
        + ["y"] * 20
        + ["x"] * 10
        + ["y"] * 90
    )
    t = composition_table(brain, region, lab, regions=["a", "b"])
    assert t.groupby(["brain", "region"]).proportion.sum().round(9).eq(1).all()
    c = composition_contrast(t, floor=0.005)
    assert c.n_caste == 4 and c.n_regional == 4
    assert c.regional_median > c.caste_median
    # a class under the floor in one group only is dropped from both arms
    one_sided = composition_table(
        ["queen"] * 1000 + ["worker"] * 1000,
        ["a"] * 2000,
        ["x"] * 999 + ["y"] + ["x"] * 900 + ["y"] * 100,
        regions=["a"],
    )
    two = pd.concat([one_sided, t[t.region == "b"]])
    assert composition_contrast(two).n_caste == 3
