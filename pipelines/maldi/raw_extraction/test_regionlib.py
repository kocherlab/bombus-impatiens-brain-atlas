from pathlib import Path

import numpy as np
import pandas as pd
import regionlib

PIPELINE_DIR = Path(__file__).resolve().parent


def test_map_labels_to_pixels_basic():
    px = np.array([1, 2, 1, 2], dtype=np.int32)
    py = np.array([1, 1, 2, 2], dtype=np.int32)
    label_img = np.array([[5, 5], [0, 3]], dtype=np.int32)  # indexed [py-1, px-1]
    ids = regionlib.map_labels_to_pixels(px, py, label_img)
    assert ids.tolist() == [5, 5, 0, 3]


def test_map_labels_to_pixels_shape_mismatch():
    px = np.array([1])
    py = np.array([1])
    bad = np.zeros((3, 3), dtype=np.int32)
    try:
        regionlib.map_labels_to_pixels(px, py, bad, expected_shape=(2, 2))
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "shape" in str(e).lower()


def test_compute_region_stats():
    pixel_regions = pd.DataFrame(
        {
            "px": [1, 2, 1, 2],
            "py": [1, 1, 2, 2],
            "region_id": [1, 1, 1, 2],
            "region_name": ["A", "A", "A", "B"],
        }
    )
    features = pd.DataFrame(
        {
            "px": [1, 2, 1, 2],
            "py": [1, 1, 2, 2],
            "sNPF_v1": [10.0, 20.0, 30.0, 100.0],
            "Tk2": [1.0, 2.0, 3.0, 9.0],
        }
    )
    stats = regionlib.compute_region_stats(
        pixel_regions, features, peptides=["sNPF_v1", "Tk2"]
    )
    a = stats.set_index("region_name").loc["A"]
    assert a["n_pixels"] == 3
    assert a["sNPF_v1_mean"] == 20.0
    assert a["sNPF_v1_sum"] == 60.0
    assert a["sNPF_v1_median"] == 20.0
    b = stats.set_index("region_name").loc["B"]
    assert b["n_pixels"] == 1
    assert b["Tk2_sum"] == 9.0
    assert "0" not in stats["region_name"].astype(str).tolist()


def test_crop_region_image():
    img = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    mask[2, 1] = True
    crop = regionlib.crop_region_image(img, mask)
    assert crop.shape == (2, 1)
    assert crop[0, 0] == 5.0
    assert crop[1, 0] == 8.0


def test_export_draw_image(tmp_path):
    import json
    import subprocess
    import sys

    import pandas as pd
    import tifffile

    feat = pd.DataFrame(
        {
            "px": [1, 2, 1],
            "py": [1, 1, 2],
            "sNPF_v1": [10.0, 20.0, 0.0],
            "Tk2": [1.0, 0.0, 3.0],
            "TIC": [100.0, 200.0, 50.0],
        }
    )
    fp = tmp_path / "features.parquet"
    feat.to_parquet(fp, index=False)
    out_tif = tmp_path / "draw.tif"
    out_json = tmp_path / "draw.json"
    rc = subprocess.run(
        [
            sys.executable,
            "export_draw_image.py",
            str(fp),
            "testsample",
            str(out_tif),
            str(out_json),
        ],
        cwd=PIPELINE_DIR,
    )
    assert rc.returncode == 0
    img = tifffile.imread(out_tif)
    assert img.shape == (3, 2, 2)  # (channels, H, W)
    meta = json.loads(out_json.read_text())
    assert meta["shape"] == [2, 2]
    assert meta["channels"][0] == "TIC"


def test_import_regions_end_to_end(tmp_path):
    import json
    import subprocess
    import sys

    import numpy as np
    import pandas as pd
    import tifffile

    feat = pd.DataFrame(
        {
            "px": [1, 2, 1, 2],
            "py": [1, 1, 2, 2],
            "sNPF_v1": [10.0, 20.0, 30.0, 100.0],
            "Tk2": [1.0, 2.0, 3.0, 9.0],
            "TIC": [11.0, 22.0, 33.0, 109.0],
        }
    )
    fp = tmp_path / "features.parquet"
    feat.to_parquet(fp, index=False)
    label_img = np.array([[1, 1], [1, 2]], dtype=np.int32)
    lab = tmp_path / "labels.tif"
    tifffile.imwrite(lab, label_img)
    names = tmp_path / "regions.json"
    names.write_text(json.dumps({"1": "A", "2": "B"}))
    outdir = tmp_path / "out"
    rc = subprocess.run(
        [
            sys.executable,
            "import_regions.py",
            str(fp),
            str(lab),
            str(names),
            "testsample",
            str(outdir),
        ],
        cwd=PIPELINE_DIR,
    )
    assert rc.returncode == 0
    pr = pd.read_parquet(outdir / "pixel_regions_testsample.parquet")
    assert set(pr["region_name"]) == {"A", "B"}
    st = pd.read_parquet(outdir / "region_stats_testsample.parquet").set_index(
        "region_name"
    )
    assert st.loc["A", "n_pixels"] == 3
    assert st.loc["A", "sNPF_v1_sum"] == 60.0
    assert st.loc["B", "n_pixels"] == 1
    crops = list((outdir / "region_crops").glob("*.png"))
    assert len(crops) >= 1
