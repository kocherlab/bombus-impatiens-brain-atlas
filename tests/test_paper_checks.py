"""Planted failures: each integrity check must fail on the defect it guards.

The integrity checks are the last guard against a stale number or figure, so
each test here plants one defect and asserts the check catches it.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
import zlib
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "scripts" / "paper"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(REPO / "scripts"))

import manuscript_text as mt  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── \iffalse stripping ───────────────────────────────────────────────────────


def test_iffalse_nested_and_indented():
    text = "keep1\n  \\iffalse drop \\ifx\\a\\b inner \\fi still drop \\fi keep2"
    assert mt.strip_iffalse(text) == "keep1\n   keep2"


def test_iffalse_ignores_other_conditionals_outside():
    assert mt.strip_iffalse("\\ifdefined\\x a\\fi b") == "\\ifdefined\\x a\\fi b"


def test_printed_drops_comments_not_escaped_percent():
    assert mt.printed("9.2\\% of cells % 9.3 in a comment") == "9.2\\% of cells "


# ── numerals ─────────────────────────────────────────────────────────────────


def test_numerals_normalise_latex():
    lit = "1{,}300{,}192 cells, $-0.596$, 9.2\\%, $4.4\\times10^{-5}$, 0.06--0.59"
    got = mt.numerals(lit)
    for want in ("1,300,192", "-0.596", "9.2", "4.4e-5", "0.06", "0.59"):
        assert want in got
    assert "-0.59" not in got  # a range dash is not a minus sign


# ── figures: any DIFFER is fatal ─────────────────────────────────────────────


class _Fig(NamedTuple):
    pairs: list

    def files(self):
        return self.pairs


def test_figure_differ_is_fatal(tmp_path):
    (tmp_path / "ms").mkdir()
    (tmp_path / "res").mkdir()
    (tmp_path / "ms" / "a.png").write_bytes(b"old")
    (tmp_path / "res" / "a.png").write_bytes(b"new")
    (tmp_path / "ms" / "b.png").write_bytes(b"same")
    (tmp_path / "res" / "b.png").write_bytes(b"same")
    figs = [_Fig([("a.png", "a.png"), ("b.png", "b.png")])]
    n_match, n_differ, fatal = mt.compare_figures(
        figs, tmp_path / "ms", tmp_path / "res", {}
    )
    assert (n_match, n_differ) == (1, 1)
    assert len(fatal) == 1 and "a.png" in fatal[0]
    # a documented non-reproducible file is excused, a missing one never is
    _, _, fatal = mt.compare_figures(
        figs, tmp_path / "ms", tmp_path / "res", {"a.png": "why"}
    )
    assert fatal == []
    (tmp_path / "res" / "a.png").unlink()
    _, _, fatal = mt.compare_figures(
        figs, tmp_path / "ms", tmp_path / "res", {"a.png": "why"}
    )
    assert len(fatal) == 1


def test_pdf_hash_ignores_draw_date(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"%PDF /CreationDate (D:20260101) x startxref 101")
    b.write_bytes(b"%PDF /CreationDate (D:20261117120000) x startxref 1009")
    assert mt.md5(a) == mt.md5(b)


# ── sync: a reshaped figure ships no file at all ─────────────────────────────


def _png(path, w, h):
    raw = b"".join(b"\x00" + b"\x00" * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _pdf(path, w, h, tag):
    path.write_bytes(b"%%PDF-1.4 /MediaBox [0 0 %d %d] %s" % (w, h, tag))


def test_sync_guards_the_pdf_before_copying(tmp_path):
    pytest.importorskip("PIL")
    sync = _load("sync_under_test", PAPER / "sync_manuscript_figures.py")
    repo, ms = tmp_path / "repo", tmp_path / "ms"
    repo.mkdir()
    ms.mkdir()
    # PNG re-rendered at the same shape; the PDF changed shape (new layout).
    _png(ms / "f.png", 100, 50)
    _png(repo / "f.png", 100, 51)
    _pdf(ms / "f.pdf", 100, 50, b"old")
    _pdf(repo / "f.pdf", 100, 100, b"new")
    pairs = [("fig", "f.pdf", "f.pdf"), ("fig", "f.png", "f.png")]
    before = (ms / "f.pdf").read_bytes(), (ms / "f.png").read_bytes()
    n = sync.sync(pairs, repo, ms, apply=True)
    assert n["block"] == 2 and n["copy"] == 0
    assert ((ms / "f.pdf").read_bytes(), (ms / "f.png").read_bytes()) == before
    n = sync.sync(pairs, repo, ms, apply=True, allow={"fig"})
    assert n["copy"] == 2


# ── provenance: data flow and the key-list baseline ──────────────────────────


@pytest.fixture(scope="module")
def prov():
    return _load("prov_under_test", PAPER / "check_data_provenance.py")


def test_spatial_label_read_is_flagged(prov):
    src = (
        "from bimp_st.paths import SPATIAL_COMBINED, CANONICAL_TANGRAM\n"
        "tg = ad.read_h5ad(CANONICAL_TANGRAM)\n"
        "a = ad.read_h5ad(SPATIAL_COMBINED)\n"
        "lab = a.obs['tangram_cell_type']\n"
    )
    assert prov.spatial_label_reads(src) == [4]


def test_helper_given_the_stale_column_is_flagged(prov):
    src = (
        "H5 = REPO / 'data' / 'combined_clustered.h5ad'\n"
        "def main():\n"
        "    a = ad.read_h5ad(H5)\n"
        "    add_canonical(a, src=('tangram_cell_type', 'cell_type'))\n"
    )
    assert prov.spatial_label_reads(src) == [4]


def test_overwritten_label_column_is_clean(prov):
    src = (
        "a = ad.read_h5ad(paths.SPATIAL_COMBINED)\n"
        "a.obs['tangram_cell_type'] = keyed_obs_column(a, 'tangram_cell_type', P)\n"
        "add_canonical(a, src='tangram_cell_type')\n"
        "x = a.obs['tangram_cell_type']\n"
    )
    assert prov.spatial_label_reads(src) == []


def test_baseline_is_a_key_list_not_a_count(prov, tmp_path):
    base = tmp_path / "baseline.txt"
    base.write_text("a.py:spatial-label  # reviewed\nb.py:obs-loc\n")
    assert prov.compare_baseline(["a.py:spatial-label", "b.py:obs-loc"], base) == 0
    # one fixed and one new: the count is unchanged, the check still fails
    assert prov.compare_baseline(["a.py:spatial-label", "c.py:obs-loc"], base) == 1
    # a fixed finding must be removed from the list
    assert prov.compare_baseline(["a.py:spatial-label"], base) == 1


def test_repo_matches_its_provenance_baseline(prov, monkeypatch):
    prov.findings.clear()
    files = prov.tracked_files()
    prov.check_superseded_reads(files)
    prov.check_obs_loc(files)
    prov.check_spatial_label_reads(files)
    prov.check_precorrection_fig5kc(files)
    keys = [k for k, _ in prov.findings]
    assert prov.compare_baseline(keys, PAPER / "provenance_baseline.txt") == 0


# ── figure map: README's generated table must match the registry ─────────────


@pytest.fixture
def figmap():
    return _load("make_figure_map", PAPER / "make_figure_map.py")


def test_readme_figure_map_is_current(figmap):
    assert figmap.missing_descriptions() == []
    readme = figmap.README.read_text()
    assert figmap.updated(readme) == readme, (
        "README.md figure map is stale: run scripts/paper/make_figure_map.py"
    )


def test_figure_map_catches_a_stale_row(figmap):
    readme = figmap.README.read_text()
    planted = readme.replace("`scripts/atlas/fig_atlas_overview.py`", "`old.py`", 1)
    assert planted != readme
    assert figmap.updated(planted) != planted
