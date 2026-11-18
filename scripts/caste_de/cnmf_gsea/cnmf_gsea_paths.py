"""Paths of the in-repo port of the joint cNMF / GSEA caste analysis.

Analysis design and notebook code: I. Traniello. Replaces the
``# ---- paths ----`` block of the analysis notebook (clean notebook cell 2,
``01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb``, md5
184efd9af8192fd09f7329f1eeaa9f17). Nothing here points into the snRNA-seq
analysis tree:
``freeze_inputs.py`` copies what the pipeline needs into
``results/caste_programs/inputs/`` (PROVENANCE.md there), and every step
reads those copies or repo outputs.

Every step takes ``--source``:

``repo`` (default)
    the repo's own joint cNMF (``pipeline/cnmf_joint``, factorized by
    ``cnmf_prepare.py`` + ``run_joint.sh``) and the repo's DESeq2 table
    (``bimp_st.paths.CASTE_DE_FINE``, from ``caste_de_blocked_all.py``). This is
    what ``fig:caste_de`` panel d reads.
``saved``
    the analysis's saved joint cNMF and ``de_all.csv.gz`` (frozen copies), so
    the port can be checked against the saved tables (``parity.py``).
"""

from dataclasses import dataclass
from pathlib import Path

from bimp_st.paths import (
    CASTE_DE_FINE,
    CASTE_PROGRAMS_DIR,  # noqa: E402
)

REPO = Path(__file__).resolve().parents[3]
# resolve the shared results/ symlink so paths written into run_joint.sh and
# PROVENANCE files do not name the worktree they were generated from
R9 = CASTE_PROGRAMS_DIR.resolve()

INPUTS = R9 / "inputs"
LOOKUP = INPUTS / "BIMP_lookup.csv"  # gene annotation table (tab-separated)
EMAPPER = INPUTS / "BIMP_v4.3.emapper.annotations"  # eggNOG-mapper GO terms
GO_OBO = INPUTS / "go-basic.obo"
# On-disk names of the saved-analysis copies, the saved-source run directories
# and the program_map.csv column of the saved run's numbers: caches and the
# Supplementary Data steps key on them, so they are spelled here only.
SAVED_DIRNAME = "ian"
SAVED_PROGRAM_COL = "ian_program"
_SAVED_INPUTS_DIRNAME = "ian_saved"
SAVED = INPUTS / _SAVED_INPUTS_DIRNAME  # saved outputs: parity and --source saved
SAVED_CNMF = SAVED / "cnmf_joint"
SAVED_DE = SAVED / "de_all.csv.gz"
# the repo's fine-type DE (caste_de_blocked_all.py), as fig:caste_de reads it
REPO_DE = CASTE_DE_FINE

RUN = R9 / "pipeline"  # everything this code writes
REPO_CNMF = RUN / "cnmf_joint"
PROGRAM_MAP = RUN / "program_map.csv"  # saved run's program numbers -> repo run's
FIG3_INPUTS = RUN / "fig3_inputs"
PARITY = RUN / "parity"
VERBATIM = R9  # outputs of the verbatim copy (commit cd8d4278): parity reference

CNMF_NAME = "joint"  # cell 21 OUT_J_NAME
FIG_PROGS = [18, 20, 6, 7, 12]  # figure programs (cells 27, 30), saved numbering


@dataclass(frozen=True)
class Source:
    """Inputs and output directory of one pipeline run."""

    name: str
    de_csv: Path  # cell 6 DE_CSV
    cnmf_dir: Path  # cell 2 OUT_J: cNMF output_dir
    out_dir: Path  # cell 2 OUT_DIR / results: tables and plots

    @property
    def gsea_csv(self) -> Path:
        """Cell 15 cache: per-cell-type GSEA."""
        return self.out_dir / "gsea_all.csv.gz"

    @property
    def program_go_csv(self) -> Path:
        """Cell 27 cache: GO BP tables of the figure programs."""
        return self.out_dir / "joint_k50_figure_programs_go_bp.csv.gz"

    @property
    def program_go_cell109_csv(self) -> Path:
        """Original-notebook cell 109: the figure programs' GO BP tables at the
        permutation counts of the original notebook's panel d labels."""
        return self.out_dir / "joint_k50_figure_programs_go_bp_cell109.csv.gz"

    @property
    def go_labels_csv(self) -> Path:
        """Original-notebook cell 102 cache: top GO terms of every program."""
        return self.out_dir / "joint_k50_go_labels.csv"


def source(name: str) -> Source:
    """Paths for ``--source repo`` or ``--source saved``; creates the output dir."""
    if name == "repo":
        src = Source(name, REPO_DE, REPO_CNMF, RUN / "repo")
    elif name == "saved":
        src = Source(name, SAVED_DE, SAVED_CNMF, RUN / SAVED_DIRNAME)
    else:
        raise ValueError(f"unknown source {name!r}")
    src.out_dir.mkdir(parents=True, exist_ok=True)
    return src


def add_source_arg(parser) -> None:
    """Add the shared ``--source {repo,saved}`` option to a step's parser."""
    parser.add_argument(
        "--source",
        choices=("repo", "saved"),
        default="repo",
        help="repo: the repo's factorization and DESeq2 table (fig:caste_de); "
        "saved: the analysis's saved cNMF and DE, for parity",
    )
