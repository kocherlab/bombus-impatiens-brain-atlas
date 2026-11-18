"""Which brain the shared whole-brain scripts work on.

The scripts in this directory are shared by the queen (``queen/``) and worker
(``worker/``) pipelines. They read and write the brain directory's ``data/`` and
``figures/``, read its region annotations from
``data/annotations/<brain>/v1/`` at the repository root (``annotations_dir``),
and take the brain's crop-to-annotation table from its ``labeling.py``.

The brain directory is ``BIMP_BRAIN_DIR`` if set, else the current directory:
every Snakemake rule runs ``cd <brain dir> && python ../<script>``.
Anything else fails here rather than writing ``data/`` somewhere unexpected.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def resolve_brain_dir() -> Path:
    d = Path(os.environ.get("BIMP_BRAIN_DIR") or Path.cwd()).resolve()
    if not ((d / "config.yaml").is_file() and (d / "labeling.py").is_file()):
        raise SystemExit(
            f"{d} is not a whole-brain pipeline directory (it needs config.yaml "
            "and labeling.py). cd into pipelines/wholebrain/queen or "
            "pipelines/wholebrain/worker, or set BIMP_BRAIN_DIR."
        )
    return d


def load_annotation_mapping(brain: Path) -> dict[str, list[str]]:
    """The brain's ANNOTATION_MAPPING (annotation GeoJSON -> crop suffixes)."""
    spec = importlib.util.spec_from_file_location(
        f"_labeling_{brain.name}", brain / "labeling.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {brain / 'labeling.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ANNOTATION_MAPPING


def annotations_dir(brain: Path, version: str = "v1") -> Path:
    """The brain's region annotations: data/annotations/<brain name>/<version>.

    v1 is the original set the pipeline's ``label`` step was run on; v2-v4 are
    the later sets the region relabel (pipelines/analyses) reads.
    """
    return brain.parents[2] / "data" / "annotations" / brain.name / version


BRAIN_DIR = resolve_brain_dir()
