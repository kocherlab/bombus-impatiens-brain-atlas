"""Site configuration: where inputs that are not in the repository live.

Paths under the repository go through :mod:`bimp_st.paths`. Everything else
(a checkout that holds the gitignored ``data/``, the snRNA-seq analysis tree the
reference and count matrices come from, the genome release, raw MALDI exports,
the deposit staging area) resolves through :func:`get`, so no script carries a
machine-specific path.

Lookup for a key ``k``, first hit wins:

1. the environment variable ``BIMP_<K>`` (``BIMP_DATA_REPO`` for ``data_repo``);
2. the TOML file named by ``$BIMP_CONFIG``;
3. ``config/site.toml`` in the repository (gitignored);
4. ``$XDG_CONFIG_HOME/bimp-st/site.toml`` (default ``~/.config/bimp-st/``),
   which every checkout and worktree of one user shares;
5. the default in :data:`KEYS`: repository-relative, or none.

A key with no value and no default raises :class:`SiteConfigError` naming the
key, the environment variable and the file to set it in. The keys, their
meaning and a template are in ``config/site.example.toml``.

Shell scripts read a key with ``python -m bimp_st.site <key>``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: key -> (default, description). A ``None`` default means the key has to be
#: set before the scripts that read it can run.
KEYS: dict[str, tuple[str | None, str]] = {
    "data_repo": (
        str(REPO),
        "checkout whose data/ and pipelines/*/data/ hold the atlas inputs "
        "(set it when running from a second checkout without those directories)",
    ),
    "snrna_source": (
        None,
        "snRNA-seq analysis tree: CellBender outputs (out/cellbender), cNMF and "
        "GSEA outputs (out/), notebook code (src/scripts/03_analysis)",
    ),
    "snrna_shared_scripts": (
        None,
        "directory with the shared snRNA-seq helper scripts the cNMF port "
        "freezes (integrated_analyses/src/scripts)",
    ),
    "genome_dir": (
        None,
        "BIMP v4 reference genome directory (genome/BIMP_genome_v4.fasta, "
        "the protein FASTA and GFF the ortholog step reads)",
    ),
    "genome_release": (
        None,
        "BIMP v4.3 genome release directory (BIMP_OGS_v4.3.gff)",
    ),
    "genomes_dir": (
        None,
        "directory of reference genomes the deposit staging copies from",
    ),
    "maldi_raw": (
        None,
        "raw MALDI-MSI imzML exports (SCiLS), read by the MALDI extraction",
    ),
    "deposit_staging": (
        None,
        "where scripts/deposit/ builds the public data deposit",
    ),
    "isoseq_raw": (
        None,
        "SMRT Link Iso-Seq run directory (Isoseq_analysis_13887/flnc-<i>.bam and "
        "its reports), read by the EGAPx input tables",
    ),
    "busco_env": (
        None,
        "conda environment with BUSCO 6.1.0 "
        "(scripts/gene_annotation/run_genome_busco.sh)",
    ),
    "busco6_env": (
        None,
        "environment built from config/envs/busco.yml "
        "(scripts/gene_annotation/run_annotation_busco.sh)",
    ),
    "busco_downloads": (
        None,
        "BUSCO lineage download directory (run_annotation_busco.sh)",
    ),
    # Values, not paths: author-confirmed metadata the deposit staging writes.
    "snrna_individuals_per_sample": (
        None,
        "number of bees pooled in each snRNA-seq sample (deposit metadata, "
        "scripts/deposit/build_staging.py)",
    ),
    "snrna_instrument": (
        None,
        "sequencing instrument model, instrument ID and flowcell of the "
        "snRNA-seq libraries (deposit metadata, scripts/deposit/build_staging.py)",
    ),
}

#: Older environment variable names still honoured, new name -> old names.
ALIASES: dict[str, tuple[str, ...]] = {
    "data_repo": ("BIMP_MAIN_CHECKOUT", "BIMP_DATA_ROOT"),
}


class SiteConfigError(KeyError):
    """A site key has no value and no default."""


def env_name(key: str) -> str:
    return f"BIMP_{key.upper()}"


def config_files() -> list[Path]:
    """The TOML files consulted, in order (missing ones are skipped)."""
    files = []
    if os.environ.get("BIMP_CONFIG"):
        files.append(Path(os.environ["BIMP_CONFIG"]).expanduser())
    files.append(REPO / "config" / "site.toml")
    xdg = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    files.append(Path(xdg) / "bimp-st" / "site.toml")
    return files


def _read_toml(f: Path) -> dict:
    """The file's keys; a flat ``key = "value"`` reader where tomllib is missing
    (Python < 3.11: the cNMF environment)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        out = {}
        for line in f.read_text().splitlines():
            m = re.match(r'\s*([A-Za-z_][\w-]*)\s*=\s*"([^"]*)"\s*(#.*)?$', line)
            if m:
                out[m.group(1)] = m.group(2)
        return out
    with open(f, "rb") as fh:
        return tomllib.load(fh)


def _from_files(key: str) -> str | None:
    for f in config_files():
        if f.is_file():
            data = _read_toml(f)
            if key in data:
                return str(data[key])
    return None


def lookup(key: str) -> str | None:
    """The configured value of ``key``, or its default (``None`` if unset)."""
    if key not in KEYS:
        raise KeyError(f"unknown site key {key!r}; known: {', '.join(KEYS)}")
    for name in (env_name(key), *ALIASES.get(key, ())):
        if os.environ.get(name):
            return os.environ[name]
    value = _from_files(key)
    return value if value is not None else KEYS[key][0]


def value(key: str) -> str:
    """The string configured for ``key``; raises if it has neither value nor
    default. For keys that hold metadata rather than a path."""
    v = lookup(key)
    if v is None:
        raise SiteConfigError(
            f"site key {key!r} is not set ({KEYS[key][1]}). Set ${env_name(key)} "
            "or add it to config/site.toml (template: config/site.example.toml)."
        )
    return v


def get(key: str) -> Path:
    """The path configured for ``key``; raises if it has neither value nor default."""
    value = lookup(key)
    if value is None:
        raise SiteConfigError(
            f"site key {key!r} is not set ({KEYS[key][1]}). Set ${env_name(key)} "
            "or add it to config/site.toml (template: config/site.example.toml)."
        )
    return Path(value).expanduser()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m bimp_st.site <key>", file=sys.stderr)
        print("keys: " + ", ".join(KEYS), file=sys.stderr)
        return 2
    try:
        print(get(argv[0]))
    except (SiteConfigError, KeyError) as e:
        print(e.args[0], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
