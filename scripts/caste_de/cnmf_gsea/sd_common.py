"""Shared helpers of the snRNA caste Supplementary Data tables.

Table design: I. Traniello. Ports the helpers of
``out/fig4_supp/scripts/supp_full.py`` (md5
70ce9f8c44e4af04ef773d8910976545; the script behind ``Supp_program_GO_terms.csv``)
that the three table builders share: GO term annotation (``annotate``), gene-name
display of leading-edge genes (``named``), and the cell-type name clean-up. Each
builder reads either the repo's outputs (``source="repo"``, the default and what
ships) or the frozen copies of the saved outputs (``source="saved"``), so
``compare_si_tables.py`` can tell a port difference from an input difference.

Reads only CSVs, so it runs in the repo ``.venv``; the GSEA itself is
``gsea_celltypes.py`` / ``program_go_all.py`` in the pipeline env.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from cnmf_gsea_paths import LOOKUP, R9, REPO_DE, RUN, SAVED, SAVED_DE, SAVED_DIRNAME

SI = R9 / "si_tables"  # everything the Supplementary Data steps write
SI_SAVED = SI / SAVED_DIRNAME  # their --source saved output
NS = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}
SOURCES = ("repo", "saved")


def de_csv(source: str) -> Path:
    """The fine-type DESeq2 table (``~ trial + caste``) of a source."""
    return {"repo": REPO_DE, "saved": SAVED_DE}[source]


def gsea_csv(source: str) -> Path:
    """Clean-notebook cell 15's per-cell-type GSEA (``gsea_celltypes.py``)."""
    return {"repo": RUN / "repo", "saved": SAVED}[source] / "gsea_all.csv.gz"


def caste_csv(source: str) -> Path:
    """Cell 25's per-program x cell-type caste table (``cnmf_caste.py``)."""
    return {"repo": RUN / "repo", "saved": SAVED}[source] / "joint_k50_caste.csv"


def clean_ct(s: pd.Series) -> pd.Series:
    """Source cell-type names carry line breaks for plotting; collapse whitespace."""
    return s.astype(str).str.replace(r"\s+", " ", regex=True)


def parse_obo(alt_ids: bool = True) -> dict[str, tuple[str, str]]:
    """Clean-notebook cell 13 (``cnmf_gsea_common.parse_obo``): GO id ->
    (name, namespace).

    Imported lazily: ``cnmf_gsea_common`` pulls in scanpy, which the CSV-only
    callers do not otherwise need. ``alt_ids=False`` keeps primary ids only, the
    lookup the saved unfiltered cell-type table was annotated with (its 135
    secondary ids have no name there).
    """
    from cnmf_gsea_common import parse_obo as _parse
    from cnmf_gsea_paths import GO_OBO

    meta = _parse()
    if alt_ids:
        return meta
    with open(GO_OBO) as fh:
        primary = {ln[4:].strip() for ln in fh if ln.startswith("id: GO:")}
    return {k: v for k, v in meta.items() if k in primary}


def lookup() -> pd.DataFrame:
    """The gene annotation table (``BIMP_lookup.csv``, tab-separated)."""
    return pd.read_csv(LOOKUP, sep="\t")


def name_map(de: pd.DataFrame) -> pd.Series:
    """``supp_full.py`` ``nmap``: gene id -> gene name, over the DE table's genes."""
    lk = (
        lookup()[["gene_id", "gene"]]
        .rename(columns={"gene_id": "gene", "gene": "gene_name"})
        .replace({"gene_name": {"": np.nan}})
        .drop_duplicates("gene")
    )
    genes = pd.DataFrame({"gene": de["gene"].drop_duplicates()})
    return genes.merge(lk, on="gene", how="left").set_index("gene").gene_name


def named(s, nmap: pd.Series) -> str:
    """``supp_full.py`` ``named``: leading-edge ids -> names, else the bare id."""
    ids = [x for x in str(s).split(";") if x]
    out = []
    for i in ids:
        n = nmap.get(i)
        out.append(n if isinstance(n, str) else i.replace("BIMP_v4.3_", ""))
    return "; ".join(out)


def go_annotate(g: pd.DataFrame, go_meta: dict, drop: bool = True) -> pd.DataFrame:
    """``supp_full.py`` ``annotate``: GO id, term name and BP/MF/CC namespace.

    ``drop`` removes terms without a namespace (ids missing from the OBO) and
    obsolete terms, as ``annotate`` does; the unfiltered cell-type table keeps them.
    """
    g = g.copy()
    g["go_id"] = g.Term.astype(str).str.extract(r"(GO:\d{7})", expand=False)
    meta = g.go_id.map(lambda k: go_meta.get(k, (None, None)))
    g["term"] = meta.str[0]
    g["namespace"] = meta.str[1].map(NS)
    if drop:
        obsolete = g.term.fillna("").str.contains("obsolete", case=False)
        g = g[g.namespace.notna() & ~obsolete]
    for c in ["ES", "NES", "NOM p-val", "FDR q-val"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g


def add_source_arg(parser) -> None:
    """``--source repo|saved`` and ``--out`` for a table builder's CLI."""
    parser.add_argument("--source", choices=SOURCES, default="repo")
    parser.add_argument("--out", type=Path, help="CSV path (default: si_tables/)")


def write(df: pd.DataFrame, out: Path) -> None:
    """Write a table (plain UTF-8, no BOM, full float precision)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows x {df.shape[1]} columns)")
