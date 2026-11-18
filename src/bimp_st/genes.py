"""Gene names and IDs shared by the figure scripts.

``GENE_SYMBOLS`` holds the hand-curated short symbols the figures print
(lower-case BIMP v4.3 id -> symbol). The web viewer keeps its own copy of the
focal subset; ``scripts/paper/check_gene_symbol_sync.py`` checks that the two
agree.
"""

from __future__ import annotations

GENE_SYMBOLS = {
    "bimp_v4.3_009307": "oct-β2R",
    "bimp_v4.3_005257": "FTZ-F1β",
    "bimp_v4.3_009612": "oct-β1R",
    "bimp_v4.3_006191": "GlyRα1",
    "bimp_v4.3_004137": "HSP90",
    "bimp_v4.3_005222": "CaMKII",
    "bimp_v4.3_008309": "ERR1",
    "bimp_v4.3_005730": "sNPF",
    "bimp_v4.3_005433": "Kr-h1",
    # Kenyon-cell volcano tags with a verifiable short symbol (FlyBase/NCBI Gene):
    # "protein takeout" is `to` (FBgn0040091), "Innexin family member shaking B"
    # is `shakB` (FBgn0085387), "NHS actin remodeling regulator GUK-holder" is
    # `gukh` (FBgn0026239). Genes with no confirmable symbol fall back to their
    # cleaned description.
    "bimp_v4.3_002609": "to",
    "bimp_v4.3_005261": "shakB",
    "bimp_v4.3_010814": "gukh",
}


def read_panel_crosswalk():
    """The panel gene-ID crosswalk (``paths.PANEL_CROSSWALK``) as a DataFrame."""
    import pandas as pd

    from bimp_st.paths import PANEL_CROSSWALK

    return pd.read_csv(PANEL_CROSSWALK)
