"""BIMP-ST: Bumblebee brain spatial transcriptomics analysis pipeline."""

__version__ = "0.1.0"

# No eager submodule imports: cluster_labels/utils pull in numpy, and
# bimp_st.paths must stay importable without it (e.g. from a Snakefile in a bare
# snakemake environment). Consumers import the submodule they need explicitly.
