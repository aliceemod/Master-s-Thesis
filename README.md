# Multimodal Group Interaction | Master's Thesis

This repository contains the analysis for my master's thesis using data from the AffectAI project. I study how patterns in group conversation and multimodal signals relate to participants' perceived effectiveness of a group session. The work brings together transcript, audio, eye-tracking, and physiological features, with both direct prediction and latent-state approaches to modeling group interaction.

The broader AffectAI work was developed in a shared repository under a company account, with contributions from other project members. This separate repository collects the material for my thesis. It is not a complete copy of the collaborative project.

## Repository guide

| Directory | Contents |
| --- | --- |
| `analysis/` | Exploratory notebooks and validation analyses. |
| `data/` | Metadata, transcripts, stimulus responses, and derived data used in the analysis. |
| `feature_engineering/` | Extraction and preprocessing for audio, dialogue, eye tracking, physiology, and other features. |
| `models/` | Latent-state and outcome-prediction analyses. |
| `results/` | Generated figures, tables, and reports. |
| `software/` | Pipeline tools, configuration, and tests. |
| `archive/` | Earlier data-processing material and scripts retained for reference. |

The directories follow the path from data and feature extraction through analysis and modeling to reported results. Notebooks document exploratory work as well as later analyses; their presence does not imply that every experiment is part of the final thesis findings.

## Reproducibility

The Python package configuration is in `software/pyproject.toml`. The project requires Python 3.10 or newer. Analysis dependencies are grouped under the `analysis` extra, while development dependencies include `pytest` and `ruff`.

The usual workflow is:

1. Prepare the data under `data/`.
2. Run the relevant feature pipelines in `feature_engineering/`.
3. Run the HMM or prediction notebooks in `models/`.
4. Review generated tables, figures, and reports under `results/`.

Most notebooks expect to be run from the repository root and use paths relative to this structure. Large raw recordings and machine-specific environments are not part of the repository.
