# Multimodal Group Interaction | Master's Thesis

This repository contains the analysis for my master's thesis using data from the AffectAI project. The thesis asks whether observable patterns in group interaction can help explain or predict how effective participants perceive their group to be. The analyses combine conversational and transcript features with audio, eye-tracking, physiological, and semantic measures.

Two complementary approaches are used: prediction models relate measured features to perceived outcomes, while latent-state models look for recurring patterns in group interaction over time. The repository includes exploratory and validation work as well as later analyses; not every experiment represents a final thesis result.

The broader AffectAI project was developed in a shared repository under a company account, with contributions from other project members. This repository is a separate, thesis-focused selection of analyses and supporting materials from that larger project. It is not the complete AffectAI project, and its contents should not be read as a full account of the shared project or all of its contributors' work.

## Start here

For a quick route through the project:

1. [Feature audit](analysis/exploratory/feature_eda_and_selection.ipynb): review feature coverage and selection diagnostics.
2. [Outcome construction](analysis/exploratory/target_index_eda.ipynb): see how perceived-effectiveness measures are examined and combined.
3. [HMM analysis](models/hmm/collective_states_hmm_updated_overlaps.ipynb): inspect the latent-state analysis of group interaction.
4. [Prediction analysis](models/prediction/effectiveness_prediction_comprehensive.ipynb): compare feature sets for perceived outcomes.
5. [Results summary](analysis/exploratory/thesis_final_results_summary.ipynb): review the curated findings and their qualifications.

These notebooks are a guide to the main analyses, not a single automatic run sequence. Some rely on derived data already present in the repository.

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

The directories follow the project from data and feature extraction through analysis and modeling to reported results. The `archive/` directory contains retained reference material and is not part of the main analysis path.

## Reproducibility

The tools under `software/` have a Python package configuration in `software/pyproject.toml` and require Python 3.10 or newer. That configuration is not a complete environment specification for every analysis notebook; notebook dependencies vary by modality and workflow.

At a high level, the analysis workflow is:

1. Prepare the data under `data/`.
2. Run the relevant feature pipelines in `feature_engineering/`.
3. Run the HMM or prediction notebooks in `models/`.
4. Review generated tables, figures, and reports under `results/`.

Most notebooks use paths relative to the repository and expect to be run from its root. Derived datasets are organized under `data/derived/`, while generated figures, tables, and reports are organized under `results/`. Large raw recordings and machine-specific environments are not included, so a full rerun may require access to the original data and modality-specific dependencies.
