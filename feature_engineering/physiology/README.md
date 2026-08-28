# Feature Extraction Tools

Task-aware physiology and semantic biomarker extraction for GroupAffect-4 session data.

## Scripts
- `extract_physio_features.py`: EmotiBit participant-task and rolling-window features.
- `extract_audio_features.py`: DPA close-talk speech/prosody features with bleed rejection.
- `analyze_individual_audio.py`: participant-level audio/prosody task profiles,
  baseline deltas, paired task tests, QC summaries, and figures.
- `analyze_physio_paper.py`: paper-facing physio usability, task-effect, correlation,
  and temporal-profile summaries.
- `analyze_autonomic_paper.py`: combined EmotiBit + Tobii pupil paper summaries and figures.
- `extract_pupil_features.py`: Tobii pupil participant-task and rolling-window features.
- `compute_group_dynamics.py`: dyad/group synchrony metrics from window tables.
- `build_semantic_biomarkers.py`: semantic biomarker composites from extracted features.
- `build_participant_group_comparisons.py`: participant joins with answers/annotations + group pooled comparisons.
- `analyze_multimodal_statistics.py`: downstream mixed models, paired task tests,
  cross-modal correlations, group checks, and dyad synchrony tests.
- `run_feature_pipeline.py`: runs all scripts in sequence.
- `visualize_physio_features.py`: quick PNG summaries for physio feature/QC review.

## Typical Run
```bash
python tools/features/run_feature_pipeline.py \
  --data-root GroupAffect-4-data-processing-seed/data \
  --out-dir data/derived_features \
  --window-s 30 \
  --step-s 15
```

Quick physio figures:
```bash
python tools/features/visualize_physio_features.py \
  --features-dir features \
  --out-dir figures/physio
```

Paper physio analysis:
```bash
python tools/features/analyze_physio_paper.py \
  --features-dir features \
  --results-dir results/physio \
  --figures-dir figures/physio
```

Pupil + physio paper analysis:
```bash
python tools/features/extract_pupil_features.py \
  --data-root F:/processed_data/sub-01 \
  --out-dir features \
  --window-s 30 \
  --step-s 15

python tools/features/analyze_autonomic_paper.py \
  --features-dir features \
  --results-dir results/autonomic \
  --figures-dir figures/autonomic
```

Audio features:
```bash
python tools/features/extract_audio_features.py \
  --audio-root F:/bids_release_no_video \
  --out-dir data/derived_features

python tools/features/analyze_individual_audio.py \
  --features-dir data/derived_features \
  --out-dir results/audio
```

Inferential multimodal statistics:
```bash
python -m pip install -e ".[analysis]"

python tools/features/analyze_multimodal_statistics.py \
  --features-dir data/derived_features \
  --personality-dir results/personality \
  --out-dir results/statistics
```
This also writes visual summaries to `results/statistics/figures/` unless
`--no-figures` is passed. If `audio_participant_task.tsv` exists under
`--features-dir` (or is passed with `--audio-features`), audio/prosody features
are merged into the participant, group, model, correlation, and figure outputs.

## Key Outputs
- `physio_participant_task.tsv` (canonical paper-ready EmotiBit table)
- `physio_qc_summary.tsv` (participant-task QC and missingness table)
- `physio_window_30s.tsv` (canonical rolling-window EmotiBit table)
- `audio_participant_task.tsv`
- `audio_qc_summary.tsv`
- `results/audio/individual_audio_task.tsv`
- `results/audio/individual_audio_summary.tsv`
- `results/audio/individual_audio_qc.tsv`
- `results/audio/individual_audio_task_pairwise_tests.tsv`
- `results/audio/figures/individual_audio_task_heatmap.png`
- `results/audio/figures/individual_speaking_fraction_by_task.png`
- `results/audio/figures/individual_audio_profiles.png`
- `results/audio/figures/individual_audio_qc_coverage.png`
- `physio_feature_definitions.tsv`
- `results/physio/physio_paper_feature_usability.tsv`
- `results/physio/physio_task_delta_stats.tsv`
- `results/physio/physio_session_task_summary.tsv`
- `results/physio/physio_qc_flag_counts.tsv`
- `results/physio/physio_feature_correlations.tsv`
- `results/physio/physio_temporal_profile.tsv`
- `features_pupil_participant_task.tsv`
- `features_pupil_window_30s.tsv`
- `results/autonomic/autonomic_task_delta_stats.tsv`
- `results/autonomic/autonomic_modality_coverage.tsv`
- `results/autonomic/autonomic_pupil_physio_links.tsv`
- `results/autonomic/autonomic_paper_key_findings.tsv`
- `features_physio_participant_task.tsv` (legacy alias, unless disabled)
- `features_physio_window_30s.tsv` (legacy alias, unless disabled)
- `features_pupil_participant_task.tsv`
- `features_pupil_window_30s.tsv`
- `features_group_dynamics_window_30s.tsv`
- `features_group_dynamics_task.tsv`
- `semantic_biomarkers_participant_task.tsv`
- `semantic_biomarkers_window_30s.tsv`
- `participant_features_answers_annotations.tsv`
- `group_pool_task_summary.tsv`
- `participant_vs_group_comparison.tsv`
- `biomarker_vad_label_comparison.tsv`
- `biomarker_vad_performance_by_participant.tsv`
- `biomarker_annotation_performance_by_participant.tsv`
- `results/statistics/analysis_dataset_participant_task.tsv`
- `results/statistics/audio_feature_status.tsv`
- `results/statistics/analysis_combination_catalog.tsv`
- `results/statistics/mixed_model_results.tsv`
- `results/statistics/participant_task_pairwise_tests.tsv`
- `results/statistics/participant_cross_modal_correlations.tsv`
- `results/statistics/group_task_pairwise_tests.tsv`
- `results/statistics/pair_synchrony_task_tests.tsv`
- `results/statistics/figures/analysis_combination_coverage.png`
- `results/statistics/figures/participant_task_effects_heatmap.png`
- `results/statistics/figures/audio_task_feature_heatmap.png`
- `results/statistics/figures/cross_modal_correlation_heatmap.png`
- `results/statistics/figures/correlation_family_*.png`
- `results/statistics/figures/mixed_model_forest.png`
- `results/statistics/figures/mixed_model_coefficient_heatmap.png`
- `results/statistics/figures/mixed_model_screening_volcano.png`
- `results/statistics/figures/pair_synchrony_by_task.png`
- `results/statistics/figures/pair_synchrony_context_heatmap.png`
- `results/statistics/figures/group_trait_correlation_heatmap.png`

## Notes
- Uses participant IDs `P1`-`P4` from split files (`*_acq-P*_*.tsv.gz`).
- Assumes task-split files already exist (`task-T0` to `task-T4`).
- Default channel indices can be overridden via CLI flags.
- For paper coverage tables, run physio extraction with `--include-missing-qc`.
- The statistics pipeline is exploratory by default. Treat raw p-values as
  screening results and prioritize effect sizes, confidence intervals, and
  FDR-adjusted q-values.

