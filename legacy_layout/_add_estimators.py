"""Add Lasso + ElasticNet + RF results alongside existing Ridge results."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

OUT = Path("analysis/results")

# Load existing Ridge results
ridge_df = pd.read_csv(OUT / "prediction_model_comparison.tsv", sep="\t")
ridge_df["Estimator"] = "Ridge"
print(f"Existing Ridge results: {len(ridge_df)} rows")

# Load model_df3 (same pipeline as _full_analysis.py)
REPO_ROOT = Path(".")
ICMI_RESULTS = REPO_ROOT / "icmi_paper" / "results"
TASKS = ["T1", "T2", "T3"]

HMM_FEATS = ["group_hr_mean_bpm_mean", "group_eda_phasic_rate_hz_mean",
             "group_temp_mean_mean", "group_et_pupil_mean_mean",
             "tr_silence_duration_s", "tr_backchannel_count", "tr_laughter_count",
             "tr_n_active_speakers", "tr_ovl_count", "tr_ovl_contested",
             "tr_ovl_time_s", "tr_ovl_collaboration_index"]
AUDIO_FEATS = ["audio_energy_mean", "audio_pitch_mean", "audio_pitch_sd", "audio_hnr_mean"]
KEY = ["group_id", "task_id", "window_index"]

hmm_win = pd.read_csv(ICMI_RESULTS / "hmm_input_features_final.tsv", sep="\t")
audio_win = pd.read_csv(ICMI_RESULTS / "audio_window_features.tsv", sep="\t")
merged = hmm_win[KEY + HMM_FEATS].merge(audio_win[KEY + AUDIO_FEATS], on=KEY, how="inner")
merged = merged[merged["task_id"].isin(TASKS)].reset_index(drop=True)
enh = pd.read_csv(ICMI_RESULTS / "enhanced_features_final.tsv", sep="\t")
recovered = enh[(enh["group_id"] == "grp-12") & (enh["task_id"] == "T3")].copy()
recovered["tr_ovl_contested"] = recovered["tr_ovl_competitive_timing"] + recovered["tr_ovl_floor_fight"]
recovered = recovered[KEY + HMM_FEATS]
for c in AUDIO_FEATS:
    recovered[c] = np.nan
merged = pd.concat([merged, recovered], ignore_index=True)

PHYSIO_Z = ["group_hr_mean_bpm_mean", "group_eda_phasic_rate_hz_mean",
            "group_temp_mean_mean", "group_et_pupil_mean_mean"]
LOG_FEATS = ["tr_silence_duration_s", "tr_backchannel_count", "tr_laughter_count",
             "tr_ovl_count", "tr_ovl_contested", "tr_ovl_time_s", "tr_ovl_collaboration_index"]
clean = merged.copy()
for col in PHYSIO_Z + AUDIO_FEATS:
    clean[col] = clean.groupby("group_id")[col].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-9))
for col in LOG_FEATS:
    clean[col] = np.log1p(clean[col].clip(lower=0))

feat_task = clean.groupby(["group_id", "task_id"])[HMM_FEATS + AUDIO_FEATS].mean().reset_index().rename(columns={"task_id": "task"})
target_participant = pd.read_csv("analysis/results/perceived_effectiveness_index_participant.tsv", sep="\t")
available_outcomes = [c for c in ["perceived_effectiveness_z", "team_coordination", "cooperative", "satisfaction", "decision_confidence", "idea_quality"] if c in target_participant.columns]
target_task = target_participant.groupby(["group_id", "task"])[available_outcomes].mean().reset_index()
model_df = feat_task.merge(target_task, on=["group_id", "task"], how="inner")
task_dummies = pd.get_dummies(model_df["task"], prefix="task", drop_first=True)
model_df = pd.concat([model_df, task_dummies], axis=1)
TASK_DUMMY_COLS = list(task_dummies.columns)

sr = pd.read_csv("analysis/results/collective_features_v20260812/collective_task_selfreport.tsv", sep="\t").rename(columns={"task_id": "task"})
SR_OUTCOMES = [c for c in ["voice_inclusion_mean", "mental_demand_mean", "team_coordination_mean",
                            "cooperative_mean", "satisfaction_mean", "decision_confidence_mean", "idea_quality_mean"] if c in sr.columns]
model_df3 = model_df.merge(sr[["group_id", "task"] + SR_OUTCOMES], on=["group_id", "task"], how="left")
for col, tasks in [("team_coordination_mean", ["T1", "T3"]), ("cooperative_mean", ["T2"])]:
    mask = model_df3["task"].isin(tasks)
    vals = model_df3.loc[mask, col]
    model_df3.loc[mask, "_coord_coop_z"] = (vals - vals.mean()) / (vals.std(ddof=1) + 1e-9)

# HMM states
_old_hmm = pd.read_csv(ICMI_RESULTS / "hmm_prediction_results.tsv", sep="\t").rename(columns={"task_id": "task"})
_raw_old = [c for c in ["S0_pct","S1_pct","S2_pct","S3_pct","S4_pct"] if c in _old_hmm.columns]
_old_hmm = _old_hmm.rename(columns={c: f"old_{c}" for c in _raw_old})
HMM_OLD_STATE_FEATS = [f"old_{c}" for c in _raw_old]
model_df3 = model_df3.merge(_old_hmm[["group_id","task"] + HMM_OLD_STATE_FEATS], on=["group_id","task"], how="left")
_new_hmm = pd.read_csv("analysis/results/hmm_state_proportions_expanded.tsv", sep="\t").rename(columns={"task_id": "task"})
_raw_new = [c for c in _new_hmm.columns if c.endswith("_pct")]
_new_hmm = _new_hmm.rename(columns={c: f"new_{c}" for c in _raw_new})
HMM_NEW_STATE_FEATS = [f"new_{c}" for c in _raw_new]
model_df3 = model_df3.merge(_new_hmm[["group_id","task"] + HMM_NEW_STATE_FEATS], on=["group_id","task"], how="left")

# Enhanced/lexical/DA/semantic features
_enh_extra = ["tr_speaking_entropy", "tr_cooperative_overlap", "tr_collaborative_overlap",
              "tr_competitive_overlap", "tr_overlap_count", "tr_filled_pause_count",
              "tr_conflict_overlap_ratio", "tr_ovl_smooth", "tr_ovl_floor_fight",
              "tr_ovl_completion_index", "group_hrv_rmssd_ms_mean", "group_eda_tonic_mean_mean"]
_enh_all = [c for c in _enh_extra if c in enh.columns]
enh_task = enh.rename(columns={"task_id": "task"}).groupby(["group_id", "task"])[_enh_all].mean().reset_index()
for df_m in [enh_task]:
    new_cols = [c for c in df_m.columns if c not in {"group_id", "task"} and c not in model_df3.columns]
    if new_cols:
        model_df3 = model_df3.merge(df_m[["group_id", "task"] + new_cols], on=["group_id", "task"], how="left")

_lex = pd.read_csv("analysis/results/lexical_window_30s.tsv", sep="\t").rename(columns={"task_id": "task"})
LEX_THEORY = [c for c in ["lex_we_i_ratio", "lex_turn_cohesion", "lex_word_gini",
              "lex_hedging_count", "lex_agreement_count", "lex_question_count",
              "lex_suggestion_count", "lex_social_composite", "lex_sentiment_ratio"] if c in _lex.columns]
lex_task = _lex.groupby(["group_id", "task"])[LEX_THEORY].mean().reset_index()
new_cols = [c for c in LEX_THEORY if c not in model_df3.columns]
if new_cols:
    model_df3 = model_df3.merge(lex_task[["group_id", "task"] + new_cols], on=["group_id", "task"], how="left")

_da = pd.read_csv("analysis/results/da_keyword_window_30s.tsv", sep="\t").rename(columns={"task_id": "task"})
DA_FEATS = [c for c in ["da_agreement", "da_disagreement", "da_compromise", "da_proposal",
            "da_question", "da_hedge", "da_backchannel"] if c in _da.columns]
da_task = _da.groupby(["group_id", "task"])[DA_FEATS].mean().reset_index()
new_cols = [c for c in DA_FEATS if c not in model_df3.columns]
if new_cols:
    model_df3 = model_df3.merge(da_task[["group_id", "task"] + new_cols], on=["group_id", "task"], how="left")

_sem = pd.read_csv("analysis/results/semantic_continuity_window_30s.tsv", sep="\t").rename(columns={"task_id": "task"})
SEM_FEATS = ["tr_semantic_continuity", "tr_cross_speaker_similarity"]
sem_task = _sem.groupby(["group_id", "task"])[SEM_FEATS].mean().reset_index()
_cr = pd.read_csv("analysis/results/completion_repetition_window_30s.tsv", sep="\t").rename(columns={"task_id": "task"})
CR_FEATS = ["tr_completion_count", "tr_repetition_count"]
cr_task = _cr.groupby(["group_id", "task"])[CR_FEATS].mean().reset_index()
for df_m in [sem_task, cr_task]:
    new_cols = [c for c in df_m.columns if c not in {"group_id", "task"} and c not in model_df3.columns]
    if new_cols:
        model_df3 = model_df3.merge(df_m[["group_id", "task"] + new_cols], on=["group_id", "task"], how="left")

# ── Feature sets (same as Ridge run) ─────────────────────────────────────────
PHYSIO_ONLY = [c for c in ["group_hr_mean_bpm_mean", "group_eda_phasic_rate_hz_mean",
               "group_temp_mean_mean", "group_hrv_rmssd_ms_mean", "group_eda_tonic_mean_mean"] if c in model_df3.columns]
ET_ONLY = [c for c in ["group_et_pupil_mean_mean"] if c in model_df3.columns]
TURN_TAKING = [c for c in ["tr_silence_duration_s", "tr_backchannel_count",
    "tr_n_active_speakers", "tr_speaking_entropy", "tr_overlap_count",
    "tr_competitive_overlap", "tr_cooperative_overlap", "tr_collaborative_overlap",
    "tr_filled_pause_count", "tr_conflict_overlap_ratio",
    "tr_ovl_smooth", "tr_ovl_floor_fight", "tr_ovl_completion_index"] if c in model_df3.columns]
EXPANDED_RAW = [c for c in [
    "group_hr_mean_bpm_mean", "group_eda_phasic_rate_hz_mean",
    "group_hrv_rmssd_ms_mean", "group_eda_tonic_mean_mean",
    "group_temp_mean_mean", "group_et_pupil_mean_mean",
    "tr_silence_duration_s", "tr_backchannel_count", "tr_overlap_count",
    "tr_competitive_overlap", "tr_laughter_count", "tr_n_active_speakers",
    "tr_speaking_entropy", "tr_cooperative_overlap", "tr_collaborative_overlap",
    "lex_word_gini", "lex_we_i_ratio", "lex_turn_cohesion", "lex_hedging_count",
] if c in model_df3.columns]

MODALITY_SETS = {
    "Physio only":            PHYSIO_ONLY,
    "Eye-tracking only":      ET_ONLY,
    "Physio + ET":            PHYSIO_ONLY + ET_ONLY,
    "Audio only":             AUDIO_FEATS,
    "Turn-taking only":       TURN_TAKING,
    "Lexical (theory)":       LEX_THEORY,
    "DA keywords":            DA_FEATS,
    "Semantic + completion":  SEM_FEATS + CR_FEATS,
    "HMM states (K=4)":      HMM_NEW_STATE_FEATS,
    "HMM states (K=5)":      HMM_OLD_STATE_FEATS,
    "Expanded raw (19)":      EXPANDED_RAW,
    "States + Turn-taking":   HMM_NEW_STATE_FEATS + TURN_TAKING,
    "States + Lexical":       HMM_NEW_STATE_FEATS + LEX_THEORY,
    "States + DA":            HMM_NEW_STATE_FEATS + DA_FEATS,
    "States + Lex + DA":      HMM_NEW_STATE_FEATS + LEX_THEORY + DA_FEATS,
    "Turn + Lex + DA":        TURN_TAKING + LEX_THEORY + DA_FEATS,
}

TARGETS = {
    "effectiveness_z":     (["T1","T2","T3"], "perceived_effectiveness_z"),
    "voice_inclusion":     (["T1","T2","T3"], "voice_inclusion_mean"),
    "mental_demand":       (["T1","T2","T3"], "mental_demand_mean"),
    "coord+coop":          (["T1","T2","T3"], "_coord_coop_z"),
    "cooperative":         (["T2"],           "cooperative_mean"),
    "team_coordination":   (["T1","T3"],      "team_coordination_mean"),
    "satisfaction":        (["T2","T3"],       "satisfaction_mean"),
    "decision_confidence": (["T1"],            "decision_confidence_mean"),
    "idea_quality":        (["T3"],            "idea_quality_mean"),
}

# ── Estimator definitions ────────────────────────────────────────────────────
ESTIMATORS = {
    "Lasso": lambda: make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 2, 20), cv=5, max_iter=10000, random_state=42)),
    "ElasticNet": lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 2, 20), cv=5, max_iter=10000, random_state=42)),
    "RF": lambda: make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42)),
}

def run_model(estimator_fn, feature_cols, target_col, task_filter):
    df = model_df3[model_df3["task"].isin(task_filter)].copy()
    if target_col not in df.columns or df[target_col].isna().all():
        return {"R2": np.nan, "MAE": np.nan, "rho": np.nan, "perm_p": np.nan}
    task_cols = [c for c in TASK_DUMMY_COLS if df[c].astype(float).nunique() > 1]
    cols = [c for c in feature_cols if c in df.columns] + task_cols
    if not cols:
        return {"R2": np.nan, "MAE": np.nan, "rho": np.nan, "perm_p": np.nan}
    X = df[cols].values.astype(float)
    y = df[target_col].values.astype(float)
    g = df["group_id"].values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_m, y_m, g_m = X[mask], y[mask], g[mask]
    if mask.sum() < max(3, len(np.unique(g_m))):
        return {"R2": np.nan, "MAE": np.nan, "rho": np.nan, "perm_p": np.nan}
    try:
        preds = cross_val_predict(estimator_fn(), X_m, y_m, cv=LeaveOneGroupOut(), groups=g_m)
    except Exception:
        return {"R2": np.nan, "MAE": np.nan, "rho": np.nan, "perm_p": np.nan}
    r2 = r2_score(y_m, preds)
    mae = mean_absolute_error(y_m, preds)
    rho_val, _ = spearmanr(y_m, preds)
    # 50 permutations (faster, still informative)
    rng = np.random.default_rng(42)
    n_better = 0
    for _ in range(50):
        y_s = rng.permutation(y_m)
        try:
            p_s = cross_val_predict(estimator_fn(), X_m, y_s, cv=LeaveOneGroupOut(), groups=g_m)
            if r2_score(y_s, p_s) >= r2:
                n_better += 1
        except Exception:
            pass
    perm_p = (n_better + 1) / 51
    return {"R2": round(r2, 3), "MAE": round(mae, 3), "rho": round(float(rho_val), 3), "perm_p": round(perm_p, 3)}

# ── Run Lasso, ElasticNet, RF ─────────────────────────────────────────────────
new_rows = []
total = len(ESTIMATORS) * len(MODALITY_SETS) * len(TARGETS)
done = 0
for est_name, est_fn in ESTIMATORS.items():
    for m_name, feats in MODALITY_SETS.items():
        for t_name, (tasks, t_col) in TARGETS.items():
            done += 1
            if done % 30 == 0:
                print(f"  [{done}/{total}] {est_name} / {m_name} / {t_name}")
            res = run_model(est_fn, feats, t_col, tasks)
            res["Estimator"] = est_name
            res["Model"] = m_name
            res["Target"] = t_name
            new_rows.append(res)

new_df = pd.DataFrame(new_rows)
print(f"\nNew results: {len(new_df)} rows")

# Combine with existing Ridge
all_df = pd.concat([ridge_df, new_df], ignore_index=True)
all_df.to_csv(OUT / "prediction_all_estimators.tsv", sep="\t", index=False)
print(f"Saved combined results: {len(all_df)} rows -> prediction_all_estimators.tsv")

# ── Print comparison: best estimator per target ──────────────────────────────
print("\n=== Best R² per target × estimator (significant only, p<0.05) ===")
sig = all_df[all_df["perm_p"] < 0.05]
for target in TARGETS:
    sub = sig[sig["Target"] == target].sort_values("R2", ascending=False)
    if len(sub) == 0:
        print(f"\n{target}: no significant models")
        continue
    print(f"\n{target}:")
    for est in ["Ridge", "Lasso", "ElasticNet", "RF"]:
        e_sub = sub[sub["Estimator"] == est]
        if len(e_sub) > 0:
            best = e_sub.iloc[0]
            print(f"  {est:12s}  {best['Model']:25s}  R²={best['R2']:.3f}  ρ={best['rho']:.3f}  p={best['perm_p']:.3f}")
        else:
            print(f"  {est:12s}  —")

# ── Robustness summary: do Ridge and Lasso agree? ────────────────────────────
print("\n=== Robustness check: Ridge vs Lasso on significant Ridge models ===")
ridge_sig = ridge_df[ridge_df["perm_p"] < 0.05]
for _, r in ridge_sig.sort_values("R2", ascending=False).head(10).iterrows():
    lasso_match = new_df[(new_df["Estimator"] == "Lasso") & (new_df["Model"] == r["Model"]) & (new_df["Target"] == r["Target"])]
    en_match = new_df[(new_df["Estimator"] == "ElasticNet") & (new_df["Model"] == r["Model"]) & (new_df["Target"] == r["Target"])]
    rf_match = new_df[(new_df["Estimator"] == "RF") & (new_df["Model"] == r["Model"]) & (new_df["Target"] == r["Target"])]
    l_r2 = lasso_match.iloc[0]["R2"] if len(lasso_match) > 0 else np.nan
    e_r2 = en_match.iloc[0]["R2"] if len(en_match) > 0 else np.nan
    rf_r2 = rf_match.iloc[0]["R2"] if len(rf_match) > 0 else np.nan
    print(f"  {r['Model']:25s} → {r['Target']:20s}  Ridge={r['R2']:.3f}  Lasso={l_r2:.3f}  EN={e_r2:.3f}  RF={rf_r2:.3f}")

print("\nDone.")
