import json, pathlib

path = pathlib.Path("analysis/effectiveness_prediction_models.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

# Fix: the HMM states modeling cell merges _hmm_pred into model_df3 again,
# but the columns are already there via model_df2. Just set HMM_STATE_FEATS = HMM_OLD_STATE_FEATS
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if ("HMM_STATE_FEATS = ['S0_pct', 'S1_pct'" in src and
        "model_df3 = model_df3.merge(_hmm_pred" in src and
        "run_cv" in src):
        # Replace the load+merge with just a reference to existing columns
        new_src = src.replace(
            "# ── HMM state proportions (from icmi_paper HMM run) ─────────────────────────\n"
            "_hmm_pred = pd.read_csv(REPO_ROOT / 'icmi_paper' / 'results' / 'hmm_prediction_results.tsv', sep='\\t')\n"
            "_hmm_pred = _hmm_pred.rename(columns={'task_id': 'task'})\n"
            "HMM_STATE_FEATS = ['S0_pct', 'S1_pct', 'S2_pct', 'S3_pct', 'S4_pct']\n"
            "model_df3 = model_df3.merge(_hmm_pred[['group_id', 'task'] + HMM_STATE_FEATS],\n"
            "                            on=['group_id', 'task'], how='left')\n"
            "print(f'HMM state feats NaNs: {model_df3[HMM_STATE_FEATS].isnull().sum().sum()}')\n"
            "print(f'composite_actual already in model_df3: {\"composite_actual\" in model_df3.columns}')\n",
            "# Old HMM states already merged via model_df2 → model_df3\n"
            "HMM_STATE_FEATS = [c for c in HMM_OLD_STATE_FEATS if c in model_df3.columns]\n"
            "print(f'HMM_STATE_FEATS available: {HMM_STATE_FEATS}')\n"
            "print(f'HMM_NEW_STATE_FEATS available: {HMM_NEW_STATE_FEATS}')\n"
            "print(f'composite_actual in model_df3: {\"composite_actual\" in model_df3.columns}')\n"
        )
        if new_src != src:
            cell["source"] = new_src
            cell["outputs"] = []
            cell["execution_count"] = None
            print("Fixed: removed redundant HMM merge from modeling cell")
        else:
            print("WARNING: pattern not found")
            # Print a snippet to debug
            idx = src.find("HMM_STATE_FEATS")
            print(repr(src[max(0,idx-50):idx+200]))
        break

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Saved")
