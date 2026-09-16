import pandas as pd, numpy as np, re
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

R = Path("icmi_paper/results")
STIM_DIR = Path("metadata/extracted stimuli answers")
HMM_GROUPS = [f"grp-{i:02d}" for i in range(7, 17)]
GRPRE = re.compile(r"grp[_-]?(\d+)", re.IGNORECASE)

def load_item(item_key):
    rows = []
    for f in sorted(STIM_DIR.glob("*_extracted.tsv")):
        m = GRPRE.search(f.name)
        if not m: continue
        grp = "grp-%02d" % int(m.group(1))
        if grp not in HMM_GROUPS: continue
        df = pd.read_csv(f, sep="\t", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns: df[c] = df[c].str.strip()
        pb = df[(df["response_type"] == "postblock") & (df["item_key"] == item_key)]
        for _, row in pb.iterrows():
            v = pd.to_numeric(row["item_value"], errors="coerce")
            if pd.notna(v):
                rows.append({"group_id": grp, "task": row["task"], "participant": row["participant"], "score": v})
    return pd.DataFrame(rows).drop_duplicates(subset=["group_id","task","participant"])

tc = load_item("team_coordination")
coop = load_item("cooperative")
mc2 = load_item("manipcheck_t2")
print(f"manipcheck_t2: n={len(mc2)}, range=[{mc2.score.min()},{mc2.score.max()}], mean={mc2.score.mean():.2f}")

# within-T2 convergent check: does manipcheck_t2 relate to cooperative at all?
m = coop.merge(mc2, on=["group_id","participant"], suffixes=("_coop","_mc2"))
r, p = spearmanr(m["score_coop"], m["score_mc2"])
print(f"cooperative vs manipcheck_t2 @ T2 (same task/time): rho={r:+.3f} p={p:.3f} n={len(m)}")

# build composite with manipcheck_t2 as T2 stand-in, z-scored within item
tc["z"] = (tc["score"] - tc["score"].mean()) / tc["score"].std(ddof=1)
mc2["z"] = (mc2["score"] - mc2["score"].mean()) / mc2["score"].std(ddof=1)
composite = pd.concat([tc, mc2], ignore_index=True).rename(columns={"z":"coordination_composite"})
comp_task = composite.groupby(["group_id","task"])["coordination_composite"].mean().reset_index()
print(f"\ncomposite (group x task): n={len(comp_task)}")
print(comp_task.groupby("task")["coordination_composite"].agg(["mean","std","count"]))

task_d = pd.get_dummies(comp_task["task"], prefix="task", drop_first=True)
preds0 = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV()), task_d.values, comp_task["coordination_composite"].values,
                            cv=GroupKFold(min(5, comp_task.group_id.nunique())), groups=comp_task["group_id"].values)
print(f"Task-dummies-ONLY CV R2 (sanity, should be ~0): {r2_score(comp_task['coordination_composite'].values, preds0):.3f}")

assign = pd.read_csv(R / "hmm_cluster_assignments_k5_updated_overlaps.tsv", sep="\t")
feat = pd.read_csv(R / "hmm_input_features_final.tsv", sep="\t")
fa = feat.merge(assign, on=["group_id", "task_id"])
pct = fa.groupby(["group_id", "task_id"])["hmm_state"].value_counts(normalize=True).mul(100).reset_index()
pct.columns = ["group_id", "task_id", "hmm_state", "pct"]
pct_wide = pct.pivot_table(index=["group_id", "task_id"], columns="hmm_state", values="pct", fill_value=0).reset_index()
pct_wide.columns = ["group_id", "task_id"] + [f"S{k}_pct" for k in range(4)]
data = pct_wide.rename(columns={"task_id": "task"}).merge(comp_task, on=["group_id", "task"], how="inner")
print(f"\nn={len(data)} groups={data.group_id.nunique()}")
groups = data["group_id"].values
y = data["coordination_composite"].values
n_splits = min(5, data["group_id"].nunique())
for cols in [["S1_pct"], ["S2_pct"], ["S1_pct","S2_pct"]]:
    Xs = data[cols].values
    preds = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV()), Xs, y, cv=GroupKFold(n_splits), groups=groups)
    print(f"CV R2 ({'+'.join(cols)}): {r2_score(y, preds):.3f}")
