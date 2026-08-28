"""Benchmark GaussianHMM.fit timing on the real feature data to estimate notebook runtime."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(".")
WINDOW_FILE = ROOT / "analysis" / "results" / "collective_features_v20260812" / "collective_window_features.tsv"
windows = pd.read_csv(WINDOW_FILE, sep="\t")

PHYSIO_FEATURES = [
    "group_hr_mean_bpm_mean", "group_eda_phasic_rate_hz_mean", "group_hrv_rmssd_ms_mean",
    "group_eda_tonic_mean_mean", "group_temp_mean_mean", "group_et_pupil_mean_mean",
]
TRANSCRIPT_FEATURES = [
    "tr_silence_duration_s", "tr_backchannel_count", "tr_overlap_count",
    "tr_competitive_overlap", "tr_laughter_count", "tr_n_active_speakers", "tr_speaking_entropy",
]
CONTEXT_OVERLAP_FEATURES = ["tr_cooperative_overlap", "tr_collaborative_overlap"]
LEXICAL_FEATURES = ["lex_word_gini", "lex_we_i_ratio", "lex_turn_cohesion", "lex_hedging_count"]
ALL_FEATURES = [c for c in PHYSIO_FEATURES + TRANSCRIPT_FEATURES + CONTEXT_OVERLAP_FEATURES + LEXICAL_FEATURES
                if c in windows.columns]

df = windows.dropna(subset=["tr_spk_duration_s"]).copy()
parts = []
for (grp, task), gdf in df.groupby(["group_id", "task_id"]):
    active_idx = gdf.loc[gdf["tr_spk_duration_s"] > 0, "window_index"]
    if active_idx.empty:
        continue
    parts.append(gdf[(gdf["window_index"] >= active_idx.min()) & (gdf["window_index"] <= active_idx.max())])
df = pd.concat(parts, ignore_index=True)

keep = [c for c in ALL_FEATURES if df[c].isna().mean() <= 0.4]
ALL_FEATURES = keep
X_raw = SimpleImputer(strategy="mean").fit_transform(df[ALL_FEATURES])
X = StandardScaler().fit_transform(X_raw)
meta = df[["group_id", "task_id", "window_index"]].reset_index(drop=True)

pca = PCA(n_components=min(10, len(ALL_FEATURES)))
X_pca = pca.fit_transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_pcs = int(np.searchsorted(cumvar, 0.80)) + 1
X_reduced = X_pca[:, :n_pcs]
print(f"n_pcs={n_pcs}, X_reduced shape={X_reduced.shape}")

groups = sorted(meta["group_id"].unique())
tasks = sorted(meta["task_id"].unique())
sequences_X = []
for grp in groups:
    for task in tasks:
        mask = (meta["group_id"] == grp) & (meta["task_id"] == task)
        idx = meta[mask].sort_values("window_index").index
        if len(idx) < 2:
            continue
        sequences_X.append(X_reduced[idx])
print(f"n_sequences={len(sequences_X)}, total_T={sum(len(s) for s in sequences_X)}")


class GaussianHMM:
    def __init__(self, n_states=3, n_iter=100, tol=1e-4, random_state=42):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _log_emit(self, X):
        T, D = X.shape
        log_p = np.zeros((T, self.K))
        for k in range(self.K):
            log_p[:, k] = multivariate_normal.logpdf(X, mean=self.means_[k], cov=np.diag(self.covs_[k]))
        return log_p

    def _forward(self, log_emit):
        T, K = log_emit.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(self.pi_ + 1e-300) + log_emit[0]
        log_A = np.log(self.A_ + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_A[:, k]) + log_emit[t, k]
        return log_alpha

    def _backward(self, log_emit):
        T, K = log_emit.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(self.A_ + 1e-300)
        for t in range(T-2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(log_A[k] + log_emit[t+1] + log_beta[t+1])
        return log_beta

    def fit(self, sequences):
        D = sequences[0].shape[1]
        K = self.K
        all_X = np.vstack(sequences)
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(all_X)
        self.means_ = km.cluster_centers_.copy()
        self.covs_ = np.array([np.var(all_X[labels == k], axis=0).clip(1e-4) for k in range(K)])
        self.pi_ = np.ones(K) / K
        self.A_ = np.ones((K, K)) / K

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            pi_num, A_num = np.zeros(K), np.zeros((K, K))
            mean_num, cov_num = np.zeros((K, D)), np.zeros((K, D))
            weight_s, total_ll = np.zeros(K), 0.0

            for X in sequences:
                T = len(X)
                if T < 2:
                    continue
                log_e = self._log_emit(X)
                log_a = self._forward(log_e)
                log_b = self._backward(log_e)
                ll = logsumexp(log_a[-1])
                total_ll += ll

                log_gamma = log_a + log_b
                log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
                gamma = np.exp(log_gamma)

                log_A = np.log(self.A_ + 1e-300)
                log_xi = (log_a[:-1, :, None] + log_A[None, :, :] + log_e[1:, None, :] + log_b[1:, None, :])
                log_xi -= logsumexp(log_xi.reshape(T-1, -1), axis=1, keepdims=True).reshape(T-1, 1, 1)
                xi = np.exp(log_xi)

                pi_num += gamma[0]
                A_num += xi.sum(axis=0)
                weight_s += gamma.sum(axis=0)
                mean_num += gamma.T @ X
                for k in range(K):
                    cov_num[k] += (gamma[:, k:k+1] * (X - self.means_[k]) ** 2).sum(axis=0)

            self.pi_ = np.maximum(pi_num / pi_num.sum(), 1e-10)
            self.A_ = np.maximum(A_num / A_num.sum(axis=1, keepdims=True), 1e-10)
            for k in range(K):
                if weight_s[k] > 1e-6:
                    self.means_[k] = mean_num[k] / weight_s[k]
                    self.covs_[k] = np.maximum(cov_num[k] / weight_s[k], 1e-4)

            if abs(total_ll - prev_ll) < self.tol:
                break
            prev_ll = total_ll

        self.log_likelihood_ = total_ll
        self.n_iter_run_ = iteration + 1
        return self


for k in [2, 5, 8]:
    t0 = time.perf_counter()
    m = GaussianHMM(n_states=k, n_iter=200, tol=1e-5, random_state=42)
    m.fit(sequences_X)
    dt = time.perf_counter() - t0
    print(f"k={k}: {dt:.2f}s for {m.n_iter_run_} iterations ({dt/m.n_iter_run_*1000:.1f} ms/iter)")
