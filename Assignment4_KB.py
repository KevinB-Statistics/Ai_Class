"""
Pipeline (closed-form ESF):
1) Load simulation CSVs.
2) Compute KL divergence from Ewens using CLOSED-FORM E[a_j] per (run_id, H_id, theta).
3) Merge features (mu, rho, theta, K_T, mean_phi_t, var_phi_t).
4) Make 4+ exploratory plots.
"""
# ===================== 0) Imports & config =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

sns.set_context("notebook")
plt.rcParams["figure.dpi"] = 120
# ===================== 1) Closed-form Ewens expectations =====================
def ewens_expected_aj(theta: float, n: int) -> np.ndarray:
    """
    Closed-form expectation under Ewens:
      E[a_j | n, theta] = (theta/j) * (n!/(n-j)!) * Gamma(theta+n-j)/Gamma(theta+n)
    Returned vector has length n with index j-1 corresponding to E[a_j].
    Uses log-gamma for numerical stability.
    """
    j = np.arange(1, n + 1, dtype=float)
    log_fall = gammaln(n + 1.0) - gammaln(n - j + 1.0)
    log_gamma_ratio = gammaln(theta + n - j) - gammaln(theta + n)
    log_E = np.log(theta) - np.log(j) + log_fall + log_gamma_ratio
    return np.exp(log_E)

def ewens_expected_probs(theta: float, n: int) -> np.ndarray:
    """
    Normalize E[a_j] to a probability distribution over j (sums to 1).
    Useful for KL comparisons on size-frequency vectors.
    """
    E = ewens_expected_aj(theta, n)
    S = E.sum()
    return E / S if S > 0 else E

# ===================== 2) Load raw CSVs =====================
# sample_results.csv: run_id,H_id,j,a_j_sample,n_s,theta
# summary_results.csv: run_id,H_id,theta,mu,rho,K_T,mean_phi_t,var_phi_t,...
sample = pd.read_csv("sample_results.csv")
summary = pd.read_csv("summary_results.csv")
aggregate = pd.read_csv("aggregates_results.csv")

# ensure numeric typing for counts
sample["j"] = sample["j"].astype(int)
sample["a_j_sample"] = sample["a_j_sample"].astype(int)
#%%
# ===================== 3) Compute KL per (run_id, H_id, theta) =====================
rows = []
eps = 1e-12 # smoothing to avoid log(0)
E_cache: dict[tuple, np.ndarray] = {}  #cache E[a_j] by (theta, n_s)

for (run_id, H_id, theta), grp in sample.groupby(["run_id", "H_id", "theta"]):
    # build empirical a_j for j=1..n_s
    n_s = int((grp["j"] * grp["a_j_sample"]).sum())
    emp_aj = np.zeros(n_s, float)
    emp_aj[grp["j"].to_numpy() - 1] = grp["a_j_sample"].to_numpy()

    # normalize to probability over j (size-frequency distribution)
    p_emp = emp_aj / emp_aj.sum()

    # closed-form Ewens expectation at same (n_s, theta)
    key = (float(theta), n_s)
    if key not in E_cache:
        E_cache[key] = ewens_expected_probs(theta=float(theta), n=n_s)
    p_esf = E_cache[key]

    # KL(p_emp || p_esf)
    kl = np.sum((p_emp + eps) * (np.log(p_emp + eps) - np.log(p_esf + eps)))

    rows.append({"run_id": run_id, "H_id": H_id, "theta": float(theta), "n_s": n_s, "KL": kl})

dev = pd.DataFrame(rows)

# ===================== 4) Merge features =====================
# Each file contains complementary information:
# - dev: KL divergences (run_id, H_id, theta, KL)
# - summary: base parameters (theta, mu, rho)
# - aggregates: per-generation stats (N_t, K_t, mean_phi_t, var_phi_t)

# --- build final-generation aggregates (no FutureWarning) ---
# pick the row with max t per (run_id,H_id,theta)
idx = aggregate.groupby(["run_id", "H_id", "theta"])["t"].idxmax()
agg_final = (
    aggregate.loc[idx, ["run_id", "H_id", "theta", "K_t", "mean_phi_t", "var_phi_t"]]
    .rename(columns={"K_t": "K_T"})
    .reset_index(drop=True)
)

# --- merge dev + summary + final aggregates ---
summary_key = summary.drop_duplicates(subset=["run_id", "H_id", "theta"])
df = (
    dev.merge(summary_key, on=["run_id", "H_id", "theta"], how="left")
       .merge(agg_final,    on=["run_id", "H_id", "theta"], how="left")
)

# --- resolve duplicate columns created upstream ---
# prefer n_s from dev; fall back to summary if needed
if "n_s_x" in df.columns or "n_s_y" in df.columns:
    df["n_s"] = df.get("n_s_x", df.get("n_s_y"))
    df.drop([c for c in ["n_s_x","n_s_y"] if c in df.columns], axis=1, inplace=True)

# prefer K_T from aggregates; if not present, use any existing K_T_*
if "K_T" not in df.columns:
    if "K_T_y" in df.columns: df.rename(columns={"K_T_y":"K_T"}, inplace=True)
    elif "K_T_x" in df.columns: df.rename(columns={"K_T_x":"K_T"}, inplace=True)
# drop any leftover K_T_x/y
for c in ["K_T_x","K_T_y"]:
    if c in df.columns: df.drop(columns=c, inplace=True)

# --- keep only rows with core fields ---
core_needed = ["theta", "mu", "rho", "KL"]
df = df.dropna(subset=core_needed).reset_index(drop=True)

print("✅ Combined dataset shape:", df.shape)
print("✅ Columns:", df.columns.tolist())

# ===================== 5) Exploratory plots =====================

# 5.1 Scatter: theta vs KL (colored by rho)
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="theta", y="KL", hue="rho", palette="viridis", s=60)
plt.xscale("log")
plt.xlabel(r"$\theta$ (innovation rate)")
plt.ylabel("KL divergence from Ewens")
plt.title("Scatter: θ vs KL (colored by rho)")
plt.legend(title=r"rho", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()

# 5.2 Boxplot: KL by hypothesis family
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="H_id", y="KL", color="lightblue")
plt.xlabel("Hypothesis ID (H)")
plt.ylabel("KL divergence from Ewens")
plt.title("Boxplot: KL by Hypothesis")
plt.tight_layout(); plt.show()

# 5.3 Histogram: distribution of KL
plt.figure(figsize=(7,4))
plt.hist(df["KL"], bins=20, color="steelblue", edgecolor="black", alpha=0.8)
plt.xlabel("KL divergence"); plt.ylabel("Frequency")
plt.title("Histogram of KL Deviations")
plt.tight_layout(); plt.show()

# 5.4 Correlation heatmap among numeric variables
num_cols = ["theta", "mu", "rho", "K_T", "mean_phi_t", "var_phi_t", "KL"]
corr = df[num_cols].corr(method="pearson")
plt.figure(figsize=(7,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink":0.8})
plt.title("Correlation Heatmap: Params and KL")
plt.tight_layout(); plt.show()

# 5.5 Line graph: K_T over theta per hypothesis 
plt.figure(figsize=(7,5))
sns.lineplot(data=df, x="theta", y="K_T", hue="H_id", marker="o", alpha=0.85)
plt.xscale("log")
plt.xlabel(r"$\theta$"); plt.ylabel(r"$K_T$")
plt.title("K_T vs θ by Hypothesis")
plt.legend(title="H_id", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()

# ----- choose feature columns -----
feature_candidates = ["theta", "mu", "rho", "K_T", "mean_phi_t", "var_phi_t"]
feat_cols = [c for c in feature_candidates if c in df.columns]

# ----- drop rows with missing values in features/target -----
df_rf = df.dropna(subset=feat_cols + ["KL"]).copy()

# ----- build model inputs -----
# DataFrame of features (n_samples x n_features)
X = df_rf[feat_cols] 
# 1D target array (n_samples,)
y = df_rf["KL"].to_numpy()

# ----- train/test split for later modeling -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes to verify correct model input structure
print("Feature matrix shape (X):", X_train.shape)
print("Target vector shape (y):", y_train.shape)

# Display the first few rows of the training data
print("\nFirst 5 training samples:")
print(X_train.head())

print("\nFirst 5 corresponding KL targets:")
print(y_train[:5])