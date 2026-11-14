"""
================================================================================
ECONOMETRIC & MICRO-TARGETING DECISION ENGINE: GRASSROOTS ELECTORAL ANALYTICS
Dataset: gram_panchayat_5012_voters_cleaned_v2.csv (N = 5,012 | Households = 1,232)
Models:
  1. Descriptive Statistics, Whipple Age-Heaping & Household Clustering Audit
  2. Ordinary Least Squares (OLS) / Linear Probability Model (Standard & Clustered SE)
  3. Multicollinearity Diagnostics (Variance Inflation Factors & Tolerance)
  4. Binary Logistic Regression (Logit MLE), Odds Ratios & Average Marginal Effects
  5. Probit Regression (MLE) & Average Marginal Effects (Specification Robustness)
  6. Predictive Classification Diagnostics (Confusion Matrix, ROC-AUC, Sensitivity/Specificity)
  7. Probability-Based Electorate Segmentation & Ward-Level Strategic Targeting
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# ------------------------------------------------------------------------------
# 1. DATA INGESTION & STRUCTURAL INTEGRITY AUDIT
# ------------------------------------------------------------------------------
DATA_PATH = "data/gram_panchayat_5012_voters_cleaned_v2.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("1. DATASET OVERVIEW & STRUCTURAL AUDIT")
print("=" * 80)
print(f"Total Observations (Voters) : {len(df):,}")
print(f"Unique Households           : {df['household_id'].nunique():,}")
print(f"Administrative Wards        : {df['ward_no'].nunique()}")
print(f"Missing Values              : {df.isnull().sum().sum()}")

# Incumbent baseline support
inc_votes = df["vote_incumbent"].sum()
opp_votes = len(df) - inc_votes
inc_share = df["vote_incumbent"].mean() * 100
print(f"Incumbent Support           : {inc_votes:,} ({inc_share:.2f}%)")
print(f"Opposition Support          : {opp_votes:,} ({100 - inc_share:.2f}%)")

# Whipple's Index for Age Heaping (ages 23 to 62 ending in 0 or 5)
age_sub = df[(df["age"] >= 23) & (df["age"] <= 62)]["age"]
whipple_idx = (age_sub.apply(lambda x: x % 5 == 0).sum() / (0.2 * len(age_sub))) * 100
print(f"Whipple Age-Heaping Index   : {whipple_idx:.1f} (100 = uniform; >150 = rural self-report clustering)")

# Household structure & split-voting analysis
hh_sizes = df.groupby("household_id").size()
single_hh = (hh_sizes == 1).sum()
multi_hh = (hh_sizes >= 2).sum()
split_flag = df.groupby("household_id")["is_split_household"].first()
split_multi = ((hh_sizes >= 2) & (split_flag == 1)).sum()
unanimous_multi = ((hh_sizes >= 2) & (split_flag == 0)).sum()

female_heads = len(df[(df["is_head"] == 1) & (df["female"] == 1)])
male_heads = len(df[(df["is_head"] == 1) & (df["female"] == 0)])

print(f"Household Composition       : {single_hh} Single-Voter | {multi_hh:,} Multi-Member")
print(f"Household Leadership        : {male_heads:,} Male Heads | {female_heads} Female Heads")
print(f"Unanimous Multi-Member HH   : {unanimous_multi:,} ({unanimous_multi / multi_hh * 100:.1f}%)")
print(f"Split-Voting Multi-Member HH: {split_multi:,} ({split_multi / multi_hh * 100:.1f}%)")

# ------------------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS & MULTICOLLINEARITY (VIF / TOLERANCE)
# ------------------------------------------------------------------------------
y_var = "vote_incumbent"
x_vars = [
    "age",
    "female",
    "land_acres",
    "caste_match",
    "dbt_cash_beneficiary",
    "housing_beneficiary",
    "mgnrega_distress",
    "water_drainage_eval",
    "roads_lights_eval",
    "pradhan_approachable",
    "anti_incumbency"
]

desc_stats = df[[y_var] + x_vars].agg(["count", "mean", "std", "min", "median", "max"]).T
desc_stats.columns = ["N", "Mean", "Std_Dev", "Min", "Median", "Max"]

# Compute VIF and Tolerance
X_const = sm.add_constant(df[x_vars])
vif_records = []
for idx, col in enumerate(x_vars):
    vif_val = variance_inflation_factor(X_const.values, idx + 1)
    vif_records.append({
        "Variable": col,
        "Tolerance": 1.0 / vif_val,
        "VIF": vif_val
    })
vif_df = pd.DataFrame(vif_records).set_index("Variable")

print("\n" + "=" * 80)
print("2. DESCRIPTIVE STATISTICS & COLLINEARITY DIAGNOSTICS")
print("=" * 80)
print(desc_stats.round(4).to_string())
print("\nMulticollinearity Diagnostics (VIF < 2.0 confirms orthogonality):")
print(vif_df.round(4).to_string())

desc_stats.to_csv(os.path.join(OUTPUT_DIR, "table1_descriptive_stats.csv"))
vif_df.to_csv(os.path.join(OUTPUT_DIR, "table2_vif_diagnostics.csv"))

# ------------------------------------------------------------------------------
# 3. LINEAR PROBABILITY MODEL (OLS: STANDARD vs. HOUSEHOLD-CLUSTERED SE)
# ------------------------------------------------------------------------------
formula = f"{y_var} ~ " + " + ".join(x_vars)

ols_standard = smf.ols(formula, data=df).fit()
ols_clustered = smf.ols(formula, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["household_id"]}
)

# Standardized Beta weights
std_y = df[y_var].std()
std_betas = {v: ols_standard.params[v] * (df[v].std() / std_y) for v in x_vars}

# Check out-of-bounds predictions [0, 1]
ols_preds = ols_standard.predict(df)
below_zero = (ols_preds < 0).sum()
above_one = (ols_preds > 1).sum()

print("\n" + "=" * 80)
print("3. LINEAR PROBABILITY MODEL (OLS) & INTRA-HOUSEHOLD CLUSTERING")
print("=" * 80)
print(f"R-squared                   : {ols_standard.rsquared:.4f}")
print(f"Adjusted R-squared          : {ols_standard.rsquared_adj:.4f}")
print(f"F-statistic (Standard)      : {ols_standard.fvalue:.3f} (p = {ols_standard.f_pvalue:.4e})")
print(f"F-statistic (Clustered)     : {ols_clustered.fvalue:.3f} (p = {ols_clustered.f_pvalue:.4e})")
print(f"Out-of-Bounds Predictions   : {below_zero} below 0.0 ({below_zero/len(df)*100:.2f}%), {above_one} above 1.0")

# ------------------------------------------------------------------------------
# 4. BINARY LOGISTIC REGRESSION (LOGIT MLE) & ODDS RATIOS
# ------------------------------------------------------------------------------
logit_model = smf.logit(formula, data=df).fit(disp=False)
logit_ame = logit_model.get_margeff(at="overall")

null_logit = smf.logit(f"{y_var} ~ 1", data=df).fit(disp=False)
ll_ratio = 2 * (logit_model.llf - null_logit.llf)
mcfadden_r2 = 1 - (logit_model.llf / null_logit.llf)
cox_snell_r2 = 1 - np.exp((2 / len(df)) * (null_logit.llf - logit_model.llf))
nagelkerke_r2 = cox_snell_r2 / (1 - np.exp((2 / len(df)) * null_logit.llf))

print("\n" + "=" * 80)
print("4. BINARY LOGISTIC REGRESSION (MLE) GOODNESS-OF-FIT")
print("=" * 80)
print(f"-2 Log Likelihood           : {-2 * logit_model.llf:.2f}")
print(f"Omnibus LR Chi-Square       : {ll_ratio:.2f} (df = {int(logit_model.df_model)}, p = {logit_model.llr_pvalue:.4e})")
print(f"McFadden Pseudo R-squared   : {mcfadden_r2:.4f}")
print(f"Cox & Snell Pseudo R-squared: {cox_snell_r2:.4f}")
print(f"Nagelkerke Pseudo R-squared : {nagelkerke_r2:.4f}")

# ------------------------------------------------------------------------------
# 5. PROBIT REGRESSION (MLE) & TRI-MODEL SPECIFICATION COMPARISON
# ------------------------------------------------------------------------------
probit_model = smf.probit(formula, data=df).fit(disp=False)
probit_ame = probit_model.get_margeff(at="overall")

logit_ci = np.exp(logit_model.conf_int())

comparison_df = pd.DataFrame({
    "OLS_B": ols_standard.params.drop("Intercept"),
    "OLS_Std_SE": ols_standard.bse.drop("Intercept"),
    "OLS_Cluster_SE": ols_clustered.bse.drop("Intercept"),
    "OLS_Cluster_p": ols_clustered.pvalues.drop("Intercept"),
    "Std_Beta": [std_betas[v] for v in x_vars],
    "Logit_B": logit_model.params.drop("Intercept"),
    "Logit_SE": logit_model.bse.drop("Intercept"),
    "Logit_OddsRatio": np.exp(logit_model.params.drop("Intercept")),
    "OR_95CI_Low": logit_ci.loc[x_vars, 0],
    "OR_95CI_High": logit_ci.loc[x_vars, 1],
    "Logit_AME": logit_ame.margeff,
    "Probit_B": probit_model.params.drop("Intercept"),
    "Probit_AME": probit_ame.margeff,
    "VIF": vif_df["VIF"]
})

print("\n" + "=" * 80)
print("5. TRI-MODEL ECONOMETRIC COMPARISON TABLE (OLS vs. LOGIT vs. PROBIT)")
print("=" * 80)
print(comparison_df.round(4).to_string())
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "table3_trimodel_comparison.csv"))

# ------------------------------------------------------------------------------
# 6. CLASSIFICATION & DISCRIMINATION DIAGNOSTICS (ROC-AUC & CONFUSION MATRIX)
# ------------------------------------------------------------------------------
df["logit_prob"] = logit_model.predict(df)
df["probit_prob"] = probit_model.predict(df)
df["pred_vote_50"] = (df["logit_prob"] >= 0.50).astype(int)

cm = confusion_matrix(df[y_var], df["pred_vote_50"])
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / len(df) * 100
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
roc_auc = roc_auc_score(df[y_var], df["logit_prob"])
brier = brier_score_loss(df[y_var], df["logit_prob"])

print("\n" + "=" * 80)
print("6. CLASSIFICATION & PREDICTIVE ACCURACY (CUTOFF = 0.50)")
print("=" * 80)
print(f"Confusion Matrix            : TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
print(f"Overall Accuracy            : {accuracy:.2f}%")
print(f"Sensitivity (Recall)        : {sensitivity:.2f}%")
print(f"Specificity                 : {specificity:.2f}%")
print(f"ROC-AUC Score               : {roc_auc:.4f}")
print(f"Brier Calibration Score     : {brier:.4f}")

# ------------------------------------------------------------------------------
# 7. PROBABILITY SEGMENTATION, GRIEVANCE AUDIT & WARD TARGETING
# ------------------------------------------------------------------------------
def classify_segment(p):
    if p >= 0.65:
        return "Core Incumbent Base"
    elif p < 0.35:
        return "Core Opposition Base"
    else:
        return "Swing / Persuadable"

df["computed_segment"] = df["logit_prob"].apply(classify_segment)

seg_summary = df.groupby("computed_segment").agg(
    Voters=("voter_id", "count"),
    Share_Pct=("voter_id", lambda x: len(x) / len(df) * 100),
    Mean_Pred_Prob=("logit_prob", "mean"),
    Actual_Incumbent_Vote=("vote_incumbent", lambda x: x.mean() * 100),
    Split_HH_Share=("is_split_household", lambda x: x.mean() * 100)
)

ward_summary = df.groupby("ward_no").agg(
    Total_Voters=("voter_id", "count"),
    Households=("household_id", "nunique"),
    Incumbent_Share_Pct=("vote_incumbent", lambda x: x.mean() * 100),
    Swing_Voters=("computed_segment", lambda x: (x == "Swing / Persuadable").sum()),
    MGNREGA_Distress_Pct=("mgnrega_distress", lambda x: x.mean() * 100)
)

print("\n" + "=" * 80)
print("7. ELECTORATE SEGMENTATION & WARD-LEVEL STRATEGIC MATRIX")
print("=" * 80)
print(seg_summary.round(2).to_string())
print("\nWard-Level Priority Matrix:")
print(ward_summary.round(2).to_string())

seg_summary.to_csv(os.path.join(OUTPUT_DIR, "table4_voter_segments.csv"))
ward_summary.to_csv(os.path.join(OUTPUT_DIR, "table5_ward_targeting_matrix.csv"))

# ------------------------------------------------------------------------------
# 8. PUBLICATION-GRADE VISUALIZATIONS
# ------------------------------------------------------------------------------
# Figure 1: Ward Vote Share
fig, ax = plt.subplots(figsize=(8.5, 4.5))
w_colors = ["#d9534f" if v < 40 else ("#f0ad4e" if v < 50 else "#5cb85c") for v in ward_summary["Incumbent_Share_Pct"]]
bars = ax.bar(ward_summary.index.astype(str), ward_summary["Incumbent_Share_Pct"], color=w_colors, edgecolor="black", alpha=0.88)
ax.axhline(50, color="navy", linestyle="--", linewidth=1.2, label="50% Victory Threshold")
ax.set_title("Incumbent Vote Share Across 10 Administrative Wards (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Ward Number", fontsize=11)
ax.set_ylabel("Incumbent Support (%)", fontsize=11)
ax.set_ylim(0, 65)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_ward_vote_share.png"), dpi=300)
plt.close()

# Figure 2: Average Marginal Effects Forest Plot (Logit vs Probit)
fig, ax = plt.subplots(figsize=(9, 5.2))
ame_sorted = comparison_df.sort_values("Logit_AME")
y_pos = np.arange(len(ame_sorted))
ax.barh(y_pos - 0.17, ame_sorted["Logit_AME"] * 100, height=0.34, label="Logit AME (pp)", color="#2b5c8f", edgecolor="black")
ax.barh(y_pos + 0.17, ame_sorted["Probit_AME"] * 100, height=0.34, label="Probit AME (pp)", color="#e68a00", edgecolor="black")
ax.axvline(0, color="black", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(ame_sorted.index, fontsize=10)
ax.set_xlabel("Average Marginal Effect on Vote Probability (Percentage Points)", fontsize=10.5)
ax.set_title("Specification Robustness: Logit vs. Probit Average Marginal Effects", fontsize=12.5, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_marginal_effects_comparison.png"), dpi=300)
plt.close()

# Figure 3: ROC Discrimination Curve
fpr, tpr, _ = roc_curve(df[y_var], df["logit_prob"])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(fpr, tpr, color="#2b5c8f", lw=2.2, label=f"11-Variable Logit (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Baseline (AUC = 0.500)")
ax.set_title("ROC Curve: Electoral Choice Discrimination", fontsize=12.5, fontweight="bold")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_roc_curve.png"), dpi=300)
plt.close()

# Figure 4: Predicted Probability Distribution & Segmentation Thresholds
fig, ax = plt.subplots(figsize=(8.5, 4.5))
sns.histplot(df["logit_prob"], bins=35, kde=True, color="#337ab7", edgecolor="black", alpha=0.65, ax=ax)
ax.axvline(0.35, color="#d9534f", linestyle="--", linewidth=1.8, label="Opposition Cutoff (P = 0.35)")
ax.axvline(0.65, color="#5cb85c", linestyle="--", linewidth=1.8, label="Core Base Cutoff (P = 0.65)")
ax.set_title("Predicted Vote Probability Distribution & Swing Zone (0.35 <= P < 0.65)", fontsize=12.5, fontweight="bold")
ax.set_xlabel("Predicted Probability of Voting Incumbent", fontsize=10.5)
ax.set_ylabel("Voter Count", fontsize=10.5)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_probability_segmentation.png"), dpi=300)
plt.close()

print("\nAll tables and high-resolution figures saved to:", os.path.abspath(OUTPUT_DIR))
