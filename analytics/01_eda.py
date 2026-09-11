"""
Zepto Capstone - Module 2, Part A: Profiling, Cleaning, EDA on Titanic dataset.
Run: python 01_eda.py
Produces: titanic.csv (offline fallback), several .png charts in ./charts/
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("charts", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD (exactly once) + PROFILE
# ---------------------------------------------------------------------------
df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)  # offline fallback for grading
print("Loaded titanic dataset. Shape:", df.shape)

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() ---")
print(df.describe(include="all"))

print("\n--- Missing value % per column ---")
missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print(missing_pct)

# ---------------------------------------------------------------------------
# 2. MISSING VALUE HANDLING (threshold rule)
# ---------------------------------------------------------------------------
# Measured percentages (printed above) drive the decisions below.
# age        ~19.87%  -> 5-30% band -> impute (median)
# embarked   ~0.22%   -> <5% band  -> drop rows
# embark_town ~0.22%  -> <5% band -> drop rows
# deck       ~77.2%   -> too high for reliable imputation -> encode "missing" as own category

df_clean = df.copy()

# age: 5-30% missing -> impute with median
age_missing_pct = df["age"].isna().mean() * 100
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
print(f"\nage missing = {age_missing_pct:.2f}% -> median-imputed "
      f"(median={df['age'].median()})")

# embarked / embark_town: <5% missing -> drop those rows
embarked_missing_pct = df["embarked"].isna().mean() * 100
df_clean = df_clean.dropna(subset=["embarked", "embark_town"])
print(f"embarked missing = {embarked_missing_pct:.2f}% -> rows dropped")

# deck: too high missing (~77%) -> encode "missing" as its own category
deck_missing_pct = df["deck"].isna().mean() * 100
df_clean["deck"] = df_clean["deck"].astype("object").fillna("Missing")
print(f"deck missing = {deck_missing_pct:.2f}% -> too high to impute reliably; "
      f"encoded as its own 'Missing' category")

print("\nShape after cleaning:", df_clean.shape)

# ---------------------------------------------------------------------------
# 3. UNIVARIATE ANALYSIS: age & fare
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df_clean["age"], kde=True, ax=axes[0, 0]).set_title("Age Histogram")
sns.boxplot(x=df_clean["age"], ax=axes[0, 1]).set_title("Age Boxplot")
sns.histplot(df_clean["fare"], kde=True, ax=axes[1, 0]).set_title("Fare Histogram")
sns.boxplot(x=df_clean["fare"], ax=axes[1, 1]).set_title("Fare Boxplot")
plt.tight_layout()
plt.savefig("charts/univariate_age_fare.png")
plt.close()


def iqr_outlier_count(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum(), lower, upper


age_outliers, age_lo, age_hi = iqr_outlier_count(df_clean["age"])
fare_outliers, fare_lo, fare_hi = iqr_outlier_count(df_clean["fare"])
print(f"\nAge outliers (IQR rule): {age_outliers} (bounds: [{age_lo:.2f}, {age_hi:.2f}])")
print(f"Fare outliers (IQR rule): {fare_outliers} (bounds: [{fare_lo:.2f}, {fare_hi:.2f}])")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
print(f"\nFare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
skew_note = (
    "RIGHT-SKEWED (mean > median > mode), which is expected because a small number "
    "of first-class passengers paid very high fares, pulling the mean upward while "
    "most passengers clustered at lower fares."
    if fare_mean > fare_median > fare_mode
    else "See printed mean/median/mode above to judge skew direction."
)
print("Skewness interpretation:", skew_note)

# ---------------------------------------------------------------------------
# 4. BIVARIATE ANALYSIS
# ---------------------------------------------------------------------------
survival_by_sex = df_clean.groupby("sex")["survived"].mean()
survival_by_pclass = df_clean.groupby("pclass")["survived"].mean()
survival_by_sex_pclass = df_clean.groupby(["sex", "pclass"])["survived"].mean()

print("\nSurvival rate by sex:\n", survival_by_sex)
print("\nSurvival rate by pclass:\n", survival_by_pclass)
print("\nSurvival rate by sex & pclass:\n", survival_by_sex_pclass)

# boolean-masking version (explicit &/| as required)
female_survival = df_clean.loc[df_clean["sex"] == "female", "survived"].mean()
male_survival = df_clean.loc[df_clean["sex"] == "male", "survived"].mean()
female_class1 = df_clean.loc[(df_clean["sex"] == "female") & (df_clean["pclass"] == 1), "survived"].mean()
print(f"\n[boolean mask] female survival = {female_survival:.3f}, male survival = {male_survival:.3f}")
print(f"[boolean mask] female & pclass==1 survival = {female_class1:.3f}")

# correlation matrix on exactly these 6 columns
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_clean[corr_cols].corr()
plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (6 numeric columns)")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png")
plt.close()

# find top-2 strongest off-diagonal correlations
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ["feat1", "feat2", "corr"]
corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
top2 = corr_pairs.sort_values("abs_corr", ascending=False).head(2)
print("\nTop 2 strongest correlations:\n", top2)

# ---------------------------------------------------------------------------
# 5. MULTIVARIATE "DATA STORY" (4+ charts)
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Class and Sex")
plt.tight_layout()
plt.savefig("charts/story_1_class_sex.png")
plt.close()
print("\nChart 1 interpretation: Female passengers had a much higher survival rate "
      "than male passengers across all classes, and 1st class females survived at "
      "the highest rate overall. This reflects the 'women and children first' "
      "evacuation protocol combined with class-based cabin proximity to lifeboats.")

plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("charts/story_2_age_survival.png")
plt.close()
print("Chart 2 interpretation: Survivors show a slightly younger median age than "
      "non-survivors, with a visible cluster of young children among survivors, "
      "consistent with children being prioritized during evacuation.")

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_clean, x="fare", y="age", hue="survived", alpha=0.6)
plt.title("Fare vs Age, colored by Survival")
plt.tight_layout()
plt.savefig("charts/story_3_fare_age_scatter.png")
plt.close()
print("Chart 3 interpretation: Passengers who paid higher fares (likely 1st class) "
      "show a higher density of survival markers, reinforcing that socio-economic "
      "class was linked to survival odds, independent of age.")

plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x="embarked", hue="survived")
plt.title("Survival Count by Embarkation Port")
plt.tight_layout()
plt.savefig("charts/story_4_embarked_survival.png")
plt.close()
print("Chart 4 interpretation: Passengers embarking from Cherbourg (C) show a "
      "relatively higher survival proportion than those from Southampton (S), "
      "which correlates with a higher share of 1st class passengers boarding at C.")

# ---------------------------------------------------------------------------
# 6. EXPLORATORY Z-SCORE STANDARDIZATION CHECK (does NOT feed into modeling)
# ---------------------------------------------------------------------------
before_stats = df_clean[["age", "fare"]].agg(["mean", "std"])
print("\nBefore standardization:\n", before_stats)

z_age = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
z_fare = (df_clean["fare"] - df_clean["fare"].mean()) / df_clean["fare"].std()
after_stats = pd.DataFrame({"age_z": z_age, "fare_z": z_fare}).agg(["mean", "std"])
print("\nAfter standardization (z-score, EDA sanity check only):\n", after_stats)

print("\n01_eda.py complete. titanic.csv and charts/ saved.")
