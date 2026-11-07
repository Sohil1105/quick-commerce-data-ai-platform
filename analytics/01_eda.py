import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = OUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV_PATH = OUT_DIR / "titanic.csv"
CLEAN_CSV_PATH = OUT_DIR / "titanic_cleaned.csv"
EDA_LOG_PATH = OUT_DIR / "eda_report.txt"


def load_titanic_once() -> pd.DataFrame:
    """Load the Titanic dataset once via sns.load_dataset('titanic'), save titanic.csv, and fall back to CSV if offline."""
    try:
        df = sns.load_dataset("titanic")
        df.to_csv(RAW_CSV_PATH, index=False)
        return df
    except Exception:
        return pd.read_csv(RAW_CSV_PATH)


def main() -> None:
    lines = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    # --- Task 1: Load & Profile ---
    df_raw = load_titanic_once()
    log("=== TASK 1: DATASET PROFILING ===")
    log(f"Shape: {df_raw.shape}")

    buf = io.StringIO()
    df_raw.info(buf=buf)
    log("\n--- df.info() ---")
    log(buf.getvalue().strip())

    log("\n--- df.describe(include='all') ---")
    log(df_raw.describe(include="all").to_string())

    missing_pct = (df_raw.isna().mean() * 100).round(2)
    affected = missing_pct[missing_pct > 0]
    log("\n--- Missing Value Percentages (Columns with Missing Data) ---")
    for col, pct in affected.items():
        log(f"  {col}: {pct:.2f}% ({df_raw[col].isna().sum()} / {len(df_raw)} rows)")

    # --- Task 2: Missing-Value Handling per Column ---
    log("\n=== TASK 2: MISSING-VALUE HANDLING (THRESHOLD RULE) ===")
    df_clean = df_raw.copy()

    # 1. Under 5% missing -> drop rows (embarked: 0.22%, embark_town: 0.22%)
    low_missing_cols = [c for c, pct in affected.items() if pct < 5.0]
    for col in low_missing_cols:
        log(f"  {col} ({affected[col]:.2f}% < 5%): Dropping rows with missing '{col}'.")
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(subset=low_missing_cols).reset_index(drop=True)
    log(f"  Rows after dropping <5% missing columns ({low_missing_cols}): {rows_before} -> {len(df_clean)}")

    # 2. 5% - 30% missing -> impute (age: 19.87%)
    mid_missing_cols = [c for c, pct in affected.items() if 5.0 <= pct <= 30.0]
    for col in mid_missing_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        log(f"  {col} ({affected[col]:.2f}% in [5%, 30%]): Imputed with median = {median_val:.2f}.")

    # 3. Over 30% missing -> drop column or encode 'missing' (deck: 77.22%)
    high_missing_cols = [c for c, pct in affected.items() if pct > 30.0]
    for col in high_missing_cols:
        log(
            f"  {col} ({affected[col]:.2f}% > 30%): Dropped column '{col}' because {affected[col]:.2f}% of records are missing, "
            "making statistical imputation unreliable and mostly correlated with ticket class (pclass)."
        )
        df_clean = df_clean.drop(columns=[col])

    df_clean.to_csv(CLEAN_CSV_PATH, index=False)
    log(f"  Cleaned DataFrame shape: {df_clean.shape} (remaining nulls: {int(df_clean.isna().sum().sum())})")

    # --- Task 3: Univariate Analysis (Age & Fare) ---
    log("\n=== TASK 3: UNIVARIATE ANALYSIS & IQR OUTLIERS ===")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df_clean["age"], kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Age Histogram")
    sns.boxplot(x=df_clean["age"], ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Age Box Plot")

    sns.histplot(df_clean["fare"], kde=True, ax=axes[1, 0], color="darkorange")
    axes[1, 0].set_title("Fare Histogram")
    sns.boxplot(x=df_clean["fare"], ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title("Fare Box Plot")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_univariate_age_fare.png", dpi=150)
    plt.close(fig)

    for col in ["age", "fare"]:
        q1, q3 = df_clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers_clean = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()

        s_raw = df_raw[col].dropna()
        rq1, rq3 = s_raw.quantile([0.25, 0.75])
        riqr = rq3 - rq1
        outliers_raw = ((s_raw < (rq1 - 1.5 * riqr)) | (s_raw > (rq3 + 1.5 * riqr))).sum()
        log(
            f"  {col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Bounds=[{lower:.2f}, {upper:.2f}] -> "
            f"Outliers (cleaned df) = {outliers_clean} (raw non-null df = {outliers_raw})"
        )

    fare_mean = df_clean["fare"].mean()
    fare_median = df_clean["fare"].median()
    fare_mode = df_clean["fare"].mode().iloc[0]
    log(
        f"  Fare Central Tendency (cleaned): Mean = {fare_mean:.2f}, Median = {fare_median:.2f}, Mode = {fare_mode:.2f}"
    )
    log(
        f"  Fare Skewness Conclusion: Strongly RIGHT-SKEWED (positively skewed) because Mean ({fare_mean:.2f}) > "
        f"Median ({fare_median:.2f}) > Mode ({fare_mode:.2f})."
    )

    # --- Task 4: Bivariate Analysis & 6x6 Correlation Matrix ---
    log("\n=== TASK 4: BIVARIATE SURVIVAL RATES (BOOLEAN MASKING) & 6x6 CORRELATION MATRIX ===")
    log("  (a) Survival Rate by Sex:")
    for sex_val in ["female", "male"]:
        mask = df_clean["sex"] == sex_val
        log(f"      sex == '{sex_val}': {df_clean.loc[mask, 'survived'].mean() * 100:.2f}% (n={mask.sum()})")

    log("  (b) Survival Rate by Pclass:")
    for pclass_val in [1, 2, 3]:
        mask = df_clean["pclass"] == pclass_val
        log(f"      pclass == {pclass_val}: {df_clean.loc[mask, 'survived'].mean() * 100:.2f}% (n={mask.sum()})")

    log("  (c) Survival Rate by Sex & Pclass (using & boolean masking):")
    for sex_val in ["female", "male"]:
        for pclass_val in [1, 2, 3]:
            mask = (df_clean["sex"] == sex_val) & (df_clean["pclass"] == pclass_val)
            log(
                f"      (sex == '{sex_val}') & (pclass == {pclass_val}): "
                f"{df_clean.loc[mask, 'survived'].mean() * 100:.2f}% (n={mask.sum()})"
            )

    high_priority_mask = (df_clean["sex"] == "female") | (df_clean["pclass"] == 1)
    log(
        f"      (sex == 'female') | (pclass == 1) combined survival rate: "
        f"{df_clean.loc[high_priority_mask, 'survived'].mean() * 100:.2f}% (n={high_priority_mask.sum()})"
    )

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df_clean[corr_cols].corr()
    log("\n--- 6x6 Correlation Matrix (adult_male and alone excluded) ---")
    log(corr_matrix.round(4).to_string())

    # Rank off-diagonal pairs by abs(correlation) using upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
    ranked_pairs = upper.reindex(upper.abs().sort_values(ascending=False).index)
    log("\n--- Top Off-Diagonal Correlation Pairs by |r| ---")
    for (c1, c2), raw_v in ranked_pairs.head(4).items():
        log(f"  {c1} <-> {c2}: r = {raw_v:.4f} (|r| = {abs(raw_v):.4f})")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("6x6 Correlation Heatmap (Numeric Features & Survived)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Task 5: Multivariate Data Story (4 Distinct Charts) ---
    log("\n=== TASK 5: MULTIVARIATE DATA STORY (4 CHARTS GENERATED) ===")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", palette="Set2", ax=ax)
    ax.set_title("Chart 1: Survival Rate by Passenger Class and Sex")
    ax.set_ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_multivariate_bar_sex_pclass.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_clean, x="pclass", y="age", hue="survived", palette="Set1", ax=ax)
    ax.set_title("Chart 2: Age Distribution Across Passenger Class and Survival")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_multivariate_box_age_pclass_survived.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_clean, x="age", y="fare", hue="survived", style="sex", alpha=0.75, palette="viridis", ax=ax
    )
    ax.set_title("Chart 3: Age vs. Fare by Survival Status and Sex")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_multivariate_scatter_age_fare.png", dpi=150)
    plt.close(fig)

    df_story = df_clean.assign(family_size=(df_clean["sibsp"] + df_clean["parch"]).clip(upper=4))
    pivot_surv = df_story.pivot_table(index="family_size", columns="pclass", values="survived", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot_surv, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Chart 4: Survival Rate by Family Size (0 to 4+) and Pclass")
    ax.set_ylabel("Family Size (sibsp + parch, capped at 4)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_multivariate_heatmap_family_pclass.png", dpi=150)
    plt.close(fig)
    log("  Saved 4 multivariate story charts to analytics/charts/.")

    # --- Task 6: Exploratory Z-Score Standardization Check ---
    log("\n=== TASK 6: EXPLORATORY Z-SCORE STANDARDIZATION CHECK (AGE & FARE) ===")
    z_df = pd.DataFrame(
        StandardScaler().fit_transform(df_clean[["age", "fare"]]),
        columns=["age_z", "fare_z"],
    )

    for orig_col, z_col in [("age", "age_z"), ("fare", "fare_z")]:
        b_mean, b_std = df_clean[orig_col].mean(), df_clean[orig_col].std(ddof=0)
        a_mean, a_std = z_df[z_col].mean(), z_df[z_col].std(ddof=0)
        log(
            f"  {orig_col} -> Before: mean={b_mean:.4f}, std={b_std:.4f} | "
            f"After ({z_col}): mean={a_mean:.6f}, std={a_std:.6f}"
        )
        assert abs(a_mean) < 1e-12 and abs(a_std - 1.0) < 1e-12

    EDA_LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved complete EDA report to {EDA_LOG_PATH}")


if __name__ == "__main__":
    main()
