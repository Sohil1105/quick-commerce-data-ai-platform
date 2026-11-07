# Module 2: Analytics & Predictive Modeling Pipeline (`/analytics`)

## Overview
End-to-end exploratory data analysis (EDA), statistical profiling, classification, and regression pipeline built on the Titanic dataset (`sns.load_dataset('titanic')`).
The dataset is loaded **once** in `01_eda.py` and immediately saved to `analytics/titanic.csv` as an offline fallback; `02_modeling.py` reads directly from `analytics/titanic.csv` without ever calling `sns.load_dataset('titanic')` a second time.

## Setup & Execution
From the repository root:
```bash
pip install -r requirements.txt
python analytics/01_eda.py
python analytics/02_modeling.py
```

---

## Part A — Profiling, Cleaning, and the Data Story

### 1. Dataset Profile & Missing-Value Percentages
- **Raw Shape**: `891 rows × 15 columns`
- **Columns with Missing Values**:
  - `age`: **19.87%** (`177 / 891` rows)
  - `embarked`: **0.22%** (`2 / 891` rows)
  - `embark_town`: **0.22%** (`2 / 891` rows)
  - `deck`: **77.22%** (`688 / 891` rows)

### 2. Missing-Value Handling Strategy (Threshold Rule)
Following the required percentage threshold rule:
- **`embarked` (`0.22%` < 5%) & `embark_town` (`0.22%` < 5%)**: **Dropped rows** (`2` rows removed, leaving `889` rows) because the missing rate is well below the 5% threshold.
- **`age` (`19.87%`, between 5% and 30%)**: **Imputed with median (`28.00` years)** because `age` falls in the 5%–30% band and median imputation is robust to age outliers.
- **`deck` (`77.22%` > 30%)**: **Dropped the column** because over three-quarters of all records are missing (`77.22%`), making statistical imputation unreliable and redundant given that cabin deck is largely captured by `pclass` and `fare`.

### 3. Univariate Analysis: IQR Outliers & Fare Skewness
- **Saved Plot**: `analytics/charts/01_univariate_age_fare.png`
- **`age` IQR Outliers**:
  - Cleaned dataset (`n = 889`, post-median imputation): $Q_1 = 22.00$, $Q_3 = 35.00$, $\text{IQR} = 13.00$, bounds $[2.50, 54.50]$ $\rightarrow$ **65 outliers** (on raw non-null `age`, $Q_1 = 20.125, Q_3 = 38.00, \text{IQR} = 17.875$ $\rightarrow$ **11 outliers**).
- **`fare` IQR Outliers**:
  - Cleaned dataset (`n = 889`): $Q_1 = 7.90$, $Q_3 = 31.00$, $\text{IQR} = 23.10$, bounds $[-26.76, 65.66]$ $\rightarrow$ **114 outliers** (on raw `fare`, $n = 891$ $\rightarrow$ **116 outliers**).
- **`fare` Central Tendency & Skewness Conclusion**:
  - **Mean** = `32.10` (`32.20` raw), **Median** = `14.45`, **Mode** = `8.05`.
  - Because **$\text{Mean } (32.10) > \text{Median } (14.45) > \text{Mode } (8.05)$**, the `fare` distribution is **strongly right-skewed (positively skewed)**, pulled upward by a small number of luxury First-Class suite tickets (up to `512.33`).

### 4. Bivariate Analysis (Boolean Masking) & 6×6 Correlation Matrix
- **Survival Rate Breakdowns**:
  - **(a) By `sex`**:
    - `sex == 'female'`: **74.04%** (`n = 312`)
    - `sex == 'male'`: **18.89%** (`n = 577`)
  - **(b) By `pclass`**:
    - `pclass == 1`: **62.62%** (`n = 214`)
    - `pclass == 2`: **47.28%** (`n = 184`)
    - `pclass == 3`: **24.24%** (`n = 491`)
  - **(c) By `sex` and `pclass` together (`&` boolean mask)**:
    - `(sex == 'female') & (pclass == 1)`: **96.74%** (`n = 92`)
    - `(sex == 'female') & (pclass == 2)`: **92.11%** (`n = 76`)
    - `(sex == 'female') & (pclass == 3)`: **50.00%** (`n = 144`)
    - `(sex == 'male') & (pclass == 1)`: **36.89%** (`n = 122`)
    - `(sex == 'male') & (pclass == 2)`: **15.74%** (`n = 108`)
    - `(sex == 'male') & (pclass == 3)`: **13.54%** (`n = 347`)
    - `(sex == 'female') | (pclass == 1)` (`|` boolean mask): **63.59%** (`n = 434`)

- **6×6 Correlation Matrix (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`)** (`adult_male` and `alone` excluded as redundant derived flags; heatmap saved to `analytics/charts/02_correlation_heatmap.png`):

| Feature | `survived` | `pclass` | `age` | `sibsp` | `parch` | `fare` |
|---|---:|---:|---:|---:|---:|---:|
| **`survived`** | 1.0000 | -0.3355 | -0.0698 | -0.0340 | 0.0832 | 0.2553 |
| **`pclass`** | -0.3355 | 1.0000 | -0.3365 | 0.0817 | 0.0168 | -0.5482 |
| **`age`** | -0.0698 | -0.3365 | 1.0000 | -0.2325 | -0.1715 | 0.0937 |
| **`sibsp`** | -0.0340 | 0.0817 | -0.2325 | 1.0000 | 0.4145 | 0.1609 |
| **`parch`** | 0.0832 | 0.0168 | -0.1715 | 0.4145 | 1.0000 | 0.2175 |
| **`fare`** | 0.2553 | -0.5482 | 0.0937 | 0.1609 | 0.2175 | 1.0000 |

- **Two Strongest Off-Diagonal Correlations (by $|r|$)**:
  1. **`pclass` and `fare` ($r = -0.5482$, $|r| = 0.5482$)**: Strong negative correlation because `pclass` is coded ordinally from `1` (First Class, highest ticket prices) to `3` (Third Class, lowest ticket prices). Higher-tier cabins near the boat deck commanded much higher fares.
  2. **`sibsp` and `parch` ($r = 0.4145$, $|r| = 0.4145$)**: Moderate-to-strong positive correlation because passengers traveling with spouses or siblings (`sibsp`) frequently traveled as part of larger family units that also included parents or children (`parch`).

### 5. Multivariate "Data Story" (4 Charts & Written Interpretations)

1. **Chart 1 — Bar Plot (`analytics/charts/03_multivariate_bar_sex_pclass.png`): Survival Rate by Passenger Class and Sex**
   Female passengers achieved dramatically higher survival rates than male passengers across every cabin class, reflecting the strict *"women and children first"* evacuation protocol. However, socioeconomic status heavily modulated access to lifeboats: nearly all First-Class (`96.74%`) and Second-Class (`92.11%`) women survived, whereas Third-Class female survival dropped to `50.00%`. Among men, First-Class men survived at more than double the rate (`36.89%`) of Second-Class (`15.74%`) and Third-Class (`13.54%`) men.

2. **Chart 2 — Box Plot (`analytics/charts/04_multivariate_box_age_pclass_survived.png`): Age Distribution Across Passenger Class and Survival**
   First-Class passengers were significantly older on average than Second- and Third-Class passengers, explaining the negative correlation between `age` and `pclass` ($r = -0.3365$). Within every passenger class, the median age of survivors (`survived = 1`) is lower than that of non-survivors (`survived = 0`), confirming that younger passengers and children were prioritized during lifeboat loading regardless of ticket tier.

3. **Chart 3 — Scatter Plot (`analytics/charts/05_multivariate_scatter_age_fare.png`): Age vs. Fare Colored by Survival and Styled by Sex**
   Passengers paying premium fares above `GBP 65` (the upper IQR threshold) cluster heavily in the survivor group (`survived = 1`) and are predominantly female or First-Class travelers across a wide range of ages. Conversely, the dense cluster at the bottom of the plot (`fare < GBP 15`, ages `18–35`) consists overwhelmingly of Third-Class male passengers who experienced the highest mortality rate.

4. **Chart 4 — Heatmap (`analytics/charts/06_multivariate_heatmap_family_pclass.png`): Survival Rate by Family Size (`sibsp + parch`) and Passenger Class**
   Passengers traveling in small-to-medium family groups (`1` to `3` relatives) had higher survival rates across all three classes than either solo travelers (`family_size = 0`) or large families (`family_size >= 4`). Solo Third-Class passengers were predominantly young single men bunked far from the boat deck, while large Third-Class families struggled to locate all family members and navigate crowded corridors together before lifeboats launched.

### 6. Exploratory Z-Score Standardization Check (`age` & `fare`)
Applying $z = (x - \mu)/\sigma$ on the full cleaned dataset (`n = 889`):
- **`age`**: Before $\rightarrow \mu = 29.3152, \sigma = 12.9776$ | After (`age_z`) $\rightarrow \mu = 0.000000, \sigma = 1.000000$.
- **`fare`**: Before $\rightarrow \mu = 32.0967, \sigma = 49.6695$ | After (`fare_z`) $\rightarrow \mu = 0.000000, \sigma = 1.000000$.

---

## Part B — Predictive Modeling Pipeline

### 7. Stratified Train/Test Split & Justification
- **Class Balance**: `549` Not Survived (`61.62%`) vs. `342` Survived (`38.38%`).
- **Justification for Stratification**: Because the target class is moderately imbalanced (`61.62%` vs. `38.38%`) on a relatively small dataset (`891` rows), an unstratified random split risks skewing the positive-class ratio between train and test sets, which would distort threshold-sensitive metrics like Precision, Recall, and F1. Using `stratify=y` preserves the exact survival ratio in both splits (`Train: 38.34%`, `Test: 38.55%`).

### 8. Leakage-Free Preprocessing (`ColumnTransformer` + `Pipeline`)
All preprocessing is encapsulated inside a scikit-learn `ColumnTransformer` fit **exclusively on `X_train`** and applied in `transform`-only mode to `X_test`:
- **Numeric features (`pclass`, `age`, `sibsp`, `parch`, `fare`)**: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical features (`sex`, `embarked`)**: `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.

### 9 & 10. Three Classifiers Evaluation & Decision Tree Visualization
- **ROC Curve Plot**: `analytics/charts/07_roc_curves.png`
- **Decision Tree Visualization (`plot_tree` with labeled features/classes)**: `analytics/charts/08_decision_tree_plot.png`

| Classifier | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---|
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 | **0.8437** | `[[98, 12], [23, 46]]` |
| **Decision Tree (`max_depth=4`)** | 0.7933 | **0.8636** | 0.5507 | 0.6726 | 0.8292 | `[[104, 6], [31, 38]]` |
| **Random Forest (`n_estimators=100`)** | **0.8156** | 0.8000 | **0.6957** | **0.7442** | 0.8287 | `[[98, 12], [21, 48]]` |

### 11. Imbalance Handling Comparison (Logistic Regression)
SMOTE is applied **strictly inside `imblearn.pipeline.Pipeline` on the training fold only**, preventing synthetic oversampling leakage into the test fold:

| Imbalance Strategy | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **(a) Baseline (No Handling)** | 0.8045 | **0.7931** | 0.6667 | 0.7244 |
| **(b) `class_weight='balanced'`** | 0.8045 | 0.7297 | **0.7826** | 0.7552 |
| **(c) `SMOTE` (Training Fold Only)** | **0.8101** | 0.7397 | **0.7826** | **0.7606** |

**Written Conclusion on Imbalance Strategy**:
**SMOTE oversampling on the training fold** worked best overall, achieving the highest F1 score (**`0.7606`**) and highest Accuracy (**`0.8101`**) while tying `class_weight='balanced'` for the highest Recall (**`0.7826`**, up from `0.6667` in the baseline). Both cost-sensitive weighting and SMOTE substantially reduce false negatives on the minority `survived = 1` class, but interpolating synthetic minority examples in feature space (`SMOTE`) preserved slightly better precision (`0.7397` vs. `0.7297`) than uniform inverse-frequency weighting.

### 12. Hyperparameter Tuning (`GridSearchCV`) & Out-of-Bag (`OOB`) Score
Constructed with `RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)` and tuned via 5-fold `GridSearchCV` over `n_estimators` (`[100, 200]`), `max_depth` (`[4, 6, 8, None]`), and `max_features` (`['sqrt', 'log2']`):
- **Best Hyperparameters**: `{'classifier__max_depth': 8, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 100}`
- **Best 5-Fold CV F1 Score**: **`0.7417`**
- **Corresponding Out-of-Bag (`oob_score_`) Score**: **`0.8174`**
- **Test-Set Metrics**: Accuracy = `0.8045`, Precision = `0.8036`, Recall = `0.6522`, F1 = `0.7200`, ROC-AUC = `0.8398`.

### 13. Regression Side-Task: Predicting `fare` with Multivariate Linear Regression
- **Residual Plot**: `analytics/charts/09_regression_residuals.png`
- **Test Metrics**:
  - **MAE**: `20.8977`
  - **RMSE**: `30.5328`
  - **$R^2$**: `0.3975`
  - **Adjusted $R^2$**: `0.3617`
- **Heteroscedasticity Conclusion**:
  The residual plot exhibits clear **heteroscedasticity** (a non-random, cone/funnel-shaped spread of residuals whose variance grows wider as predicted `fare` increases). Because ticket prices are heavily right-skewed with extreme First-Class luxury suite outliers (`512.33`), linear regression errors are small and tightly clustered for Third-Class fares (`< 15`) but span hundreds of pounds for First-Class predictions.

### 14. Final Model Comparison Table & Deployment Recommendation

*(Classification and Regression metrics are reported in separate column groups because they operate on fundamentally different scales and targets.)*

| Model | Task / Target | Clf Accuracy | Clf Precision | Clf Recall | Clf F1 | Clf ROC-AUC | Reg MAE | Reg RMSE | Reg $R^2$ | Reg Adj $R^2$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | Classification (`survived`) | 0.8045 | 0.7931 | 0.6667 | 0.7244 | **0.8437** | — | — | — | — |
| **Decision Tree (`max_depth=4`)** | Classification (`survived`) | 0.7933 | **0.8636** | 0.5507 | 0.6726 | 0.8292 | — | — | — | — |
| **Random Forest (Baseline)** | Classification (`survived`) | **0.8156** | 0.8000 | **0.6957** | **0.7442** | 0.8287 | — | — | — | — |
| **Tuned Random Forest (`oob_score=True`)** | Classification (`survived`) | 0.8045 | 0.8036 | 0.6522 | 0.7200 | 0.8398 | — | — | — | — |
| **Multivariate Linear Regression** | Regression (`fare`) | — | — | — | — | — | **20.8977** | **30.5328** | **0.3975** | **0.3617** |

**Final Deployment Recommendation**:
For production deployment, we recommend the **Random Forest pipeline** (specifically the regularized `max_depth=8` tuned pipeline serialized in `best_titanic_pipeline.joblib`, or the baseline 100-tree ensemble). Across standard classifiers, Random Forest achieved the highest test Accuracy (`0.8156`), Recall (`0.6957`), and F1 score (`0.7442`), whereas the single Decision Tree suffered from low Recall (`0.5507`, F1 = `0.6726`) due to axis-aligned split variance. Constraining tree depth (`max_depth=8`) via `GridSearchCV` lifted the ROC-AUC from `0.8287` to `0.8398` (nearly matching Logistic Regression's `0.8437`) while achieving a strong Out-of-Bag generalization score of `0.8174` and higher precision (`0.8036`). Moreover, if maximizing survivor recall is the primary business objective, pairing Logistic Regression with training-fold SMOTE (`F1 = 0.7606`, `Recall = 0.7826`) provides an excellent lightweight alternative.

### 15. End-to-End Pipeline Export & Reload Verification
- **Saved Artifact**: `analytics/best_titanic_pipeline.joblib` (contains the full fitted `Pipeline`: `ColumnTransformer` with median/mode imputation, `StandardScaler`, and `OneHotEncoder`, chained into `RandomForestClassifier(max_depth=8, max_features='sqrt', n_estimators=100, oob_score=True)`).
- **Reload Verification on Raw Data**: Reloaded via `joblib.load("analytics/best_titanic_pipeline.joblib")` in `02_modeling.py` and verified directly on unpreprocessed DataFrame rows containing missing (`NaN`) `age` and `embarked` values (`Sample 1 -> survived=1, prob=0.9990`; `Sample 2 -> survived=0, prob=0.0842`).
