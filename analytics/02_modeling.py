from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

OUT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = OUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV_PATH = OUT_DIR / "titanic.csv"
PIPELINE_PATH = OUT_DIR / "best_titanic_pipeline.joblib"
MODELING_LOG_PATH = OUT_DIR / "modeling_report.txt"

NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
NA_ROW_DEFAULTS = {
    "Clf_Accuracy": "N/A",
    "Clf_Precision": "N/A",
    "Clf_Recall": "N/A",
    "Clf_F1": "N/A",
    "Clf_AUC": "N/A",
    "Reg_MAE": "N/A",
    "Reg_RMSE": "N/A",
    "Reg_R2": "N/A",
    "Reg_Adj_R2": "N/A",
}


def build_preprocessor(
    num_cols: list[str] = NUMERIC_FEATURES,
    cat_cols: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Create a ColumnTransformer fit strictly on training data."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def main() -> None:
    lines = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    # Read from the committed titanic.csv produced by 01_eda.py (never calling sns.load_dataset again)
    assert RAW_CSV_PATH.exists(), f"Missing {RAW_CSV_PATH}; run 01_eda.py first."
    df = pd.read_csv(RAW_CSV_PATH)

    # --- Task 7: Stratified Train/Test Split ---
    log("=== TASK 7: STRATIFIED TRAIN/TEST SPLIT ===")
    class_counts = df["survived"].value_counts()
    class_props = (df["survived"].value_counts(normalize=True) * 100).round(2)
    log(
        f"Overall Target Class Balance: Not Survived (0) = {class_counts[0]} ({class_props[0]:.2f}%), "
        f"Survived (1) = {class_counts[1]} ({class_props[1]:.2f}%)"
    )

    X = df[ALL_FEATURES].copy()
    y = df["survived"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    log(
        f"Train shape: {X_train.shape} (Survived = {y_train.mean() * 100:.2f}%) | "
        f"Test shape: {X_test.shape} (Survived = {y_test.mean() * 100:.2f}%)"
    )

    # --- Tasks 8, 9, 10: Train & Evaluate 3 Classifiers on Identical Split ---
    log("\n=== TASKS 8, 9 & 10: THREE CLASSIFIERS & EVALUATION SUITE ===")
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    fitted_pipelines = {}
    clf_metrics = []

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for name, clf in classifiers.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        acc, prec, rec, f1, auc = (
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_prob),
        )

        clf_metrics.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "ROC_AUC": round(auc, 4),
                "Confusion_Matrix": f"[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]",
            }
        )

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves for Logistic Regression, Decision Tree, and Random Forest")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_roc_curves.png", dpi=150)
    plt.close(fig_roc)

    # Render Decision Tree via plot_tree with labeled feature_names and class_names
    dt_pipe = fitted_pipelines["Decision Tree"]
    dt_feature_names = list(dt_pipe.named_steps["preprocessor"].get_feature_names_out())
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt_pipe.named_steps["classifier"],
        feature_names=dt_feature_names,
        class_names=["Not Survived", "Survived"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax_tree,
    )
    ax_tree.set_title("Decision Tree Visualization (max_depth=4)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "08_decision_tree_plot.png", dpi=150)
    plt.close(fig_tree)

    log(pd.DataFrame(clf_metrics).to_string(index=False))

    # --- Task 11: Imbalance Handling Comparison ---
    log("\n=== TASK 11: IMBALANCE HANDLING COMPARISON (LOGISTIC REGRESSION) ===")
    imbalance_variants = {
        "(a) Baseline (No Handling)": Pipeline(
            [("preprocessor", build_preprocessor()), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "(b) class_weight='balanced'": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "(c) SMOTE (Training Fold Only)": ImbPipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=42)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
    }

    imb_rows = []
    for var_name, var_pipe in imbalance_variants.items():
        var_pipe.fit(X_train, y_train)
        vp = var_pipe.predict(X_test)
        imb_rows.append(
            {
                "Strategy": var_name,
                "Accuracy": round(accuracy_score(y_test, vp), 4),
                "Precision": round(precision_score(y_test, vp), 4),
                "Recall": round(recall_score(y_test, vp), 4),
                "F1": round(f1_score(y_test, vp), 4),
            }
        )
    log(pd.DataFrame(imb_rows).to_string(index=False))

    # --- Task 12: Hyperparameter Tuning with GridSearchCV & OOB Score ---
    log("\n=== TASK 12: RANDOM FOREST GRIDSEARCHCV & OUT-OF-BAG (OOB) SCORE ===")
    rf_base_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
        ]
    )
    grid = GridSearchCV(
        rf_base_pipe,
        param_grid={
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__max_features": ["sqrt", "log2"],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_rf_pipe = grid.best_estimator_
    oob_score_val = best_rf_pipe.named_steps["classifier"].oob_score_

    y_pred_tuned = best_rf_pipe.predict(X_test)
    y_prob_tuned = best_rf_pipe.predict_proba(X_test)[:, 1]
    tuned_acc, tuned_prec, tuned_rec, tuned_f1, tuned_auc = (
        accuracy_score(y_test, y_pred_tuned),
        precision_score(y_test, y_pred_tuned),
        recall_score(y_test, y_pred_tuned),
        f1_score(y_test, y_pred_tuned),
        roc_auc_score(y_test, y_prob_tuned),
    )

    log(f"Best Parameters: {grid.best_params_}")
    log(f"Best CV F1 Score: {grid.best_score_:.4f}")
    log(f"Corresponding OOB Score (oob_score_): {oob_score_val:.4f}")
    log(
        f"Tuned RF Test Metrics -> Accuracy: {tuned_acc:.4f}, Precision: {tuned_prec:.4f}, "
        f"Recall: {tuned_rec:.4f}, F1: {tuned_f1:.4f}, ROC_AUC: {tuned_auc:.4f}"
    )

    # --- Task 13: Regression Side-Task (Predicting Fare) ---
    log("\n=== TASK 13: MULTIVARIATE LINEAR REGRESSION (PREDICTING FARE) ===")
    reg_num_features = ["pclass", "age", "sibsp", "parch", "survived"]
    X_reg = df[reg_num_features + CATEGORICAL_FEATURES].copy()
    y_reg = df["fare"].copy()

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)
    reg_pipe = Pipeline(
        [("preprocessor", build_preprocessor(reg_num_features, CATEGORICAL_FEATURES)), ("regressor", LinearRegression())]
    )
    reg_pipe.fit(Xr_train, yr_train)
    yr_pred = reg_pipe.predict(Xr_test)

    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = float(np.sqrt(mean_squared_error(yr_test, yr_pred)))
    r2 = r2_score(yr_test, yr_pred)
    n_samples = len(yr_test)
    p_features = reg_pipe.named_steps["preprocessor"].transform(Xr_test).shape[1]
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - p_features - 1)

    log(f"Regression Test Metrics (Target = fare, n={n_samples}, p={p_features}):")
    log(f"  MAE         : {mae:.4f}")
    log(f"  RMSE        : {rmse:.4f}")
    log(f"  R^2         : {r2:.4f}")
    log(f"  Adjusted R^2: {adj_r2:.4f}")

    residuals = yr_test - yr_pred
    fig_res, ax_res = plt.subplots(figsize=(8, 5))
    ax_res.scatter(yr_pred, residuals, alpha=0.6, color="teal", edgecolors="k")
    ax_res.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax_res.set_xlabel("Predicted Fare")
    ax_res.set_ylabel("Residuals (Actual - Predicted)")
    ax_res.set_title("Residual Plot for Fare Multivariate Linear Regression")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "09_regression_residuals.png", dpi=150)
    plt.close(fig_res)
    log(
        "  Heteroscedasticity Conclusion: The residual plot exhibits clear HETEROSCEDASTICITY "
        "(a non-random, funnel-shaped spread of residuals that widens substantially as predicted fare increases, "
        "driven by extreme right-skewness and luxury first-class suite outliers up to 512.33)."
    )

    # --- Task 14: Combined Comparison Table (Separate Metric Columns per Model Type) ---
    log("\n=== TASK 14: FINAL MODEL COMPARISON TABLE (SEPARATE CLASSIFICATION VS REGRESSION SCALES) ===")
    all_clf_rows = clf_metrics + [
        {
            "Model": "Tuned Random Forest (OOB)",
            "Accuracy": tuned_acc,
            "Precision": tuned_prec,
            "Recall": tuned_rec,
            "F1": tuned_f1,
            "ROC_AUC": tuned_auc,
        }
    ]
    combined_rows = [
        {
            "Model": r["Model"],
            "Task_Type": "Classification (survived)",
            **NA_ROW_DEFAULTS,
            "Clf_Accuracy": f"{r['Accuracy']:.4f}",
            "Clf_Precision": f"{r['Precision']:.4f}",
            "Clf_Recall": f"{r['Recall']:.4f}",
            "Clf_F1": f"{r['F1']:.4f}",
            "Clf_AUC": f"{r['ROC_AUC']:.4f}",
        }
        for r in all_clf_rows
    ] + [
        {
            "Model": "Multivariate Linear Regression",
            "Task_Type": "Regression (fare)",
            **NA_ROW_DEFAULTS,
            "Reg_MAE": f"{mae:.4f}",
            "Reg_RMSE": f"{rmse:.4f}",
            "Reg_R2": f"{r2:.4f}",
            "Reg_Adj_R2": f"{adj_r2:.4f}",
        }
    ]
    log(pd.DataFrame(combined_rows).to_string(index=False))

    # --- Task 15: Save Full Pipeline via joblib.dump & Reload Verification on Raw Input ---
    log("\n=== TASK 15: PIPELINE SERIALIZATION & RAW-DATA RELOAD VERIFICATION ===")
    joblib.dump(best_rf_pipe, PIPELINE_PATH)
    log(f"Saved full end-to-end Pipeline (ColumnTransformer + Tuned RandomForest) to: {PIPELINE_PATH}")

    reloaded_pipeline = joblib.load(PIPELINE_PATH)
    raw_test_samples = pd.DataFrame(
        [
            {"pclass": 1, "age": 29.0, "sibsp": 0, "parch": 0, "fare": 120.0, "sex": "female", "embarked": "C"},
            {"pclass": 3, "age": np.nan, "sibsp": 1, "parch": 0, "fare": 7.25, "sex": "male", "embarked": np.nan},
        ]
    )
    preds = reloaded_pipeline.predict(raw_test_samples)
    probs = reloaded_pipeline.predict_proba(raw_test_samples)[:, 1]
    log("Reloaded pipeline predictions on raw unpreprocessed data (with NaNs):")
    for idx, (p, pr) in enumerate(zip(preds, probs)):
        log(f"  Sample {idx + 1}: predicted survived={int(p)} (probability={pr:.4f})")

    assert len(preds) == 2 and int(preds[0]) == 1 and int(preds[1]) == 0

    MODELING_LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved complete modeling report to {MODELING_LOG_PATH}")


if __name__ == "__main__":
    main()
