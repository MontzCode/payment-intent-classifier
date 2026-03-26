"""
train.py
========
Model training for the Payment Query Intent Classifier.

Handles:
- 5 model pipelines including MLP neural network
- 5-fold stratified cross-validation
- Automatic best model selection (weighted F1 + accuracy)
- Hyperparameter tuning on the best model via GridSearchCV
- Saves tuned model to outputs/model.pkl
- Saves full results to outputs/metrics.json

Run from project root:
    python src/train.py
"""

import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.naive_bayes     import MultinomialNB
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.metrics import make_scorer, f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

TRAIN_PATH   = os.path.join("data", "train.csv")
MODEL_PATH   = os.path.join("outputs", "model.pkl")
METRICS_PATH = os.path.join("outputs", "metrics.json")

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# INTENT LABELS
# ─────────────────────────────────────────────────────────────

INTENT_LABELS = {
    0: "Card Fee Charge",
    1: "Fraud / Disputed Payment",
    2: "Deposit Delay",
    3: "ATM Partial Withdrawal",
    4: "Cash Withdrawal Charge",
    5: "Duplicate Transaction",
    6: "Declined Withdrawal",
    7: "Transfer Fee",
    8: "Transfer Delay",
    9: "Missing Money",
}

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD TRAINING DATA
# ─────────────────────────────────────────────────────────────

def load_training_data(path: str = TRAIN_PATH):
    """
    Loads the processed training CSV.
    Uses cleaned text if available, falls back to raw text.
    """
    print("[1/4] Loading training data...")
    df = pd.read_csv(path)

    text_col = "text_clean" if "text_clean" in df.columns else "text"
    X = df[text_col].astype(str)
    y = df["label"].astype(int)

    print(f"      {len(X)} training samples across {y.nunique()} classes")
    print(f"      Using column: '{text_col}'")
    return X, y


# ─────────────────────────────────────────────────────────────
# STEP 2 — DEFINE MODEL PIPELINES
# ─────────────────────────────────────────────────────────────

def build_pipelines() -> dict:
    """
    Returns a dictionary of named sklearn Pipeline objects.
    Each pipeline pairs a TF-IDF vectoriser with a classifier.
    """
    pipelines = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf",   MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf",   LinearSVC(max_iter=2000, C=1.0, random_state=42)),
        ]),
        "Logistic Regression (Char n-gram)": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=10000
            )),
            ("clf",   LogisticRegression(max_iter=1000, C=5.0, random_state=42)),
        ]),
        "MLP Neural Network": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf",   MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ]),
    }
    return pipelines


# ─────────────────────────────────────────────────────────────
# STEP 3 — CROSS-VALIDATION COMPARISON
# ─────────────────────────────────────────────────────────────

def run_cross_validation(
    pipelines: dict,
    X: pd.Series,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """
    Runs stratified k-fold cross-validation across all pipelines.
    Scores each on both weighted F1 and accuracy.
    Returns a results dictionary.
    """
    print(f"\n[2/4] Running {n_splits}-fold stratified cross-validation...")
    print(f"      Evaluating {len(pipelines)} models — this may take a minute\n")

    cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "f1"      : make_scorer(f1_score, average="weighted"),
        "accuracy": make_scorer(accuracy_score),
    }

    results = {}
    col_w   = 38

    header = (
        f"  {'Model':<{col_w}} {'F1 Mean':>9} {'F1 Std':>8} "
        f"{'Acc Mean':>10} {'Acc Std':>9} {'Time':>7}"
    )
    print(header)
    print("  " + "-" * (col_w + 47))

    for name, pipeline in pipelines.items():
        t0 = time.time()
        cv_results = cross_validate(
            pipeline, X, y,
            cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False,
        )
        elapsed = time.time() - t0

        f1_mean  = cv_results["test_f1"].mean()
        f1_std   = cv_results["test_f1"].std()
        acc_mean = cv_results["test_accuracy"].mean()
        acc_std  = cv_results["test_accuracy"].std()

        results[name] = {
            "f1_mean"  : round(f1_mean,  4),
            "f1_std"   : round(f1_std,   4),
            "acc_mean" : round(acc_mean, 4),
            "acc_std"  : round(acc_std,  4),
            "f1_scores": cv_results["test_f1"].tolist(),
            "acc_scores": cv_results["test_accuracy"].tolist(),
            "time_s"   : round(elapsed, 1),
        }

        print(
            f"  {name:<{col_w}} "
            f"{f1_mean:>9.4f} {f1_std:>8.4f} "
            f"{acc_mean:>10.4f} {acc_std:>9.4f} "
            f"{elapsed:>6.1f}s"
        )

    return results


# ─────────────────────────────────────────────────────────────
# STEP 4 — SELECT BEST MODEL + HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────

def get_param_grid(best_name: str) -> dict:
    """
    Returns the hyperparameter grid for the best model type.
    Grids are deliberately focused — wide grids on small datasets
    rarely justify the compute cost.
    """
    grids = {
        "Naive Bayes": {
            "tfidf__max_features": [3000, 5000, 8000],
            "tfidf__ngram_range" : [(1, 1), (1, 2)],
            "clf__alpha"         : [0.1, 0.5, 1.0, 2.0],
        },
        "Logistic Regression": {
            "tfidf__max_features": [3000, 5000, 8000],
            "tfidf__ngram_range" : [(1, 1), (1, 2)],
            "clf__C"             : [0.1, 0.5, 1.0, 5.0, 10.0],
        },
        "Linear SVM": {
            "tfidf__max_features": [3000, 5000, 8000],
            "tfidf__ngram_range" : [(1, 1), (1, 2)],
            "clf__C"             : [0.1, 0.5, 1.0, 5.0, 10.0],
        },
        "Logistic Regression (Char n-gram)": {
            "tfidf__max_features": [5000, 10000, 15000],
            "tfidf__ngram_range" : [(2, 4), (3, 5), (2, 5)],
            "clf__C"             : [1.0, 5.0, 10.0, 20.0],
        },
        "MLP Neural Network": {
            "tfidf__max_features"   : [3000, 5000],
            "clf__hidden_layer_sizes": [(128, 64), (256, 128), (256, 128, 64)],
            "clf__alpha"            : [0.0001, 0.001, 0.01],
        },
    }
    return grids.get(best_name, {})


def tune_best_model(
    best_name: str,
    pipelines: dict,
    X: pd.Series,
    y: pd.Series,
) -> Pipeline:
    """
    Runs GridSearchCV on the best model using weighted F1 as the scoring metric.
    Returns the fitted best estimator.
    """
    print(f"\n[3/4] Tuning hyperparameters for: {best_name}")

    param_grid = get_param_grid(best_name)
    pipeline   = pipelines[best_name]
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if not param_grid:
        print("      No param grid defined — fitting with defaults.")
        pipeline.fit(X, y)
        return pipeline

    total_fits = (
        len(list(__import__("itertools").product(*param_grid.values()))) * 5
    )
    print(f"      Grid size: {total_fits} fits — running...\n")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X, y)

    print(f"      Best params:")
    for param, val in grid_search.best_params_.items():
        print(f"        {param}: {val}")
    print(f"\n      Best CV F1 (tuned): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# ─────────────────────────────────────────────────────────────
# STEP 5 — SAVE MODEL + METRICS
# ─────────────────────────────────────────────────────────────

def save_outputs(
    model: Pipeline,
    best_name: str,
    cv_results: dict,
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_PATH,
) -> None:
    """
    Saves the tuned model as a .pkl file and writes
    all CV results plus metadata to metrics.json.
    """
    print(f"\n[4/4] Saving outputs...")

    joblib.dump(model, model_path)
    print(f"      Model saved → {model_path}")

    output = {
        "best_model"  : best_name,
        "cv_results"  : cv_results,
        "intent_labels": INTENT_LABELS,
    }
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"      Metrics saved → {metrics_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_training() -> tuple:
    """
    Runs the full training pipeline end to end.
    Returns (tuned_model, best_model_name, cv_results).
    """
    print("\n" + "=" * 55)
    print("  PAYMENT INTENT CLASSIFIER — MODEL TRAINING")
    print("=" * 55)

    X, y      = load_training_data()
    pipelines = build_pipelines()
    cv_results = run_cross_validation(pipelines, X, y)

    # Select best by weighted F1
    best_name = max(cv_results, key=lambda n: cv_results[n]["f1_mean"])
    best_f1   = cv_results[best_name]["f1_mean"]
    best_acc  = cv_results[best_name]["acc_mean"]

    print(f"\n  Best model: {best_name}")
    print(f"  CV Weighted F1: {best_f1:.4f}  |  CV Accuracy: {best_acc:.4f}")

    tuned_model = tune_best_model(best_name, pipelines, X, y)
    save_outputs(tuned_model, best_name, cv_results)

    print("\n  Training complete.\n")
    return tuned_model, best_name, cv_results


if __name__ == "__main__":
    run_training()