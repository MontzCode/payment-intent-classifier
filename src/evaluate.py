"""
evaluate.py
===========
Model evaluation for the Payment Query Intent Classifier.

Handles:
- Loading saved model and test set
- Full classification report
- Confusion matrix (normalised)
- Per-class F1 bar chart
- Precision vs Recall scatter
- Misclassified examples table
- Updates outputs/metrics.json with test results

Run from project root:
    python src/evaluate.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

MODEL_PATH   = os.path.join("outputs", "model.pkl")
TEST_PATH    = os.path.join("data", "test.csv")
METRICS_PATH = os.path.join("outputs", "metrics.json")
PLOTS_DIR    = os.path.join("outputs", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────

VISA_BLUE   = "#1A1F71"
VISA_GOLD   = "#F7B600"
ACCENT_TEAL = "#00B8B0"
LIGHT_BG    = "#F4F6FB"
GREY        = "#6B7280"
DANGER_RED  = "#E05C5C"

PALETTE = [
    VISA_BLUE, VISA_GOLD, ACCENT_TEAL, DANGER_RED, "#7C3AED",
    "#059669", "#DC7F1E", "#2563EB", "#DB2777", "#64748B",
]

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.facecolor"   : LIGHT_BG,
    "figure.facecolor" : "white",
    "axes.labelcolor"  : VISA_BLUE,
    "xtick.color"      : GREY,
    "ytick.color"      : GREY,
    "axes.titlecolor"  : VISA_BLUE,
    "axes.titleweight" : "bold",
    "axes.titlesize"   : 13,
})

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
# STEP 1 — LOAD MODEL + TEST DATA
# ─────────────────────────────────────────────────────────────

def load_model_and_data():
    """
    Loads the saved model pipeline and test CSV.
    Returns (model, X_test, y_test, df_test).
    """
    print("[1/5] Loading model and test data...")

    model = joblib.load(MODEL_PATH)
    print(f"      Model loaded from {MODEL_PATH}")

    df    = pd.read_csv(TEST_PATH)
    text_col = "text_clean" if "text_clean" in df.columns else "text"
    X_test   = df[text_col].astype(str)
    y_test   = df["label"].astype(int)

    print(f"      Test set: {len(X_test)} samples")
    return model, X_test, y_test, df


# ─────────────────────────────────────────────────────────────
# STEP 2 — GENERATE PREDICTIONS + REPORT
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Runs predictions and prints the classification report.
    Returns (y_pred, report_dict).
    """
    print("\n[2/5] Evaluating on test set...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n      Test Accuracy  : {acc:.4f}")
    print(f"      Weighted F1   : {f1:.4f}")
    print("\n      Classification Report:")
    print("      " + "-" * 65)

    report = classification_report(
        y_test, y_pred,
        target_names=list(INTENT_LABELS.values()),
    )
    for line in report.split("\n"):
        print(f"      {line}")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=list(INTENT_LABELS.values()),
        output_dict=True,
    )

    return y_pred, report_dict, acc, f1


# ─────────────────────────────────────────────────────────────
# PLOT 1 — CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, acc, f1):
    print("\n[3/5] Generating evaluation plots...")
    print("      Plotting confusion matrix...")

    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    labels  = list(INTENT_LABELS.values())

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalised Recall")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(10):
        for j in range(10):
            val      = cm[i, j]
            norm_val = cm_norm[i, j]
            color    = "white" if norm_val > 0.55 else VISA_BLUE
            weight   = "bold" if i == j else "normal"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xlabel("Predicted Intent", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual Intent", fontsize=11, labelpad=10)
    ax.set_title(
        f"Confusion Matrix — Held-Out Test Set (n={len(y_test)})\n"
        f"Accuracy: {acc:.1%}  |  Weighted F1: {f1:.3f}  |  "
        f"Diagonal = correct predictions, off-diagonal = routing errors",
        fontsize=12, pad=15,
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "06_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"        Saved → {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 2 — PER-CLASS F1 BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_per_class_f1(report_dict):
    print("      Plotting per-class F1 scores...")

    intents = list(INTENT_LABELS.values())
    f1_vals = [report_dict[i]["f1-score"] for i in intents]
    colors  = [
        VISA_GOLD   if v >= 0.95 else
        ACCENT_TEAL if v >= 0.85 else
        DANGER_RED
        for v in f1_vals
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(intents)), f1_vals,
                  color=colors, edgecolor="white", width=0.65)

    for bar, val in zip(bars, f1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=VISA_BLUE,
        )

    ax.set_xticks(range(len(intents)))
    ax.set_xticklabels(intents, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.axhline(y=0.90, color=GREY, linestyle="--",
               alpha=0.5, linewidth=1.5, label="0.90 threshold")
    ax.set_facecolor(LIGHT_BG)
    ax.set_title(
        "Per-Class F1 Score — Test Set\n"
        "Each bar represents how reliably the model routes that query type. "
        "Low F1 = higher customer misrouting risk.",
        fontsize=12,
    )

    legend_handles = [
        mpatches.Patch(color=VISA_GOLD,   label="F1 ≥ 0.95  Excellent"),
        mpatches.Patch(color=ACCENT_TEAL, label="0.85 ≤ F1 < 0.95  Good"),
        mpatches.Patch(color=DANGER_RED,  label="F1 < 0.85  Needs attention"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "07_per_class_f1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"        Saved → {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 3 — PRECISION VS RECALL SCATTER
# ─────────────────────────────────────────────────────────────

def plot_precision_recall(report_dict):
    print("      Plotting precision vs recall scatter...")

    intents   = list(INTENT_LABELS.values())
    precision = [report_dict[i]["precision"] for i in intents]
    recall    = [report_dict[i]["recall"]    for i in intents]
    f1_vals   = [report_dict[i]["f1-score"]  for i in intents]

    fig, ax = plt.subplots(figsize=(9, 7))

    scatter = ax.scatter(
        recall, precision,
        c=f1_vals, cmap="RdYlGn",
        s=180, edgecolors=VISA_BLUE,
        linewidths=1.2, vmin=0.7, vmax=1.0, zorder=3,
    )
    plt.colorbar(scatter, ax=ax, label="F1 Score", shrink=0.85)

    for i, intent in enumerate(intents):
        short = intent.replace(" / ", "/")
        ax.annotate(
            short,
            (recall[i], precision[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=VISA_BLUE,
        )

    ax.axhline(y=0.90, color=GREY, linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=0.90, color=GREY, linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Recall  (coverage — how many of each class we catch)", fontsize=10)
    ax.set_ylabel("Precision  (accuracy — when we predict a class, are we right?)", fontsize=10)
    ax.set_xlim(0.75, 1.05)
    ax.set_ylim(0.75, 1.05)
    ax.set_facecolor(LIGHT_BG)
    ax.set_title(
        "Precision vs Recall by Intent Class\n"
        "Top-right quadrant = best performance. "
        "Classes bottom-left carry the highest business risk.",
        fontsize=12,
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_precision_recall_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"        Saved → {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 4 — MISCLASSIFIED EXAMPLES TABLE
# ─────────────────────────────────────────────────────────────

def plot_misclassified(df_test, y_pred):
    print("      Plotting misclassified examples...")

    text_col = "text_clean" if "text_clean" in df_test.columns else "text"

    df_errors = df_test.copy()
    df_errors["predicted"] = y_pred
    df_errors = df_errors[df_errors["label"] != df_errors["predicted"]].copy()
    df_errors["actual_intent"]    = df_errors["label"].map(INTENT_LABELS)
    df_errors["predicted_intent"] = df_errors["predicted"].map(INTENT_LABELS)

    # Sample up to 12 errors, spread across classes
    sample = (
        df_errors.groupby("actual_intent", group_keys=False)
        .apply(lambda g: g.sample(min(2, len(g)), random_state=42))
        .head(12)
        .reset_index(drop=True)
    )

    table_data = []
    for _, row in sample.iterrows():
        query = str(row[text_col])
        if len(query) > 60:
            query = query[:57] + "..."
        table_data.append([
            query,
            row["actual_intent"],
            row["predicted_intent"],
        ])

    fig, ax = plt.subplots(figsize=(15, len(table_data) * 0.6 + 2))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=["Query", "Actual Intent", "Predicted Intent"],
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.23, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE3F0")
        cell.set_height(0.1)
        if row == 0:
            cell.set_facecolor(VISA_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2:
            cell.set_facecolor("#FFF0F0")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF0FA")
        else:
            cell.set_facecolor("white")

    total_errors = len(df_errors)
    error_rate   = total_errors / len(df_test)

    ax.set_title(
        f"Misclassified Query Examples — {total_errors} errors from {len(df_test)} test queries "
        f"({error_rate:.1%} error rate)\n"
        "Predicted Intent (red) = where the model would have routed the query incorrectly. "
        "Most errors occur between semantically similar classes.",
        fontsize=11, color=VISA_BLUE, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "09_misclassified_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"        Saved → {path}")

    print(f"\n      Total misclassified: {total_errors} / {len(df_test)} "
          f"({error_rate:.1%} error rate)")


# ─────────────────────────────────────────────────────────────
# STEP 5 — UPDATE METRICS.JSON
# ─────────────────────────────────────────────────────────────

def update_metrics(acc, f1, report_dict):
    """
    Appends test set results to the existing metrics.json.
    """
    print("\n[5/5] Updating metrics.json with test results...")

    existing = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            existing = json.load(f)

    per_class = {
        intent: {
            "precision": round(report_dict[intent]["precision"], 3),
            "recall"   : round(report_dict[intent]["recall"],    3),
            "f1"       : round(report_dict[intent]["f1-score"],  3),
            "support"  : int(report_dict[intent]["support"]),
        }
        for intent in INTENT_LABELS.values()
    }

    existing["test_results"] = {
        "accuracy"        : round(acc, 4),
        "weighted_f1"     : round(f1,  4),
        "per_class"       : per_class,
        "macro_f1"        : round(report_dict["macro avg"]["f1-score"], 4),
        "macro_precision" : round(report_dict["macro avg"]["precision"], 4),
        "macro_recall"    : round(report_dict["macro avg"]["recall"],    4),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"      Saved → {METRICS_PATH}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 55)
    print("  PAYMENT INTENT CLASSIFIER — EVALUATION")
    print("=" * 55)

    model, X_test, y_test, df_test = load_model_and_data()
    y_pred, report_dict, acc, f1   = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(y_test, y_pred, acc, f1)
    plot_per_class_f1(report_dict)
    plot_precision_recall(report_dict)
    plot_misclassified(df_test, y_pred)
    update_metrics(acc, f1, report_dict)

    print(f"\n  Evaluation complete.")
    print(f"  Test Accuracy: {acc:.4f}  |  Weighted F1: {f1:.4f}")
    print(f"  Plots saved to {PLOTS_DIR}/\n")


if __name__ == "__main__":
    run_evaluation()