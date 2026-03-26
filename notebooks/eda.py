"""
eda.py
======
Exploratory Data Analysis for the Payment Query Intent Classifier.

Loads processed data from SQLite and produces 5 publication-quality
plots saved to outputs/plots/.

Run from project root:
    python notebooks/eda.py
"""

import os
import re
import sqlite3
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

DB_PATH      = os.path.join("data", "intent_queries.db")
PLOTS_DIR    = os.path.join("outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────

VISA_BLUE    = "#1A1F71"
VISA_GOLD    = "#F7B600"
ACCENT_TEAL  = "#00B8B0"
LIGHT_BG     = "#F4F6FB"
GREY         = "#6B7280"
DANGER_RED   = "#E05C5C"

PALETTE = [
    VISA_BLUE, VISA_GOLD, ACCENT_TEAL, DANGER_RED, "#7C3AED",
    "#059669", "#DC7F1E", "#2563EB", "#DB2777", "#64748B",
]

plt.rcParams.update({
    "font.family"        : "DejaVu Sans",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.facecolor"     : LIGHT_BG,
    "figure.facecolor"   : "white",
    "axes.labelcolor"    : VISA_BLUE,
    "xtick.color"        : GREY,
    "ytick.color"        : GREY,
    "axes.titlecolor"    : VISA_BLUE,
    "axes.titleweight"   : "bold",
    "axes.titlesize"     : 13,
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

STOPWORDS = {
    "i", "my", "the", "a", "an", "is", "it", "to", "was", "and",
    "of", "in", "for", "me", "have", "that", "this", "on", "why",
    "did", "do", "not", "be", "are", "has", "had", "with", "at",
    "when", "there", "what", "how", "can", "get", "been", "from",
    "but", "or", "its", "so", "if", "they", "we", "you", "he",
    "she", "their", "which", "will", "would", "could", "should",
    "just", "also", "up", "out", "by", "about", "some", "as",
    "am", "were", "who", "no", "more", "all", "one", "too",
}

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_data(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT text, label FROM queries", conn)
    conn.close()
    df["intent"]     = df["label"].map(INTENT_LABELS)
    df["text_clean"] = df["text"].str.lower()
    df["text_clean"] = df["text_clean"].apply(
        lambda t: re.sub(r"[^\w\s']", " ", t)
    )
    df["text_clean"] = df["text_clean"].apply(
        lambda t: re.sub(r"\s+", " ", t).strip()
    )
    df["word_count"] = df["text_clean"].str.split().str.len()
    df["char_count"] = df["text_clean"].str.len()
    return df

# ─────────────────────────────────────────────────────────────
# PLOT 1 — INTENT DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_intent_distribution(df: pd.DataFrame) -> None:
    print("  Plotting intent distribution...")

    counts = df.groupby("intent")["label"].count().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        counts.index, counts.values,
        color=PALETTE[:len(counts)], edgecolor="white", height=0.65,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            val + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=10,
            color=VISA_BLUE, fontweight="bold",
        )

    ax.set_xlabel("Number of Queries", fontsize=11)
    ax.set_title(
        "Query Volume by Intent Category\n"
        "Well-balanced dataset — each class between 211 and 227 queries",
        fontsize=13,
    )
    ax.set_xlim(0, max(counts.values) * 1.12)
    ax.set_facecolor(LIGHT_BG)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_intent_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")

# ─────────────────────────────────────────────────────────────
# PLOT 2 — QUERY LENGTH BY INTENT
# ─────────────────────────────────────────────────────────────

def plot_query_length(df: pd.DataFrame) -> None:
    print("  Plotting query length distribution...")

    intent_order = (
        df.groupby("intent")["word_count"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Word count boxplot
    data_words = [df[df["intent"] == i]["word_count"].values for i in intent_order]
    bp = axes[0].boxplot(
        data_words, vert=False, patch_artist=True,
        medianprops=dict(color=VISA_GOLD, linewidth=2.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_yticklabels(intent_order, fontsize=9)
    axes[0].set_xlabel("Word Count", fontsize=10)
    axes[0].set_title("Word Count per Query", fontsize=12)
    axes[0].set_facecolor(LIGHT_BG)

    # Char count violin
    plot_data = [df[df["intent"] == i]["char_count"].values for i in intent_order]
    vp = axes[1].violinplot(
        plot_data, vert=False, showmedians=True,
    )
    for body, color in zip(vp["bodies"], PALETTE):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    vp["cmedians"].set_color(VISA_GOLD)
    vp["cmedians"].set_linewidth(2)
    axes[1].set_yticks(range(1, len(intent_order) + 1))
    axes[1].set_yticklabels(intent_order, fontsize=9)
    axes[1].set_xlabel("Character Count", fontsize=10)
    axes[1].set_title("Character Count per Query", fontsize=12)
    axes[1].set_facecolor(LIGHT_BG)

    plt.suptitle(
        "Query Length Analysis by Intent\n"
        "Transfer and fraud-related queries tend to be longer and more detailed",
        fontsize=13, fontweight="bold", color=VISA_BLUE, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_query_length_by_intent.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")

# ─────────────────────────────────────────────────────────────
# PLOT 3 — TOP TERMS PER INTENT
# ─────────────────────────────────────────────────────────────

def plot_top_terms(df: pd.DataFrame) -> None:
    print("  Plotting top terms per intent...")

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()

    for idx, (label, intent) in enumerate(INTENT_LABELS.items()):
        texts = df[df["label"] == label]["text_clean"].tolist()
        words = []
        for t in texts:
            words.extend([
                w for w in t.split()
                if w not in STOPWORDS and len(w) > 2
            ])

        top_words = Counter(words).most_common(12)
        terms  = [w for w, _ in top_words][::-1]
        counts = [c for _, c in top_words][::-1]

        color = PALETTE[idx % len(PALETTE)]
        axes[idx].barh(
            range(len(terms)), counts,
            color=color, alpha=0.85, edgecolor="white",
        )
        axes[idx].set_yticks(range(len(terms)))
        axes[idx].set_yticklabels(terms, fontsize=8.5)
        axes[idx].set_title(
            f"{intent}", fontsize=9,
            color=VISA_BLUE, fontweight="bold",
        )
        axes[idx].set_facecolor(LIGHT_BG)
        axes[idx].tick_params(axis="x", labelsize=8)

    plt.suptitle(
        "Top Discriminative Terms per Intent Class\n"
        "Stopwords removed — shows the language patterns unique to each query type",
        fontsize=14, fontweight="bold", color=VISA_BLUE, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_top_terms_per_intent.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")

# ─────────────────────────────────────────────────────────────
# PLOT 4 — CLASS OVERLAP HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_class_overlap(df: pd.DataFrame) -> None:
    print("  Plotting class overlap heatmap...")

    def get_top_words(texts, n=30):
        words = []
        for t in texts:
            words.extend([
                w for w in t.split()
                if w not in STOPWORDS and len(w) > 2
            ])
        return set([w for w, _ in Counter(words).most_common(n)])

    intents  = list(INTENT_LABELS.values())
    n        = len(intents)
    overlap  = np.zeros((n, n))

    top_word_sets = {}
    for label, intent in INTENT_LABELS.items():
        texts = df[df["label"] == label]["text_clean"].tolist()
        top_word_sets[intent] = get_top_words(texts, n=40)

    for i, intent_a in enumerate(intents):
        for j, intent_b in enumerate(intents):
            set_a = top_word_sets[intent_a]
            set_b = top_word_sets[intent_b]
            if len(set_a | set_b) == 0:
                overlap[i, j] = 0
            else:
                overlap[i, j] = len(set_a & set_b) / len(set_a | set_b)

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.eye(n, dtype=bool)

    sns.heatmap(
        overlap,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        xticklabels=intents,
        yticklabels=intents,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Jaccard Similarity (word overlap)"},
        vmin=0, vmax=0.6,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(
        "Vocabulary Overlap Between Intent Classes\n"
        "Higher values mean classes share more language — harder to distinguish",
        fontsize=13, pad=15,
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_class_overlap_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")

# ─────────────────────────────────────────────────────────────
# PLOT 5 — SUMMARY STATS TABLE
# ─────────────────────────────────────────────────────────────

def plot_summary_table(df: pd.DataFrame) -> None:
    print("  Plotting summary stats table...")

    stats = df.groupby("intent").agg(
        Count=("text", "count"),
        Avg_Words=("word_count", "mean"),
        Avg_Chars=("char_count", "mean"),
        Max_Words=("word_count", "max"),
    ).reset_index()

    stats["Avg_Words"] = stats["Avg_Words"].round(1)
    stats["Avg_Chars"] = stats["Avg_Chars"].round(1)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    table_data = stats.values.tolist()
    col_labels = ["Intent", "Count", "Avg Words", "Avg Chars", "Max Words"]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.35, 0.12, 0.14, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(VISA_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF0FA")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#DDE3F0")
        cell.set_height(0.09)

    ax.set_title(
        "Dataset Summary Statistics by Intent Class",
        fontsize=13, color=VISA_BLUE, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_summary_stats_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_eda() -> None:
    print("\n" + "=" * 55)
    print("  PAYMENT INTENT CLASSIFIER — EDA")
    print("=" * 55 + "\n")

    df = load_data()
    print(f"  Loaded {len(df)} queries across {df['label'].nunique()} intent classes\n")

    plot_intent_distribution(df)
    plot_query_length(df)
    plot_top_terms(df)
    plot_class_overlap(df)
    plot_summary_table(df)

    print(f"\n  EDA complete. All plots saved to {PLOTS_DIR}/\n")


if __name__ == "__main__":
    run_eda()