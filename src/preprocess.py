"""
preprocess.py
=============
Data layer for the Payment Query Intent Classifier.

Handles:
- SQLite ingestion from raw CSV
- PySpark local session for data profiling
- Text cleaning
- Label mapping to business intent names
- Stratified train/test split
"""

import os
import re
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

RAW_CSV    = os.path.join("data", "ds_task_dataset.csv")
DB_PATH    = os.path.join("data", "intent_queries.db")
TRAIN_PATH = os.path.join("data", "train.csv")
TEST_PATH  = os.path.join("data", "test.csv")

# ─────────────────────────────────────────────────────────────
# LABEL MAPPING
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
# STEP 1 — INGEST CSV INTO SQLITE
# ─────────────────────────────────────────────────────────────

def ingest_to_sqlite(csv_path: str = RAW_CSV, db_path: str = DB_PATH) -> None:
    """
    Reads the raw CSV and writes it into a SQLite database.
    Creates the database and table if they don't already exist.
    """
    print("[1/5] Ingesting CSV into SQLite...")

    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)
    df.to_sql("queries", conn, if_exists="replace", index=False)
    conn.close()

    print(f"      Saved {len(df)} rows to {db_path} → table: queries")


# ─────────────────────────────────────────────────────────────
# STEP 2 — LOAD FROM SQLITE USING SQL
# ─────────────────────────────────────────────────────────────

def load_from_sqlite(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Queries all records from the SQLite database.
    Returns a pandas DataFrame.
    """
    print("[2/5] Loading data from SQLite...")

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT
            text,
            label,
            LENGTH(text)                        AS char_length,
            LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1 AS word_count
        FROM queries
        ORDER BY label
        """,
        conn,
    )

    conn.close()
    print(f"      Loaded {len(df)} rows across {df['label'].nunique()} intent classes")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — PYSPARK PROFILING
# ─────────────────────────────────────────────────────────────

def profile_with_spark(db_path: str = DB_PATH) -> None:
    """
    Spins up a local PySpark session and profiles the dataset.
    Prints class distribution and average query length per intent.
    This mirrors how the pipeline would run at scale on a cluster.
    """
    print("[3/5] Profiling data with PySpark (local mode)...")

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, count, avg, length, split, size

        spark = (
            SparkSession.builder
            .appName("IntentClassifierEDA")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )

        # Suppress verbose Spark logs
        spark.sparkContext.setLogLevel("ERROR")

        # Read directly from CSV for Spark
        # (Spark doesn't read SQLite natively without a JDBC driver)
        sdf = spark.read.csv(
            RAW_CSV,
            header=True,
            inferSchema=True,
        )

        # Add text length and word count columns
        sdf = sdf.withColumn("char_length", length(col("text")))
        sdf = sdf.withColumn("word_count", size(split(col("text"), " ")))

        print("\n      ── Class Distribution ──")
        sdf.groupBy("label") \
           .agg(count("*").alias("query_count")) \
           .orderBy("label") \
           .show(truncate=False)

        print("      ── Avg Query Length per Class ──")
        sdf.groupBy("label") \
           .agg(
               avg("char_length").alias("avg_chars"),
               avg("word_count").alias("avg_words"),
           ) \
           .orderBy("label") \
           .show(truncate=False)

        spark.stop()
        print("      PySpark session closed.")

    except Exception as e:
        print(f"      PySpark profiling skipped: {e}")
        print("      Continuing without Spark — pipeline will still run.")


# ─────────────────────────────────────────────────────────────
# STEP 4 — TEXT CLEANING + LABEL MAPPING
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans a single query string:
    - Lowercases
    - Removes punctuation except apostrophes
    - Collapses extra whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning and maps numeric labels to
    business-readable intent names.
    """
    print("[4/5] Cleaning text and mapping intent labels...")

    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df["intent"]     = df["label"].map(INTENT_LABELS)

    print(f"      Text cleaned. Sample:")
    for _, row in df.sample(3, random_state=42).iterrows():
        print(f"        [{row['intent']}] {row['text_clean']}")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

def split_and_save(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 train/test split.
    Saves both splits to CSV and returns them.
    """
    print("[5/5] Splitting and saving train/test sets...")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"      Train: {len(train_df)} rows → {train_path}")
    print(f"      Test:  {len(test_df)} rows  → {test_path}")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# MAIN — run all steps in order
# ─────────────────────────────────────────────────────────────

def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the full preprocessing pipeline end to end.
    Returns (train_df, test_df).
    """
    print("\n" + "=" * 55)
    print("  PAYMENT INTENT CLASSIFIER — DATA PREPROCESSING")
    print("=" * 55 + "\n")

    ingest_to_sqlite()
    df = load_from_sqlite()
    profile_with_spark()
    df = clean_and_map(df)
    train_df, test_df = split_and_save(df)

    print("\n  Preprocessing complete.\n")
    return train_df, test_df


if __name__ == "__main__":
    run_preprocessing()