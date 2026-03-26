# Payment Query Intent Classifier 💳

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://payment-intent-classifier-tool.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.5%25-brightgreen)

---

## The Business Problem

Every day a payments business receives thousands of inbound customer queries — card fees, disputed transactions, failed withdrawals, missing transfers. Manually reading and routing each one to the right team is slow, error-prone, and expensive. Misrouted queries mean longer resolution times, frustrated customers, and unnecessary operational cost.

This project builds an end-to-end NLP pipeline that automatically classifies incoming payment queries into 10 intent categories and routes them to the appropriate team — with a confidence score that flags low-certainty cases for human review. The result is faster triage, lower handling cost, and a better customer experience.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Test Accuracy | **94.5%** |
| Weighted F1 | **94.5%** |
| Number of Intent Classes | 10 |
| Training Samples | 1,735 |
| Test Samples | 434 |
| Best Model | Logistic Regression (Char n-gram TF-IDF) |
| Error Rate | 5.5% (24 / 434 queries) |

The model outperformed an MLP neural network and three other classifiers in cross-validation. The winning approach — character-level n-gram TF-IDF with Logistic Regression — handles the short, informal language of customer queries better than word-level methods because it captures sub-word patterns like "withdraw", "withdrew", "withdrawal" without needing a large training set.

---

## Live Demo

**[→ Try the app here](https://payment-intent-classifier-tool.streamlit.app)**

Type any payment-related query or click an example. The app returns the predicted intent, a confidence score, a routing instruction, and the top 3 most likely categories.

---

## How It Works

Raw customer query text is ingested from CSV into a **SQLite database** so all data access goes through SQL queries rather than flat file reads. A local **PySpark session** profiles the data — computing class distributions and average query lengths — mirroring the pattern you'd use on a distributed cluster against large-scale transaction data.

Text is cleaned (lowercased, punctuation stripped, whitespace normalised) and five model pipelines are trained and evaluated using **5-fold stratified cross-validation**: Naive Bayes, Logistic Regression (word n-grams), Linear SVM, Logistic Regression (character n-grams), and an **MLP neural network**. The best performer is automatically selected and tuned using **GridSearchCV** before being saved as a deployable artefact.

The saved model is loaded once at startup by the inference module, which applies identical preprocessing to new queries and returns a prediction, calibrated confidence score, and routing instruction. The Streamlit app sits on top of this inference module — any query typed into the UI goes through the same pipeline end to end.

---

## Tech Stack

| Tool | Role | Why |
|---|---|---|
| Python 3.11 | Core language | Industry standard for data science |
| SQLite + SQL | Data layer | Real query-based data access, scalable pattern |
| PySpark (local) | Data profiling | Same code runs on a distributed cluster at scale |
| scikit-learn | ML pipeline | TF-IDF vectorisation, model training, CV, tuning |
| MLP (sklearn) | Neural baseline | Benchmarked against classical approaches |
| GridSearchCV | Hyperparameter tuning | Systematic optimisation of the best model |
| joblib | Model serialisation | Saves trained pipeline for deployment |
| Streamlit | Deployment | Live public demo accessible from a URL |
| GitHub | Version control | Full commit history, deployable from repo |

---

## Intent Categories

| Class | Intent | Routing |
|---|---|---|
| 0 | Card Fee Charge | Card Services |
| 1 | Fraud / Disputed Payment | Fraud & Disputes — high priority |
| 2 | Deposit Delay | Deposit Operations |
| 3 | ATM Partial Withdrawal | ATM & Cash Services |
| 4 | Cash Withdrawal Charge | Card Services |
| 5 | Duplicate Transaction | Transaction Disputes |
| 6 | Declined Withdrawal | Card Operations |
| 7 | Transfer Fee | Payments & Transfers |
| 8 | Transfer Delay | Payments & Transfers |
| 9 | Missing Money | Account Operations — review urgently |

---

## Project Structure

```
payment-intent-classifier/
│
├── data/
│   ├── ds_task_dataset.csv       # raw labelled queries
│   ├── intent_queries.db         # SQLite database
│   ├── train.csv                 # processed training split
│   └── test.csv                  # processed test split
│
├── src/
│   ├── preprocess.py             # SQL ingestion, PySpark profiling, cleaning, split
│   ├── train.py                  # CV model comparison, MLP, hyperparameter tuning
│   ├── evaluate.py               # test set evaluation, plots, misclassified examples
│   └── predict.py                # inference module — load model, classify, return result
│
├── notebooks/
│   └── eda.py                    # exploratory analysis — 5 plots
│
├── outputs/
│   ├── plots/                    # all EDA and evaluation charts
│   ├── model.pkl                 # saved tuned model pipeline
│   └── metrics.json              # CV results and test metrics
│
├── app/
│   └── app.py                    # Streamlit UI
│
├── runtime.txt                   # Python version for Streamlit Cloud
├── requirements.txt
└── README.md
```

---

## Run It Locally

```bash
# 1. Clone the repo
git clone https://github.com/CCLeyton/payment-intent-classifier.git
cd payment-intent-classifier

# 2. Create and activate environment
conda create -n intent-classifier python=3.11 -y
conda activate intent-classifier

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run preprocessing (creates SQLite DB, train/test splits)
python src/preprocess.py

# 5. Train models and save the best one
python src/train.py

# 6. Evaluate on the test set
python src/evaluate.py

# 7. Test a single query from the terminal
python src/predict.py "why was I charged twice for the same transaction"

# 8. Launch the Streamlit app
streamlit run app/app.py
```

---

## Model Performance Detail

```
                          precision    recall  f1-score   support

         Card Fee Charge       0.93      0.91      0.92        46
Fraud / Disputed Payment       0.94      0.98      0.96        45
           Deposit Delay       1.00      1.00      1.00        44
  ATM Partial Withdrawal       1.00      0.89      0.94        44
  Cash Withdrawal Charge       0.86      1.00      0.92        43
   Duplicate Transaction       0.98      0.95      0.96        43
     Declined Withdrawal       0.95      0.98      0.97        43
            Transfer Fee       0.90      0.88      0.89        42
          Transfer Delay       0.93      0.93      0.93        42
           Missing Money       0.97      0.93      0.95        42

                accuracy                           0.95       434
               macro avg       0.95      0.94      0.94       434
            weighted avg       0.95      0.94      0.94       434
```

Transfer Fee is the weakest class at F1 = 0.89, driven by vocabulary overlap with Transfer Delay and Card Fee Charge. Analysis of misclassified examples confirms that most errors occur between semantically adjacent classes — the query language genuinely overlaps, and the same ambiguity would challenge a human agent. A production system would flag these low-confidence predictions for manual review rather than auto-routing.

---

## Scaling to Production

This pipeline is built on patterns that translate directly to production scale. Replacing SQLite with **Apache Hive** or a cloud data warehouse gives the same SQL interface over billions of rows. The PySpark session — currently running in local mode — is cluster-ready: pointing `SparkSession.master` at a **YARN** or **Databricks** cluster requires no code changes beyond configuration. The TF-IDF and classification steps could be replaced with **Spark MLlib** pipelines for distributed training on full transaction corpora.

For a VisaNet-scale deployment the architecture would look like: raw transaction metadata and contact centre logs ingested via **Kafka** into a **Hadoop HDFS** data lake, profiled and featurised using **Spark**, trained on a cluster, and served via a REST API with the Streamlit interface calling the endpoint rather than loading a local `.pkl` file.

---

## Dataset

Banking NLP intent dataset — 2,169 labelled customer queries across 10 payment-related intent classes. Classes are well-balanced (211–227 samples each), making weighted and macro F1 directly comparable.

---

*Built as part of a data science portfolio targeting client-facing analytics roles in financial services.*