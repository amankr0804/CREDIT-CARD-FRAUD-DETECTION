## Credit Card Fraud Detection Using Anomaly Detection Techniques

An end-to-end machine learning project that detects fraudulent credit card transactions, compares supervised and unsupervised models, and serves predictions through a web API with an interactive Transaction Risk Dashboard.

## Table of Contents

- [1. Overview](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-overview)

- [2. Key Features](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-key-features)

- [3. Tech Stack](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-tech-stack)

- [4. System Architecture](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-system-architecture)

- [5. Project Structure](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-project-structure)

- [6. Dataset](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-dataset)

- [7. Installation & Setup](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-installation--setup)

- [8. Machine Learning Pipeline](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-machine-learning-pipeline)

- [9. Model Comparison](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-model-comparison)

- [10. Evaluation Metrics](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-evaluation-metrics)

- [11. Database (SQL) Design](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-database-sql-design)

- 12. 13. 14. Dashboard

- [Results](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-results)

- [15. Future Improvements](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-future-improvements)

- [API Reference](https://claude.ai/chat/81dd428e-cda5-435f-83d8-b32bf4f70d38#-api-reference)

## Overview

Credit card fraud is rare but costly. In typical datasets, fewer than 0.2% of transactions are fraudulent, which makes this a highly imbalanced classification / anomaly detection problem where plain accuracy is misleading.

This project:

- Trains and compares four models: Logistic Regression, Random Forest, XGBoost, and Isolation Forest.

- Handles class imbalance using class weights, SMOTE, and threshold tuning.

- Evaluates with metrics that matter for fraud: Precision, Recall, F1-score, ROC- AUC, PR-AUC, and Confusion Matrix.

- Outputs a fraud probability and risk level for every transaction.

- Stores transactions and predictions in a SQL database.

- Exposes a Flask/FastAPI backend and a HTML/CSS/JavaScript dashboard with Chart.js/Plotly visualizations.


## Key Features

## Feature

Multi-model

comparison

Imbalance handling

Rich evaluation

Fraud probability

Risk classification

REST API

SQL storage

Interactive dashboard Live charts, filters, model comparison, risk table

Explainability

Reproducibility

## Description

Logistic Regression vs Random Forest vs XGBoost vs Isolation Forest

class_weight, scale_pos_weight, SMOTE, undersampling

Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix

Per-transaction probability score (0–1)

Low / Medium / High / Critical risk bands

Single and batch prediction endpoints

Transaction history, predictions, and model runs

Feature importance (and optional SHAP values)

Fixed seeds, saved models, requirements.txt

## Tech Stack

## Language & Data

- Python 3.9+

- Pandas, NumPy

## Machine Learning

- Scikit-learn (Logistic Regression, Random Forest, Isolation Forest, metrics, preprocessing)

- XGBoost

- imbalanced-learn (SMOTE)

- Joblib (model persistence)

## Backend

- Flask or FastAPI (choose one)

- SQLAlchemy + SQLite / PostgreSQL / MySQL

## Frontend

- HTML5, CSS3, JavaScript

- Chart.js and/or Plotly.js

## Database

- SQL (SQLite for development, PostgreSQL/MySQL for production)


## System Architecture

## Dataset

## [Source: Credit Card Fraud Detection – Kaggle (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Property Transactions 284,807 Fraudulent 492 (≈ 0.172%)

Period

Features Time, V1–V28 (PCA-transformed), Amount

Target

Class (0 = legitimate, 1 = fraud)

Features V1–V28 are anonymized PCA components. Only Time and Amount are in their

original form

Download

## Value

2 days (European cardholders, September 2013)

creditcard.csv

and place it in

data/raw/.


## Project Structure

credit-card-fraud-detection/

│

├── data/

│ ├── raw/

│ └── processed/

│

├── notebooks/

│ ├── 01_eda.ipynb │ ├── 02_preprocessing.ipynb │ └── 03_model_comparison.ipynb

│

├── src/

│ ├── data_preprocessing.py # Cleaning, scaling, splitting, SMOTE

│ ├── train.py

│ ├── evaluate.py

│ ├── predict.py

│ └── utils.py

│

├── models/

│ ├── logistic_regression.pkl

│ ├── random_forest.pkl

│ ├── xgboost.pkl

│ ├── isolation_forest.pkl

│ └── scaler.pkl

│

├── app/

│ ├── main.py

│ ├── routes.py

│ ├── database.py

│ ├── schemas.py

│ ├── templates/

│

│ └── static/

│

│

│

├── sql/

│ ├── schema.sql

│ └── queries.sql

│

├── reports/

│ └── figures/

│

├── tests/

│ └── test_api.py

│

├── requirements.txt ├── .gitignore

└── README.md

\# Original dataset (creditcard.csv)

\# Cleaned / split data

\# Exploratory data analysis

│ └── dashboard.html

├── css/style.css └── js/dashboard.js

\# Train all models # Metrics, plots, comparison table

\# Inference + risk scoring

\# Flask / FastAPI entry point

\# DB connection & ORM models

\# Request/response schemas (FastAPI)

\# Chart.js / Plotly logic

\# Table definitions

\# Analytical queries

\# ROC curves, confusion matrices, etc.


## Installation & Setup

## 1. Clone the repository

```
git clone https://github.com/<your-username>/credit-card-fraud-
detection.git
cd credit-card-fraud-detection
```

## 2. Create a virtual environment

```
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

## 3. Install dependencies

```
pip install -r requirements.txt
```

## requirements.txt

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
matplotlib
seaborn
plotly
shap
flask # or: fastapi + uvicorn
sqlalchemy
python-dotenv
pytest
```

## 4. Add the dataset

```
mkdir -p data/raw
# Place creditcard.csv inside data/raw/
```

## 5. Initialize the database

```
sqlite3 fraud.db < sql/schema.sql
```

## 6. Train the models

python src/train.py


## 7. Evaluate and compare models

python src/evaluate.py

## 8. Run the application

## Flask

```
python app/main.py
```

## FastAPI

```
uvicorn app.main:app --reload
```

Open http://localhost:5000 (Flask) or http://localhost:8000 (FastAPI) in your browser.

## Machine Learning Pipeline

## 1. Exploratory Data Analysis

- Class distribution and imbalance ratio

- Distribution of Amount and Time for fraud vs. legitimate

- Correlation heatmap of features

- Outlier analysis

## 2. Preprocessing

- Scale Amount and Time (StandardScaler / RobustScaler)

- Remove duplicates, check for missing values

- Stratified train/test split (e.g., 80/20) so fraud appears in both sets

## 3. Handling Class Imbalance

| Technique | Used With |
| --- | --- |
| class_weight='balanced' Logistic Regression, Random Forest |   |
| scale_pos_weight | XGBoost |
| SMOTE (training set only) Optional for all supervised models |   |
| Random undersampling | Optional baseline |
| Threshold tuning | All models |


Apply SMOTE only to the training set after the split, to avoid data leakage.

## 4. Model Training

- Hyperparameter tuning with GridSearchCV / RandomizedSearchCV

- Stratified K-Fold cross-validation

- Scoring on PR-AUC / F1, not accuracy

## 5. Risk Scoring

Each transaction receives a fraud probability, mapped to a risk level:

## Fraud Probability Risk Level

| 0.00 – 0.30 | Low |
| --- | --- |
| 0.30 – 0.60 | Medium |
| 0.60 – 0.85 | High |
| 0.85 – 1.00 | Critical |

(Thresholds are configurable in src/predict.py.)

## Model Comparison

| Model | Type | Strengths | Limitations |
| --- | --- | --- | --- |
| Logistic |   |   | Struggles with non-linear |
|   |   | Supervised (linear) Fast, interpretable baseline |   |
| Regression |   |   | patterns |
|   |   | Robust, handles non- |   |
| Random | Supervised |   | Slower, larger memory |
|   |   | linearity, feature |   |
| Forest | (ensemble) |   | footprint |
|   |   | importance |   |
|   |   | Typically best |   |
|   | Supervised |   | More hyperparameters to |
| XGBoost |   | performance on tabular |   |
|   | (boosting) |   | tune |
|   |   | data |   |
| Isolation | Unsupervised | Needs no labels, detects | Lower precision, sensitive |
| Forest | (anomaly detection) | novel fraud patterns | to contamination |

Isolation Forest note: it outputs an anomaly score rather than a probability. Scores are normalized (e.g., min–max) to a 0–1 range so they can be compared and displayed alongside the supervised models. Set contamination close to the real fraud rate (≈ 0.0017).


## Evaluation Metrics

Accuracy is not used as the main metric, since a model predicting "not fraud" every time would score ~99.8%.

Metric

Precision

Recall

Harmonic mean of Precision and Recall

F1-Score

ROC-AUC

PR-AUC

Confusion

Shows TP, FP, TN, FN counts directly

Matrix

Fraud

Per-transaction risk score for ranking and alerting

Probability

## Why It Matters

Of the transactions flagged as fraud, how many really are? (Avoids false alarms)

Of all real frauds, how many were caught? (Avoids missed fraud)

Ability to separate fraud from legitimate across all thresholds

More informative than ROC-AUC on highly imbalanced data

```
from sklearn.metrics import (
precision_score, recall_score, f1_score,
roc_auc_score, average_precision_score, confusion_matrix
)
y_pred = (y_proba >= threshold).astype(int)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
```

## Generated visuals (saved to reports/figures/):

- ROC curves (all models on one plot)

- PrecisionRecall curves

- Confusion matrix heatmaps

- Feature importance charts

- Metrics comparison bar chart


## Database (SQL) Design

## sql/schema.sql

```
CREATE TABLE IF NOT EXISTS transactions (
transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
time_elapsed REAL,
amount REAL NOT NULL,
v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL, v6 REAL, v7 REAL,
v8 REAL, v9 REAL, v10 REAL, v11 REAL, v12 REAL, v13 REAL, v14 REAL,
v15 REAL, v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL, v21 REAL,
v22 REAL, v23 REAL, v24 REAL, v25 REAL, v26 REAL, v27 REAL, v28 REAL,
actual_class INTEGER, -- ground truth, if known
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
transaction_id INTEGER NOT NULL,
model_name TEXT NOT NULL,
fraud_probability REAL NOT NULL,
predicted_class INTEGER NOT NULL,
risk_level TEXT CHECK (risk_level IN
('Low','Medium','High','Critical')),
predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);
CREATE TABLE IF NOT EXISTS model_metrics (
run_id INTEGER PRIMARY KEY AUTOINCREMENT,
model_name TEXT NOT NULL,
precision_ REAL,
recall REAL,
f1_score REAL,
roc_auc REAL,
pr_auc REAL,
trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
Example analytical queries (sql/queries.sql)
-- Top 10 highest-risk transactions
SELECT t.transaction_id, t.amount, p.model_name, p.fraud_probability, p.risk_level
FROM predictions p
JOIN transactions t ON t.transaction_id = p.transaction_id
ORDER BY p.fraud_probability DESC
LIMIT 10;
-- Count of transactions per risk level
SELECT risk_level, COUNT(*) AS total
FROM predictions
GROUP BY risk_level;
-- Average fraud amount vs. legitimate amount
SELECT actual_class, ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY actual_class;
-- Latest metrics per model
SELECT * FROM model_metrics ORDER BY trained_at DESC;
```


## API Reference

Base URL: http://localhost:5000 (Flask) / http://localhost:8000 (FastAPI)

| Method | Endpoint | Description |
| --- | --- | --- |
| GET / |   | Serves the dashboard |
| POST | /api/predict | Predict fraud for a single transaction |
| POST | /api/predict/batch Predict fraud for multiple transactions (JSON/CSV) |   |
| GET | /api/metrics | Return evaluation metrics for all models |
| GET | /api/transactions | List stored transactions with risk scores |
| GET | /api/stats | Summary stats (totals, fraud rate, risk distribution) |
| GET | Health check /api/health |   |

## Example: Single Prediction

## Request

```
curl -X POST http://localhost:8000/api/predict \
-H "Content-Type: application/json" \
-d '{
"model": "xgboost",
"features": {
"Time": 406, "Amount": 149.62,
"V1": -1.36, "V2": -0.07, "...": "...", "V28": -0.02
}
}'
```

## Response

```
{
"model": "xgboost",
"fraud_probability": 0.9231,
"prediction": "Fraud",
"risk_level": "Critical"
}
```

FastAPI users get auto-generated interactive docs at /docs.


## Dashboard

The web dashboard (HTML/CSS/JavaScript + Chart.js/Plotly) provides:

- KPI cards: total transactions, flagged frauds, fraud rate, total amount at risk

- Model comparison chart: grouped bar chart of Precision, Recall, F1, ROC-AUC

- ROC & PrecisionRecall curves: interactive, all models overlaid

- Confusion matrix: heatmap with model selector dropdown

- Risk distribution: donut/pie chart of Low / Medium / High / Critical

- Transaction risk table: sortable and filterable, with fraud probability and color- coded risk badges

- Live prediction form: enter or upload a transaction and get an instant risk score

- Time series view: fraud activity over time

- Feature importance: top contributing features per model


## Results

Fill this table in after running python src/evaluate.py. Values depend on the random seed, split, and tuning.

| Model | Precision Recall F1-Score ROC-AUC PR-AUC |   |   |   |   |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression – |   | – | – | – | – |
| Random Forest | – | – | – | – | – |
| XGBoost | – | – | – | – | – |
| Isolation Forest | – | – | – | – | – |

Key takeaways (to be completed with your findings):

- Which model gives the best Recall/Precision trade-off?

- How did SMOTE / class weights affect performance?

- How does the unsupervised Isolation Forest compare to supervised models?

- What is the cost-optimal decision threshold?

## Future Improvements

- [ ] SHAP / LIME explainability for individual predictions

- [ ] Autoencoder / One-Class SVM / Local Outlier Factor comparison

- [ ] Cost-sensitive learning based on transaction amount

- [ ] Real-time streaming with Kafka

- [ ] Model monitoring and drift detection

- [ ] Docker & docker-compose deployment

- [ ] CI/CD with GitHub Actions

- [ ] User authentication for the dashboard

- [ ] Cloud deployment (AWS / GCP / Azure / Render)
