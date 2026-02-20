#  Enterprise Customer Lifetime Value (CLV) Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)](https://scikit-learn.org/)
[![Lifetimes](https://img.shields.io/badge/Lifetimes-BTYD-success.svg)](#)

## Executive Summary & Business Problem
In e-commerce, predicting future customer spend is a notoriously difficult machine learning challenge. Traditional regression fails because retail data is **zero-inflated** (many customers never return) and highly skewed. 

This repository implements a production-grade **Hybrid Probabilistic-Machine Learning Pipeline** to predict 90-day Customer Lifetime Value (CLV) based on historical transaction logs. By accurately forecasting future spend, businesses can optimize customer acquisition costs (CAC) and aggressively target high-value retention campaigns.

## Technical Methodology
To handle the heavy-tailed, zero-inflated nature of spend data, this pipeline avoids standard linear regression in favor of a hybrid approach:

1. **BTYD Probabilistic Modeling:** Utilizes the `Lifetimes` library to fit Beta-Geometric/Negative Binomial (BG/NBD) and Gamma-Gamma models. This extracts deep behavioral features such as *Probability Alive* and *Conditional Expected Value*.
2. **Feature Engineering:** Combines probabilistic outputs with standard RFM (Recency, Frequency, Monetary) metrics and Interpurchase Time Standard Deviation.
3. **Advanced Regression (Tweedie Loss):** Evaluates a Model Zoo, specifically leveraging Tweedie objective functions to mathematically model sparse, non-negative continuous spend data.

## Repository Architecture
Structured adhering to strict MLOps and Separation of Concerns principles:

```text
clv-prediction-engine/
│
├── data/                      # Local data storage (Git-ignored)
├── notebooks/
│   └── main_execution.ipynb   # Colab control center & pipeline orchestration
├── src/                       # Modularized core logic
│   ├── __init__.py
│   ├── config.py              # Centralized hyperparameters and I/O paths
│   ├── data_ingestion.py      # Automated fetching and preprocessing
│   ├── feature_engineering.py # BTYD and RFM feature generation
│   ├── modeling.py            # Model zoo, training, and GridSearch tuning
│   └── evaluation.py          # Business metric calculations and artifact saving
├── artifacts/                 # Serialized champion models and EDA plots (Git-ignored)
├── .gitignore                 # Version control exclusions
└── README.md                  # Project documentation


📊 Performance & Results
Models were evaluated on a strict out-of-time test set. The Random Forest Regressor emerged as the champion model, capturing complex non-linear relationships and outperforming the Naive Baseline (historical average) by a massive margin.

Model                   ,RMSE ($),MAE ($),R² Score
Random Forest (Champion),1057.78 ,593.69 ,0.640
Linear Regression       ,1153.79 ,639.03 ,0.572
Ridge (L2)              ,1159.00 ,640.82 ,0.568
ElasticNet              ,1240.52 ,671.10 ,0.505
Tweedie Regressor       ,1271.45 ,661.78 ,0.480
Naive Baseline          ,1732.31 ,792.59 ,0.036

Final Business Metric: WAPE (Weighted Absolute Percentage Error): 63.16%

How to Run the Pipeline
This pipeline is designed to be executed via Google Colab with Google Drive integration for remote execution.

Clone this repository to your local machine.

Upload the clv-prediction-engine folder to the root of your Google Drive.

Open notebooks/main_execution.ipynb in Google Colab.

Execute the notebook to automatically mount your drive, fetch the dataset, train the models, and serialize the champion pipeline to the artifacts/ directory.