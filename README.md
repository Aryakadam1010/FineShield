# FinShield — AI-Powered Fraud Detection System

> Built to detect fraudulent financial transactions in real time using machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it does

FinShield takes a financial transaction (amount, time, behavioral features) and predicts in real time whether it is **legitimate or fraudulent**, along with a fraud probability score and human-readable risk factors.

This mirrors the core problem solved by platforms like IDfy's **OneRisk** — proactive fraud prevention at scale.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend API | Flask (Python) | Lightweight, perfect for ML model serving |
| ML Model | Random Forest + Logistic Regression | Compared both; RF captures non-linear fraud patterns better |
| Imbalance handling | SMOTE | Real fraud data is 0.17% fraud — SMOTE creates synthetic fraud samples for balanced training |
| Frontend | HTML + CSS + Vanilla JS | No framework overhead; fast and portable |
| Model persistence | Pickle | Standard way to serialize trained sklearn models |

---

## Key ML Decisions

### Why SMOTE?
The Kaggle Credit Card Fraud dataset has **284,807 transactions — only 492 are fraud (0.17%)**. A naive model that always predicts "not fraud" achieves 99.8% accuracy but is completely useless. SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud samples in the training set so the model actually learns to distinguish fraud.

> ⚠️ SMOTE is applied **only on training data**, never on test data. Applying it to test data would be data leakage — your model would appear to perform better than it actually is on real-world data.

### Why Random Forest over Logistic Regression?
Fraud patterns are non-linear and involve **combinations** of conditions:
- High amount **AND** late night **AND** unusual merchant → fraud
- High amount alone → might be legit (a laptop purchase)

Logistic Regression assumes linear relationships between features and outcome. Random Forest builds 100 decision trees, each capturing different combinations. The ensemble is robust to outliers and handles non-linear patterns without feature engineering.

### Why F1-score, not Accuracy?
- **False negative** (missing actual fraud) → customer loses money
- **False positive** (blocking legit transaction) → customer frustrated

F1-score is the harmonic mean of precision and recall, penalizing both failure modes. Accuracy is misleading on imbalanced datasets.

---

## Project Structure

```
finshield/
├── app.py              # Flask server — routes, API endpoints, prediction logic
├── requirements.txt    # Python dependencies
├── model/
│   ├── train.py        # Full training pipeline with comments explaining each decision
│   ├── fraud_model.pkl # Trained model (auto-generated)
│   └── scaler.pkl      # StandardScaler fitted on training data
└── templates/
    └── index.html      # Frontend UI — transaction form + live results
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Aryakadam1010/FinShield.git
cd FinShield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates fraud_model.pkl and scaler.pkl)
python model/train.py

# 4. Start the Flask server
python app.py

# 5. Open http://localhost:5000 in your browser
```

---

## Model Performance

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.94 | 0.89 | 0.91 |
| **Random Forest** | **0.97** | **0.93** | **0.95** |

*Trained on synthetic data mirroring Kaggle Credit Card Fraud dataset distribution.*

---

## API Reference

**POST** `/predict`

```json
// Request
{
  "amount": 4200,
  "hour": 2,
  "v1": -3.5,
  "v2": 2.8,
  "v3": -4.2,
  "v4": 1.9,
  "v5": -2.6
}

// Response
{
  "prediction": "FRAUD",
  "fraud_probability": 0.91,
  "fraud_probability_pct": "91.0%",
  "risk_level": "HIGH",
  "risk_factors": [
    "Unusual transaction time (2:00 AM)",
    "High transaction amount (₹4200)",
    "Anomalous spending pattern (V1)"
  ]
}
```

---

## What I Learned

- Handling class imbalance is often more impactful than model choice
- The scaler fitted on training data must be reused at inference — not re-fitted
- REST API design: keeping the prediction endpoint stateless and fast
- The cost asymmetry of false positives vs false negatives in fraud detection

---

## About

Built by **Arya Kadam** — B.E. AI & Data Science, D.Y. Patil College of Engineering, Akurdi  
[GitHub](https://github.com/Aryakadam1010) · [LinkedIn](https://linkedin.com/in/arya-kadam)

Inspired by real-world fraud prevention systems like IDfy OneRisk, Razorpay Shield, and Stripe Radar.
