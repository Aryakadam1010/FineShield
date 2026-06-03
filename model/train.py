import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle

# ─────────────────────────────────────────────
# STEP 1: Create fake transaction data
# ─────────────────────────────────────────────

# Legitimate transactions — normal behaviour
legit_data = {
    'amount': np.random.exponential(scale=80,  size=5000),  # small amounts
    'hour':   np.random.randint(6, 23,          size=5000),  # daytime
    'v1':     np.random.normal(0, 1,            size=5000),  # normal pattern
    'v2':     np.random.normal(0, 1,            size=5000),
    'v3':     np.random.normal(0, 1,            size=5000),
    'v4':     np.random.normal(0, 1,            size=5000),
    'v5':     np.random.normal(0, 1,            size=5000),
    'is_fraud': np.zeros(5000)                               # label = 0 = not fraud
}

# Fraudulent transactions — suspicious behaviour
fraud_data = {
    'amount': np.random.exponential(scale=200, size=100),   # high amounts
    'hour':   np.random.randint(0, 6,          size=100),   # late night
    'v1':     np.random.normal(-3, 1,          size=100),   # extreme patterns
    'v2':     np.random.normal( 2, 1,          size=100),
    'v3':     np.random.normal(-4, 1,          size=100),
    'v4':     np.random.normal( 1, 1,          size=100),
    'v5':     np.random.normal(-2, 1,          size=100),
    'is_fraud': np.ones(100)                                 # label = 1 = fraud
}

# Convert dictionaries to DataFrames (like Excel tables)
df_legit = pd.DataFrame(legit_data)
df_fraud = pd.DataFrame(fraud_data)

# Join both tables into one and shuffle the rows
df = pd.concat([df_legit, df_fraud])
df = df.sample(frac=1, random_state=42)  # shuffle

print("Total transactions:", len(df))
print("Fraud transactions:", int(df['is_fraud'].sum()))
print("Legit transactions:", int((df['is_fraud'] == 0).sum()))
print()

# ─────────────────────────────────────────────
# STEP 2: Separate inputs and output
# ─────────────────────────────────────────────

# X = what we give the model to look at (the inputs)
# y = what we want the model to predict (fraud or not)
X = df[['amount', 'hour', 'v1', 'v2', 'v3', 'v4', 'v5']]
y = df['is_fraud']

# ─────────────────────────────────────────────
# STEP 3: Scale the numbers
# ─────────────────────────────────────────────
# amount can be 5000. hour is 0-23. Very different scales.
# StandardScaler brings everything to the same scale (mean=0, std=1)
# so the model treats them fairly.

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['amount', 'hour']] = scaler.fit_transform(X[['amount', 'hour']])

# ─────────────────────────────────────────────
# STEP 4: Split into train and test
# ─────────────────────────────────────────────
# Train set = data the model learns from (80%)
# Test set  = data we use to check how good it is (20%)
# We never test on data the model already saw — that would be cheating

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 20% goes to testing
    random_state=42     # same split every time
)

print("Training samples:", len(X_train))
print("Testing samples: ", len(X_test))
print()

# ─────────────────────────────────────────────
# STEP 5: Fix the imbalance using SMOTE
# ─────────────────────────────────────────────
# Problem: we have 5000 legit but only 100 fraud
# The model will just always say "not fraud" and be 98% accurate — useless
# SMOTE creates new fake fraud samples so both classes are equal

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print("  Fraud samples:", int(y_train.sum()))
print("  Legit samples:", int((y_train == 0).sum()))
print()

# ─────────────────────────────────────────────
# STEP 6: Train two models and compare
# ─────────────────────────────────────────────

# Model 1: Logistic Regression — simple, draws a straight line
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_f1   = f1_score(y_test, lr_pred)

print("Logistic Regression F1 Score:", round(lr_f1, 4))
print(classification_report(y_test, lr_pred, target_names=['Legit', 'Fraud']))

# Model 2: Random Forest — 100 decision trees voting together
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_f1   = f1_score(y_test, rf_pred)

print("Random Forest F1 Score:", round(rf_f1, 4))
print(classification_report(y_test, rf_pred, target_names=['Legit', 'Fraud']))

# ─────────────────────────────────────────────
# STEP 7: Save the better model
# ─────────────────────────────────────────────
# pickle saves a Python object to a file
# so Flask can load it later without retraining

if rf_f1 >= lr_f1:
    best_model = rf_model
    print("Winner: Random Forest")
else:
    best_model = lr_model
    print("Winner: Logistic Regression")

with open('model/fraud_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print()
print("Model saved to model/fraud_model.pkl")
print("Scaler saved to model/scaler.pkl")
print("Done! Now run: python app.py")