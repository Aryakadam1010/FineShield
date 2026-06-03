"""
FinShield - Flask Web Application
===================================
This is the backend server. It does two things:
1. Serves the frontend HTML page (the UI the user sees)
2. Exposes a /predict API endpoint that takes transaction data and returns fraud probability

WHY FLASK?
Flask is a micro web framework — "micro" means it gives you the basics without forcing
a structure on you. For a small project like this it's perfect. Django would be overkill.

ROUTE EXPLANATION:
- @app.route('/') -> when someone visits the homepage, run this function
- @app.route('/predict', methods=['POST']) -> when the frontend sends transaction data,
  run this function, process it, return JSON with the prediction
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load model and scaler once at startup
# ─────────────────────────────────────────────
# We load these once when the server starts, not on every request.
# Loading a pickle file on every API call would be very slow.

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'fraud_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print("Model and scaler loaded successfully.")


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint: receives transaction data, returns fraud prediction.
    
    INPUT (JSON):
    {
        "amount": 250.00,
        "hour": 23,
        "v1": -2.3,
        "v2": 1.8,
        "v3": -3.1,
        "v4": 1.2,
        "v5": -1.7
    }
    
    OUTPUT (JSON):
    {
        "prediction": "FRAUD" or "LEGITIMATE",
        "fraud_probability": 0.87,
        "risk_level": "HIGH" / "MEDIUM" / "LOW",
        "risk_factors": [...]
    }
    
    WHY JSON?
    JSON (JavaScript Object Notation) is the standard format for web APIs.
    The frontend sends a fetch() request with JSON, we process it, send JSON back.
    This is called a REST API pattern.
    """
    try:
        data = request.get_json()

        # Extract and validate inputs
        amount = float(data.get('amount', 0))
        hour = int(data.get('hour', 12))
        v1 = float(data.get('v1', 0))
        v2 = float(data.get('v2', 0))
        v3 = float(data.get('v3', 0))
        v4 = float(data.get('v4', 0))
        v5 = float(data.get('v5', 0))

        # Scale amount and hour the same way we did during training
        # CRITICAL: we must use the SAME scaler fitted on training data
        # If we re-fit the scaler here, the model would get different numbers than it was trained on
        scaled_vals = scaler.transform(pd.DataFrame([[amount, hour]], columns=['amount','hour']))[0]
        amount_scaled = scaled_vals[0]
        hour_scaled = scaled_vals[1]

        # Build feature array in the same order as training
        features = pd.DataFrame([[amount_scaled, hour_scaled, v1, v2, v3, v4, v5]], columns=['amount','hour','v1','v2','v3','v4','v5'])

        # Get prediction and probability
        prediction = model.predict(features)[0]
        # predict_proba returns [P(legit), P(fraud)]
        fraud_prob = model.predict_proba(features)[0][1]

        # Determine risk level from probability
        if fraud_prob >= 0.7:
            risk_level = "HIGH"
        elif fraud_prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Generate human-readable risk factors for the UI
        risk_factors = []
        if hour < 6:
            risk_factors.append(f"Unusual transaction time ({hour}:00 AM)")
        if amount > 500:
            risk_factors.append(f"High transaction amount (₹{amount:.0f})")
        if v1 < -2:
            risk_factors.append("Anomalous spending pattern (V1)")
        if v3 < -3:
            risk_factors.append("Unusual merchant category pattern (V3)")
        if not risk_factors and prediction == 0:
            risk_factors.append("All patterns within normal range")

        return jsonify({
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': round(float(fraud_prob), 4),
            'fraud_probability_pct': f"{fraud_prob * 100:.1f}%",
            'risk_level': risk_level,
            'risk_factors': risk_factors
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # debug=True means Flask auto-reloads when you change code — great for development
    # In production you'd use gunicorn or similar, never Flask's dev server
    app.run(debug=True, port=5000)
