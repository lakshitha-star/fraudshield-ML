import streamlit as st
import pandas as pd
import joblib
import os

st.title("FraudShield Dashboard")

# Model selection
model_choice = st.selectbox("Choose model", ["Logistic Regression", "Random Forest", "XGBoost"])
model_map = {
    "Logistic Regression": "log_reg_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# Load feature names
feature_names = joblib.load(os.path.join("models", "feature_names.pkl"))

# Load model
model_path = os.path.join("models", model_map[model_choice])
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file {model_map[model_choice]} not found. Please train first.")
    st.stop()

# Load scaler for Logistic Regression
scaler = None
if model_choice == "Logistic Regression":
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))

st.subheader("Interactive Fraud Prediction")

# Transaction details
sender = st.text_input("Sender ID")
receiver = st.text_input("Receiver ID")
txn_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM", "Bank Transfer"])
device = st.selectbox("Device Used", ["Mobile", "Desktop", "Unknown"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
time = st.number_input("Transaction Time", min_value=0.0, value=1000.0)

# Build input row with correct columns
input_row = [0] * len(feature_names)
input_dict = dict(zip(feature_names, input_row))
input_dict["Time"] = time
input_dict["Amount"] = amount
input_data = pd.DataFrame([input_dict], columns=feature_names)

if st.button("Predict"):
    if not sender or not receiver:
        st.error("⚠️ Sender ID and Receiver ID are required!")
    else:
        if model_choice == "Logistic Regression" and scaler is not None:
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]
            probability = model.predict_proba(input_data_scaled)[0][1]
        else:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        # Unified JSON output
        st.json({
            "sender_id": sender,
            "receiver_id": receiver,
            "transaction_type": txn_type,
            "device": device,
            "amount": amount,
            "prediction": "Fraud" if prediction == 1 else "Not Fraud",
            "probability": float(probability)
        })
