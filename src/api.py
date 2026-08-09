from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

# Load default model (XGBoost for demo)
model = joblib.load(os.path.join("models", "xgb_model.pkl"))
feature_names = joblib.load(os.path.join("models", "feature_names.pkl"))

@app.post("/score")
def score_transaction(transaction: dict):
    # Build dataframe with correct features
    input_row = [0] * len(feature_names)
    input_dict = dict(zip(feature_names, input_row))
    input_dict["Time"] = transaction.get("Time", 0)
    input_dict["Amount"] = transaction.get("Amount", 0)
    df = pd.DataFrame([input_dict], columns=feature_names)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "sender_id": transaction.get("sender_id"),
        "receiver_id": transaction.get("receiver_id"),
        "prediction": "Fraud" if pred == 1 else "Not Fraud",
        "probability": float(prob)
    }


