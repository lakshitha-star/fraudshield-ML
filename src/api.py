from fastapi import FastAPI
import pandas as pd
from src.models.ml_models import get_ml_models

app = FastAPI()
models = get_ml_models()

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    model = models["RandomForest"]
    pred = model.predict(df)
    return {"fraud_prediction": int(pred[0])}


