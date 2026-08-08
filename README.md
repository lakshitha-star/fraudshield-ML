# FraudShield

An Explainable Hybrid ML/DL Framework for Real-Time Online Transaction Anomaly Detection.

## Features
- Preprocessing (scaling, encoding, missing values)
- Imbalance handling (SMOTE)
- Feature engineering (velocity, merchant risk, time since last)
- ML models (LR, RF, XGBoost)
- DL models (Autoencoder, LSTM)
- GNN skeleton
- Evaluation (Precision, Recall, F1, AUPRC)
- Explainability (SHAP)
- Streamlit dashboard
- FastAPI scoring service
- MLflow experiment tracking

## Run
```bash
pip install -r requirements.txt
python -m src.train
streamlit run src/app.py
uvicorn src.api:app --reload --port 8000

