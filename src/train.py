import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

    print("Loading dataset from:", data_path)
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)

    # Features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Save feature names for later use
    feature_names = X.columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure models folder exists
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save feature names
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))

    mlflow.set_experiment("fraudshield_experiment")

    # Logistic Regression
    print("\n=== Training Logistic Regression ===")
    with mlflow.start_run(run_name="LogisticRegression"):
        log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
        log_reg.fit(X_train_scaled, y_train)
        preds = log_reg.predict(X_test_scaled)
        auc = roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1])
        print("Logistic Regression AUC:", auc)
        print(classification_report(y_test, preds))
        mlflow.log_metric("AUC", auc)
        mlflow.sklearn.log_model(log_reg, "log_reg_model")
        joblib.dump(log_reg, os.path.join(models_dir, "log_reg_model.pkl"))
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # Random Forest
    print("\n=== Training Random Forest ===")
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        print("Random Forest AUC:", auc)
        print(classification_report(y_test, preds))
        mlflow.log_metric("AUC", auc)
        mlflow.sklearn.log_model(rf, "rf_model")
        joblib.dump(rf, os.path.join(models_dir, "rf_model.pkl"))

    # XGBoost
    print("\n=== Training XGBoost ===")
    with mlflow.start_run(run_name="XGBoost"):
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            use_label_encoder=False,
            eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        print("XGBoost AUC:", auc)
        print(classification_report(y_test, preds))
        mlflow.log_metric("AUC", auc)
        mlflow.xgboost.log_model(xgb, "xgb_model")
        joblib.dump(xgb, os.path.join(models_dir, "xgb_model.pkl"))

if __name__ == "__main__":
    main()

