from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def get_ml_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric="logloss")
    }

