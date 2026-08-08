from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

def evaluate_model(y_true, y_pred, y_prob=None):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_prob) if y_prob is not None else None
    }

