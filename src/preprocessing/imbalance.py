from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

