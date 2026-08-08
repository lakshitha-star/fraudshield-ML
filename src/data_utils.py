import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=5000, fraud_ratio=0.01, random_state=42):
    np.random.seed(random_state)

    # Basic fields
    user_ids = np.random.randint(1, 500, size=n_samples)
    merchant_ids = np.random.randint(1, 100, size=n_samples)
    amounts = np.random.exponential(scale=100, size=n_samples).round(2)
    times = np.arange(n_samples)  # simple sequential time

    # Fraud labels (rare events)
    fraud = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, int(n_samples * fraud_ratio), replace=False)
    fraud[fraud_indices] = 1

    df = pd.DataFrame({
        "user_id": user_ids,
        "merchant_id": merchant_ids,
        "amount": amounts,
        "time": times,
        "fraud": fraud.astype(int)
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/raw/synthetic.csv", index=False)
    print("Synthetic dataset saved to data/raw/synthetic.csv")

