import pandas as pd

def add_features(df):
    df["transaction_velocity"] = df.groupby("user_id")["amount"].transform("count")
    df["time_since_last"] = df.groupby("user_id")["time"].diff().fillna(0)
    df["merchant_risk"] = df.groupby("merchant_id")["fraud"].transform("mean").fillna(0)
    return df

