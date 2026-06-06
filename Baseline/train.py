import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]

def load_and_prepare_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

    df['das_income_bucket'] = df['digital_activity_score'] ** (df['income_bucket'] + 1)

    is_bad_rows = df.loc[df['avg_tx_amount'] > 500].index
    df.drop(is_bad_rows, axis=0, inplace=True)
    df['avg_tx_amount'] = np.log(df['avg_tx_amount'])

    target_cols = [f"product_{p}" for p in PRODUCTS]
    feature_cols = [c for c in df.columns if c not in ["user_id"] + target_cols]

    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[[col for col in X.columns if col not in ['has_child', 'is_salary_client']]])
    X_binary = X[['has_child', 'is_salary_client']].values
    X_scaled = np.hstack([X_scaled, X_binary])

    return X_scaled, y, feature_cols, target_cols


def train_models(X, y):
    models = {}

    for col in y.columns:
        model = LogisticRegression(
            solver='saga',
            l1_ratio=0.8,
            C=10.0,
            max_iter=150,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X, y[col])

        models[col] = model

    return models

if __name__ == "__main__":
    X, y, feat_cols, target_cols = load_and_prepare_data()

    models = train_models(X, y)

    model_pack = {
        "feature_columns": feat_cols,
        "model": models,
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
