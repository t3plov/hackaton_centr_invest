import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]

BEST_PARAMS = {
    "credit_card": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    "mortgage": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    "deposit": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    "investment": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    "insurance": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    "p2p_transfer": {'C': 1.0, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    "cashback": {'C': 0.1, 'penalty': None, 'solver': 'lbfgs', 'max_iter': 300, 'l1_ratio': 0.5},
    "premium_account": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    "business_loan": {'C': 0.1, 'penalty': None, 'solver': 'sag', 'max_iter': 300, 'l1_ratio': 0.5}
}


def load_and_prepare_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

    target_cols = [f"product_{p}" for p in PRODUCTS]
    feature_cols = [c for c in df.columns if c not in ["user_id"] + target_cols]

    X = df[feature_cols].copy()
    y = df[target_cols].values

    for col in ['has_child', 'is_salary_client']:
        if col in X.columns:
            X[col] = X[col].astype(int)

    scaler = StandardScaler()
    X[['age', 'tenure_months', 'tx_count_30d', 'avg_tx_amount']] = scaler.fit_transform(X[['age', 'tenure_months', 'tx_count_30d', 'avg_tx_amount']])

    return X, y, scaler, feature_cols


def train_models(X, y):
    models = {}

    for idx, prod in enumerate(PRODUCTS):
        lr = LogisticRegression(**BEST_PARAMS[prod], random_state=42)
        lr.fit(X, y[:, idx])

        models[prod] = lr

    return models


if __name__ == "__main__":
    X, y, scaler, feat_cols = load_and_prepare_data()

    models = train_models(X, y)

    model_pack = {
        "scaler": scaler,
        "feature_columns": feat_cols,
        "models": models,
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
