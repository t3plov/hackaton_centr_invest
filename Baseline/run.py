import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help='test data path')
    parser.add_argument('--output-path', type=str, required=True, help='output file')
    args = parser.parse_args()

    test_data = pd.read_csv(args.input_path)

    pack = joblib.load(Path("Baseline/baseline_model.joblib"))
    scaler = pack["scaler"]
    feat_cols = pack["feature_columns"]
    models = pack["models"]

    X = test_data[feat_cols].copy()

    for col in ['has_child', 'is_salary_client']:
        if col in X.columns:
            X[col] = X[col].astype(int)

    cols_to_scale = ['age', 'tenure_months', 'tx_count_30d', 'avg_tx_amount']
    X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    X_values = X.values

    y_proba = np.column_stack([
        models[p].predict_proba(X_values)[:, 1] for p in PRODUCTS
    ])

    result = pd.DataFrame(y_proba)
    result.to_csv(args.output_path, header=False, index=False)
