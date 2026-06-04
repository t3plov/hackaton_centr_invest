import pandas as pd
import sklearn
print("Sklearn version in run:", sklearn.__version__)
import joblib
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    test_data = pd.read_csv(args.input_path)

    pack = joblib.load(Path("baseline_model.joblib"))
    scaler = pack["scaler"]
    feat_cols = pack["feature_columns"]
    chain_model = pack["chain_model"]

    X = test_data[feat_cols].copy()

    for col in ['has_child', 'is_salary_client']:
        if col in X.columns: X[col] = X[col].astype(int)
    if 'income_bucket' in X.columns: X['income_bucket'] = X['income_bucket'].astype(int)

    cols_to_scale = ['age', 'tenure_months', 'tx_count_30d', 'avg_tx_amount', 'digital_activity_score']
    X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    y_proba = chain_model.predict_proba(X)

    result = pd.DataFrame(y_proba)
    result.to_csv(args.output_path, header=False, index=False)
