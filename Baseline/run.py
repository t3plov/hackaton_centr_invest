import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse

def build_features(df):
    df['das_income_bucket'] = df['digital_activity_score'] ** (df['income_bucket'] + 1)
    bins = [0, 50, 200, 500, float('inf')]
    labels = ['tx_low', 'tx_medium', 'tx_high', 'tx_very_high']
    df['avg_tx_amount_bin'] = pd.cut(df['avg_tx_amount'], bins=bins, labels=labels, right=False)
    df = pd.get_dummies(df, columns=['avg_tx_amount_bin'], drop_first=False)
    df.drop('avg_tx_amount', axis=1, inplace=True)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    test_data = pd.read_csv(args.input_path)
    test_data = build_features(test_data)

    pack = joblib.load(Path("baseline_model.joblib"))
    feat_cols = pack["feature_columns"]
    models = pack["model"]
    scaler = pack["scaler"]

    X = test_data[feat_cols].copy()
    for col in ['has_child', 'is_salary_client', 'avg_tx_amount_bin_tx_low', 'avg_tx_amount_bin_tx_medium',
                'avg_tx_amount_bin_tx_high', 'avg_tx_amount_bin_tx_very_high']:
        X[col] = X[col].astype(int)

    X_scaled = scaler.transform(X[[col for col in X.columns if col not in ['has_child', 'is_salary_client', 'avg_tx_amount_bin_tx_low', 'avg_tx_amount_bin_tx_medium',
           'avg_tx_amount_bin_tx_high', 'avg_tx_amount_bin_tx_very_high']]])
    X_binary = X[['has_child', 'is_salary_client', 'avg_tx_amount_bin_tx_low', 'avg_tx_amount_bin_tx_medium',
                  'avg_tx_amount_bin_tx_high', 'avg_tx_amount_bin_tx_very_high']].values
    X_scaled = np.hstack([X_scaled, X_binary])

    proba_list = []
    for col, model in models.items():
        if col in ['product_credit_card', 'product_cashback']:
            proba = model.predict_proba(X_scaled)[:, 1]
            proba_list.append(proba)
        else:
            proba = model.predict_proba(X)[:, 1]
            proba_list.append(proba)

    result = np.column_stack(proba_list)
    result = pd.DataFrame(result)
    result.to_csv(args.output_path, header=False, index=False)
