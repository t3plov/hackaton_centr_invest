import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]


def create_features(df):
    df = df.copy()

    # Взаимодействия числовых признаков
    df['tx_per_month'] = df['tx_count_30d'] / (df['tenure_months'] + 1)
    df['total_tx_volume'] = df['tx_count_30d'] * df['avg_tx_amount']
    df['age_income'] = df['age'] * df['income_bucket']
    df['digital_tx'] = df['digital_activity_score'] * df['tx_count_30d']
    df['salary_premium'] = df['is_salary_client'] * df['income_bucket']
    df['child_income'] = df['has_child'] * df['income_bucket']

    # Бинарные флаги
    df['is_young_active'] = ((df['age'] < 30) & (df['digital_activity_score'] > 5)).astype(int)
    df['is_high_income'] = (df['income_bucket'] >= 6).astype(int)
    df['is_loyal'] = (df['tenure_months'] > 60).astype(int)

    # Возрастные группы
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 50, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='test data path')
    parser.add_argument('--output-path', type=str, help='output file')
    args = parser.parse_args()

    test_data = pd.read_csv(args.input_path)

    test_data = create_features(test_data)

    pack = joblib.load(Path("baseline_model.joblib"))
    scaler = pack["scaler"]
    feat_cols = pack["feature_columns"]
    models = pack["models"]

    X = test_data[feat_cols].values
    X_scaled = scaler.transform(X)

    y_proba = np.column_stack([
        models[p].predict_proba(X_scaled)[:, 1] for p in PRODUCTS
    ])

    result = pd.DataFrame(y_proba)
    result.to_csv(args.output_path, header=False, index=False)