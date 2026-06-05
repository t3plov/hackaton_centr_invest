import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse

def build_features(df):
    df["tx_per_tenure"] = df["tx_count_30d"] / (df["tenure_months"] + 1)
    df["avg_monthly_spend"] = df["tx_count_30d"] * df["avg_tx_amount"]
    df["spend_per_tenure"] = df["avg_monthly_spend"] / (df["tenure_months"] + 1)
    df["digital_intensity"] = df["digital_activity_score"] * df["tx_per_tenure"]
    df["spend_per_income"] = df["avg_monthly_spend"] / (df["income_bucket"] + 1)
    df["activity_per_income"] = df["tx_per_tenure"] / (df["income_bucket"] + 1)
    df["log_tx"] = np.log1p(df["tx_count_30d"])
    df["log_income"] = np.log1p(df["income_bucket"])
    df["behavior_index"] = df["log_tx"] * df["digital_activity_score"]
    df["log_avg_tx_amount"] = np.log1p(df["avg_tx_amount"])
    df['age_bin'] = pd.cut(
        df['age'],
        bins=[18, 25, 35, 45, 55, 70],
        labels=[
            '18_25',
            '26_35',
            '36_45',
            '46_55',
            '56_70'
        ],
        include_lowest=True
    )
    age_dummies = pd.get_dummies(
        df['age_bin'],
        prefix='age',
        dtype=int
    )
    df = pd.concat([df, age_dummies], axis=1)
    df = df.drop(columns=['age', 'age_bin'])
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

    X = test_data[feat_cols].copy()

    proba_list = []
    for col, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        proba_list.append(proba)

    result = np.column_stack(proba_list)
    result = pd.DataFrame(result)
    result.to_csv(args.output_path, header=False, index=False)
