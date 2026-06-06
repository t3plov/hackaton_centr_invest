import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse

def build_features(df):
    df['das_income_bucket'] = df['digital_activity_score'] ** (df['income_bucket'] + 1)
    is_bad_rows = df.loc[df['avg_tx_amount'] > 500].index
    df.drop(is_bad_rows, axis=0, inplace=True)
    df['avg_tx_amount'] = np.log(df['avg_tx_amount'])

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
