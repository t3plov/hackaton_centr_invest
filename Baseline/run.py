import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
from sklearn.preprocessing import OneHotEncoder


def build_features(df, pack):
    """Воспроизводит ВСЕ шаги предобработки из train.py"""

    if 'user_id' in df.columns:
        df = df.drop('user_id', axis=1)

    df['avg_tx_amount'] = np.log10(df['avg_tx_amount'])


    for col in ['has_child', 'is_salary_client']:
        df[col] = df[col].astype(int)

    scaler = pack["scaler"]
    features = pack["feature_columns"]

    X = df[features]

    cols_to_scale = [col for col in features if col not in
                     ['has_child', 'is_salary_client']]
    X[cols_to_scale] = scaler.transform(df[cols_to_scale])


    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    # Загружаем данные и модель
    test_data = pd.read_csv(args.input_path)
    pack = joblib.load(Path("baseline_model.joblib"))

    X = build_features(test_data, pack)

    proba_list = []
    for col in pack["target_columns"]:
        model = pack["models"][col]
        proba = model.predict_proba(X)[:, 1]
        proba_list.append(proba)

    result = np.column_stack(proba_list)
    result = pd.DataFrame(result)
    result.to_csv(args.output_path, header=False, index=False)

    print(f"Предсказания сохранены в {args.output_path}")