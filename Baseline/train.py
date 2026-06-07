import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]


def load_and_prepare_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path, skipfooter=700, engine='python')
    print(df.shape)

    if 'user_id' in df.columns:
        df.drop('user_id', axis=1, inplace=True)
    df = df[(df['tx_count_30d'] >= 5) & (df['tx_count_30d'] <= 26)]
    bad_rows = df.loc[df['avg_tx_amount'] > 850].index
    df.drop(bad_rows, axis=0, inplace=True)
    bad_rows = df.loc[df['digital_activity_score'] > 0.7542490209948082].index
    df.drop(bad_rows, axis=0, inplace=True)
    df['avg_tx_amount'] = np.log10(df['avg_tx_amount'])


    target_cols = [f"product_{p}" for p in PRODUCTS]

    cols_to_scale = [col for col in df.columns
                       if col not in target_cols + ['has_child', 'is_salary_client']]

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    for col in ['has_child', 'is_salary_client']:
        df[col] = df[col].astype(int)

    features = [col for col in df.columns if col not in target_cols]
    X = df[features]
    y = df[target_cols]

    return (X, y, features, target_cols,
            scaler)


def train_models(X, y, target_cols):
    models = {}

    for col in target_cols:
        print(f"Обучение LR {col}")
        model = LogisticRegression(
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.8,
            C=0.01,
            max_iter=150,
            random_state=42,
            class_weight=None
        )
        model.fit(X, y[col])
        models[col] = model


    return models


if __name__ == "__main__":
    (X, y, features, target_cols, scaler) = load_and_prepare_data()

    models = train_models(X, y, target_cols)

    model_pack = {
        "feature_columns": features,
        "target_columns": target_cols,
        "models": models,
        "scaler": scaler
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
    print(f"\nМодель сохранена в {output_path}")