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
    bad_rows = df.loc[df['avg_tx_amount'] > 1000].index
    df.drop(bad_rows, axis=0, inplace=True)
    bad_rows = df.loc[df['digital_activity_score'] > 0.8043961733674306].index
    df.drop(bad_rows, axis=0, inplace=True)

    df['avg_tx_amount'] = np.log1p(df['avg_tx_amount'])

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    income_encoded = encoder.fit_transform(df[['income_bucket']])
    income_columns = encoder.get_feature_names_out(['income_bucket'])
    income_df = pd.DataFrame(income_encoded, columns=income_columns, index=df.index)
    df = df.drop('income_bucket', axis=1)
    df = pd.concat([df, income_df], axis=1)

    # 2. Определяем таргеты
    target_cols = [f"product_{p}" for p in PRODUCTS]

    cols_to_scale = [col for col in df.columns
                       if col not in target_cols + ['has_child', 'is_salary_client', 'income_bucket_1', 'income_bucket_2', 'income_bucket_3']]

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    for col in ['has_child', 'is_salary_client', 'income_bucket_1', 'income_bucket_2', 'income_bucket_3']:
        df[col] = df[col].astype(int)

    features = [col for col in df.columns if col not in target_cols]
    X = df[features]
    y = df[target_cols]

    return (X, y, features, target_cols,
            scaler, encoder)


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
    (X, y, features, target_cols, scaler, encoder) = load_and_prepare_data()

    models = train_models(X, y, target_cols)

    model_pack = {
        "feature_columns": features,
        "target_columns": target_cols,
        "models": models,
        "scaler": scaler,
        "encoder": encoder
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
    print(f"\nМодель сохранена в {output_path}")