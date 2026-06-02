import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

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


def load_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

    df = create_features(df)

    feature_cols = [
        c for c in df.columns
        if c not in ["user_id"] + [f"product_{p}" for p in PRODUCTS]
    ]
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = df[[f"product_{p}" for p in PRODUCTS]].values
    return X_scaled, y, scaler, feature_cols


def train_models(X, y):
    models = {}
    for idx, prod in enumerate(PRODUCTS):

        hgb = HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.01,
            max_depth=3,
            min_samples_leaf=17,
            l2_regularization=0.1,
            random_state=42,
            verbose=0
        )
        hgb.fit(X, y[:, idx])
        models[prod] = hgb
    return models


if __name__ == "__main__":
    X, y, scaler, feat_cols = load_data()

    models = train_models(X, y)

    model_pack = {
        "scaler": scaler,
        "feature_columns": feat_cols,
        "models": models,
    }
    joblib.dump(model_pack, Path("./baseline_model.joblib"))
