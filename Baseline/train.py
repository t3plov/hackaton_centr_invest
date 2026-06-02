import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import joblib

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]


def load_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

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
        print(f"Обучение модели для: {prod}")

        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=-1,
            num_leaves=7,
            subsample=0.8,
            eval_metric='auc',
            objective='binary',
            min_child_samples=30,
            is_unbalance=True,
            verbose=-1,
            random_state=42,
            n_jobs=-1
        )
        lgbm.fit(X, y[:, idx])
        models[prod] = lgbm
    return models


if __name__ == "__main__":   # ИСПРАВЛЕНО: было "if name == "main":"
    X, y, scaler, feat_cols = load_data()

    models = train_models(X, y)

    model_pack = {
        "scaler": scaler,
        "feature_columns": feat_cols,
        "models": models,
    }
    joblib.dump(model_pack, Path("./baseline_model.joblib"))
