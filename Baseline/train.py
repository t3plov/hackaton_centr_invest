import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
import joblib

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]

def load_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)
    feature_cols = [c for c in df.columns if c not in ["user_id"] + [f"product_{p}" for p in PRODUCTS]]
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df[[f"product_{p}" for p in PRODUCTS]].values
    return X_scaled, y, scaler, feature_cols

def train_models(X, y):
    models = {}
    for idx, prod in enumerate(PRODUCTS):
        hgb = HistGradientBoostingClassifier(
            max_iter=1200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=15,
            l2_regularization=0.1,
            random_state=42,
            verbose=1
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
