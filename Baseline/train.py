import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]


def load_and_prepare_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

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

    target_cols = [f"product_{p}" for p in PRODUCTS]
    feature_cols = [c for c in df.columns if c not in ["user_id"] + target_cols]

    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    return X, y, feature_cols, target_cols


def train_models(X, y, target_cols):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42
    )

    models = {}

    for col in target_cols:
        model = CatBoostClassifier(
            iterations=4000,  # больше итераций + early stopping
            learning_rate=0.02,  # ниже = стабильнее обобщение

            depth=5,  # sweet spot для таблички
            l2_leaf_reg=8,  # сильнее регуляризация (важно!)
            random_strength=2,  # шум в сплитах (борьба с overfit)

            bagging_temperature=0.8,  # стохастичность выборки
            subsample=0.8,  # row sampling

            loss_function="Logloss",
            eval_metric="AUC",

            leaf_estimation_iterations=5,
            leaf_estimation_method="Gradient",

            grow_policy="Lossguide",  # часто лучше на табличке
            min_data_in_leaf=50,  # ключевая защита от переобучения

            od_type="Iter",
            od_wait=200,

            verbose=200,
            random_seed=42
        )

        model.fit(
            X_train, y_train[col],
            eval_set=(X_val, y_val[col]),
            use_best_model=True,
            verbose=200
        )

        models[col] = model

    return models


if __name__ == "__main__":
    import sklearn
    print("Sklearn version in train:", sklearn.__version__)
    X, y, feat_cols, target_cols = load_and_prepare_data()

    models = train_models(X, y, target_cols)

    model_pack = {
        "feature_columns": feat_cols,
        "model": models,
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
