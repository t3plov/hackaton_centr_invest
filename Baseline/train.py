import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]

# BEST_PARAMS = {
#     "credit_card": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
#     "mortgage": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
#     "deposit": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
#     "investment": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
#     "insurance": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
#     "p2p_transfer": {'C': 1.0, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
#     "cashback": {'C': 0.1, 'penalty': None, 'solver': 'lbfgs', 'max_iter': 300, 'l1_ratio': 0.5},
#     "premium_account": {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
#     "business_loan": {'C': 0.1, 'penalty': None, 'solver': 'sag', 'max_iter': 300, 'l1_ratio': 0.5}
# }


def load_and_prepare_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

    target_cols = [f"product_{p}" for p in PRODUCTS]
    feature_cols = [c for c in df.columns if c not in ["user_id"] + target_cols]

    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    for col in ['has_child', 'is_salary_client']:
        if col in X.columns:
            X[col] = X[col].astype(int)

    columns_to_scaled = ['age', 'tenure_months', 'tx_count_30d', 'avg_tx_amount', 'digital_activity_score']
    scaler = StandardScaler()
    X[columns_to_scaled] = scaler.fit_transform(X[columns_to_scaled])

    return X, y, scaler, feature_cols, target_cols


def train_models(X, y, target_cols):
    chain_order_names = [
        'product_credit_card',
        'product_cashback',
        'product_p2p_transfer',
        'product_deposit',
        'product_investment',
        'product_insurance',
        'product_mortgage',
        'product_business_loan',
        'product_premium_account'
    ]
    order_indices = [target_cols.index(name) for name in chain_order_names]

    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=10,
        max_depth=2,
        lambda_l1=0.001,
        is_unbalance=True,
        eval_metric='auc',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

    chain_model = ClassifierChain(
        lgbm,
        order=order_indices,
        random_state=42,
        chain_method='predict_proba'
    )
    chain_model.fit(X, y)

    return chain_model


if __name__ == "__main__":
    import sklearn
    print("Sklearn version in train:", sklearn.__version__)
    X, y, scaler, feat_cols, target_cols = load_and_prepare_data()

    chain_model = train_models(X, y, target_cols)

    model_pack = {
        "scaler": scaler,
        "feature_columns": feat_cols,
        "chain_model": chain_model,
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
