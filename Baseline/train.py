import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]

def load_and_prepare_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

    df['das_income_bucket'] = df['digital_activity_score'] ** (df['income_bucket'] + 1)

    bins = [0, 50, 200, 500, float('inf')]
    labels = ['tx_low', 'tx_medium', 'tx_high', 'tx_very_high']
    df['avg_tx_amount_bin'] = pd.cut(df['avg_tx_amount'], bins=bins, labels=labels, right=False)
    df = pd.get_dummies(df, columns=['avg_tx_amount_bin'], drop_first=False)
    df.drop('avg_tx_amount', axis=1, inplace=True)



    target_cols = [f"product_{p}" for p in PRODUCTS]
    feature_cols = [c for c in df.columns if c not in ["user_id"] + target_cols]

    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    for col in ['has_child', 'is_salary_client', 'avg_tx_amount_bin_tx_low', 'avg_tx_amount_bin_tx_medium',
                'avg_tx_amount_bin_tx_high', 'avg_tx_amount_bin_tx_very_high']:
        X[col] = X[col].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[[col for col in X.columns if col not in ['has_child', 'is_salary_client', 'avg_tx_amount_bin_tx_low', 'avg_tx_amount_bin_tx_medium',
           'avg_tx_amount_bin_tx_high', 'avg_tx_amount_bin_tx_very_high']]])
    X_binary = X[['has_child', 'is_salary_client', 'avg_tx_amount_bin_tx_low', 'avg_tx_amount_bin_tx_medium',
           'avg_tx_amount_bin_tx_high', 'avg_tx_amount_bin_tx_very_high']].values
    X_scaled = np.hstack([X_scaled, X_binary])

    return X, X_scaled, y, feature_cols, target_cols, scaler


def train_models(X, X_scaled, y):
    models = {}
    cat_features = ['has_child', 'is_salary_client', 'avg_tx_amount_bin_tx_low', 'avg_tx_amount_bin_tx_medium',
                    'avg_tx_amount_bin_tx_high',
                    'avg_tx_amount_bin_tx_very_high']

    model_product_credit_card = LogisticRegression(
        solver='saga',
        l1_ratio=0.8,
        C=2.5,
        max_iter=200,
        random_state=42,
        class_weight='balanced'
    )
    model_product_credit_card.fit(X_scaled, y['product_credit_card'])
    models['product_credit_card'] = model_product_credit_card


    catboost_params = {
        'iterations': 800,
        'learning_rate': 0.01,
        'depth': 8,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'l2_leaf_reg': 6.0,
        'min_data_in_leaf': 50,
        'verbose': False,
    }
    model_product_mortgage = CatBoostClassifier(**catboost_params)
    train_pool = Pool(
        X,
        y['product_mortgage'],
        cat_features=cat_features if cat_features else None
    )
    model_product_mortgage.fit(train_pool)
    models['product_mortgage'] = model_product_mortgage


    catboost_params = {
        'iterations': 600,
        'learning_rate': 0.05,
        'depth': 3,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'l2_leaf_reg': 1.0,
        'verbose': False,
    }
    model_product_deposit = CatBoostClassifier(**catboost_params)
    train_pool = Pool(
        X,
        y['product_deposit'],
        cat_features=cat_features if cat_features else None
    )
    model_product_deposit.fit(train_pool)
    models['product_deposit'] = model_product_deposit


    catboost_params = {
        'iterations': 600,
        'learning_rate': 0.03,
        'depth': 2,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'l2_leaf_reg': 1.0,
        'verbose': False,
    }
    model_product_investment = CatBoostClassifier(**catboost_params)
    train_pool = Pool(
        X,
        y['product_investment'],
        cat_features=cat_features if cat_features else None
    )
    model_product_investment.fit(train_pool)
    models['product_investment'] = model_product_investment


    catboost_params = {
        'iterations': 500,
        'learning_rate': 0.005,
        'depth': 2,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'l2_leaf_reg': 2.0,
        'verbose': False,
    }
    model_product_insurance = CatBoostClassifier(**catboost_params)
    train_pool = Pool(
        X,
        y['product_insurance'],
        cat_features=cat_features if cat_features else None
    )
    model_product_insurance.fit(train_pool)
    models['product_insurance'] = model_product_insurance


    catboost_params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 2,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'l2_leaf_reg': 0.8,
        'verbose': False,
    }
    model_product_p2p_transfer = CatBoostClassifier(**catboost_params)
    train_pool = Pool(
        X,
        y['product_p2p_transfer'],
        cat_features=cat_features if cat_features else None
    )
    model_product_p2p_transfer.fit(train_pool)
    models['product_p2p_transfer'] = model_product_p2p_transfer


    model_product_cashback = LogisticRegression(
        solver='newton-cg',
        l1_ratio=0.0,
        C=1.5,
        max_iter=100,
        random_state=42,
        class_weight='balanced'
    )
    model_product_cashback.fit(X_scaled, y['product_cashback'])
    models['product_cashback'] = model_product_cashback


    catboost_params = {
        'iterations': 900,
        'learning_rate': 0.05,
        'depth': 2,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'l2_leaf_reg': 0.8,
        'verbose': False,
    }
    model_product_premium_account = CatBoostClassifier(**catboost_params)
    train_pool = Pool(
        X,
        y['product_premium_account'],
        cat_features=cat_features if cat_features else None
    )
    model_product_premium_account.fit(train_pool)
    models['product_premium_account'] = model_product_premium_account


    catboost_params = {
        'iterations': 500,
        'learning_rate': 0.01,
        'depth': 2,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'l2_leaf_reg': 0.6,
        'verbose': False,
    }
    model_product_business_loan = CatBoostClassifier(**catboost_params)
    train_pool = Pool(
        X,
        y['product_business_loan'],
        cat_features=cat_features if cat_features else None
    )
    model_product_business_loan.fit(train_pool)
    models['product_business_loan'] = model_product_business_loan


    return models

if __name__ == "__main__":
    X, X_scaled, y, feat_cols, target_cols, scaler = load_and_prepare_data()

    models = train_models(X, X_scaled, y)

    model_pack = {
        "feature_columns": feat_cols,
        "model": models,
        "scaler": scaler
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
