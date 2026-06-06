import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]


def load_and_prepare_data():
    train_path = Path("./train_data.csv")
    df = pd.read_csv(train_path)

    if 'user_id' in df.columns:
        df.drop('user_id', axis=1, inplace=True)

    # 1. Клиппинг и логарифмирование avg_tx_amount
    upper_limit = df['avg_tx_amount'].quantile(0.99)
    df['avg_tx_amount'] = np.clip(df['avg_tx_amount'], a_min=0, a_max=upper_limit)
    df['avg_tx_amount'] = np.log1p(df['avg_tx_amount'])  # log1p безопаснее для нулей

    # 2. Определяем таргеты
    target_cols = [f"product_{p}" for p in PRODUCTS]

    # 3. Первый скейлер — для кластеризации
    cols_to_scale_1 = [col for col in df.columns
                       if col not in target_cols + ['has_child', 'is_salary_client']]

    scaler1 = StandardScaler()
    X_for_kmeans = scaler1.fit_transform(df[cols_to_scale_1])

    # Бинарные признаки для склейки
    binary_cols_1 = ['has_child', 'is_salary_client']
    X_binary_1 = df[binary_cols_1].values
    X_for_kmeans = np.hstack([X_for_kmeans, X_binary_1])

    # 4. Кластеризация
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['client_cluster'] = kmeans.fit_predict(X_for_kmeans)

    # 5. One-Hot Encoding кластеров
    df = pd.get_dummies(
        df,
        columns=['client_cluster'],
        prefix='cluster',
        drop_first=True
    )

    # Приводим к int
    for col in ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 6. Финальные признаки и таргеты
    features = [col for col in df.columns if col not in target_cols]
    X = df[features]
    y = df[target_cols]

    # 7. Второй скейлер — финальный
    binary_cols_2 = ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
                     'has_child', 'is_salary_client']
    cols_to_scale_2 = [col for col in X.columns if col not in binary_cols_2]

    scaler2 = StandardScaler()
    X_scaled = scaler2.fit_transform(X[cols_to_scale_2])
    X_binary_2 = X[binary_cols_2].values
    X_final = np.hstack([X_scaled, X_binary_2])

    # 8. Разбиение
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    return (X_train, y_train, X_test, y_test, features, target_cols,
            scaler1, scaler2, kmeans, upper_limit, cols_to_scale_1, cols_to_scale_2,
            binary_cols_1, binary_cols_2)


def train_models(X_train, y_train, target_cols):
    models = {}
    auc_scores = {}

    for col in target_cols:
        print(f"Обучение LR {col}")
        model = LogisticRegression(
            solver='saga',  # saga поддерживает l1_ratio
            penalty='elasticnet',
            l1_ratio=0.8,
            C=5.0,
            max_iter=200,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train[col])
        models[col] = model

        preds = model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train[col], preds)
        auc_scores[col] = auc
        print(f"  Train AUC: {auc:.5f}")

    return models, auc_scores


if __name__ == "__main__":
    (X_train, y_train, X_test, y_test, features, target_cols,
     scaler1, scaler2, kmeans, upper_limit,
     cols_to_scale_1, cols_to_scale_2,
     binary_cols_1, binary_cols_2) = load_and_prepare_data()

    models, auc_scores = train_models(X_train, y_train, target_cols)

    # Оценка на тесте
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ НА ТЕСТЕ:")
    print("=" * 60)
    test_auc_scores = {}
    for col in target_cols:
        preds = models[col].predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test[col], preds)
        test_auc_scores[col] = auc
        print(f"  {col}: {auc:.5f}")

    macro_auc = np.mean(list(test_auc_scores.values()))
    print(f"\nMACRO ROC-AUC: {macro_auc:.5f}")

    # Сохраняем ВСЁ необходимое для инференса
    model_pack = {
        "feature_columns": features,
        "target_columns": target_cols,
        "models": models,
        "scaler1": scaler1,
        "scaler2": scaler2,
        "kmeans": kmeans,
        "upper_limit": upper_limit,
        "cols_to_scale_1": cols_to_scale_1,
        "cols_to_scale_2": cols_to_scale_2,
        "binary_cols_1": binary_cols_1,
        "binary_cols_2": binary_cols_2,
    }

    output_path = Path("./baseline_model.joblib")
    joblib.dump(model_pack, output_path)
    print(f"\nМодель сохранена в {output_path}")