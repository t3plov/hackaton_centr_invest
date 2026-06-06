import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse


def build_features(df, pack):
    """Воспроизводит ВСЕ шаги предобработки из train.py"""

    # Удаляем user_id если есть
    if 'user_id' in df.columns:
        df = df.drop('user_id', axis=1)

    # 1. Клиппинг и логарифмирование (используем ПОРОГ из обучения!)
    upper_limit = pack["upper_limit"]
    df['avg_tx_amount'] = np.clip(df['avg_tx_amount'], a_min=0, a_max=upper_limit)
    df['avg_tx_amount'] = np.log1p(df['avg_tx_amount'])

    # 2. Первый скейлер (для кластеризации)
    scaler1 = pack["scaler1"]
    cols_to_scale_1 = pack["cols_to_scale_1"]
    binary_cols_1 = pack["binary_cols_1"]

    X_for_kmeans = scaler1.transform(df[cols_to_scale_1])
    X_binary_1 = df[binary_cols_1].values.astype(float)
    X_for_kmeans = np.hstack([X_for_kmeans, X_binary_1])

    # 3. Кластеризация (PREDICT, не fit_predict!)
    kmeans = pack["kmeans"]
    df['client_cluster'] = kmeans.predict(X_for_kmeans)

    # 4. One-Hot Encoding кластеров
    df = pd.get_dummies(
        df,
        columns=['client_cluster'],
        prefix='cluster',
        drop_first=True
    )

    # Убеждаемся, что все кластерные колонки есть (на случай если в тесте нет какого-то кластера)
    for i in range(1, 5):
        col_name = f'cluster_{i}'
        if col_name not in df.columns:
            df[col_name] = 0
        df[col_name] = df[col_name].astype(int)

    # 5. Второй скейлер (финальный)
    scaler2 = pack["scaler2"]
    cols_to_scale_2 = pack["cols_to_scale_2"]
    binary_cols_2 = pack["binary_cols_2"]

    X_scaled = scaler2.transform(df[cols_to_scale_2])
    X_binary_2 = df[binary_cols_2].values.astype(float)
    X_final = np.hstack([X_scaled, X_binary_2])

    return X_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    # Загружаем данные и модель
    test_data = pd.read_csv(args.input_path)
    pack = joblib.load(Path("baseline_model.joblib"))

    # Строим признаки
    X = build_features(test_data, pack)

    # Предсказания для каждого продукта
    proba_list = []
    for col in pack["target_columns"]:
        model = pack["models"][col]
        proba = model.predict_proba(X)[:, 1]
        proba_list.append(proba)

    # Сохраняем результат
    result = np.column_stack(proba_list)
    result = pd.DataFrame(result)
    result.to_csv(args.output_path, header=False, index=False)

    print(f"Предсказания сохранены в {args.output_path}")