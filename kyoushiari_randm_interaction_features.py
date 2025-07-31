#5かける3で15この特徴量


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. データ準備 ---
print("--- 1. データの準備を開始 ---")
try:
    # ★★★ 作成済みの交互作用特徴量を直接読み込む ★★★
    X_df = pd.read_csv("interaction_features.csv")
    
    # ★★★ 正解ラベルを取得するために元のファイルも読み込む ★★★
    y_df = pd.read_csv("penguins_processed.csv")
except FileNotFoundError:
    print("エラー: 'interaction_features.csv' または 'penguins_processed.csv' が見つかりません。")
    exit()

# 特徴量 (X) とターゲット (y) を定義
X = X_df
y = y_df['Sex']

print(f"使用する特徴量の数: {len(X.columns)}")

# 前処理
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. GridSearchCVで最適なパラメータを一度だけ見つける ---
print("\n--- 2. 最適なハイパーパラメータを探索します ---")
# 探索用にデータを一度だけ分割
X_train_search, _, y_train_search, _ = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 3]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_search, y_train_search)
best_params = grid_search.best_params_
print(f"\nGridSearchCVで見つかった最適なパラメータ: {best_params}")

# --- 3. 最適なモデルで1000回の試行と精度評価 ---
print("\n--- 3. 最適なモデルで1000回の試行を開始します ---")
n_runs = 1000
accuracy_scores = []

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=i, stratify=y_encoded
    )
    model = RandomForestClassifier(**best_params, random_state=i)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)
    if (i + 1) % 100 == 0:
        print(f"実行回数 {i+1}/{n_runs} 完了...")

# --- 4. 最終結果の表示 ---
print("\n" + "="*50)
print("--- 4. 最終結果 ---")
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
print(f"実行回数: {n_runs}回")
print(f"平均正解率 (Mean Accuracy): {mean_accuracy:.4f}")
print(f"精度の標準偏差 (Standard Deviation): {std_accuracy:.4f}")
print("="*50)