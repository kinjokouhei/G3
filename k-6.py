import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. データ準備と特徴量作成 ---
print("--- 1. データの準備と特徴量作成 ---")
df = pd.read_csv("penguins_no_outliers.csv")
original_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'
df_clean = df[original_features + [target]].dropna().reset_index(drop=True)
df_clean = df_clean[df_clean['Sex'] != '.'].reset_index(drop=True)

# 全ての特徴量を作成
df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
df_clean['new'] = df_clean['Culmen Aspect Ratio'] * df_clean['Body Mass (g)']

# 特徴量選択（前回の分析で重要度が高かったもの）
selected_features = [
    'Culmen Depth (mm)',
    'Body Mass (g)',
    'Culmen Length (mm)',
    'new'
]

X = df_clean[selected_features]
y = df_clean[target]

# 前処理
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. GridSearchCVで最適なパラメータの「組み合わせ」を一度だけ見つける ---
print("\n--- 2. 最適なハイパーパラメータの組み合わせを探索します ---")
# 訓練データとテストデータに一度だけ分割（GridSearchCV用）
X_train_for_search, _, y_train_for_search, _ = train_test_split(
    X_scaled, y_encoded, test_size=0.2
)

param_grid = {
    'n_neighbors': range(3, 16),
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_for_search, y_train_for_search)

# 最適なパラメータを取得
best_params = grid_search.best_params_
print(f"\nGridSearchCVで見つかった最適なパラメータ: {best_params}")

# --- 3. 最適なモデルで1000回の試行と精度評価 ---
print("\n--- 3. 最適なモデルで1000回の試行を開始します ---")
n_runs = 1000
accuracy_scores = []

for i in range(n_runs):
    # 毎回異なるデータ分割を行う
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=i
    )

    # 見つけておいた最適なパラメータでモデルを作成
    model = KNeighborsClassifier(**best_params) # **best_params で辞書を展開して引数に渡す
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

    # 100回ごとに進行状況を表示
    if (i + 1) % 100 == 0:
        print(f"実行回数 {i+1}/{n_runs} 完了...")

# --- 4. 最終結果の表示 ---
print("\n" + "="*50)
print("--- 4. 最終結果 ---")
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print(f"探索で見つかった最適なkの値: {best_params['n_neighbors']}")
print(f"\n実行回数: {n_runs}回")
print(f"平均正解率 (Mean Accuracy): {mean_accuracy:.4f}")
print(f"精度の標準偏差 (Standard Deviation): {std_accuracy:.4f}")
print("="*50)