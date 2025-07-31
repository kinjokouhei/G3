import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# --- 1. データ準備 (一度だけ実行) ---
print("--- 1. データの準備を開始 ---")
df = pd.read_csv("clean_penguins.csv")
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'
df_clean = df[features + [target]].dropna()
df_clean = df_clean[df_clean['Sex'] != '.']
X_df = df_clean[features]
y_s = df_clean[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y_s)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# --- 2. 1000回の試行ループ ---
print("\n--- 2. 1000回の試行を開始します ---")
n_runs = 1000
final_accuracies = []  # 各回の最終的な精度を保存するリスト
optimal_k_list = []    # 各回で見つかった最適なkを保存するリスト

for i in range(n_runs):
    # random_stateを指定せず、毎回ランダムにデータを分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2
    )

    # --- 最適なkを交差検証で探す ---
    k_range = range(1, 10)
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # この試行での最適なkを取得
    current_optimal_k = k_range[np.argmax(cv_scores)]
    optimal_k_list.append(current_optimal_k)

    # --- 最適kで再学習・評価 ---
    knn_opt = KNeighborsClassifier(n_neighbors=current_optimal_k)
    knn_opt.fit(X_train, y_train)
    predictions = knn_opt.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    final_accuracies.append(accuracy)

    # 100回ごとに進行状況を表示
    if (i + 1) % 100 == 0:
        print(f"実行回数 {i+1}/{n_runs} 完了... (今回の最適k: {current_optimal_k}, 今回の精度: {accuracy:.4f})")


# --- 3. 最終結果の集計と表示 ---
print("\n" + "="*50)
print("--- 3. 最終結果 ---")

# 最も頻繁に選ばれたkの値を計算
k_counts = Counter(optimal_k_list)
most_common_k = k_counts.most_common(1)[0] # (値, 回数) のタプル

# 精度の平均と標準偏差を計算
mean_accuracy = np.mean(final_accuracies)
std_accuracy = np.std(final_accuracies)

print(f"実行回数: {n_runs}回")
print(f"\n最も頻繁に最適だと選ばれたkの値: {most_common_k[0]} ({most_common_k[1]}回)")
print(f"\n平均正解率 (Mean Accuracy): {mean_accuracy:.4f}")
print(f"精度の標準偏差 (Standard Deviation): {std_accuracy:.4f}")
print("="*50)