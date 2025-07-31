import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier # RandomForestClassifierをインポート
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib

# 日本語フォントを指定（Mac標準のヒラギノ角ゴ）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号を正しく表示

#--- データ読み込みと前処理 ---
df = pd.read_csv("penguins_no_outliers.csv")

#特徴量とターゲット
original_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'

#欠損値・性別不明データを除外
df_clean = df[original_features + [target]].dropna()
df_clean = df_clean[df_clean['Sex'] != '.']

# --- 新しい比率特徴量の追加 ---
df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
df_clean['Flipper per Body Mass'] = df_clean['Flipper Length (mm)'] / df_clean['Body Mass (g)'] 
df_clean['Culmen Area per Body Mass'] = (df_clean['Culmen Length (mm)'] * df_clean['Culmen Depth (mm)']) / df_clean['Body Mass (g)'] 
df_clean['Flipper per Culmen Length'] = df_clean['Flipper Length (mm)'] / df_clean['Culmen Length (mm)']
df_clean['Culmen Aspect Ratio / Body Mass'] = df_clean['Culmen Aspect Ratio'] / df_clean['Body Mass (g)']
df_clean['Culmen Length / Body Mass'] = df_clean['Culmen Length (mm)'] / df_clean['Body Mass (g)']
df_clean['Culmen Depth / Body Mass'] = df_clean['Culmen Depth (mm)'] / df_clean['Body Mass (g)']



#--
# --- 新しい比率特徴量の追加 ---
#df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
#df_clean['new0'] = df_clean['Culmen Aspect Ratio'] / df_clean['Body Mass (g)']
#df_clean['new1'] = df_clean['Culmen Aspect Ratio'] * df_clean['Body Mass (g)']
#df_clean['new2'] = df_clean['Culmen Length (mm)'] / df_clean['Body Mass (g)']
#df_clean['new3'] = df_clean['Culmen Length (mm)'] * df_clean['Body Mass (g)']
# 全ての特徴量をリスト化
#all_features = original_features + ['new0', 'new1','new2','new3']







# 全ての特徴量をリスト化
all_features = original_features + ['Culmen Aspect Ratio',  'Culmen Aspect Ratio / Body Mass','Culmen Length / Body Mass','Culmen Depth / Body Mass']

#特徴量とターゲットに分ける
X = df_clean[all_features]
y = df_clean[target]

#ラベルエンコード（FEMALE→0, MALE→1）
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#標準化 (RandomForestでは必須ではないが、K近傍のために残しておく)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 標準化されたデータ

#訓練・テストに分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# --- 最終的なレポート表示のためのラベル定義 ---
labels_for_report = sorted(np.unique(y_encoded))
target_names_for_report = le.inverse_transform(labels_for_report)


# --- RandomForestClassifier で特徴量重要度を算出 ---
print("--- RandomForestClassifierによる特徴量重要度の算出 ---")
# RandomForestClassifierのインスタンス化と学習
# random_state を固定することで再現性を確保
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 特徴量重要度を取得
feature_importances = rf_model.feature_importances_

# 特徴量名と重要度をペアにしてソート
features_and_importances = sorted(zip(all_features, feature_importances), key=lambda x: x[1], reverse=True)

print("\n--- 各特徴量の重要度 ---")
for feature, importance in features_and_importances:
    print(f"{feature}: {importance:.4f}")

# 特徴量重要度をグラフで可視化
plt.figure(figsize=(12, 6))
plt.bar([f[0] for f in features_and_importances], [f[1] for f in features_and_importances], color='lightsteelblue')
plt.xlabel("特徴量")
plt.ylabel("重要度")
plt.title("RandomForestClassifierによる特徴量重要度")
plt.xticks(rotation=45, ha='right') # ラベルが重ならないように回転
plt.tight_layout() # レイアウトの自動調整
plt.grid(axis='y', linestyle='--')
plt.show()

print("\n--- RandomForestClassifier のテストデータでの評価 ---")
rf_pred = rf_model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, rf_pred))
print("RandomForest Classification Report:")
print(classification_report(y_test, rf_pred, labels=labels_for_report, target_names=target_names_for_report))



































# --- 準備：上記のRandomForestのコードを実行し、最適な4つの特徴量を特定した後、以下のコードを実行 ---

# 仮に、RandomForestで以下の4つの特徴量が最も重要だと判断されたとする
# 実際に実行した結果に基づいて、このリストを更新してください！
selected_features = [
    'Culmen Depth (mm)',
    'Body Mass (g)',
    'Flipper per Body Mass',
    'Culmen Length (mm)',
    'Flipper Length (mm)',
    'Culmen Aspect Ratio'
]
print(f"\n--- 選択された特徴量: {selected_features} ---")

# --- 選択された特徴量でデータを再構築 ---
# df_clean は上記コードで既に比率特徴量が追加されたものを使用
X_selected = df_clean[selected_features]
y_selected = df_clean[target] # ターゲットは同じ

# ラベルエンコード（yは変更なしなので再実行不要だが、念のため記載）
y_encoded_selected = le.fit_transform(y_selected)

# 標準化（選択された特徴量のみを対象）
scaler_selected = StandardScaler()
X_scaled_selected = scaler_selected.fit_transform(X_selected)

# 訓練・テストに分割
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    X_scaled_selected, y_encoded_selected, test_size=0.2, random_state=42
)

# --- K近傍法モデルの最適化（選択された特徴量のみで） ---

# --- 1. 最適なkを交差検証で探す ---
print("\n--- 1. 最適なkの探索（選択特徴量のみ）---")
k_range = range(1, 10)
cv_scores_k_selected = []

for k in k_range:
    knn_temp_k = KNeighborsClassifier(n_neighbors=k)
    scores_k = cross_val_score(knn_temp_k, X_train_selected, y_train_selected, cv=5, scoring='accuracy')
    cv_scores_k_selected.append(scores_k.mean())

optimal_k_selected = k_range[np.argmax(cv_scores_k_selected)]
print(f"最適なk: {optimal_k_selected}, クロスバリデーション精度: {max(cv_scores_k_selected):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, cv_scores_k_selected, marker='o')
plt.title("KNNのkごとの交差検証精度（選択特徴量のみ）")
plt.xlabel("kの値")
plt.ylabel("平均精度")
plt.xticks(k_range)
plt.grid(True)
plt.show()

# --- 2. 最適なmetricを交差検証で探す ---
print("\n--- 2. 最適なmetricの探索（選択特徴量のみ）---")
metrics_to_evaluate = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
metric_scores_selected = {}

for metric_name in metrics_to_evaluate:
    knn_temp_metric = KNeighborsClassifier(n_neighbors=optimal_k_selected, metric=metric_name)
    scores_metric = cross_val_score(knn_temp_metric, X_train_selected, y_train_selected, cv=5, scoring='accuracy')
    metric_scores_selected[metric_name] = scores_metric.mean()

optimal_metric_selected = max(metric_scores_selected, key=metric_scores_selected.get)
print(f"\n最適なmetric: {optimal_metric_selected}, クロスバリデーション精度: {metric_scores_selected[optimal_metric_selected]:.4f}")

metrics_names_selected = list(metric_scores_selected.keys())
metrics_accuracy_selected = list(metric_scores_selected.values())

plt.figure(figsize=(10, 5))
plt.bar(metrics_names_selected, metrics_accuracy_selected, color='skyblue')
plt.xlabel("距離尺度 (Metric)")
plt.ylabel("平均精度")
plt.title("KNNにおける各距離尺度ごとの交差検証精度（選択特徴量のみ）")
plt.ylim(min(metrics_accuracy_selected) * 0.95, max(metrics_accuracy_selected) * 1.05)
plt.grid(axis='y', linestyle='--')
plt.show()

# --- 3. 最適なweightsを交差検証で探す ---
print("\n--- 3. 最適なweightsの探索（選択特徴量のみ）---")
weights_to_evaluate = ['uniform', 'distance']
weights_scores_selected = {}

for weight_type in weights_to_evaluate:
    knn_temp_weights = KNeighborsClassifier(
        n_neighbors=optimal_k_selected,
        metric=optimal_metric_selected,
        weights=weight_type
    )
    scores_weights = cross_val_score(knn_temp_weights, X_train_selected, y_train_selected, cv=5, scoring='accuracy')
    weights_scores_selected[weight_type] = scores_weights.mean()

optimal_weights_selected = max(weights_scores_selected, key=weights_scores_selected.get)
print(f"\n最適なweights: {optimal_weights_selected}, クロスバリデーション精度: {weights_scores_selected[optimal_weights_selected]:.4f}")

weights_names_selected = list(weights_scores_selected.keys())
weights_accuracy_selected = list(weights_scores_selected.values())

plt.figure(figsize=(7, 5))
plt.bar(weights_names_selected, weights_accuracy_selected, color=['lightcoral', 'lightgreen'])
plt.xlabel("重み付け (Weights)")
plt.ylabel("平均精度")
plt.title("KNNにおける各重み付けごとの交差検証精度（選択特徴量のみ）")
plt.ylim(min(weights_accuracy_selected) * 0.95, max(weights_accuracy_selected) * 1.05)
plt.grid(axis='y', linestyle='--')
plt.show()

# --- 4. 最適なk, metric, weightsで最終モデルを学習・評価（選択特徴量のみ） ---
print(f"\n--- 4. 最適なk={optimal_k_selected}, metric={optimal_metric_selected}, weights={optimal_weights_selected} で最終モデルを評価（選択特徴量のみ）---")
knn_final_selected = KNeighborsClassifier(
    n_neighbors=optimal_k_selected,
    metric=optimal_metric_selected,
    weights=optimal_weights_selected
)
knn_final_selected.fit(X_train_selected, y_train_selected)
knn_final_pred_selected = knn_final_selected.predict(X_test_selected)

print(f"KNN（最適k={optimal_k_selected}, 最適metric={optimal_metric_selected}, 最適weights={optimal_weights_selected}）のAccuracy:", accuracy_score(y_test_selected, knn_final_pred_selected))
print("KNN（最適k, 最適metric, 最適weights）Classification Report:")
print(classification_report(
    y_test_selected, knn_final_pred_selected,
    labels=labels_for_report, # ここは共通のラベル定義を使用
    target_names=target_names_for_report
))