import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# --- 日本語フォントの設定 ---
try:
    plt.rcParams['font.family'] = 'Hiragino Sans' # Mac
except:
    plt.rcParams['font.family'] = 'sans-serif'

# --- 1. データ準備と特徴量作成 ---
df = pd.read_csv("clean_penguins.csv")
original_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'
df_clean = df[original_features + [target]].dropna().reset_index(drop=True)
df_clean = df_clean[df_clean['Sex'] != '.'].reset_index(drop=True)
# 全ての特徴量を作成
df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
df_clean['Flipper per Body Mass'] = df_clean['Flipper Length (mm)'] / df_clean['Body Mass (g)'] * 1000
df_clean['Culmen Area per Body Mass'] = (df_clean['Culmen Length (mm)'] * df_clean['Culmen Depth (mm)']) / df_clean['Body Mass (g)'] * 1000
df_clean['Flipper per Culmen Length'] = df_clean['Flipper Length (mm)'] / df_clean['Culmen Length (mm)']
df_clean['new'] = df_clean['Culmen Aspect Ratio'] / df_clean['Body Mass (g)']

all_features = original_features + ['Culmen Aspect Ratio', 'Flipper per Body Mass', 'Culmen Area per Body Mass', 'Flipper per Culmen Length', 'new']
X = df_clean[all_features]
y = df_clean[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- 2. 特徴量の重要度を計算し、ランキング順に並べる ---
print("--- 2. 特徴量の重要度を計算 ---")
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_scaled, y_encoded)
importances = rf.feature_importances_
# 重要度順に特徴量名を取得
ranked_features = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
sorted_feature_names = [feature for feature, importance in ranked_features]


# --- 3. 特徴量の数を変えながら精度を測定 ---
print("\n--- 3. 特徴量の数を変えながら最適な個数を探索 ---")
# 試す特徴量の数 (1個から全個数まで)
num_features_range = range(1, len(all_features) + 1)
cv_scores = []

for k in num_features_range:
    # 上位k個の特徴量名を取得
    top_k_features = sorted_feature_names[:k]
    print(f"上位 {k} 個の特徴量で検証中: {top_k_features}")
    
    # 該当する特徴量の列だけを抽出
    X_subset = df_clean[top_k_features]
    
    # 再度スケーリング
    scaler_subset = StandardScaler()
    X_subset_scaled = scaler_subset.fit_transform(X_subset)
    
    # k-NNモデルで交差検証
    knn = KNeighborsClassifier(n_neighbors=9) # kは仮で5に固定（後で最適化も可能）
    scores = cross_val_score(knn, X_subset_scaled, y_encoded, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())


# --- 4. 結果を可視化 ---
optimal_num_features = num_features_range[np.argmax(cv_scores)]
best_score = max(cv_scores)

print(f"\n最適な特徴量の数: {optimal_num_features}個, その時のCVスコア: {best_score:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(num_features_range, cv_scores, marker='o')
plt.title("使用する特徴量の数とモデル精度の関係")
plt.xlabel("重要度上位から使用した特徴量の数")
plt.ylabel("交差検証での平均精度")
plt.xticks(num_features_range)
plt.axvline(x=optimal_num_features, color='r', linestyle='--', label=f'最適: {optimal_num_features}個')
plt.legend()
plt.grid(True)
plt.show()