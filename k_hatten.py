import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- 日本語フォントの設定 (お使いの環境に合わせてください) ---
try:
    plt.rcParams['font.family'] = 'Hiragino Sans' # Mac
except:
    plt.rcParams['font.family'] = 'sans-serif'


# --- 1. データ準備 ---
print("--- 1. データの準備を開始 ---")
try:
    df = pd.read_csv("clean_penguins.csv")
except FileNotFoundError:
    print("エラー: 'clean_penguins.csv' が見つかりません。")
    exit()

original_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'
df_clean = df[original_features + [target]].dropna().reset_index(drop=True)
df_clean = df_clean[df_clean['Sex'] != '.'].reset_index(drop=True)

# 比率特徴量を追加
df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
df_clean['Flipper per Body Mass'] = df_clean['Flipper Length (mm)'] / df_clean['Body Mass (g)'] * 1000

features_to_use = original_features + ['Culmen Aspect Ratio', 'Flipper per Body Mass']

X = df_clean[features_to_use]
y = df_clean[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 訓練データとテストデータに分割
train_indices, test_indices = train_test_split(
    df_clean.index, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train = X_scaled[train_indices]
y_train = y_encoded[train_indices]


# --- 2. GridSearchCVで最適なモデルを探す ---
print("--- 2. GridSearchCVで最適なパラメータの組み合わせを探します ---")
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
grid_search.fit(X_train, y_train)

print(f"\n見つかった最適なパラメータ: {grid_search.best_params_}")
model = grid_search.best_estimator_


# --- 3. 全データで予測を実行し、結果を統合 ---
print("\n--- 3. 最適なモデルで全データに対する予測を実行 ---")
all_predictions_encoded = model.predict(X_scaled)
results_df = df_clean.copy()
results_df['Predicted_Sex'] = le.inverse_transform(all_predictions_encoded)
results_df['Prediction_Result'] = np.where(y_encoded == all_predictions_encoded, 'Success', 'Failure')
results_df['Split'] = 'Train'
results_df.loc[test_indices, 'Split'] = 'Test'

# 結果を一つのCSVファイルとして保存（オプション）
output_filename = 'prediction_results.csv'
results_df.to_csv(output_filename, index=False)
print(f"--- 全ての結果を '{output_filename}' に保存しました。 ---")


# --- 4. ペアプロットで結果を可視化・分析 ---
print("--- 4. ペアプロットを作成してエラー分析を行います ---")

# 可視化する特徴量を絞る
features_to_plot = [
    'Culmen Length (mm)', 
    'Culmen Depth (mm)', 
    'Body Mass (g)', 
    'Culmen Aspect Ratio',
    'Prediction_Result' # 色分けに使うため、これも含める
]

# ペアプロットを作成
sns.pairplot(results_df[features_to_plot], hue='Prediction_Result', markers=['o', 's'], palette={'Success':'gray', 'Failure':'red'})
plt.suptitle('予測結果のペアプロット分析 (Failure=赤x)', y=1.02, fontsize=16)
plt.savefig('grahu.png')
plt.show()