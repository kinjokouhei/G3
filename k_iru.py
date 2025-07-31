import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib

# 日本語フォントを指定（Mac標準のヒラギノ角ゴ）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号を正しく表示

#--- データ読み込みと前処理 ---
df = pd.read_csv("clean_penguins.csv")

#特徴量とターゲット
original_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'

#欠損値・性別不明データを除外
df_clean = df[original_features + [target]].dropna()
df_clean = df_clean[df_clean['Sex'] != '.']

# --- 比率特徴量の追加（選択する特徴量に含まれる可能性があるため、ここで計算しておく） ---
df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
df_clean['Flipper per Body Mass'] = df_clean['Flipper Length (mm)'] / df_clean['Body Mass (g)'] * 1000
df_clean['Culmen Area per Body Mass'] = (df_clean['Culmen Length (mm)'] * df_clean['Culmen Depth (mm)']) / df_clean['Body Mass (g)'] * 1000
df_clean['Flipper per Culmen Length'] = df_clean['Flipper Length (mm)'] / df_clean['Culmen Length (mm)']

# --- ご指定の4つの特徴量を選択 ---
selected_features = [
    'Culmen Depth (mm)',
    'Body Mass (g)',
    'Flipper per Body Mass', # 比率特徴量
    'Culmen Length (mm)'
]
print(f"\n--- 使用する特徴量: {selected_features} ---")


#特徴量とターゲットに分ける（選択された特徴量のみを使用）
X = df_clean[selected_features]
y = df_clean[target]

#ラベルエンコード（FEMALE→0, MALE→1）
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#訓練・テストに分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, #random_state=27
)

# --- 最終的なレポート表示のためのラベル定義 ---
labels_for_report = sorted(np.unique(y_encoded))
target_names_for_report = le.inverse_transform(labels_for_report)


# --- ここから、あなたが最適なハイパーパラメータの値を直接設定 ---
# 過去のグリッドサーチや手動探索で得られた最適な値をここに設定してください。
# 例:
optimal_k = 3       # あなたがKの探索で見つけた最適な値
optimal_metric = 'manhattan' # あなたがmetricの探索で見つけた最適な値
optimal_weights = 'uniform' # あなたがweightsの探索で見つけた最適な値
# -----------------------------------------------------------------


# --- 固定された最適なハイパーパラメータでK近傍法モデルを学習・評価 ---
print(f"\n--- 最適なk={optimal_k}, metric={optimal_metric}, weights={optimal_weights} でK近傍法モデルを評価（選択特徴量込み）---")

knn_fixed_params = KNeighborsClassifier(
    n_neighbors=optimal_k,
    metric=optimal_metric,
    weights=optimal_weights
)

knn_fixed_params.fit(X_train, y_train)
final_pred = knn_fixed_params.predict(X_test)

print(f"K近傍法（固定パラメータ）のAccuracy:", accuracy_score(y_test, final_pred))
print("K近傍法（固定パラメータ）Classification Report:")
print(classification_report(
    y_test, final_pred,
    labels=labels_for_report,
    target_names=target_names_for_report
))