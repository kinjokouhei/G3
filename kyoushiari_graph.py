#all_time_failures.csvの中に誤差があるか判断するために、グラフにプロット

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- 日本語フォントの設定 ---
try:
    plt.rcParams['font.family'] = 'Hiragino Sans' # Mac
except:
    plt.rcParams['font.family'] = 'sans-serif'

# --- 1. データの準備 ---
df = pd.read_csv("penguins_processed.csv")
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'

df['Culmen Aspect Ratio'] = df['Culmen Length (mm)'] / df['Culmen Depth (mm)']

X_df = df[features]
y_s = df[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y_s)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# --- 2. モデルの学習と予測 ---
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_scaled, y_encoded, df.index, test_size=0.2, random_state=42, stratify=y_encoded
)

model = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='uniform')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- 3. 失敗事例を特定 ---
failure_mask = predictions != y_test
failure_indices = indices_test[failure_mask]

# --- 4. データの表示と可視化 ---
test_df = df.loc[indices_test].copy()
failures_df = test_df.loc[failure_indices]

# 全テストデータを表示
print("\n" + "="*50)
print("--- プロットに使用した全テストデータ ---")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(test_df)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
print("="*50)

# ★★★★★ ここから追加 ★★★★★
print("\n" + "="*50)
print("--- 予測に失敗したデータの詳細 ---")
if failures_df.empty:
    print("失敗したデータはありませんでした。")
else:
    # 失敗したデータだけを抜き出して表示
    print(failures_df)
print("="*50 + "\n")
# ★★★★★ ここまで追加 ★★★★★

print(f"--- テストデータ{len(test_df)}件中、{len(failures_df)}件の失敗をプロットします ---")

# グラフ描画
plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=test_df,
    x='Culmen Aspect Ratio',
    y='Body Mass (g)',
    hue='Sex',
    palette=['red', 'blue'], # FEMALE=red, MALE=blue
    style='Sex',
    s=80,
    alpha=0.7
)

plt.scatter(
    failures_df['Culmen Aspect Ratio'],
    failures_df['Body Mass (g)'],
    s=200,
    facecolors='none', 
    edgecolors='black', 
    linewidth=2,
    label='Prediction Failure'
)

plt.title("テストデータにおける予測失敗事例の分布", fontsize=16)
plt.xlabel("くちばしのアスペクト比 (Culmen Aspect Ratio)")
plt.ylabel("体重 (Body Mass g)")
plt.legend()
plt.grid(True)
plt.show()