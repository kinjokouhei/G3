import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("penguins_lter.csv")

# 使用するカラムを選択（特徴量＋Species）
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
df_clean = df[features + ['Species']].dropna()  # 欠損値を除去

# 特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# PCA 実行（2次元まで）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA結果をDataFrameに追加
df_clean['PC1'] = X_pca[:, 0]
df_clean['PC2'] = X_pca[:, 1]

# PCA 結果を「種」ごとに色分けして可視化
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='PC1', y='PC2', hue='Species', palette='Set2')
plt.title('PCA of Penguin Features by Species')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Species')
plt.grid(True)
plt.show()
