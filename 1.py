import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- データ読み込みと前処理 ---
df = pd.read_csv("clean_penguins.csv")

# 使用する特徴量とターゲット
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'

# 欠損値と性別が不明なデータを除外
df_clean = df[features + [target]].dropna()
df_clean = df_clean[df_clean['Sex'] != '.']

# 特徴量とターゲットに分ける
X = df_clean[features]
y = df_clean[target]

# ラベルエンコーディング（FEMALE → 0, MALE → 1）
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# --- SVC モデル ---
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

# ラベルと文字ラベル取得
labels_svc = sorted(np.unique(np.concatenate((y_test, svc_pred))))
target_names_svc = le.inverse_transform(labels_svc)

print("SVC Accuracy:", accuracy_score(y_test, svc_pred))
print("SVC Report:\n", classification_report(
    y_test, svc_pred, labels=labels_svc, target_names=target_names_svc
))

# --- Random Forest モデル ---
rf = RandomForestClassifier(42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

labels_rf = sorted(np.unique(np.concatenate((y_test, rf_pred))))
target_names_rf = le.inverse_transform(labels_rf)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Report:\n", classification_report(
    y_test, rf_pred, labels=labels_rf, target_names=target_names_rf
))

# --- PCA 可視化 ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# PCAの各主成分における特徴量の重み
print("PCA components:")
print("特徴量:", features)
print("PC1の重み:", pca.components_[0])
print("PC2の重み:", pca.components_[1])
print("PCAの寄与率:",pca.explained_variance_ratio_
)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='coolwarm', alpha=0.7)

# 凡例を表示（色をcoolwarmに合わせて手動で指定）
colors = ['#3b4cc0', '#b40426']  # coolwarmに対応（青→赤）
patches = [mpatches.Patch(color=colors[i], label=le.classes_[i]) for i in range(len(le.classes_))]
plt.legend(handles=patches, title="Sex")

plt.title("PCA of Penguin Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()


