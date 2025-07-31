import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# --- データ読み込みと前処理 ---
df = pd.read_csv("clean_penguins.csv")

# 特徴量とターゲット
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'

# 欠損値・性別不明データを除外
df_clean = df[features + [target]].dropna()
df_clean = df_clean[df_clean['Sex'] != '.']

# 特徴量とターゲットに分ける
X = df_clean[features]
y = df_clean[target]

# ラベルエンコード（FEMALE→0, MALE→1）
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 訓練・テストに分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2,random_state= 27
)

# --- K近傍 ---
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
labels_knn = sorted(np.unique(np.concatenate((y_test, knn_pred))))
target_names_knn = le.inverse_transform(labels_knn)

print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Classification Report:")
print(classification_report(y_test, knn_pred, labels=labels_knn, target_names=target_names_knn))

# --- 線形SVM（LinearSVC）---
svc = LinearSVC(C = 1.2,max_iter=10000)  # 収束しやすくするため反復回数を増やす
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

labels_svc = sorted(np.unique(np.concatenate((y_test, svc_pred))))
target_names_svc = le.inverse_transform(labels_svc)

print("LinearSVC Accuracy:", accuracy_score(y_test, svc_pred))
print("LinearSVC Classification Report:")
print(classification_report(y_test, svc_pred, labels=labels_svc, target_names=target_names_svc))

