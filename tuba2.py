import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- データ読み込みと前処理 ---
df = pd.read_csv("clean_penguins.csv")

features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'

df_clean = df[features + [target]].dropna()
df_clean = df_clean[df_clean['Sex'] != '.']

X = df_clean[features]
y = df_clean[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# --- ランダムサーチ用のパラメータ範囲 ---
param_dist = {
    'n_estimators': np.arange(100, 3001, 100),
    'max_depth': [None] + list(np.arange(5, 20, 2)),
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features':[None]
}

# --- RandomizedSearchCV で最適化 ---
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=200,             # ランダムに30パターン試す
    cv=5,                  # 5分割交差検証
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

# --- 結果の表示 ---
print("最適なパラメータ:", random_search.best_params_)

best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("最適化後の正解率:", accuracy_score(y_test, y_pred))
print("分類レポート:\n", classification_report(y_test, y_pred, target_names=le.classes_))