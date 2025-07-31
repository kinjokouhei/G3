import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

# --- ハイパーパラメータのグリッドサーチ ---
param_grid = {
    'n_estimators': [10, 50, 100, 200, 300, 400, 500, 600],
    'max_depth': [None, 5, 10, 15, 20, 30, 40],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 4, 6, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5分割交差検証
    n_jobs=-1,  # 並列処理
    verbose=1
)

grid_search.fit(X_train, y_train)

# --- 結果の表示 ---
print("最適なパラメータ:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("最適化後の正解率:", accuracy_score(y_test, y_pred))
print("分類レポート:\n", classification_report(y_test, y_pred, target_names=le.classes_))