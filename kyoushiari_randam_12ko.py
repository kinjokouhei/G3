import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. データ準備と特徴量作成 ---
print("--- 1. データの準備と特徴量作成 ---")
df = pd.read_csv("penguins_processed.csv")
df['Culmen Aspect Ratio'] = df['Culmen Length (mm)'] / df['Culmen Depth (mm)']
df['Culmen Aspect Ratio / Body Mass'] = df['Culmen Aspect Ratio'] / df['Body Mass (g)']

# --- 2. 種ごとの交互作用特徴量を作成 ---
print("\n--- 2. 指定された特徴量から交互作用特徴量を作成します ---")
species_dummies = pd.get_dummies(df['Species'], prefix='Species')
df_with_dummies = pd.concat([df, species_dummies], axis=1)
numeric_features = [
    'Culmen Aspect Ratio / Body Mass', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Body Mass (g)'
]
species_cols = species_dummies.columns
interaction_features = pd.DataFrame()
for feature in numeric_features:
    for species_col in species_cols:
        new_col_name = f"{species_col.split('_')[1]}_{feature.replace(' (mm)', '').replace(' (g)', '')}"
        interaction_features[new_col_name] = df_with_dummies[species_col] * df_with_dummies[feature]

# --- 3. 学習の準備 ---
X = interaction_features
y = df['Sex']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. GridSearchCVで最適なパラメータを一度だけ見つける ---
print("\n--- 4. 最適なハイパーパラメータを探索します ---")
X_train_search, _, y_train_search, _ = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 3]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_search, y_train_search)
best_params = grid_search.best_params_
print(f"\nGridSearchCVで見つかった最適なパラメータ: {best_params}")

# --- 5. 最適なモデルで1000回の試行と精度評価 ---
print("\n--- 5. 最適なモデルで1000回の試行を開始します ---")
n_runs = 1000
accuracy_scores = []

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=i, stratify=y_encoded
    )
    model = RandomForestClassifier(**best_params, random_state=i)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)
    if (i + 1) % 100 == 0:
        print(f"実行回数 {i+1}/{n_runs} 完了...")

# --- 6. 最終結果の表示 ---
print("\n" + "="*50)
print("--- 6. 最終結果 ---")
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
print(f"実行回数: {n_runs}回")
print(f"平均正解率 (Mean Accuracy): {mean_accuracy:.4f}")
print(f"精度の標準偏差 (Standard Deviation): {std_accuracy:.4f}")
print("="*50)