import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. データ準備 ---
print("--- 1. データの準備を開始 ---")
df = pd.read_csv("penguins_processed.csv")

# --- 2. 種ごとの特徴量を作成 ---
print("\n--- 2. 種ごとの特徴量を作成します ---")

# ★★★★★ ここから追加 ★★★★★
# 'Culmen Aspect Ratio' を計算してDataFrameに追加
df['Culmen Aspect Ratio'] = df['Culmen Length (mm)'] / df['Culmen Depth (mm)']
# ★★★★★★★★★★★★★★★

# ワンホットエンコーディングでSpecies列を変換
species_dummies = pd.get_dummies(df['Species'], prefix='Species')
df_with_dummies = pd.concat([df, species_dummies], axis=1)

# ★★★★★ ここを修正 ★★★★★
# 元の測定特徴量リストに 'Culmen Aspect Ratio' を追加
numeric_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Culmen Aspect Ratio']
# ★★★★★★★★★★★★★★★

species_cols = species_dummies.columns
interaction_features = pd.DataFrame()

# 各測定特徴量と、各種の列を掛け合わせる
for feature in numeric_features:
    for species_col in species_cols:
        new_col_name = f"{species_col.split('_')[1]}_{feature}"
        interaction_features[new_col_name] = df_with_dummies[species_col] * df_with_dummies[feature]

# (前略：forループが終わった直後)

print(f"作成された新しい特徴量の数: {len(interaction_features.columns)}")

# ★★★★★ こちらの方法を推奨します ★★★★★
output_filename = "interaction_features.csv"
interaction_features.to_csv(output_filename, index=False)

print(f"\n--- 変換後のデータセットを '{output_filename}' に保存しました。 ---")
print("ExcelやGoogleスプレッドシートで開いて内容を確認してください。")
# ★★★★★★★★★★★★★★★★★★★★★★★


print(f"作成された新しい特徴量の数: {len(interaction_features.columns)}")

# --- 3. 学習の準備 ---
X = interaction_features
y = df['Sex']

# 前処理
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 4. 最適なモデルを探索・評価 ---
print("\n--- 4. 新しい特徴量セットで最適なモデルを探索・評価します ---")

param_grid = {
    'n_neighbors': range(3, 16),
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\n見つかった最適なパラメータ: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# --- 5. 最終評価 ---
print("\n" + "="*50)
print("--- 5. 最終的な評価結果 ---")
print(f"最終モデルのAccuracy: {accuracy_score(y_test, predictions):.4f}")
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=le.classes_))
print("="*50)