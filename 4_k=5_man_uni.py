import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. データ準備と全15個の特徴量作成 ---
print("--- 1. 全15個の交互作用特徴量を作成します ---")
df = pd.read_csv("penguins_processed.csv")
df['Culmen Aspect Ratio'] = df['Culmen Length (mm)'] / df['Culmen Depth (mm)']

species_dummies = pd.get_dummies(df['Species'], prefix='Species')
df_with_dummies = pd.concat([df, species_dummies], axis=1)

numeric_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Culmen Aspect Ratio']
species_cols = species_dummies.columns
interaction_features = pd.DataFrame()

for feature in numeric_features:
    for species_col in species_cols:
        new_col_name = f"{species_col.split('_')[1]}_{feature}"
        interaction_features[new_col_name] = df_with_dummies[species_col] * df_with_dummies[feature]

# --- 2. 全15個の特徴量のうち、重要なものを選択 ---
print("\n--- 2. 特徴量の重要度を計算し、上位を選択します ---")
# RandomForestで重要度を計算
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(interaction_features, df['Sex'])
importances = rf.feature_importances_
feature_names = interaction_features.columns

# 重要度順にソート
ranked_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

# 上位8個の特徴量を選択（この数は調整可能）
num_top_features = 8
top_features = [feature for feature, importance in ranked_features[:num_top_features]]
print(f"選択された上位{num_top_features}個の特徴量: {top_features}")

X = interaction_features[top_features]
y = df['Sex']

# --- 3. 学習の準備と実行 ---
print("\n--- 3. 選択した特徴量でモデルを学習・評価します ---")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='uniform')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- 4. 最終評価 ---
print("\n" + "="*50)
print(f"--- 4. 上位{num_top_features}個の特徴量での最終評価 ---")
accuracy = accuracy_score(y_test, predictions)
print(f"最終モデルのAccuracy: {accuracy:.4f}")
print("="*50)