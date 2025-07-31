import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. データ準備と特徴量作成 ---
print("--- 1. データの準備と特徴量作成 ---")
df = pd.read_csv("clean_penguins.csv")
original_features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'
df_clean = df[original_features + [target]].dropna().reset_index(drop=True)
df_clean = df_clean[df_clean['Sex'] != '.'].reset_index(drop=True)
df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
df_clean['new'] = df_clean['Culmen Aspect Ratio'] / df_clean['Body Mass (g)']

# --- 2. 外れ値の特定と除去 ---
print("\n--- 2. 外れ値を特定し、除去します ---")
male_df = df_clean[df_clean['Sex'] == 'MALE']
female_df = df_clean[df_clean['Sex'] == 'FEMALE']

outlier_indices = []
for sex, group_df in [('MALE', male_df), ('FEMALE', female_df)]:
    Q1 = group_df['new'].quantile(0.25)
    Q3 = group_df['new'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    # ★★★★★ ここから追加 ★★★★★
    print(f"\n--- {sex} グループの外れ値判定基準 ---")
    print(f"下限値 (これより小さい値が外れ値): {lower_bound:.6f}")
    print(f"上限値 (これより大きい値が外れ値): {upper_bound:.6f}")
    # ★★★★★ ここまで追加 ★★★★★

    outliers = group_df[(group_df['new'] < lower_bound) | (group_df['new'] > upper_bound)]
    outlier_indices.extend(outliers.index)

unique_outlier_indices = sorted(list(set(outlier_indices)))
print("\n--- 除去対象となる外れ値の詳細 ---")
if not unique_outlier_indices:
    print("外れ値は見つかりませんでした。")
else:
    outlier_details = df_clean.loc[unique_outlier_indices]
    print(outlier_details)

print(f"\n合計 {len(unique_outlier_indices)} 件の外れ値をデータセットから除去します。")
df_no_outliers = df_clean.drop(unique_outlier_indices)
print(f"元のデータ数: {len(df_clean)}, 外れ値除去後のデータ数: {len(df_no_outliers)}")

# --- 3. 外れ値除去後のデータで、再度モデルを評価 ---
print("\n--- 3. 外れ値除去後のデータでモデル評価を開始 ---")
selected_features = ['new', 'Culmen Depth (mm)', 'Body Mass (g)', 'Culmen Length (mm)']
X = df_no_outliers[selected_features]
y = df_no_outliers[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_runs = 50 
accuracy_scores = [] 

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=i+5 
    )
    model = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='uniform')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)