import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. データの準備 ---
print("--- 1. データの準備を開始 ---")
df = pd.read_csv("penguins_processed.csv")


features = ['Culmen Aspect Ratio / Body Mass', 'Culmen Depth (mm)', 'Body Mass (g)', 'Culmen Length (mm)']
target = 'Sex' 

X = df[features]
y = df[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. 訓練データとテストデータに分割 ---
# ★ random_stateを固定して、毎回同じ分割結果になるようにする
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_scaled, y_encoded, df.index, test_size=0.2
)

# --- 3. モデルの学習と予測 ---
print("\n--- 3. k-NNモデルで学習と予測を実行 ---")
model = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='uniform')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- 4. 失敗したデータの特定と保存 ---
print("\n--- 4. 失敗したデータを特定し、CSVに保存します ---")

# 予測 (predictions) と正解 (y_test) が異なる箇所のインデックス(True/False)を取得
failure_mask = predictions != y_test

# 失敗したデータの元のDataFrameでのインデックスを取得
original_failure_indices = indices_test[failure_mask]

if len(original_failure_indices) > 0:
    # 元のDataFrameから失敗した行を抽出
    failed_df = df.loc[original_failure_indices].copy()
    
    # 失敗した予測結果（エンコードされた値）を取得
    incorrect_predictions_encoded = predictions[failure_mask]
    
    # 予測結果を元のラベル（'FEMALE', 'MALE'）に戻して、新しい列として追加
    failed_df['Predicted_Sex'] = le.inverse_transform(incorrect_predictions_encoded)
    
    # CSVファイルとして保存
    output_filename = 'test_failures.csv'
    failed_df.to_csv(output_filename, index=False)
    
    print(f"\n予測に失敗したデータ ({len(failed_df)}件) を '{output_filename}' に保存しました。")
    print("\n▼ 失敗したデータの内容:")
    print(failed_df)
else:
    print("\nテストデータでの失敗はありませんでした。パーフェクトです！")

# --- 5. 全体の結果評価 ---
print("\n" + "="*50)
print("--- 5. 最終的な評価結果 ---")
accuracy = accuracy_score(y_test, predictions)
print(f"モデルの正解率 (Accuracy): {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=le.classes_))
print("="*50)