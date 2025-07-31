#このコードは、特徴量の生成プロセスを省略し、2つのファイルを読み込みます。

#interaction_features.csv: モデルが学習する特徴量（X）として使用します。

#penguins_processed.csv: 正解ラベルである性別（y）を取得するために使用します。
#
#




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. データ準備 ---
print("--- 1. データの準備を開始 ---")
try:
    # 交互作用特徴量と、元のファイルを両方読み込む
    X_df = pd.read_csv("interaction_features.csv")
    y_df = pd.read_csv("penguins_processed.csv")
except FileNotFoundError:
    print("エラー: 'interaction_features.csv' または 'penguins_processed.csv' が見つかりません。")
    exit()

# 特徴量 (X) とターゲット (y) を定義
X = X_df
y = y_df['Sex']

print(f"使用する特徴量の数: {len(X.columns)}")

# --- 2. 前処理とデータ分割 ---
# 前処理
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割 (再現性のためにrandom_stateを42に固定)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 3. モデルの学習と評価 ---
print("\n--- 3. モデルを学習・評価します ---")

# 事前に見つけた最適なパラメータを使用
model = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='uniform')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- 4. 最終評価 ---
print("\n" + "="*50)
print("--- 4. 最終的な評価結果 ---")
accuracy = accuracy_score(y_test, predictions)
print(f"最終モデルのAccuracy: {accuracy:.4f}")

# 詳細レポート
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=le.classes_))
print("="*50)