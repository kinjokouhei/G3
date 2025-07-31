import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. データの準備 ---
print("--- 1. データの準備を開始 ---")
try:
    # 外れ値除去済みのデータを読み込む
    df = pd.read_csv("penguins_no_outliers.csv")
except FileNotFoundError:
    print("エラー: 'penguins_no_outliers.csv' が見つかりません。")
    exit()

# 特徴量とターゲット（ラベル）を定義
# (ファイルに含まれる特徴量から、以前の分析で有効だったものを選択)
features = ['Culmen Aspect Ratio / Body Mass', 'Culmen Depth (mm)', 'Body Mass (g)', 'Culmen Length (mm)']
target = 'Sex'

print(f"使用する特徴量: {features}")

# 特徴量 (X) とターゲット (y) に分ける
X = df[features]
y = df[target]

# ターゲットを数値に変換 (Label Encoding)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 特徴量を標準化（スケーリング）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- 2. 訓練データとテストデータに分割 ---
print("\n--- 2. データを訓練用とテスト用に分割 ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)


# --- 3. モデルの学習と予測 ---
print("\n--- 3. k-NNモデルで学習と予測を実行 ---")
# 事前のチューニングで最適だったパラメータを使用
model = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='uniform')
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# --- 4. 結果の評価 ---
print("\n" + "="*50)
print("--- 4. 最終的な評価結果 ---")
accuracy = accuracy_score(y_test, predictions)
print(f"モデルの正解率 (Accuracy): {accuracy:.4f}")

# より詳細なレポート
target_names = le.classes_
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=target_names))
print("="*50)