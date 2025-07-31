import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- 1. データの準備 (一度だけ実行) ---
print("--- 1. データの準備を開始 ---")
try:
    df = pd.read_csv("penguins_processed.csv")
except FileNotFoundError:
    print("エラー: 'penguins_processed.csv' が見つかりません。")
    exit()

features = ['Culmen Aspect Ratio / Body Mass', 'Culmen Depth (mm)', 'Body Mass (g)', 'Culmen Length (mm)']
target = 'Sex'
X_df = df[features]
y_s = df[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y_s)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# --- 2. 1000回の試行ループで、失敗したデータのインデックスを収集 ---
print("\n--- 2. 1000回の試行を開始し、失敗事例を収集します ---")
n_runs = 1000
# 失敗したデータの元のインデックスを保存するset（重複を自動で防ぐ）
all_failure_indices = set()

for i in range(n_runs):
    # random_stateを指定せず、毎回ランダムにデータを分割
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_scaled, y_encoded, df.index, test_size=0.2
    )

    # モデルの学習と予測
    model = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='uniform')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 失敗したデータの元のインデックスを取得
    failure_mask = predictions != y_test
    original_failure_indices = indices_test[failure_mask]
    
    # setに今回の失敗事例のインデックスを追加
    all_failure_indices.update(original_failure_indices)

    # 100回ごとに進行状況を表示
    if (i + 1) % 100 == 0:
        print(f"実行回数 {i+1}/{n_runs} 完了...")

# --- 3. 収集した全失敗事例をファイルに保存 ---
print("\n" + "="*50)
print("--- 3. 収集した全失敗事例をファイルに保存します ---")

if not all_failure_indices:
    print("1000回の試行で一度も失敗はありませんでした。")
else:
    # setをソートされたリストに変換
    sorted_failure_indices = sorted(list(all_failure_indices))
    
    # 元のDataFrameから、失敗したことがある行だけを抽出
    all_failures_df = df.loc[sorted_failure_indices]
    
    # CSVファイルとして保存
    output_filename = 'all_time_failures.csv'
    all_failures_df.to_csv(output_filename, index=False)
    
    print(f"実行回数: {n_runs}回")
    print(f"一度でも失敗したユニークなデータ数: {len(all_failures_df)}件")
    print(f"全失敗事例を '{output_filename}' に保存しました。")
    
    print("\n▼ 収集された失敗事例（一部）:")
    print(all_failures_df.head())

print("="*50)