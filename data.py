import pandas as pd

# 1. データ読み込み
df = pd.read_csv("penguins_lter.csv")

# 2. 使用する特徴量とターゲット列を指定
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'

# 3. 欠損値を含む行を削除
df_clean = df[features + [target]].dropna()

# 4. 結果の確認
print(f"元のデータ件数: {len(df)}")
print(f"欠損除去後の件数: {len(df_clean)}")

# 5. 新しいCSVとして保存（例：clean_penguins.csv）
df_clean.to_csv("clean_penguins.csv", index=False)
print("クリーンデータを 'clean_penguins.csv' に保存しました。")
