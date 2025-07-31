import pandas as pd

# 1. データ読み込み
df = pd.read_csv("penguins_lter.csv")

# 2. 使用する特徴量とターゲット列を指定
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Sex'
group_col = 'Species'  # グループ分けに使う列名（列名はファイルに応じて大文字小文字を正確に）

# 3. Speciesごとに処理
grouped = df[[group_col] + features + [target]].dropna().groupby(group_col)

# 4. 各グループごとにCSV出力
for species_name, group_df in grouped:
    filename = f"clean_penguins_{species_name.replace(' ', '_')}.csv"
    group_df.to_csv(filename, index=False)
    print(f"{species_name} のクリーンデータを '{filename}' に保存しました。")

# 5. 件数表示（任意）
print(f"元のデータ件数: {len(df)}")
print(f"欠損除去後の合計件数: {sum(len(g) for _, g in grouped)}")

