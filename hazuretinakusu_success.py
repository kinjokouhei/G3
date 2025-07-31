import pandas as pd
import numpy as np

# --- 1. データ準備 ---
print("--- 1. データの準備を開始 ---")
df = pd.read_csv("clean_penguins.csv")
df_clean = df.dropna()
df_clean = df_clean[df_clean['Sex'] != '.']

# 特徴量を作成
df_clean['Culmen Aspect Ratio'] = df_clean['Culmen Length (mm)'] / df_clean['Culmen Depth (mm)']
df_clean['new'] = df_clean['Culmen Aspect Ratio'] / df_clean['Body Mass (g)']

# --- 2. 外れ値を特定する ---
print("--- 2. 外れ値を特定します ---")
male_df = df_clean[df_clean['Sex'] == 'MALE']
female_df = df_clean[df_clean['Sex'] == 'FEMALE']

outlier_indices = []

for sex, group_df in [('MALE', male_df), ('FEMALE', female_df)]:
    Q1 = group_df['new'].quantile(0.25)
    Q3 = group_df['new'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    outliers = group_df[(group_df['new'] < lower_bound) | (group_df['new'] > upper_bound)]
    outlier_indices.extend(outliers.index)

# --- 3. 特定した外れ値を除去する ---
print("--- 3. 外れ値を除去します ---")
unique_outlier_indices = sorted(list(set(outlier_indices)))
print(f"合計 {len(unique_outlier_indices)} 件の外れ値をデータセットから除去します。")

df_no_outliers = df_clean.drop(unique_outlier_indices)

print(f"元のデータ数: {len(df_clean)}")
print(f"外れ値除去後のデータ数: {len(df_no_outliers)}")


# --- ★ 4. 新しいCSVファイルとして保存 ---
print("\n" + "="*50)
print("--- 4. 新しいファイルに保存します ---")

# 新しいファイル名を定義
output_filename = "penguins_no_outliers.csv"

# .to_csv() を使ってファイルに書き出す
# index=False は、DataFrameのインデックスをファイルに含めないための重要な設定
df_no_outliers.to_csv(output_filename, index=False)

print(f"外れ値除去後のデータセットを '{output_filename}' という名前で保存しました。")
print("="*50)