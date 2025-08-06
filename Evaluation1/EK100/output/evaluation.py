#カレントディレクトリのcsvを読み込み、評価を行うスクリプト
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SEC = "10"
DIR_PATH = f"/home/yamaguchi/vmlserver06/Experiment/output/EK100/"

def load_csv(file_path):
    try:
        return pd.read_csv(file_path, delimiter=',', quoting=1, on_bad_lines='warn')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def combine_csvs(csv_list):
    combined_df = pd.DataFrame()
    cnt = 1
    index = 1
    for _, df in enumerate(csv_list):
        if cnt < 3:
            if df is not None:
                df['file_name'] = f"rgb0{index}-{cnt}"
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            cnt += 1
            if cnt == 3: cnt = 1;index += 1
    return combined_df

P01_11 = load_csv(DIR_PATH + SEC + f"sec/evaluation_P01_11@{SEC}.csv")
P01_12 = load_csv(DIR_PATH + SEC + f"sec/evaluation_P01_12@{SEC}.csv")
P01_13 = load_csv(DIR_PATH + SEC + f"sec/evaluation_P01_13@{SEC}.csv")
P01_14 = load_csv(DIR_PATH + SEC + f"sec/evaluation_P01_14@{SEC}.csv")
P01_15 = load_csv(DIR_PATH + SEC + f"sec/evaluation_P01_15@{SEC}.csv")

csv_list = [P01_11, P01_12, P01_13, P01_14, P01_15]

combined_df = combine_csvs(csv_list)
# "file_name"を除く各列ごとに平均、標準偏差、最大値、最小値を計算
stats = combined_df.drop(columns=['file_name']).agg(['mean', 'std', 'max', 'min']).T
stats.reset_index(inplace=True)
stats.columns = ["metrics", 'mean', 'std', 'max', 'min']

# 0,1,2,3行目はintで、それ以外は小数第3位まで表示
stats.iloc[0:4, 1:] = stats.iloc[0:4, 1:].astype(int)
stats.iloc[4:, 1:] = stats.iloc[4:, 1:].round(3)

# 結果を表示
stats = stats.T
print(stats)
# 結果をCSVとして保存
output_path = DIR_PATH + SEC + f"sec/evaluation_stats@{SEC}.csv"
stats.to_csv(output_path, index=True, sep=',')
print(f"Evaluation statistics saved to: {output_path}")