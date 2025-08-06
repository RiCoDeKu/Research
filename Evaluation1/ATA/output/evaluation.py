#カレントディレクトリのcsvを読み込み、評価を行うスクリプト
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DIR_PATH = "/home/yamaguchi/vmlserver06/Experiment/ATA/output/"

def load_csv(file_path):
    try:
        return pd.read_csv(file_path, delimiter=',', quoting=1, on_bad_lines='warn', nrows=1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def combine_csvs(csv_list):
    combined_df = pd.DataFrame()
    i = 1
    j = 1
    k = 1
    for _, df in enumerate(csv_list):
        if k < 3:
            if df is not None:
                df['file_name'] = f"eval_P{i}_{j}_T{k}"
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            k += 1
            if k == 3: k = 1; j += 1
            if j == 3: j = 1; i += 1
    return combined_df

for SEC in [1,2,3,5,7]:
    SEC = str(SEC)
    evalP1_1_T1 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_1_T1@{SEC}.csv")
    evalP1_1_T2 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_1_T2@{SEC}.csv")
    evalP1_1_T3 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_1_T3@{SEC}.csv")
    evalP1_2_T1 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_2_T1@{SEC}.csv")
    evalP1_2_T2 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_2_T2@{SEC}.csv")
    evalP1_2_T3 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_2_T3@{SEC}.csv")
    evalP1_3_T1 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_3_T1@{SEC}.csv")
    evalP1_3_T2 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_3_T2@{SEC}.csv")
    evalP1_3_T3 = load_csv(DIR_PATH + SEC + f"sec/eval_P1_3_T3@{SEC}.csv")
    evalP2_1_T1 = load_csv(DIR_PATH + SEC + f"sec/eval_P2_1_T1@{SEC}.csv")
    evalP2_1_T2 = load_csv(DIR_PATH + SEC + f"sec/eval_P2_1_T2@{SEC}.csv")
    evalP2_1_T3 = load_csv(DIR_PATH + SEC + f"sec/eval_P2_1_T3@{SEC}.csv")
    evalP2_2_T1 = load_csv(DIR_PATH + SEC + f"sec/eval_P2_2_T1@{SEC}.csv")
    evalP2_2_T2 = load_csv(DIR_PATH + SEC + f"sec/eval_P2_2_T2@{SEC}.csv")
    evalP2_2_T3 = load_csv(DIR_PATH + SEC + f"sec/eval_P2_2_T3@{SEC}.csv")

    csv_list = [
        evalP1_1_T1, evalP1_1_T2, evalP1_1_T3,
        evalP1_2_T1, evalP1_2_T2, evalP1_2_T3,
        evalP1_3_T1, evalP1_3_T2, evalP1_3_T3,
        evalP2_1_T1, evalP2_1_T2, evalP2_1_T3,
        evalP2_2_T1, evalP2_2_T2, evalP2_2_T3
    ]

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