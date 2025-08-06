import pandas as pd
import os

DIR_PATH = "/home/yamaguchi/vmlserver06/Experiment/50salads/gt/"
FILE_NAME = "gt_03_2_5fps"
#csvをデータフレームとして読み込む
df = pd.read_csv(os.path.join(DIR_PATH, f'{FILE_NAME}.csv'), delimiter=';', quoting=1, on_bad_lines='warn')

# "prep"を語尾に含む行を抽出
convert_rows = df[df.iloc[:, 2].str.contains('prep', case=False, na=False)]

#convert_rowsの2列目から"_prep"を削除"
convert_rows.iloc[:, 2] = convert_rows.iloc[:, 2].str.replace('_prep', '', regex=False)

for index, row in convert_rows.iterrows():
    #rowの内容を含む行を抽出
    matching_rows = df[df.iloc[:, 2].str.contains(row.iloc[2], case=False, na=False)]
    #もし、matching_rowsの3行目の内容が"post"で終わる場合、3行目の2列目の値を1行目の2列目の値に置き換える
    while True:
        if not matching_rows.empty and len(matching_rows) >= 3 and matching_rows.iloc[2, 2].endswith('post'):
            matching_rows.iloc[0, 1] = matching_rows.iloc[2, 1]
            matching_rows.iloc[0, 2] = row['narration']
            #matching_rowsの1行目、2行目のみを削除
            df = df.drop(index=matching_rows.index[1:3])
            matching_rows = matching_rows.drop(matching_rows.index[1:3])
            df.loc[matching_rows.index[0], df.columns[1]] = matching_rows.iloc[0, 1]
            df.loc[matching_rows.index[0], df.columns[2]] = matching_rows.iloc[0, 2]
            matching_rows = matching_rows.drop(matching_rows.index[0])
        #そうでなく、2行目の内容が"core"で終わる場合、2行目の2列目の値を1行目の2列目の値に置き換える
        elif not matching_rows.empty and len(matching_rows) >= 2 and matching_rows.iloc[1, 2].endswith('core'):
            matching_rows.iloc[0, 1] = matching_rows.iloc[1, 1]
            matching_rows.iloc[0, 2] = row['narration']
            df = df.drop(index=matching_rows.index[1])
            matching_rows = matching_rows.drop(matching_rows.index[1])
            df.loc[matching_rows.index[0], df.columns[1]] = matching_rows.iloc[0, 1]
            df.loc[matching_rows.index[0], df.columns[2]] = matching_rows.iloc[0, 2]
            matching_rows = matching_rows.drop(matching_rows.index[0])
        else:
            matching_rows.drop(matching_rows.index[0])
            matching_rows = matching_rows.iloc[0:0]
        if matching_rows.empty:
            break
        
#dfのindexを振りなおす
df.reset_index(drop=True, inplace=True)

#dfをcsvとして保存
output_path = os.path.join(DIR_PATH, f'{FILE_NAME}_modified.csv')
df.to_csv(output_path, index=False, sep=';')
print(f"Modified CSV saved to: {output_path}")