# カレントディレクトリにあるEPIC_100_validation.csvから、対象のgtを出力するソースコード
import pandas as pd
import os
import argparse

overall = pd.read_csv("EPIC_100_validation.csv", delimiter=',', quoting=1, on_bad_lines='warn')

def parse_args():
    parser = argparse.ArgumentParser(description="Extract ground truth data for EPIC-KITCHEN-100.")
    parser.add_argument('--id', type=str, required=True, help='ID of the ground truth to extract.')
    return parser.parse_args()

def main():
    argparse = parse_args()
    id = argparse.id
    
    # IDに基づいて行を抽出
    gt = overall[overall['video_id'] == id]
    # gtからvideo_id, start_frame, stop_frame, narrationの列を抽出
    gt = gt[['start_frame', 'stop_frame', 'narration']]
    # 列名を変更
    gt.columns = ['start_frame', 'end_frame', 'narration']
    # start_frameとend_frameを整数型に変換
    gt['start_frame'] = gt['start_frame'].astype(int)
    gt['end_frame'] = gt['end_frame'].astype(int)
    #start_frameでsort
    gt = gt.sort_values(by='start_frame').reset_index(drop=True)
    
    # 出力ディレクトリを作成
    output_dir = f"./"
    os.makedirs(output_dir, exist_ok=True)
    # 出力ファイル名を設定
    output_file = os.path.join(output_dir, f'gt_{id}.csv')
    # gtをCSVとして保存
    gt.to_csv(output_file, index=False, sep=',', quoting=1)
    print(f"Ground truth data for ID {id} saved to: {output_file}")
    
if __name__ == "__main__":
    main()