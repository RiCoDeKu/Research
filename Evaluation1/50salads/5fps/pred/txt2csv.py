#カレントディレクトリの全てのtxtファイルに列名を追加し、csvファイルに変換するスクリプト
import os
import pandas as pd

def txt_to_csv(txt_path):
    try:
        # txtファイルを読み込み、列名を追加
        df = pd.read_csv(txt_path, delimiter=';', quoting=1, on_bad_lines='warn', header=None)
        df.columns = ['start_frame', 'stop_frame', 'narration']
        
        # 出力ファイル名を設定
        output_path = txt_path.replace('.txt', '.csv')
        
        # csvファイルとして保存
        df.to_csv(output_path, index=False, sep=';', quoting=3)
        print(f"Converted {txt_path} to {output_path}")
        # txtファイルを削除
        os.remove(txt_path)
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
        
if __name__ == "__main__":
    # カレントディレクトリの全てのtxtファイルを取得
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    for txt_file in txt_files:
        txt_to_csv(txt_file)
    print("Conversion completed.")