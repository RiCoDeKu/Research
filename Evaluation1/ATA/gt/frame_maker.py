import os

# カレントディレクトリのtxtファイルを一覧で取得
def get_txt_files():
    current_dir = os.getcwd()
    txt_files = [f for f in os.listdir(current_dir) if f.endswith('.txt')]
    return txt_files

data_files = get_txt_files()

for f in data_files:
    with open(f, 'r') as file:
        lines = file.readlines()
    file_name = os.path.splitext(f)[0]
    output_file = f"{file_name}.csv"
    with open(output_file, 'w') as out_file:
        out_file.write("start_frame;stop_frame;narration\n")
    # 各行がフレームに相当し、その行にはnarration内容が含まれています。これを"start_frame;stop_frame;narration"の形式に変換します。
    # narrationの区切りまでの行数をカウントし、start_frameとstop_frameを設定します。
    
    for i, line in enumerate(lines):
        narration = line.strip()
        if i == 0:
            start_frame = i
            stop_frame = start_frame
        else:
            if narration == prev_narration:
                stop_frame = stop_frame + 1
            else:
                with open(output_file, 'a') as out_file:
                    out_file.write(f"{start_frame};{stop_frame};{prev_narration}\n")
                start_frame = stop_frame + 1
                stop_frame += 1
        prev_narration = narration
    # 最後のnarrationを出力
    if 'prev_narration' in locals():
        with open(output_file, 'a') as out_file:
            out_file.write(f"{start_frame};{stop_frame};{prev_narration}\n")
            
    print(f"{output_file} created successfully.")
print("All files processed.")