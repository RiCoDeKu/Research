# 50salads, ATA, Epic-Kitchensの動画をリストで格納する

import pandas as pd
import os

# カレントディレクトリの.mp4ファイルをリストで取得
def get_video_files(directory):
    video_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            video_files.append(os.path.join(directory, filename))
    return video_files

def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(directory, filename))
    return csv_files

#動画リストを取得
salads_videos = get_video_files('/home/yamaguchi/vmlserver06/dataset/50salads/30fps/video/frame')
ata_videos = get_video_files('/home/yamaguchi/vmlserver06/dataset/ATA/video/frame')
epic_videos = get_video_files('/home/yamaguchi/vmlserver06/dataset/EK100/video')

#csvリストを取得 (Ground Truth)
gt_salads_csvs = get_csv_files('/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt')
gt_ata_csvs = get_csv_files('/home/yamaguchi/vmlserver06/Research/Evaluation1/ATA/gt')
gt_epic_csvs = get_csv_files('/home/yamaguchi/vmlserver06/Research/Evaluation1/EK100/gt')

#csvリストを取得 (Prediction)
pred_salads_csvs = get_csv_files('/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred')
pred_ata_csvs = get_csv_files('/home/yamaguchi/vmlserver06/Research/Evaluation1/ATA/prediction')
pred_epic_csvs = get_csv_files('/home/yamaguchi/vmlserver06/Research/Evaluation1/EK100/pred/frame')

print(f"50salads: {len(salads_videos)} videos, {len(gt_salads_csvs)} GT CSVs, {len(pred_salads_csvs)} Pred CSVs")
print(f"ATA: {len(ata_videos)} videos, {len(gt_ata_csvs)} GT CSVs, {len(pred_ata_csvs)} Pred CSVs")
print(f"Epic-Kitchens: {len(epic_videos)} videos, {len(gt_epic_csvs)} GT CSVs, {len(pred_epic_csvs)} Pred CSVs")

# print("【Phase】50salads videos:")
# for index, video in enumerate(salads_videos):
#     output_dir='/home/yamaguchi/vmlserver06/Experiment/scene_movie/output/50salads'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     print(f"video {index}: {video}")
#     gt_csv = gt_salads_csvs[index] if index < len(gt_salads_csvs) else None
#     pred_csv = pred_salads_csvs[index] if index < len(pred_salads_csvs) else None
#     print(f"  GT CSV: {gt_csv}")
#     os.system(f"CUDA_VISIBLE_DEVICES=1,2 python multi_gpu_creater.py --video {video} --csv {gt_csv} --gpus 2 --output {output_dir}")
#     print(f"  Pred CSV: {pred_csv}")
#     os.system(f"CUDA_VISIBLE_DEVICES=1,2 python multi_gpu_creater.py --video {video} --csv {pred_csv} --gpus 2 --output {output_dir}")

# print("【Phase】ATA videos:")    
# for index, video in enumerate(ata_videos):
#     output_dir='/home/yamaguchi/vmlserver06/Experiment/scene_movie/output/ATA'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     print(f"video {index}: {video}")
#     gt_csv = gt_ata_csvs[index] if index < len(gt_ata_csvs) else None
#     pred_csv = pred_ata_csvs[index] if index < len(pred_ata_csvs) else None
#     print(f"  GT CSV: {gt_csv}")
#     os.system(f"CUDA_VISIBLE_DEVICES=1,2 python multi_gpu_creater.py --video {video} --csv {gt_csv} --gpus 2 --output {output_dir}")
#     print(f"  Pred CSV: {pred_csv}")
#     os.system(f"CUDA_VISIBLE_DEVICES=1,2 python multi_gpu_creater.py --video {video} --csv {pred_csv} --gpus 2 --output {output_dir}")

# print("【Phase】Epic-Kitchens videos:")
# for index, video in enumerate(epic_videos):
#     output_dir='/home/yamaguchi/vmlserver06/Experiment/scene_movie/output/EK100'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     print(f"video {index}: {video}")
#     gt_csv = gt_epic_csvs[index] if index < len(gt_epic_csvs) else None
#     pred_csv = pred_epic_csvs[index] if index < len(pred_epic_csvs) else None
#     print(f"  GT CSV: {gt_csv}")
#     os.system(f"CUDA_VISIBLE_DEVICES=1,2 python multi_gpu_creater.py --video {video} --csv {gt_csv} --gpus 2 --output {output_dir}")
#     print(f"  Pred CSV: {pred_csv}")
#     os.system(f"CUDA_VISIBLE_DEVICES=1,2 python multi_gpu_creater.py --video {video} --csv {pred_csv} --gpus 2 --output {output_dir}")

# print("【Successfully】All videos processed!!")
#DEBUG
for i in [4,5,6]:
    print(f"##########################\nProcessing CSV: {i}\n##########################")
    os.system(f"CUDA_VISIBLE_DEVICES=1,2 python multi_gpu_creater.py --video /home/yamaguchi/vmlserver06/dataset/50salads/30fps/video/frame/rgb-01-1f.mp4 --csv /home/yamaguchi/vmlserver06/Research/scene_movie/result_videoRAG_gemini_{i}.csv --gpus 2 --output ./output")