# IoU/CapIoUのcsvをまとめて評価するスクリプト
import os
import csv
import pandas as pd
import numpy as np

DATASET_LIST= ["50salads", "ATA", "EK100"]
cnt=0
for DATASET in DATASET_LIST:
	print(f"\n########## DATASET: {DATASET} ##########")
	## VIDEO_NAME ##
	if DATASET 		== "50salads": VIDEO_LIST 	= ["rgb_01_1", "rgb_01_2", "rgb_02_1"]
	elif DATASET 	== "ATA": VIDEO_LIST 		= ["P1_1_T1", "P1_1_T2", "P1_1_T3"]
	elif DATASET 	== "EK100": VIDEO_LIST 		= ["P01_11", "P01_12", "P01_13"]
	MODE = "CapIou" # "CapIoU" or "IoU"

	## FPS ##
	if DATASET == "EK100": fps=60
	else: fps = 30

	for type in ["llava", "gemini", "rag", "hoi"]:
		for iou in [0.3, 0.5, 0.7]:
			print(f"\n### TYPE: {type} | IoU: {iou} ###")
			MODE = "CapIou"
			csv1 = f"/home/yamaguchi/vmlserver06/Research/Evaluation1/{DATASET}/{fps}fps/output/{MODE}/{type}/{iou}/cap_iou_{VIDEO_LIST[0]}.csv"
			csv2 = f"/home/yamaguchi/vmlserver06/Research/Evaluation1/{DATASET}/{fps}fps/output/{MODE}/{type}/{iou}/cap_iou_{VIDEO_LIST[1]}.csv"
			csv3 = f"/home/yamaguchi/vmlserver06/Research/Evaluation1/{DATASET}/{fps}fps/output/{MODE}/{type}/{iou}/cap_iou_{VIDEO_LIST[2]}.csv"
			# 1行目をヘッダーとして、csvをdataframeに変換
			df1 = pd.read_csv(csv1, header=0)
			df2 = pd.read_csv(csv2, header=0)
			df3 = pd.read_csv(csv3, header=0)
			#df1~df3に要素として含まれる"f1_combined"について、平均を計算
			mean_f1_combined = (df1["f1_combined"] + df2["f1_combined"] + df3["f1_combined"]) / 3
			MODE = "IoU"
			csv1 = f"/home/yamaguchi/vmlserver06/Research/Evaluation1/{DATASET}/{fps}fps/output/{MODE}/{type}/{iou}/iou_{VIDEO_LIST[0]}.csv"
			csv2 = f"/home/yamaguchi/vmlserver06/Research/Evaluation1/{DATASET}/{fps}fps/output/{MODE}/{type}/{iou}/iou_{VIDEO_LIST[1]}.csv"
			csv3 = f"/home/yamaguchi/vmlserver06/Research/Evaluation1/{DATASET}/{fps}fps/output/{MODE}/{type}/{iou}/iou_{VIDEO_LIST[2]}.csv"
			# 1行目をヘッダーとして、csvをdataframeに変換
			df1 = pd.read_csv(csv1, header=0)
			df2 = pd.read_csv(csv2, header=0)
			df3 = pd.read_csv(csv3, header=0)
			#df1~df3に要素として含まれる"f1"について、平均を計算
			mean_f1 = (df1["f1"] + df2["f1"] + df3["f1"]) / 3
			print("mean_f1_combined: ",mean_f1_combined)
			print("mean_f1: ",mean_f1)

			# mean_f1_combinedとmean_f1をcsvに保存
			output_dir = f"/home/yamaguchi/vmlserver06/Research/Evaluation1/output/summary/{DATASET}/{fps}fps/{type}"
			os.makedirs(output_dir, exist_ok=True)
			output_csv = f"{output_dir}/summary_{DATASET}_{type}_{iou}.csv"
			with open(output_csv, "w", newline="") as f:
				writer = csv.writer(f)
				writer.writerow(["mean_f1_combined", "mean_f1"])
				writer.writerow([np.round(mean_f1_combined.values[0], 4), np.round(mean_f1.values[0], 4)])
			cnt+=1
print(f"\n########## All Done! Processed {cnt} combinations. ##########\n")