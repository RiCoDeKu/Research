import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the EPIC-KITCHEN-100 dataset.")
    parser.add_argument('--gt', type=str, required=True, help='Path to the ground truth CSV file.')
    parser.add_argument('--pred', type=str, required=True, help='Path to the predictions CSV file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the evaluation results.')
    return parser.parse_args()

def load_data(path):
    data = pd.read_csv(path, delimiter=';', quoting=1, on_bad_lines='warn')
    return data

def evaluate(gt, pred):
    gt_frames = gt["start_frame"]
    pred_frames = pred["start_frame"]
    # 各フレームの前後5フレームにpredictionsがあるか確認
    results = []
    TP = []
    TP_d = []
    FP = []
    FN = []
    allow_range = 600
    for y in pred_frames:
        for x in gt_frames:
            if x-allow_range <= y <= x+allow_range:
                if y not in TP: TP.append(y)
                if x not in TP_d: TP_d.append(x)
        if y not in TP: FP.append(y)
    for x in gt_frames:
        if x not in TP_d:
            FN.append(x)
            
    print(f"TP: {TP}\n TP_d: {TP_d}\n FP: {FP}\n FN: {FN}\n")
    beta= 2
    Precision= len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
    Recall= len(TP_d) / (len(TP_d) + len(FN)) if (len(TP_d) + len(FN)) > 0 else 0
    F2 = (1+beta**2) * Precision * Recall / (beta**2 * Precision + Recall)
    Recall_F1 = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
    F1 = 2 * ((Precision * Recall_F1) / (Precision + Recall_F1)) if (Precision + Recall_F1) > 0 else 0

    results.append({
        "TP": len(TP),
        "TP_d": len(TP_d),
        "FP": len(FP),
        "FN": len(FN),
        "Precision": np.round(Precision, 4),
        "Recall": np.round(Recall, 4),
        "F2": np.round(F2, 4),
        "F1": np.round(F1, 4),
    })
    return pd.DataFrame(results), allow_range

if __name__ == "__main__":
    args = parse_args()
    gt = load_data(args.gt)
    pred = load_data(args.pred)

    # Example of processing the data
    print("Ground Truth Data:")
    print(gt["start_frame"].head())

    print("\nPredictions Data:")
    print(pred["start_frame"].head())

    # Perform evaluation
    results, allow_range = evaluate(gt, pred)
    print("\nEvaluation Results:")
    print(results.head())
    
    # Save results to CSV
    file_range = str(int(allow_range / 60))
    output_path = args.output.replace(".csv", f"@{file_range}.csv")
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Further evaluation logic can be added here