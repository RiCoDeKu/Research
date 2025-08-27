#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_args():
    p = argparse.ArgumentParser(description="Temporal segment evaluation by IoU (50Salads etc.)")
    p.add_argument("--gt", required=True, help="Ground-truth CSV path")
    p.add_argument("--pred", required=True, help="Predictions CSV path")
    p.add_argument("--output", required=True, help="Output CSV (metrics). Sidecar with pairs/unmatched will be created.")
    p.add_argument("--fps", type=float, default=25.0, help="FPS when converting seconds to frames")
    p.add_argument("--sep", default=";", help="CSV delimiter (default=';')")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold to count a TP")
    p.add_argument("--matching", choices=["greedy", "hungarian"], default="greedy",
                   help="Assignment method for 1-1 matching (default: greedy).")
    p.add_argument("--viz", action="store_true", help="Save timeline & IoU-matrix figures next to output CSV")
    p.add_argument("--time-unit", choices=["frames","seconds"], default="frames",
                   help="Timeline x-axis unit for visualization")
    p.add_argument("--max-matrix", type=int, default=200,
                   help="Max size for IoU matrix visualization (rows/cols); larger is subsampled")
    return p.parse_args()

def _to_frames(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # frames（start_frame, stop_frame）を想定
    if "start_frame" in cols and "stop_frame" in cols:
        s = df[cols["start_frame"]].astype(float)
        e = df[cols["stop_frame"]].astype(float)
    # 秒（start_t, end_t）にも対応したい場合は以下を有効化
    elif "start_t" in cols and "end_t" in cols:
        s = np.round(df[cols["start_t"]].astype(float) * fps)
        e = np.round(df[cols["end_t"]].astype(float) * fps)
    else:
        raise ValueError("CSV must contain start/end columns: start_frame,stop_frame or start_t,end_t.")
    s = s.astype(int).to_numpy()
    e = e.astype(int).to_numpy()
    s, e = np.minimum(s, e), np.maximum(s, e)  # safety
    return pd.DataFrame({"start_frame": s, "stop_frame": e})

def load_csv(path, sep, fps):
    df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="warn")
    return _to_frames(df, fps)

def seg_iou(s1, e1, s2, e2, inclusive=True):
    if inclusive:
        inter = max(0, min(e1, e2) - max(s1, s2) + 1)
        len1  = (e1 - s1 + 1)
        len2  = (e2 - s2 + 1)
    else:
        inter = max(0, min(e1, e2) - max(s1, s2))
        len1  = max(0, e1 - s1)
        len2  = max(0, e2 - s2)
    union = len1 + len2 - inter
    return inter / union if union > 0 else 0.0

def iou_matrix(pred_segs, gt_segs):
    P, G = len(pred_segs), len(gt_segs)
    M = np.zeros((P, G), dtype=float)
    for i, (ps, pe) in enumerate(pred_segs):
        for j, (gs, ge) in enumerate(gt_segs):
            M[i, j] = seg_iou(ps, pe, gs, ge, inclusive=True)
    return M

def greedy_match(M, thr):
    pairs = []
    if M.size == 0:
        return pairs
    idxs = np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1], M.shape))[0]
    used_p, used_g = set(), set()
    for pi, gi in idxs:
        if pi in used_p or gi in used_g:
            continue
        iou = M[pi, gi]
        if iou >= thr:
            pairs.append((int(pi), int(gi), float(iou)))
            used_p.add(int(pi)); used_g.add(int(gi))
    return pairs

def hungarian_match(M, thr):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return greedy_match(M, thr)
    if M.size == 0:
        return []
    cost = 1.0 - M
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        iou = float(M[r, c])
        if iou >= thr:
            pairs.append((int(r), int(c), iou))
    return pairs

def evaluate(gt_df, pred_df, iou_thr=0.5, matching="greedy"):
    gt_segs   = list(zip(gt_df["start_frame"].to_numpy(),  gt_df["stop_frame"].to_numpy()))
    pred_segs = list(zip(pred_df["start_frame"].to_numpy(), pred_df["stop_frame"].to_numpy()))
    M = iou_matrix(pred_segs, gt_segs)
    if matching == "hungarian":
        pairs = hungarian_match(M, iou_thr)
    else:
        pairs = greedy_match(M, iou_thr)
    matched_pred = {p for p,_,_ in pairs}
    matched_gt   = {g for _,g,_ in pairs}
    TP = len(pairs)
    FP = len(pred_segs) - TP
    FN = len(gt_segs) - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    ious = [iou for _,_,iou in pairs]
    mean_iou   = float(np.mean(ious)) if ious else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    details = {
        "pairs": [{"pred_idx": p, "gt_idx": g, "iou": round(i, 6)} for p,g,i in pairs],
        "unmatched_pred": [i for i in range(len(pred_segs)) if i not in matched_pred],
        "unmatched_gt":   [i for i in range(len(gt_segs))   if i not in matched_gt],
    }
    metrics = {
        "num_gt": len(gt_segs),
        "num_pred": len(pred_segs),
        "tp": TP, "fp": FP, "fn": FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou_matched": round(mean_iou, 4),
        "median_iou_matched": round(median_iou, 4),
        "iou_thr": iou_thr,
        "matching": matching,
    }
    return metrics, details

# ---------- Visualization ----------
def _plot_timeline(gt_df, pred_df, details, fps, out_png, time_unit="frames"):
    factor = 1.0 if time_unit == "frames" else 1.0/float(max(fps, 1e-6))
    # x range
    xmin = 0
    xmax = max(gt_df["stop_frame"].max() if len(gt_df) else 0,
               pred_df["stop_frame"].max() if len(pred_df) else 0)
    xmax = xmax * factor

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["PRED", "GT"])
    ax.set_xlabel("time ({})".format("frames" if time_unit=="frames" else "sec"))
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # colors
    c_match = "#4CAF50"  # green
    c_pred_un = "#F44336"  # red
    c_gt_un = "#2196F3"  # blue

    matched_pred = {d["pred_idx"] for d in details["pairs"]}
    matched_gt   = {d["gt_idx"]  for d in details["pairs"]}

    # GT bars
    for i, row in gt_df.reset_index(drop=True).iterrows():
        s, e = row["start_frame"]*factor, row["stop_frame"]*factor
        color = c_match if i in matched_gt else c_gt_un
        ax.add_patch(patches.Rectangle((s, 0.8), e - s, 0.35, color=color, alpha=0.7))

    # Pred bars
    for i, row in pred_df.reset_index(drop=True).iterrows():
        s, e = row["start_frame"]*factor, row["stop_frame"]*factor
        color = c_match if i in matched_pred else c_pred_un
        ax.add_patch(patches.Rectangle((s, -0.15), e - s, 0.35, color=color, alpha=0.7))

    # Optional: draw thin connectors for matched pairs (center lines)
    for d in details["pairs"]:
        p = d["pred_idx"]; g = d["gt_idx"]
        ps = pred_df.loc[p, "start_frame"]*factor
        pe = pred_df.loc[p, "stop_frame"]*factor
        gs = gt_df.loc[g, "start_frame"]*factor
        ge = gt_df.loc[g, "stop_frame"]*factor
        cxp = 0.5*(ps+pe)
        cxg = 0.5*(gs+ge)
        ax.plot([cxp, cxg], [0.2, 1.0], color="k", alpha=0.25, linewidth=1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def _subsample_indices(n, k):
    if n <= k:
        return np.arange(n)
    idx = np.linspace(0, n-1, num=k)
    return np.unique(idx.astype(int))

def _plot_iou_matrix(pred_df, gt_df, out_png, max_matrix=200):
    pred_segs = list(zip(pred_df["start_frame"].to_numpy(), pred_df["stop_frame"].to_numpy()))
    gt_segs   = list(zip(gt_df["start_frame"].to_numpy(),  gt_df["stop_frame"].to_numpy()))
    if len(pred_segs)==0 or len(gt_segs)==0:
        # 空ならダミーを保存
        fig, ax = plt.subplots(figsize=(4,3))
        ax.text(0.5, 0.5, "Empty IoU matrix", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig); return

    M = iou_matrix(pred_segs, gt_segs)
    # 大きすぎる行列は間引き表示
    pr_idx = _subsample_indices(M.shape[0], max_matrix).astype(int)
    gt_idx = _subsample_indices(M.shape[1], max_matrix).astype(int)
    Mv = M[np.ix_(pr_idx, gt_idx)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Mv, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("IoU")

    ax.set_xlabel("GT index (subsampled)" if len(gt_idx)<len(gt_segs) else "GT index")
    ax.set_ylabel("Pred index (subsampled)" if len(pr_idx)<len(pred_segs) else "Pred index")
    ax.set_title("IoU matrix (Pred x GT)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    args = parse_args()
    gt   = load_csv(args.gt, args.sep, args.fps)
    pred = load_csv(args.pred, args.sep, args.fps)

    metrics, details = evaluate(gt, pred, iou_thr=args.iou_thr, matching=args.matching)

    # Save metrics CSV
    out_csv = args.output
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    print(f"[saved] metrics -> {out_csv}")

    # Save details as sidecar CSVs
    base, ext = os.path.splitext(out_csv)
    pairs_rows = []
    for d in details["pairs"]:
        p = d["pred_idx"]; g = d["gt_idx"]; iou = d["iou"]
        pairs_rows.append({
            "pred_idx": p,
            "pred_start": int(pred.loc[p, "start_frame"]),
            "pred_end":   int(pred.loc[p, "stop_frame"]),
            "gt_idx": g,
            "gt_start": int(gt.loc[g, "start_frame"]),
            "gt_end":   int(gt.loc[g, "stop_frame"]),
            "iou": iou
        })
    pd.DataFrame(pairs_rows).to_csv(base + "_pairs.csv", index=False)
    # pd.DataFrame({"unmatched_pred_idx": details["unmatched_pred"]}).to_csv(base + "_unmatched_pred.csv", index=False)
    # pd.DataFrame({"unmatched_gt_idx": details["unmatched_gt"]}).to_csv(base + "_unmatched_gt.csv", index=False)
    print(f"[saved] pairs -> {base + '_pairs.csv'}")
    print(f"[saved] unmatched -> {base + '_unmatched_*.csv'}")

    # Visualization
    if args.viz:
        tl_png = base + "_timeline.png"
        mx_png = base + "_iomatrix.png"
        _plot_timeline(gt, pred, details, fps=args.fps, out_png=tl_png, time_unit=args.time_unit)
        _plot_iou_matrix(pred, gt, out_png=mx_png, max_matrix=args.max_matrix)
        print(f"[saved] viz -> {tl_png}, {mx_png}")

if __name__ == "__main__":
    main()
