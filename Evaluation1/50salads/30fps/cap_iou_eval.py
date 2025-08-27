#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cap_iou_eval.py
GT/PRED の "narration" 類似度で 1 対 1 マッチングし，
マッチしたペアに限定して IoU を算出．
さらに「キャプション一致 かつ IoU>=閾値」を TP とみなす統合指標も計算。
可視化: タイムライン図 & IoU行列ヒートマップをPNG保存。

出力:
  - --output <csv>: 統合メトリクス
  - <base>_pairs.csv: 1対1ペア一覧（similarity, IoU 付き）
  - <base>_unmatched_caption_pred.csv / _gt.csv: キャプション未マッチの一覧
  - <base>_below_iou.csv: 類似度は通過したが IoU 不足のペア
  - （任意）<viz_dir>/timeline_{frames|sec}.png: タイムライン図
  - （任意）<viz_dir>/iou_matrix.png: IoU行列ヒートマップ
  - （任意）<base>_sim_matrix.npy: 類似度行列の保存
"""

import argparse, os, re, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===== CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="Caption-similarity matching + IoU evaluation (+ visualization)")
    p.add_argument("--gt", required=True, help="GT CSV (start_frame;stop_frame;narration)")
    p.add_argument("--pred", required=True, help="PRED CSV (start_frame;stop_frame;narration)")
    p.add_argument("--output", required=True, help="Output CSV for metrics")
    p.add_argument("--sep", default=";", help="CSV delimiter (default=';')")
    p.add_argument("--method", choices=["tfidf","jaccard"], default="tfidf",
                   help="Text similarity (default tfidf, fallback to jaccard if sklearn missing)")
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=3)
    p.add_argument("--stopwords", choices=["none","english"], default="english")
    p.add_argument("--sim-thr", type=float, default=0.35, help="Caption similarity threshold")
    p.add_argument("--matching", choices=["hungarian","greedy"], default="hungarian",
                   help="1-1 assignment on similarity matrix")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for temporal TP")
    p.add_argument("--alpha-iou", type=float, default=0.0,
                   help="Fuse IoU into similarity: sim_fused=(1-a)*sim + a*IoU (0=off)")
    p.add_argument("--save-matrix", action="store_true", help="Save similarity matrix as .npy")

    # 可視化オプション
    p.add_argument("--viz", action="store_true", help="Save timeline & IoU matrix figures")
    p.add_argument("--viz-dir", default=None, help="Directory to save figures (default: <base>_viz)")
    p.add_argument("--time-unit", choices=["frames","sec"], default="frames", help="Timeline x-axis unit")
    p.add_argument("--fps", type=float, default=25.0, help="FPS used when time-unit=sec")
    p.add_argument("--max-matrix", type=int, default=200, help="Max size (per axis) to display in IoU matrix")
    p.add_argument("--annotate-captions", action="store_true",
                   help="On timeline, draw narration above GT bars and below Pred bars for matched pairs")
    p.add_argument("--caption-fontsize", type=int, default=7, help="Font size for narration on timeline")
    p.add_argument("--caption-maxlen", type=int, default=28, help="Max chars to show (trim with …)")
    return p.parse_args()

# ===== IO & utils =====
def load_df(path, sep):
    df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="warn")
    req = {"start_frame","stop_frame","narration"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{path} lacks columns: {miss}")
    df = df.copy()
    df["start_frame"] = df["start_frame"].astype(int)
    df["stop_frame"]  = df["stop_frame"].astype(int)
    df["narration"]   = df["narration"].astype(str).fillna("")
    # start<=stop
    s = df["start_frame"].values; e = df["stop_frame"].values
    df["start_frame"] = np.minimum(s,e); df["stop_frame"] = np.maximum(s,e)
    return df.reset_index(drop=True)

_tok_re = re.compile(r"[A-Za-z0-9]+")

def normalize_text(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").replace("_"," ").lower()).strip()

def tokens(s: str):
    return [m.group(0) for m in _tok_re.finditer(normalize_text(s))]

def seg_iou(s1,e1,s2,e2,inclusive=True):
    if inclusive:
        inter = max(0, min(e1,e2) - max(s1,s2) + 1)
        len1  = e1 - s1 + 1
        len2  = e2 - s2 + 1
    else:
        inter = max(0, min(e1,e2) - max(s1,s2))
        len1  = max(0, e1 - s1)
        len2  = max(0, e2 - s2)
    union = len1 + len2 - inter
    return inter/union if union>0 else 0.0

def iou_matrix(pred_segs, gt_segs):
    P, G = len(pred_segs), len(gt_segs)
    M = np.zeros((P, G), dtype=float)
    for i, (ps, pe) in enumerate(pred_segs):
        for j, (gs, ge) in enumerate(gt_segs):
            M[i, j] = seg_iou(ps, pe, gs, ge, inclusive=True)
    return M

# ===== Similarity =====
def sim_matrix_tfidf(gt_caps, pred_caps, ngram_min, ngram_max, stopwords):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    sw = None if stopwords=="none" else "english"
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(ngram_min, ngram_max),
        stop_words=sw, min_df=1, lowercase=True
    )
    all_txt = [normalize_text(x) for x in (gt_caps + pred_caps)]
    X = vec.fit_transform(all_txt)
    G = len(gt_caps)
    S = cosine_similarity(X[G:], X[:G])  # (P, G)
    return S

def sim_matrix_jaccard(gt_caps, pred_caps):
    G, P = len(gt_caps), len(pred_caps)
    S = np.zeros((P,G), dtype=float)
    gt_sets   = [set(tokens(t)) for t in gt_caps]
    pred_sets = [set(tokens(t)) for t in pred_caps]
    for i, ps in enumerate(pred_sets):
        for j, gs in enumerate(gt_sets):
            un = len(ps | gs); inter = len(ps & gs)
            S[i,j] = (inter/un) if un>0 else 0.0
    return S

# ===== Matching =====
def greedy_match(S, thr):
    pairs=[]; P,G = S.shape
    if P==0 or G==0: return pairs
    idxs = np.dstack(np.unravel_index(np.argsort(S.ravel())[::-1], S.shape))[0]
    used_p=set(); used_g=set()
    for pi, gi in idxs:
        if pi in used_p or gi in used_g: continue
        s = float(S[pi,gi])
        if s >= thr:
            pairs.append((int(pi), int(gi), s))
            used_p.add(int(pi)); used_g.add(int(gi))
    return pairs

def hungarian_match(S, thr):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return greedy_match(S, thr)
    if S.size==0: return []
    cost = 1.0 - S  # maximize sim
    r,c = linear_sum_assignment(cost)
    pairs=[]
    for i,j in zip(r,c):
        s = float(S[i,j])
        if s >= thr:
            pairs.append((int(i), int(j), s))
    return pairs

# ---------- Visualization ----------
def _plot_timeline(
    gt_df, pred_df, details, fps, out_png,
    time_unit="frames",
    annotate_captions=False, caption_fontsize=7, caption_maxlen=28
):
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
    c_match = "#4CAF50"     # green
    c_pred_un = "#F44336"   # red
    c_gt_un = "#2196F3"     # blue

    matched_pred = {d["pred_idx"] for d in details["pairs"]}
    matched_gt   = {d["gt_idx"]  for d in details["pairs"]}

    # GT bars
    for i, row in gt_df.reset_index(drop=True).iterrows():
        s, e = row["start_frame"]*factor, row["stop_frame"]*factor
        color = c_match if i in matched_gt else c_gt_un
        ax.add_patch(patches.Rectangle((s, 0.8), e - s, 0.35, color=color, alpha=0.7, zorder=1))

    # Pred bars
    for i, row in pred_df.reset_index(drop=True).iterrows():
        s, e = row["start_frame"]*factor, row["stop_frame"]*factor
        color = c_match if i in matched_pred else c_pred_un
        ax.add_patch(patches.Rectangle((s, -0.15), e - s, 0.35, color=color, alpha=0.7, zorder=1))

    # マッチしたペアのセンター同士を細線で結ぶ（任意）
    for d in details["pairs"]:
        p = d["pred_idx"]; g = d["gt_idx"]
        ps = pred_df.loc[p, "start_frame"]*factor; pe = pred_df.loc[p, "stop_frame"]*factor
        gs = gt_df.loc[g, "start_frame"]*factor;   ge = gt_df.loc[g, "stop_frame"]*factor
        cxp = 0.5*(ps+pe); cxg = 0.5*(gs+ge)
        ax.plot([cxp, cxg], [0.2, 1.0], color="k", alpha=0.25, linewidth=1, zorder=2)

    # ▼ 追加：マッチした区間だけキャプション注釈を描く
    if annotate_captions and len(details.get("pairs", [])) > 0:
        def _short(txt, n):
            t = str(txt or "").replace("_", " ").strip()
            return (t[:n] + "…") if len(t) > n else t

        for d in details["pairs"]:
            p = d["pred_idx"]; g = d["gt_idx"]
            # Pred（下側の余白に）
            ps = pred_df.loc[p, "start_frame"]*factor; pe = pred_df.loc[p, "stop_frame"]*factor
            cxp = 0.5*(ps+pe)
            txp = _short(d.get("pred_narration", pred_df.loc[p, "narration"]), caption_maxlen)
            ax.text(
                cxp, -0.30, txp,
                ha="center", va="top", fontsize=caption_fontsize, color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.0),
                zorder=3, clip_on=False
            )
            # GT（上側の余白に）
            gs = gt_df.loc[g, "start_frame"]*factor; ge = gt_df.loc[g, "stop_frame"]*factor
            cxg = 0.5*(gs+ge)
            txg = _short(d.get("gt_narration", gt_df.loc[g, "narration"]), caption_maxlen)
            ax.text(
                cxg, 1.30, txg,
                ha="center", va="bottom", fontsize=caption_fontsize, color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.0),
                zorder=3, clip_on=False
            )

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
    Mv = M[np.ix_(pr_idx, gt_idx)]  # ← np.ix_ に注意

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

# ===== Main =====
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    gt   = load_df(args.gt, args.sep)
    pred = load_df(args.pred, args.sep)

    gt_caps   = gt["narration"].astype(str).tolist()
    pred_caps = pred["narration"].astype(str).tolist()

    # 類似度行列
    if args.method=="tfidf":
        S = sim_matrix_tfidf(gt_caps, pred_caps, args.ngram_min, args.ngram_max, args.stopwords)
        method_used = "tfidf" if S is not None else "jaccard"
        if S is None:
            S = sim_matrix_jaccard(gt_caps, pred_caps)
    else:
        S = sim_matrix_jaccard(gt_caps, pred_caps); method_used = "jaccard"

    # IoU 行列
    P, G = len(pred_caps), len(gt_caps)
    IoU = np.zeros((P,G), dtype=float)
    pred_segs = list(zip(pred["start_frame"].to_numpy(), pred["stop_frame"].to_numpy()))
    gt_segs   = list(zip(gt["start_frame"].to_numpy(),   gt["stop_frame"].to_numpy()))
    for i,(ps,pe) in enumerate(pred_segs):
        for j,(gs,ge) in enumerate(gt_segs):
            IoU[i,j] = seg_iou(ps,pe,gs,ge,inclusive=True)

    # 類似度と IoU の融合（任意）
    if args.alpha_iou > 0.0:
        a = float(args.alpha_iou)
        S = (1.0 - a) * S + a * IoU

    # キャプションで 1対1 マッチ
    pairs = hungarian_match(S, args.sim_thr) if args.matching=="hungarian" else greedy_match(S, args.sim_thr)
    matched_p = {p for p,_,_ in pairs}
    matched_g = {g for _,g,_ in pairs}

    # ペアごとの IoU を付与
    pair_rows=[]
    below_iou_rows=[]
    for p,g,sim in pairs:
        iou = float(IoU[p,g])
        row = {
            "pred_idx": p,
            "pred_start": int(pred.loc[p,"start_frame"]),
            "pred_end":   int(pred.loc[p,"stop_frame"]),
            "pred_narration": pred.loc[p,"narration"],
            "gt_idx": g,
            "gt_start": int(gt.loc[g,"start_frame"]),
            "gt_end":   int(gt.loc[g,"stop_frame"]),
            "gt_narration": gt.loc[g,"narration"],
            "similarity": round(float(sim),6),
            "iou": round(iou,6)
        }
        pair_rows.append(row)
        if iou < args.iou_thr:
            below_iou_rows.append(row)

    # 統合メトリクス
    TP = sum(1 for r in pair_rows if r["iou"] >= args.iou_thr)
    FP = (len(pred_caps) - len(matched_p)) + sum(1 for r in pair_rows if r["iou"] < args.iou_thr)
    FN = (len(gt_caps)   - len(matched_g)) + sum(1 for r in pair_rows if r["iou"] < args.iou_thr)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    # 参考: キャプション一致だけの P/R/F1
    TP_cap = len(pairs)
    FP_cap = len(pred_caps) - TP_cap
    FN_cap = len(gt_caps)   - TP_cap
    prec_cap = TP_cap/(TP_cap+FP_cap) if (TP_cap+FP_cap)>0 else 0.0
    rec_cap  = TP_cap/(TP_cap+FN_cap) if (TP_cap+FN_cap)>0 else 0.0
    f1_cap   = 2*prec_cap*rec_cap/(prec_cap+rec_cap) if (prec_cap+rec_cap)>0 else 0.0

    # 参考: マッチ済みペア統計
    ious_matched = [r["iou"] for r in pair_rows]
    mean_iou   = float(np.mean(ious_matched)) if ious_matched else 0.0
    median_iou = float(np.median(ious_matched)) if ious_matched else 0.0
    mean_sim   = float(np.mean([r["similarity"] for r in pair_rows])) if pair_rows else 0.0

    # 保存
    base, _ = os.path.splitext(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # メトリクス CSV
    pd.DataFrame([{
        "num_gt": len(gt_caps),
        "num_pred": len(pred_caps),
        "tp_combined": TP, "fp_combined": FP, "fn_combined": FN,
        "precision_combined": round(precision,4),
        "recall_combined":    round(recall,4),
        "f1_combined":        round(f1,4),
        "sim_thr": args.sim_thr,
        "iou_thr": args.iou_thr,
        "method": method_used,
        "matching": args.matching,
        "alpha_iou": args.alpha_iou,
        "caption_tp": TP_cap, "caption_precision": round(prec_cap,4),
        "caption_recall": round(rec_cap,4), "caption_f1": round(f1_cap,4),
        "mean_iou_on_matched": round(mean_iou,4),
        "median_iou_on_matched": round(median_iou,4),
        "mean_similarity_on_matched": round(mean_sim,4)
    }]).to_csv(args.output, index=False)
    print(f"[saved] metrics -> {args.output}")

    # ペア一覧
    pd.DataFrame(pair_rows).to_csv(base + "_pairs.csv", index=False)
    print(f"[saved] pairs -> {base + '_pairs.csv'}")

    # ---- ここを詳細出力に変更（未マッチ：idx + 区間 + キャプション）----
    unmatched_pred_idx = [i for i in range(len(pred_caps)) if i not in matched_p]
    unmatched_gt_idx   = [i for i in range(len(gt_caps))   if i not in matched_g]

    unmatched_pred_rows = [{
        "pred_idx": i,
        "pred_start": int(pred.loc[i, "start_frame"]),
        "pred_end":   int(pred.loc[i, "stop_frame"]),
        "pred_narration": pred.loc[i, "narration"]
    } for i in unmatched_pred_idx]

    unmatched_gt_rows = [{
        "gt_idx": i,
        "gt_start": int(gt.loc[i, "start_frame"]),
        "gt_end":   int(gt.loc[i, "stop_frame"]),
        "gt_narration": gt.loc[i, "narration"]
    } for i in unmatched_gt_idx]

    pd.DataFrame(unmatched_pred_rows).to_csv(base + "_unmatched_caption_pred.csv", index=False)
    pd.DataFrame(unmatched_gt_rows).to_csv(base + "_unmatched_caption_gt.csv", index=False)
    print(f"[saved] unmatched (caption, detailed) -> {base + '_unmatched_caption_pred.csv'} / {base + '_unmatched_caption_gt.csv'}")

    # 類似度は閾値クリアだが IoU 不足のペア
    pd.DataFrame(below_iou_rows).to_csv(base + "_below_iou.csv", index=False)
    print(f"[saved] below-iou -> {base + '_below_iou.csv'}")

    # 類似度行列（任意）
    if args.save_matrix:
        np.save(base + "_sim_matrix.npy", S)
        print(f"[saved] sim matrix -> {base + '_sim_matrix.npy'}")

    # 可視化（任意）
    if args.viz:
        viz_dir = args.viz_dir or (base + "_viz")
        os.makedirs(viz_dir, exist_ok=True)
        # タイムライン（frames/seconds）
        _plot_timeline(
            gt, pred, {"pairs": pair_rows}, fps=args.fps,
            out_png=os.path.join(viz_dir, f"timeline_{args.time_unit}.png"),
            time_unit=args.time_unit,
            annotate_captions=args.annotate_captions,
            caption_fontsize=args.caption_fontsize,
            caption_maxlen=args.caption_maxlen
        )
        print(f"[saved] timeline -> {os.path.join(viz_dir, f'timeline_{args.time_unit}.png')}")
        # IoU行列
        _plot_iou_matrix(pred, gt, out_png=os.path.join(viz_dir, "iou_matrix.png"),
                        max_matrix=args.max_matrix)
        print(f"[saved] IoU matrix -> {os.path.join(viz_dir, 'iou_matrix.png')}")

if __name__ == "__main__":
    main()