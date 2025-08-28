#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caption_match.py
- GT/PRED の "narration" 類似度でシーンをマッチング
- 既定: TF-IDF (1-3gram) + cosine
- フォールバック: Jaccard (token set)
- 任意: 類似度と IoU を重み付き融合 (sim_fused = (1-a)*sim_text + a*IoU)

入出力:
  --output <csv> にメトリクスを書き出し
  併せて <base>_pairs.csv, <base>_unmatched_*.csv を保存
  （必要なら --save-matrix で <base>_sim_matrix.npy も保存）

例:
  python caption_match.py \
    --gt gt.csv --pred pred.csv \
    --sep ';' --fps 25 \
    --output out/cap_sim.csv \
    --sim-thr 0.35 --matching hungarian \
    --alpha-iou 0.2 --save-matrix
"""
import argparse, os, re, numpy as np, pandas as pd

# ===== CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="Match GT/PRED segments by caption similarity.")
    p.add_argument("--gt", required=True, help="Ground-truth CSV (start_frame;stop_frame;narration)")
    p.add_argument("--pred", required=True, help="Predictions CSV (start_frame;stop_frame;narration)")
    p.add_argument("--output", required=True, help="Output metrics CSV path")
    p.add_argument("--sep", default=";", help="CSV delimiter (default=';')")
    p.add_argument("--fps", type=float, default=25.0, help="For optional time conversions (not required)")
    p.add_argument("--method", choices=["tfidf","jaccard"], default="tfidf",
                   help="Text similarity method (default: tfidf, auto-fallback to jaccard if sklearn missing)")
    p.add_argument("--sim-thr", type=float, default=0.3, help="Similarity threshold to count a match")
    p.add_argument("--matching", choices=["greedy","hungarian"], default="hungarian",
                   help="1-1 assignment method (default: hungarian)")
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=3)
    p.add_argument("--stopwords", choices=["none","english"], default="english")
    p.add_argument("--alpha-iou", type=float, default=0.0,
                   help="Fuse IoU with text similarity: 0=off, 0.2=20%% IoU")
    p.add_argument("--save-matrix", action="store_true", help="Save similarity matrix as .npy sidecar")
    return p.parse_args()

# ===== IO & utils =====
def load_df(path, sep):
    df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="warn")
    req = {"start_frame","stop_frame","narration"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")
    # 安全のため型を整える
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
    # 例: "add_olive_oil" → "add olive oil"
    s = (s or "").replace("_", " ").lower()
    return re.sub(r"\s+", " ", s).strip()

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

# ===== Similarity =====
def sim_matrix_tfidf(gt_caps, pred_caps, ngram_min, ngram_max, stopwords):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        # フォールバック
        return None
    sw = None if stopwords=="none" else "english"
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(ngram_min, ngram_max),
        stop_words=sw,
        min_df=1,
        lowercase=True
    )
    all_txt = [normalize_text(x) for x in (gt_caps + pred_caps)]
    X = vec.fit_transform(all_txt)
    G = len(gt_caps)
    Xg = X[:G]
    Xp = X[G:]
    S = cosine_similarity(Xp, Xg)  # shape: (P, G)
    return S

def sim_matrix_jaccard(gt_caps, pred_caps):
    G = len(gt_caps); P = len(pred_caps)
    S = np.zeros((P,G), dtype=float)
    gt_sets = [set(tokens(t)) for t in gt_caps]
    pr_sets = [set(tokens(t)) for t in pred_caps]
    for i, ps in enumerate(pr_sets):
        for j, gs in enumerate(gt_sets):
            un = len(ps | gs)
            inter = len(ps & gs)
            S[i,j] = (inter / un) if un>0 else 0.0
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
    # maximize similarity == minimize (1 - sim)
    cost = 1.0 - S
    r,c = linear_sum_assignment(cost)
    pairs=[]
    for i,j in zip(r,c):
        s=float(S[i,j])
        if s>=thr:
            pairs.append((int(i),int(j),s))
    return pairs

# ===== Main =====
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    gt   = load_df(args.gt, args.sep)
    pred = load_df(args.pred, args.sep)

    gt_caps   = gt["narration"].astype(str).tolist()
    pred_caps = pred["narration"].astype(str).tolist()

    # 類似度行列（TF-IDF or フォールバック）
    S = None
    method_used = args.method
    if args.method=="tfidf":
        S = sim_matrix_tfidf(gt_caps, pred_caps, args.ngram_min, args.ngram_max, args.stopwords)
        if S is None:
            method_used = "jaccard"
            S = sim_matrix_jaccard(gt_caps, pred_caps)
    else:
        S = sim_matrix_jaccard(gt_caps, pred_caps)

    # IoU 融合（任意）
    if args.alpha_iou > 0.0:
        P, G = S.shape
        IoU = np.zeros_like(S, dtype=float)
        gt_segs   = list(zip(gt["start_frame"].to_numpy(),  gt["stop_frame"].to_numpy()))
        pred_segs = list(zip(pred["start_frame"].to_numpy(), pred["stop_frame"].to_numpy()))
        for i,(ps,pe) in enumerate(pred_segs):
            for j,(gs,ge) in enumerate(gt_segs):
                IoU[i,j] = seg_iou(ps,pe,gs,ge,inclusive=True)
        a = float(args.alpha_iou)
        S = (1.0 - a) * S + a * IoU

    # 1-1 マッチング
    if args.matching=="hungarian":
        pairs = hungarian_match(S, args.sim_thr)
    else:
        pairs = greedy_match(S, args.sim_thr)

    matched_p = {p for p,_,_ in pairs}
    matched_g = {g for _,g,_ in pairs}

    TP = len(pairs)
    FP = len(pred_caps) - TP
    FN = len(gt_caps)   - TP
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    sims = [s for _,_,s in pairs]
    mean_sim = float(np.mean(sims)) if sims else 0.0
    med_sim  = float(np.median(sims)) if sims else 0.0

    # 保存（メトリクス）
    base, _ = os.path.splitext(args.output)
    pd.DataFrame([{
        "num_gt": len(gt_caps),
        "num_pred": len(pred_caps),
        "tp": TP, "fp": FP, "fn": FN,
        "precision": round(prec,4),
        "recall": round(rec,4),
        "f1": round(f1,4),
        "mean_sim_matched": round(mean_sim,4),
        "median_sim_matched": round(med_sim,4),
        "sim_thr": args.sim_thr,
        "method": method_used,
        "matching": args.matching,
        "alpha_iou": args.alpha_iou
    }]).to_csv(args.output, index=False)
    print(f"[saved] metrics -> {args.output}")

    # 保存（対応ペア）
    rows=[]
    for p,g,s in pairs:
        rows.append({
            "pred_idx": p,
            "pred_start": int(pred.loc[p,"start_frame"]),
            "pred_end":   int(pred.loc[p,"stop_frame"]),
            "pred_narration": pred.loc[p,"narration"],
            "gt_idx": g,
            "gt_start": int(gt.loc[g,"start_frame"]),
            "gt_end":   int(gt.loc[g,"stop_frame"]),
            "gt_narration": gt.loc[g,"narration"],
            "similarity": round(float(s),6)
        })
    pd.DataFrame(rows).to_csv(base + "_pairs.csv", index=False)

    # 保存（未マッチ）
    pd.DataFrame({"unmatched_pred_idx":[i for i in range(len(pred_caps)) if i not in matched_p]}
                ).to_csv(base + "_unmatched_pred.csv", index=False)
    pd.DataFrame({"unmatched_gt_idx":[i for i in range(len(gt_caps)) if i not in matched_g]}
                ).to_csv(base + "_unmatched_gt.csv", index=False)
    print(f"[saved] pairs -> {base + '_pairs.csv'}")
    print(f"[saved] unmatched -> {base + '_unmatched_*.csv'}")

    # 類似度行列の保存（任意）
    if args.save_matrix:
        np.save(base + "_sim_matrix.npy", S)
        print(f"[saved] sim matrix -> {base + '_sim_matrix.npy'}")

if __name__ == "__main__":
    main()
