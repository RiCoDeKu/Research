# for i in {1..3}; do
#     for j in {1..2}; do
#         for k in 0 1 2 3 5 7; do
#             VIDEO_NAME="0${i}_${j}"
#             GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt/gt_${VIDEO_NAME}.csv"
#             PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/rgb_${VIDEO_NAME}f.csv"
#             FPS=30 # Frames per second
#             TOLERANCE=${k} # SEC
#             mkdir -p "/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output"
#             mkdir -p "/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/${TOLERANCE}sec"
#             OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/${TOLERANCE}sec/eval_${VIDEO_NAME}.csv"

#             python evaluate.py \
#                 --gt $GROUND_TRUTH_PATH \
#                 --pred $PREDICTION_PATH \
#                 --output $OUTPUT_PATH \
#                 --fps $FPS \
#                 --tolerance $TOLERANCE
#         done
#     done
# done

VIDEO_NAME="01_1"
GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt/gt_${VIDEO_NAME}.csv"
PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/l_rgb_${VIDEO_NAME}f.csv"
FPS=30 # Frames per second
DATATYPE="llava"
for sim in 0.2 0.25 0.3 0.4; do
    for iou in 0.1 0.3 0.5 0.7; do
        OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/CapIou/${DATATYPE}/${sim}/${iou}/cap_iou_llava_${VIDEO_NAME}.csv"
        VIZ_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/CapIou/${DATATYPE}/${sim}/${iou}/viz_llava_${VIDEO_NAME}.png"
        python cap_iou_eval.py \
        --gt ${GROUND_TRUTH_PATH} \
        --pred ${PREDICTION_PATH} \
        --output ${OUTPUT_PATH} \
        --viz-dir ${VIZ_PATH} \
        --sep ';' \
        --method tfidf \
        --ngram-min 1 \
        --ngram-max 3 \
        --stopwords english \
        --sim-thr ${sim} \
        --matching hungarian \
        --iou-thr ${iou} \
        --alpha-iou 0.0 \
        --save-matrix \
        --viz \
        --time-unit frames \
        --fps ${FPS} \
        --annotate-captions \
        --caption-fontsize 7 \
        --caption-maxlen 28
    done
done

# # 1) 既定 (TF-IDF + コサイン, ハンガリー, 閾値0.35)
# python caption_match.py \
#   --gt gt.csv --pred pred.csv \
#   --sep ';' --output out/cap_metrics.csv \
#   --sim-thr 0.35 --matching hungarian

# # 2) TF-IDFが無い環境 → 自動で Jaccard にフォールバック
# python caption_match.py --gt gt.csv --pred pred.csv --output out/cap_metrics.csv

# # 3) テキスト + IoUを融合（20% IoU 重み）
# python caption_match.py \
#   --gt gt.csv --pred pred.csv \
#   --output out/cap_metrics.csv \
#   --alpha-iou 0.2 --sim-thr 0.4 --save-matrix