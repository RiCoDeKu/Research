DATASET="50salads"
DATATYPE="rag"

VIDEO_NAME="01_1"
GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt/gt_${VIDEO_NAME}.csv"
PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/${VIDEO_NAME}/l_rgb_${VIDEO_NAME}f.csv"
FPS=30

for sim in 0.3; do
    for iou in 0.3 0.5 0.7; do
        OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/CapIou/${DATATYPE}/${sim}/${iou}/cap_iou_${VIDEO_NAME}.csv"
        VIZ_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/CapIou/${DATATYPE}/${VIDEO_NAME}/${sim}/${iou}/viz_${VIDEO_NAME}.png"
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