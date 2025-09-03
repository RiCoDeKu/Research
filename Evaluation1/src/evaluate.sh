DATASET="50salads"
TYPE="rag"
VIDEO_NAME="01_1"

for k in 2 3; do
    VIDEO_NAME="01_1"
    GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt/gt_${VIDEO_NAME}.csv"
    PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/c_rgb_${VIDEO_NAME}f.csv"
    TYPE="gemini"
    FPS=30 # Frames per second
    TOLERANCE=${k} # SEC
    OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/F1/${TYPE}/${TOLERANCE}sec/eval_${VIDEO_NAME}.csv"
    mkdir -p "$(dirname "$OUTPUT_PATH")"

    python evaluate.py \
        --gt $GROUND_TRUTH_PATH \
        --pred $PREDICTION_PATH \
        --output $OUTPUT_PATH \
        --fps $FPS \
        --tolerance $TOLERANCE
done