DATASET="50salads"
TYPE="rag"
VIDEO_NAME="01_1"

GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt/gt_${VIDEO_NAME}.csv"
PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/c_rgb_${VIDEO_NAME}f.csv"
FPS=30 # Frames per second
TOLERANCE=${k} # SEC

for sim in 0.3 0.5 0.7; do
    OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/Caption/${sim}/iou_gemini_${VIDEO_NAME}.csv"
    python caption_match.py \
        --gt ${GROUND_TRUTH_PATH} \
        --pred ${PREDICTION_PATH} \
        --sep ';' \
        --output ${OUTPUT_PATH} \
        --sim-thr ${sim} \
        --matching hungarian
done