DATASET="50salads"
TYPE="rag"
VIDEO_NAME="01_1"

GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt/gt_${VIDEO_NAME}.csv"
PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/r_rgb_${VIDEO_NAME}f.csv"
FPS=30 # Frames per second
TOLERANCE=${k} # SEC

for iou in 0.3 0.5 0.7; do
    OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/output/IoU/${TYPE}/${iou}/iou_${VIDEO_NAME}.csv"
    python IoU.py \
    --gt ${GROUND_TRUTH_PATH} \
    --pred ${PREDICTION_PATH} \
    --output ${OUTPUT_PATH} \
    --fps ${FPS} \
    --sep ';' \
    --iou-thr ${iou} \
    --matching hungarian \
    --viz \
    --time-unit seconds
done