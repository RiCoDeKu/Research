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
PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/r_rgb_${VIDEO_NAME}f.csv"
TYPE="rag"
FPS=30 # Frames per second
TOLERANCE=${k} # SEC

for iou in 0.25 0.5 0.7; do
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