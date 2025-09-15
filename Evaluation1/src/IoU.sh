DATASET="EK100"
VIDEO_LIST=("P01_13")
TYPE_LIST=("llava" "gemini" "rag" "hoi")
for TYPE in ${TYPE_LIST[@]}; do
    if [ "$TYPE" = "rag" ]; then
        HEAD="r"
    elif [ "$TYPE" = "gemini" ]; then
        HEAD="c"
    elif [ "$TYPE" = "hoi" ]; then
        HEAD="h"
    else
        HEAD="l"
    fi

    if [ "$DATASET" = "EK100" ]; then
        FPS=60 # Frames per second
    else
        FPS=30 # Default frames per second
    fi

    for VIDEO_NAME in ${VIDEO_LIST[@]}; do
        GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/${DATASET}/${FPS}fps/gt/gt_${VIDEO_NAME}.csv"
        PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/${DATASET}/${FPS}fps/pred/${VIDEO_NAME}f/${HEAD}_${VIDEO_NAME}f.csv"

        TOLERANCE=${k} # SEC
        for iou in 0.3 0.5 0.7; do
            OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/${DATASET}/${FPS}fps/output/IoU/${TYPE}/${iou}/iou_${VIDEO_NAME}.csv"
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
    done
done