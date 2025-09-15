DATASET_LIST=("EK100")
for DATASET in ${DATASET_LIST[@]}; do
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

        if [ "$DATASET" = "50salads" ]; then
            VIDEO_LIST=("rgb_01_1" "rgb_01_2" "rgb_02_1")
        elif [ "$DATASET" = "EK100" ]; then
            VIDEO_LIST=("P01_11" "P01_12" "P01_13")
        else
            VIDEO_LIST=("P1_1_T1" "P1_1_T2" "P1_1_T3")
        fi

        for VIDEO_NAME in ${VIDEO_LIST[@]}; do
            if [ "$DATASET" = "50salads" ]; then
                GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/gt/gt_${VIDEO_NAME}.csv"
                PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/50salads/30fps/pred/${VIDEO_NAME}f/${HEAD}_${VIDEO_NAME}f.csv"
            elif [ "$DATASET" = "EK100" ]; then
                GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/EK100/60fps/gt/gt_${VIDEO_NAME}.csv"
                PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/EK100/60fps/pred/${VIDEO_NAME}f/${HEAD}_${VIDEO_NAME}f.csv"
            else
                GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/ATA/30fps/gt/gt_${VIDEO_NAME}_C2.csv"
                PREDICTION_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/ATA/30fps/pred/${VIDEO_NAME}f/${HEAD}_${VIDEO_NAME}f.csv"
            fi

            for sim in 0.3; do
                for iou in 0.3 0.5 0.7; do
                    OUTPUT_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/${DATASET}/${FPS}fps/output/CapIou/${TYPE}/${iou}/cap_iou_${VIDEO_NAME}.csv"
                    VIZ_PATH="/home/yamaguchi/vmlserver06/Research/Evaluation1/${DATASET}/${FPS}fps/output/CapIou/${TYPE}/${iou}/viz_${VIDEO_NAME}.png"
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
        done
    done
done