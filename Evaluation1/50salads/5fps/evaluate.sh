for i in {1..3}; do
    for j in {1..2}; do
        for k in 1 2 3 5 7; do
            VIDEO_NAME="0${i}_${j}"
            GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Experiment/50salads/5fps/gt/gt_${VIDEO_NAME}_5fps_modified.csv"
            PREDICTION_PATH="/home/yamaguchi/vmlserver06/Experiment/50salads/5fps/pred/rgb_${VIDEO_NAME}f.csv"
            FPS=5 # Frames per second
            TOLERANCE=${k} # SEC
            mkdir -p "/home/yamaguchi/vmlserver06/Experiment/50salads/5fps/output"
            mkdir -p "/home/yamaguchi/vmlserver06/Experiment/50salads/5fps/output/${TOLERANCE}sec"
            OUTPUT_PATH="/home/yamaguchi/vmlserver06/Experiment/50salads/5fps/output/${TOLERANCE}sec/eval_${VIDEO_NAME}.csv"

            python evaluate.py \
                --gt $GROUND_TRUTH_PATH \
                --pred $PREDICTION_PATH \
                --output $OUTPUT_PATH \
                --fps $FPS \
                --tolerance $TOLERANCE
        done
    done
done
