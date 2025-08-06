for i in {1..2}; do
    for j in {1..3}; do
        for k in {1..3}; do
            VIDEO_NAME="P${i}_${j}_T${k}"
            GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Experiment/ATA/gt/${VIDEO_NAME}_C2.csv"
            PREDICTION_PATH="/home/yamaguchi/vmlserver06/Experiment/ATA/prediction/${VIDEO_NAME}f.txt"
            OUTPUT_PATH="/home/yamaguchi/vmlserver06/Experiment/ATA/output/eval_${VIDEO_NAME}.csv"
            FPS=30 # Frames per second
            TOLERANCE=0 # SEC

            python evaluate.py \
                --gt $GROUND_TRUTH_PATH \
                --pred $PREDICTION_PATH \
                --output $OUTPUT_PATH \
                --fps $FPS \
                --tolerance $TOLERANCE
        done
    done
done