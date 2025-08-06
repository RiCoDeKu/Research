# VIDEO_NAME="01_1"
# GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Experiment/50salads/gt/gt_${VIDEO_NAME}_5fps_modified.csv"
# PREDICTION_PATH="/home/yamaguchi/vmlserver06/Experiment/50salads/rgb_${VIDEO_NAME}s.csv"
# OUTPUT_PATH="/home/yamaguchi/vmlserver06/Experiment/output/evaluation_${VIDEO_NAME}s.csv"
# if [ ! -d "/home/yamaguchi/vmlserver06/Experiment/output" ]; then
#     mkdir -p /home/yamaguchi/vmlserver06/Experiment/output
# fi

# python evaluate.py \
#     --gt $GROUND_TRUTH_PATH \
#     --pred $PREDICTION_PATH \
#     --output $OUTPUT_PATH



for i in {11..15}; do
    VIDEO_NAME="P01_${i}"
    GROUND_TRUTH_PATH="/home/yamaguchi/vmlserver06/Experiment/EK100/gt/gt_${VIDEO_NAME}.csv"
    PREDICTION_PATH="/home/yamaguchi/vmlserver06/Experiment/EK100/${VIDEO_NAME}f.csv"
    OUTPUT_PATH="/home/yamaguchi/vmlserver06/Experiment/output/evaluation_${VIDEO_NAME}.csv"
    
    python evaluate.py \
        --gt $GROUND_TRUTH_PATH \
        --pred $PREDICTION_PATH \
        --output $OUTPUT_PATH
done