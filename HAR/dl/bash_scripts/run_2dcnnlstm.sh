#!/bin/bash

# Full path to Python executable
PYTHON="C:/Users/LAB/AppData/Local/Programs/Python/Python312/python.exe"

# Full path to your script
SCRIPT="F:/Phillip/PrevOccupAI_Dataset/main_dl.py"

# List of all sensors
SENSORS=("ACC" "GYR" "MAG" "ROT")

# List of window sizes
WINDOW_SIZES=(1 2.5 5)

# list of filters, kernel_size, and stride for the 2 convolutional layers, respectively
FILTER_SET=(64 128)
KERNEL_SIZE_CONV=(1 3 3 3)
STRIDE_CONV=(1 1 1 1)

# kernel sizes and strides for the two pooling layers
KERNEL_SIZE_POOL=(1 2 1 2)
STRIDE_POOL=(1 2 1 2)

# Outer loop over window sizes
for WIN in "${WINDOW_SIZES[@]}"; do
    echo "=== Running experiments with window_size_s=$WIN ==="

    # Reset ARGS for each window size
    ARGS=()

    # Inner loop over sensors
    for SENSOR in "${SENSORS[@]}"; do
        ARGS+=("$SENSOR")  # append the sensor to the growing list

        echo "Running with window_size_s=$WIN and sensors: ${ARGS[*]}"
        "$PYTHON" "$SCRIPT" --window_size_s "$WIN" --seq_len 10 --load_sensors "${ARGS[@]}" --kernel_size_conv "${KERNEL_SIZE_CONV[@]}" --stride_conv "${STRIDE_CONV[@]}" --kernel_size_pool "${KERNEL_SIZE_POOL[@]}" --stride_pool "${STRIDE_POOL[@]}" --filters "${FILTER_SET[@]}"  --model_type "cnnlstm2d" --norm_method "z-score" --norm_type subject --hidden_size 128 --batch_size 64
    done
done