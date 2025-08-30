#!/bin/bash

# Full path to Python executable
PYTHON="C:/Users/LAB/AppData/Local/Programs/Python/Python312/python.exe"

# Full path to your script
SCRIPT="F:/Phillip/PrevOccupAI_Dataset/main_dl.py"

# List of all sensors
SENSORS=("ACC" "GYR" "MAG" "ROT")

# List of window sizes
WINDOW_SIZES=(1 2.5 5)

# Outer loop over window sizes
for WIN in "${WINDOW_SIZES[@]}"; do
    echo "=== Running experiments with window_size_s=$WIN ==="

    # Reset ARGS for each window size
    ARGS=()

    # Inner loop over sensors
    for SENSOR in "${SENSORS[@]}"; do
        ARGS+=("$SENSOR")  # append the sensor to the growing list

        echo "Running with window_size_s=$WIN and sensors: ${ARGS[*]}"
        "$PYTHON" "$SCRIPT" --window_size_s "$WIN" --load_sensors "${ARGS[@]}" --seq_len 10 --norm_method "z-score" --norm_type subject --hidden_size 128
    done
done