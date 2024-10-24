#!/bin/bash

# Check if the data directory is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <data_directory>"
    exit 1
fi

# Store the data directory path from the command-line argument
data_dir="$1"

# Define the array of channel values
channels=(1 3 5)

# Define the array of image types
types=("float32" "uint8")

# Loop over each channel
for ch in "${channels[@]}"; do
    # Nested loop over each image type
    for type in "${types[@]}"; do
        # Command to run your program, using the provided data directory
        python -m benchmark.albucore_benchmark.benchmark \
            --num_channels $ch \
            --img_type $type \
            --markdown \
            -n 1000 \
            --show-std \
            -r 10 \
            -d "$data_dir"
    done
done
