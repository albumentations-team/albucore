#!/bin/bash

# Define the array of channel values
channels=(1 3 5)

# Define the array of image types
types=("uint8" "float32")

# Loop over each channel
for ch in "${channels[@]}"; do
    # Nested loop over each image type
    for type in "${types[@]}"; do
        # Command to run your program, e.g., a Python script
        python -m benchmark.benchmark --num_channels $ch --img_type $type --markdown -n 1000 --show-std -r 5 -d ~/data/coco/val2017
    done
done
