#!/bin/bash

# Default values
DEFAULT_CHANNELS=(1 3 5)
DEFAULT_TYPES=("float32" "uint8")
DEFAULT_NUM_IMAGES=1000
DEFAULT_REPEATS=10

# Function to print usage
print_usage() {
    echo "Usage: $0 <data_directory> [options]"
    echo "Options:"
    echo "  --channels    Comma-separated list of channel values (default: 1,3,5)"
    echo "  --types      Comma-separated list of image types (default: float32,uint8)"
    echo "  -n, --num    Number of images (default: 1000)"
    echo "  -r, --repeats Number of repeats (default: 10)"
    echo "Example:"
    echo "  $0 /path/to/data --channels 1,3 --types uint8 -n 500 -r 5"
}

# Check if at least data directory is provided
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

# Store the data directory path from the first argument
data_dir="$1"
shift  # Remove first argument

# Initialize variables with default values
channels=(${DEFAULT_CHANNELS[@]})
types=(${DEFAULT_TYPES[@]})
num_images=${DEFAULT_NUM_IMAGES}
repeats=${DEFAULT_REPEATS}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --channels)
            IFS=',' read -ra channels <<< "$2"
            shift 2
            ;;
        --types)
            IFS=',' read -ra types <<< "$2"
            shift 2
            ;;
        -n|--num)
            num_images="$2"
            shift 2
            ;;
        -r|--repeats)
            repeats="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Print configuration
echo "Running benchmark with:"
echo "Data directory: $data_dir"
echo "Channels: ${channels[*]}"
echo "Types: ${types[*]}"
echo "Number of images: $num_images"
echo "Number of repeats: $repeats"
echo "-------------------"

# Loop over each channel
for ch in "${channels[@]}"; do
    # Nested loop over each image type
    for type in "${types[@]}"; do
        echo "Running benchmark for $ch channels, type $type"
        # Command to run your program, using the provided data directory
        python -m benchmark.albucore_benchmark.benchmark \
            --num_channels "$ch" \
            --img_type "$type" \
            --markdown \
            -n "$num_images" \
            --show-std \
            -r "$repeats" \
            -d "$data_dir"
    done
done
