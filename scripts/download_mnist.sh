#!/bin/bash
# Downloads the MNIST dataset into the data/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for f in "${FILES[@]}"; do
    if [ ! -f "${f%.gz}" ]; then
        echo "Downloading $f..."
        curl -O "$BASE_URL/$f"
        echo "Extracting $f..."
        gunzip "$f"
    else
        echo "${f%.gz} already exists, skipping."
    fi
done

echo ""
echo "MNIST dataset downloaded to: $DATA_DIR"
echo "Files:"
ls -la "$DATA_DIR"
