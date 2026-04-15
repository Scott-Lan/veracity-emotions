#!/bin/bash
# setup.sh - One-command setup for veracity-emotions
# Creates venv, installs deps, and downloads both datasets

set -e

echo "Starting veracity-emotions setup..."

# python command - system dependent 
PYTHON_CMD=""

# Check if python3 exists
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
# Otherwise check if python exists 
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Could not find python3 or python on this machine."
    echo "Please install Python and try again."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_CMD"

# setup virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (Python >= 3.12)..."
    "$PYTHON_CMD" -m venv .venv
    echo "Virtual environment created."
else
    echo "Using existing virtual environment."
fi

source .venv/bin/activate
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Dependencies installed."

# get full datasets
mkdir -p data

echo "Downloading GoEmotions dataset (via Kaggle)..."
kaggle datasets download debarshichanda/goemotions -p data/GoEmotions --unzip

echo "Downloading full Twitter15/16 rumor dataset"
cd data

# using python to download without wget
"$PYTHON_CMD" -c '
import requests
import sys
from tqdm import tqdm

url = "https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1"
filename = "rumdetect2017.zip"

print("Downloading Twitter15/16 rumor dataset...")
response = requests.get(url, stream=True)
total_size = int(response.headers.get("content-length", 0))

with open(filename, "wb") as f, tqdm(
    desc=filename,
    total=total_size,
    unit="iB",
    unit_scale=True,
    unit_divisor=1024,
) as progress_bar:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            size = f.write(chunk)
            progress_bar.update(size)

print("Download complete. Unzipping...")
' 

unzip -q rumdetect2017.zip
rm rumdetect2017.zip   # clean up the zip after extraction

cd ..

echo ""
echo "Setup complete!"
echo ""
echo "Datasets are now in the 'data/' folder:"
echo "   - data/goemotions/          (emotions)"
echo "   - data/rumor_detection_acl2017/twitter15/"
echo "   - data/rumor_detection_acl2017/twitter16/"
