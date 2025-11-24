#!/bin/bash

set -e
set -o pipefail

# === CONFIGURATION ===
VIDEO_URLS=(
  "https://youtu.be/uDKO-Nk0ps4",
  "https://youtu.be/TuAW4aw4AR4"
)

DOWNLOAD_DIR="downloads"
FRAME_DIR="frames"
PATCH_ROOT="patches"
MODEL_PATH="clock.pt"
PYTHON_SCRIPT="build_patch_dataset.py"
DOWNLOAD_SCRIPT="download_video.py"
FPS=30

# === 1. Install dependencies ===
echo "📦 Installing dependencies..."
pip install -q yt-dlp opencv-python ultralytics tqdm ffmpeg-python pytubefix

# === 2. Install system packages if needed ===
if ! command -v ffmpeg &> /dev/null; then
    echo "🔧 Installing ffmpeg..."
    apt update && apt install -y ffmpeg
fi

# === 3. Create necessary folders ===
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$PATCH_ROOT"

# === 4. Loop over YouTube URLs ===
for VIDEO_URL in "${VIDEO_URLS[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    VIDEO_NAME="video_${TIMESTAMP}.mp4"
    VIDEO_PATH="${DOWNLOAD_DIR}/${VIDEO_NAME}"
    PATCH_DIR="${PATCH_ROOT}/video_${TIMESTAMP}"

    echo -e "\n▶️ Processing $VIDEO_URL"

    # --- 1. Download video ---
    echo "📥 Downloading high-quality video..."
    python3 "$DOWNLOAD_SCRIPT" --url "$VIDEO_URL" --output "$DOWNLOAD_DIR" --filename "$VIDEO_NAME"

    # --- 2. Extract frames ---
    echo "🎞 Extracting frames at ${FPS} FPS..."
    rm -rf "$FRAME_DIR"
    mkdir -p "$FRAME_DIR"
    ffmpeg -i "$VIDEO_PATH" -vf "fps=${FPS}" "$FRAME_DIR/frame_%05d.jpg" -hide_banner -loglevel error

    # --- 3. Run YOLO-based patch extraction ---
    echo "🧠 Running patch extraction into $PATCH_DIR..."
    mkdir -p "$PATCH_DIR"
    python3 "$PYTHON_SCRIPT" --input "$FRAME_DIR" --output "$PATCH_DIR" --model "$MODEL_PATH"

    echo "✅ Finished processing $VIDEO_URL"
    echo "📂 Patches saved in: $PATCH_DIR"

    rm -f "$VIDEO_PATH"
done

echo -e "\n🎉 All videos processed. Final patch structure is under: $PATCH_ROOT/"
