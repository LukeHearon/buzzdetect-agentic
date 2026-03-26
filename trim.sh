#!/usr/bin/env bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

MAX_DURATION=$((2 * 60 * 60))  # 2 hours

find "$INPUT_DIR" -type f -iname "*.mp3" -print0 | while IFS= read -r -d '' FILE; do
  REL_PATH="${FILE#$INPUT_DIR/}"
  OUT_FILE="$OUTPUT_DIR/$REL_PATH"

  mkdir -p "$(dirname "$OUT_FILE")"

  echo "Processing: $FILE"

  ffmpeg -nostdin -y -i "$FILE" -t "$MAX_DURATION" -c copy "$OUT_FILE"
done

echo "Done."
