#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# Remove output CSV files
find "$DIR" -name "*_buzzdetect.csv" -delete
find "$DIR" -name "*_buzzpart.csv" -delete

# Promote logs and profile CSVs from files/ subdirs up one level
find "$DIR" -path "*/files/*.log" -o -path "*/files/*_profile.csv" | while read f; do
  mv "$f" "$(dirname "$(dirname "$f")")/"
done

# Remove empty directories
find "$DIR" -type d -empty -delete

echo "Clean complete."
