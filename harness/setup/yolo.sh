#!/usr/bin/env bash
# YOLO-DocLayNet (hantian/yolo-doclaynet) via Ultralytics.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv yolo)
pipi "$PY" "ultralytics" "huggingface_hub" "pillow" "opencv-python-headless" "pyyaml"
assert_torch "$PY" yolo
record_env "$PY" yolo
