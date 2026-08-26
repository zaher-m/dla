#!/usr/bin/env bash
# PDF-Extract-Kit: DocLayout-YOLO (YOLOv10 fork) DocStructBench checkpoint.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv pek)
pipi "$PY" "doclayout-yolo" "huggingface_hub" "pillow" "opencv-python-headless" "pyyaml" || \
  pipi "$PY" "ultralytics" "huggingface_hub" "pillow" "opencv-python-headless" "pyyaml"
assert_torch "$PY" pek
record_env "$PY" pek
