#!/usr/bin/env bash
# Dolphin v2 (ByteDance): stage-1 layout + reading order, HF transformers.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv dolphin)
pipi "$PY" "transformers>=4.47" "accelerate" "timm" "albumentations" "opencv-python-headless" \
           "pillow" "pymupdf" "omegaconf" "pyyaml" "huggingface_hub"
assert_torch "$PY" dolphin
record_env "$PY" dolphin
