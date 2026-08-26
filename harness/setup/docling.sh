#!/usr/bin/env bash
# Docling: layout stage only (RT-DETR "heron"/"egret" object detectors).
source "$(dirname "$0")/_common.sh"
PY=$(mkenv docling)
pipi "$PY" "docling>=2.60" "docling-ibm-models" "docling-core" "pillow" "opencv-python-headless" "pyyaml"
assert_torch "$PY" docling
record_env "$PY" docling
