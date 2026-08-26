#!/usr/bin/env bash
# Surya 2: VLM layout + reading order. vLLM is inherited from the base image.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv surya)
pipi "$PY" "surya-ocr" "pyyaml"
assert_torch "$PY" surya
record_env "$PY" surya
