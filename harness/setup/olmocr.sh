#!/usr/bin/env bash
# olmOCR 2: VLM pipeline; vLLM inherited from the base image.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv olmocr)
pipi "$PY" "olmocr" "pyyaml" || pipi "$PY" --no-deps -e "$REPOS/olmocr"
assert_torch "$PY" olmocr
record_env "$PY" olmocr
