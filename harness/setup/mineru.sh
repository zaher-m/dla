#!/usr/bin/env bash
# MinerU: pipeline backend (PP-DocLayoutV2 re-implemented in PyTorch) and
# the MinerU 2.5 VLM backend (transformers).
source "$(dirname "$0")/_common.sh"
PY=$(mkenv mineru)
pipi "$PY" "mineru[core]" "pyyaml" || pipi "$PY" "mineru" "pyyaml"
assert_torch "$PY" mineru
record_env "$PY" mineru
