#!/usr/bin/env bash
# PaddleOCR layout detection (PP-DocLayout family).
#
# PyPI carries no linux-aarch64 PaddlePaddle wheel, but the *official* Paddle
# package index does (CPU build only — there is no aarch64 CUDA build).  So
# PaddleOCR runs here with its native Paddle runtime and official weights,
# on CPU.  Its timings are therefore CPU timings and are flagged as such;
# they must not be compared head-to-head with the GPU systems.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv paddle_onnx)
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cpu/"
pipi "$PY" "paddlepaddle==3.3.1" -i "$PADDLE_INDEX"
pipi "$PY" "paddleocr>=3.3" "numpy<2.3" "opencv-python-headless" "pillow" "pyyaml" "shapely"
"$PY" -c "import paddle;paddle.utils.run_check()" 2>&1 | tail -5
record_env "$PY" paddle_onnx
