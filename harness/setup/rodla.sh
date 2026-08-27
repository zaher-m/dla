#!/usr/bin/env bash
# RoDLA (InternImage-XL + DINO, mmdet 2.x).
#
# Upstream pins Python 3.7 / torch 1.10.2+cu113 / mmcv-full 1.5.0 / mmdet 2.28.1
# and ships a prebuilt DCNv3 wheel for cp37 / x86_64 / cu116, none of which
# installs on a current runtime. This builds the newest mmcv 1.x from source,
# mmdet 2.28.1 over it, and the DCNv3 CUDA op after patching APIs torch removed.

source "$(dirname "$0")/_common.sh"
PY=$(mkenv rodla)
export MAX_JOBS="${MAX_JOBS:-16}"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

# timm is pinned to 0.6.11 upstream, but 0.6.x cannot be imported on Python 3.12
# (`ValueError: mutable default ... for field conv_cfg` from maxxvit's dataclass).
# RoDLA only uses trunc_normal_/DropPath, which current timm still re-exports via
# the timm.models.layers shim, so the newest release is used instead.
# yapf is an undeclared runtime import of mmcv 1.x's Config.
pipi "$PY" "numpy<2" "pillow" "opencv-python-headless" "pyyaml" "yacs" "addict" \
           "termcolor" "packaging" "pycocotools" "shapely" "scipy" "matplotlib" \
           "tqdm" "timm" "regex" "six" "terminaltables" "yapf<0.41"

echo "[rodla] building mmcv-full 1.7.2 from source (the make-or-break step)"
pipi "$PY" --no-build-isolation --no-deps "mmcv-full==1.7.2" \
  && echo "[rodla] mmcv-full built OK" || { echo "[rodla] mmcv-full BUILD FAILED"; exit 21; }

pipi "$PY" --no-deps "mmdet==2.28.1" || { echo "[rodla] mmdet install FAILED"; exit 22; }

OPS="$REPOS/RoDLA/model/ops_dcnv3"
( cd "$OPS" && "$PY" setup.py build install ) \
   && echo "[rodla] DCNv3 kernel built" || { echo "[rodla] DCNv3 BUILD FAILED"; exit 23; }

assert_torch "$PY" rodla
record_env "$PY" rodla
