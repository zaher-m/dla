#!/usr/bin/env bash
# ndl_layout: the layout module of NDLOCR v2.1 (National Diet Library, Japan).
#
# Cascade Mask R-CNN with an mmcls ConvNeXt-tiny backbone, 17 classes, trained
# on Japanese printed material. Emits instance masks, and its training
# distribution is not left-to-right Latin script.
#
# mmdetection 2.x, so it reuses the mmcv-full 1.7.2 / mmdet 2.28.1 stack that
# harness/setup/rodla.sh already built; the only addition is mmclassification
# for the ConvNeXt backbone the config registers as `mmcls.ConvNeXt`.
#
# Checkpoint: ndl_retrainmodel.pth from lab.ndl.go.jp, CC BY 4.0.
source "$(dirname "$0")/_common.sh"
PY="$ENVS/rodla/bin/python"
if [ ! -x "$PY" ]; then
  echo "!! rodla env missing — run harness/setup/rodla.sh first" >&2
  exit 1
fi

# mmcls 0.25.0 is the last release of the mmcv-1.x line (requires
# mmcv>=1.4.2,<1.9.0), which is exactly the 1.7.2 we have.
pipi "$PY" --no-deps "mmcls==0.25.0"

DEST="$MODELS/ndl_layout"
mkdir -p "$DEST"
URL="https://lab.ndl.go.jp/dataset/ndlocr_v2/ndl_layout/ndl_retrainmodel.pth"
if [ ! -s "$DEST/ndl_retrainmodel.pth" ]; then
  curl -L --retry 3 -o "$DEST/ndl_retrainmodel.pth" "$URL"
fi
ls -l "$DEST"

# The config's backbone init_cfg points at an OpenMMLab ConvNeXt ImageNet
# checkpoint.  The adapter disables it (the detection checkpoint supersedes it
# entirely), so nothing is fetched at inference time.
"$PY" -c "import mmcls, mmdet, mmcv; print('mmcls', mmcls.__version__, 'mmdet', mmdet.__version__, 'mmcv', mmcv.__version__)"

record_env "$PY" rodla
