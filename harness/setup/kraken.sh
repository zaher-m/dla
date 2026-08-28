#!/usr/bin/env bash
# kraken 7.x (mittagessen/kraken) — the `blla` neural baseline/region segmenter.
#
# Two things kraken has that nothing else in this benchmark does:
#   * genuine right-to-left support.  It is the segmenter the Arabic, Hebrew and
#     Syriac manuscript community actually uses, so unlike every other system
#     here it was not designed on the assumption that reading order runs
#     left-to-right.
#   * pixelwise segmentation of *baselines* plus region polygons — the closest
#     available replacement for what Eynollah was supposed to provide before it
#     could not be completed.
#
# kraken 7.1 accepts torch >=2.9,<=2.13, so the container's CUDA build
# satisfies it and is not replaced.
#
# The awkward dependency is coremltools: kraken stores models in the CoreML
# .mlmodel container and imports coremltools at module scope in
# kraken/lib/vgsl/model.py, so it cannot be skipped.  PyPI publishes no
# linux-aarch64 wheel for any coremltools release, so it is built from the
# sdist here.  If that build fails the env is left incomplete on purpose and
# the system is reported as blocked with the build log, rather than being
# quietly dropped.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv kraken)

# Build coremltools first and on its own, so a failure is unambiguous.
if ! "$PY" -c "import coremltools" 2>/dev/null; then
  pipi "$PY" "coremltools==8.3.0" || {
    echo "!! [kraken] coremltools failed to build from source on aarch64" \
      | tee -a "$LOGS/setup_warnings.log"
    exit 3
  }
fi

pipi "$PY" "kraken==7.1"
assert_torch "$PY" kraken
"$PY" -c "from kraken import blla; from kraken.lib import vgsl; print('kraken import ok')"
record_env "$PY" kraken
