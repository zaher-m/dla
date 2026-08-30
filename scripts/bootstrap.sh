#!/usr/bin/env bash
# Build the environments a profile needs, and fetch their weights.
#
#   bash scripts/bootstrap.sh                     # the default profile
#   bash scripts/bootstrap.sh --profile fast
#   bash scripts/bootstrap.sh --profile full      # everything; hours, and ~40 GB
#   bash scripts/bootstrap.sh --env docling --env surya
#   bash scripts/bootstrap.sh --list
#
# Run this inside the container (`make setup` does). Each environment is
# independent: one that fails leaves the others working, and the systems it
# would have provided are reported as `env_missing` rather than silently
# disappearing from the comparison.
#
# This downloads model weights from Hugging Face, GitHub releases, ModelScope
# and (for two systems) Google Drive. Sizes range from 7 MB to 1.3 GB per
# checkpoint. Nothing is downloaded that the selected profile does not use.
set -uo pipefail

ROOT="${DLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export DLA_ROOT PYTHONPATH="$ROOT/harness"

PROFILE=""
ENVS_WANTED=()
LIST_ONLY=0
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --env)     ENVS_WANTED+=("$2"); shift 2;;
    --list)    LIST_ONLY=1; shift;;
    --force)   FORCE=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

# The harness environment is not optional: it is the pipeline and the server.
BASE_ENV=harness

envs_for_profile() {
  python - "$1" <<'PY'
import sys, os, yaml
sys.path.insert(0, os.path.join(os.environ["DLA_ROOT"], "harness"))
from core import paths
want = paths.load_profile(sys.argv[1] or None)
with open(paths.REGISTRY, encoding="utf8") as f:
    reg = yaml.safe_load(f)["systems"]
seen, out = set(), []
for s in reg:
    if want is not None and s["id"] not in want:
        continue
    if s["env"] not in seen:
        seen.add(s["env"]); out.append(s["env"])
print(" ".join(out))
PY
}

if [ ${#ENVS_WANTED[@]} -eq 0 ]; then
  # shellcheck disable=SC2207
  ENVS_WANTED=($(envs_for_profile "$PROFILE"))
fi

# Dependency: ndl_layout extends the rodla environment rather than rebuilding
# mmcv from source a second time, so rodla must exist first.
ORDER=("$BASE_ENV")
for e in "${ENVS_WANTED[@]}"; do
  [ "$e" = "$BASE_ENV" ] && continue
  ORDER+=("$e")
done
if printf '%s\n' "${ORDER[@]}" | grep -qx rodla; then
  ORDER+=(ndl_layout)      # a setup script, not an env: adds mmcls + the NDL weights
fi

echo "profile:      ${PROFILE:-<config default>}"
echo "environments: ${ORDER[*]}"
if [ "$LIST_ONLY" = 1 ]; then exit 0; fi

FAILED=()
for e in "${ORDER[@]}"; do
  script="harness/setup/${e}.sh"
  if [ ! -f "$script" ]; then
    echo "!! no setup script for '$e' ($script) — skipping"; FAILED+=("$e"); continue
  fi
  target="assets/envs/$e/bin/python"
  if [ "$FORCE" = 0 ] && [ -x "$target" ] && [ "$e" != "ndl_layout" ]; then
    echo "== $e: already built"
    continue
  fi
  echo; echo "===== $e ====="
  log="benchmark/logs/setup_${e}.log"
  mkdir -p benchmark/logs
  if bash "$script" > >(tee -a "$log") 2>&1; then
    echo "== $e: ok"
  else
    echo "!! $e: FAILED (see $log)"; FAILED+=("$e")
  fi
done

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "finished with failures: ${FAILED[*]}"
  echo "Their systems will be reported as env_missing; everything else works."
else
  echo "all environments built."
fi
echo
"$ROOT/assets/envs/harness/bin/python" -m core.runner --list ${PROFILE:+--profile "$PROFILE"} || true
exit 0
