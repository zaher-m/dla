# Shared helpers for environment setup scripts.
#
# Every env inherits the container's CUDA build of torch/torchvision from system
# site-packages, so no setup script ever re-downloads a multi-GB CUDA wheel and
# no resolver is allowed to swap it for a generic PyPI one — that would silently
# drop GPU support, and every result would be wrong without an error.
#
# Paths come from harness/core/paths.py, which reads config/dla.yaml and the
# DLA_* environment overrides, so a setup script never hardcodes a directory.
set -euo pipefail

DLA_ROOT="${DLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DLA_ROOT

_dla_path() {   # _dla_path <key>   -> absolute path from the config
  python - "$1" <<'PY' 2>/dev/null || true
import sys, os
sys.path.insert(0, os.path.join(os.environ["DLA_ROOT"], "harness"))
from core import paths
print({"envs": paths.ENVS, "models": paths.MODELS, "repos": paths.REPOS,
       "workspace": paths.WORKSPACE, "assets": paths.ASSETS}[sys.argv[1]])
PY
}

ENVS="${DLA_ENVS:-$(_dla_path envs)}";       ENVS="${ENVS:-$DLA_ROOT/assets/envs}"
MODELS="${DLA_MODELS:-$(_dla_path models)}"; MODELS="${MODELS:-$DLA_ROOT/assets/models}"
REPOS="${DLA_REPOSITORIES:-$(_dla_path repos)}"; REPOS="${REPOS:-$DLA_ROOT/assets/repositories}"
BENCH="${DLA_WORKSPACE:-$(_dla_path workspace)}"; BENCH="${BENCH:-$DLA_ROOT/benchmark}"
LOGS="$BENCH/logs"
mkdir -p "$ENVS" "$MODELS" "$REPOS" "$LOGS"

# Compile CUDA extensions for the device that is actually present. Set
# TORCH_CUDA_ARCH_LIST yourself to cross-build for a different one.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  _arch="$(python -c 'import torch;c=torch.cuda.get_device_capability();print(f"{c[0]}.{c[1]}")' 2>/dev/null || true)"
  [ -n "$_arch" ] && export TORCH_CUDA_ARCH_LIST="$_arch"
fi

SYS_TORCH="$(python -c 'import torch;print(torch.__version__)')"
SYS_TV="$(python -c 'import torchvision;print(torchvision.__version__)' 2>/dev/null || echo none)"

mkenv() {  # mkenv <name>
  local n="$1"
  if [ ! -x "$ENVS/$n/bin/python" ]; then
    # --seed gives the venv its own pip.  pip (unlike uv) honours
    # include-system-site-packages, so the inherited CUDA build of
    # torch/torchvision counts as already-satisfied and is never replaced by a
    # generic PyPI wheel that would silently drop GPU support.
    uv venv --seed --system-site-packages --python "$(which python)" "$ENVS/$n"
  fi
  echo "$ENVS/$n/bin/python"
}

# Install without ever letting a resolver swap out the inherited CUDA torch.
pipi() {  # pipi <venv-python> <args...>
  local py="$1"; shift
  "$py" -m pip install --no-cache-dir --no-input "$@"
}

# Clone an upstream repository into the shared assets tree, idempotently.
clone() {  # clone <github-org/repo> [dir] [--depth N]
  local slug="$1" dir="${2:-$(basename "$1")}"
  if [ ! -d "$REPOS/$dir/.git" ]; then
    git clone --depth 1 "https://github.com/${slug}.git" "$REPOS/$dir"
  else
    echo "have $dir"
  fi
}

assert_torch() {  # assert_torch <venv-python> <env-name>
  local py="$1" n="$2"
  local v; v="$("$py" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"
  if [ "$v" != "$SYS_TORCH" ]; then
    echo "!! [$n] torch changed: system=$SYS_TORCH env=$v (CUDA support may be lost)" \
      | tee -a "$LOGS/setup_warnings.log"
  else
    echo "ok [$n] torch=$v (inherited, CUDA $( "$py" -c 'import torch;print(torch.version.cuda)' ))"
  fi
}

record_env() {  # record_env <venv-python> <env-name>
  local py="$1" n="$2"
  mkdir -p "$BENCH/environments"
  "$py" -m pip freeze > "$BENCH/environments/${n}.pip-freeze.txt" 2>/dev/null || true
  "$py" - <<'PY' > "$BENCH/environments/${n}.runtime.json" 2>/dev/null || true
import json,sys,platform
d={"python":sys.version,"platform":platform.platform(),"machine":platform.machine()}
try:
    import torch; d["torch"]=torch.__version__; d["cuda"]=torch.version.cuda
    d["cuda_available"]=torch.cuda.is_available()
    d["device"]=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e: d["torch_error"]=str(e)
print(json.dumps(d,indent=1))
PY
}
