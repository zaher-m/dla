#!/usr/bin/env bash
# Container entrypoint.
#
# Two jobs, in this order:
#   1. make the working tree writable as the *host* user, so nothing this
#      container creates ends up owned by root on the host;
#   2. dispatch on the command — `serve` starts the web application, `setup`
#      builds environments, anything else is run verbatim.
set -e

ROOT="${DLA_ROOT:-/work}"

# Give the runtime user a real HOME inside the working tree. Without it, tools
# that write dotfiles (torch hub, matplotlib, nvidia compute cache) scatter them
# across the repository root, which then shows up in git status.
export HOME="$ROOT/assets/cache/home"
export XDG_CACHE_HOME="$ROOT/assets/cache"

# The directory shape a workspace needs. Creating them here rather than in the
# image means a bind-mounted, empty host directory works on the first run.
mkdir -p "$ROOT"/assets/{envs,models,repositories,hf-cache,cache/home} \
         "$ROOT"/data/{jobs,uploads} \
         "$ROOT"/benchmark/{logs,working,raw_outputs,normalized_outputs,visualizations,metrics,reports,environments,inventory} \
         2>/dev/null || true

if [ -n "${HOST_UID:-}" ] && [ "${HOST_UID}" != "0" ]; then
  if ! id -u dla >/dev/null 2>&1; then
    groupadd -g "${HOST_GID:-$HOST_UID}" dla 2>/dev/null || true
    useradd -u "${HOST_UID}" -g "${HOST_GID:-$HOST_UID}" -d "$ROOT" -M -s /bin/bash dla 2>/dev/null || true
  fi
  RUN_AS=(gosu "${HOST_UID}:${HOST_GID:-$HOST_UID}")
else
  RUN_AS=()
fi

run() {
  if [ ${#RUN_AS[@]} -gt 0 ]; then
    exec "${RUN_AS[@]}" "$@" 2>/dev/null \
      || exec setpriv --reuid "${HOST_UID}" --regid "${HOST_GID:-$HOST_UID}" --clear-groups "$@"
  fi
  exec "$@"
}

# Prefer the dedicated harness virtualenv; fall back to the image's own Python,
# which carries the same dependencies. That is what lets `make up` serve a
# working UI on a fresh clone, before `make setup` has built anything.
HARNESS_PY="$ROOT/assets/envs/harness/bin/python"
[ -x "$HARNESS_PY" ] || HARNESS_PY="$(command -v python3 || command -v python)"

case "${1:-bash}" in
  serve)
    cd "$ROOT"
    if [ ! -x "$ROOT/assets/envs/harness/bin/python" ]; then
      echo "No model environments are built yet — the UI will start and report that."
      echo "Build them with:  make setup"
    fi
    run "$HARNESS_PY" -m app.server
    ;;
  setup)
    shift
    cd "$ROOT"
    run bash scripts/bootstrap.sh "$@"
    ;;
  *)
    run "$@"
    ;;
esac
