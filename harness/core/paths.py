#!/usr/bin/env python3
"""Configuration and path resolution — the one place that knows where things live.

Every other module imports from here rather than hardcoding a directory, which
is what makes the same code serve three callers with different layouts:

  * a corpus run, whose workspace is ``benchmark/`` by default;
  * the web application, where each uploaded PDF gets its own workspace under
    ``data/jobs/<job_id>/``;
  * a developer running one stage by hand against either of those.

Resolution order, most specific first:

  1. an environment variable (``DLA_WORKSPACE``, ``DLA_SERVER_PORT``, …);
  2. the YAML file at ``DLA_CONFIG`` (default ``<root>/config/dla.yaml``);
  3. the built-in defaults in ``DEFAULTS`` below, so the package still works
     with no config file at all.

The distinction that matters most: **assets are shared, workspaces are not.**
Virtualenvs, model weights and cloned repositories are large, slow to build and
identical for every job, so they live outside any workspace and are never
copied. Everything a run produces — renders, raw outputs, metrics, reports —
lives inside the workspace and can be deleted without losing anything that
cost a download.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))          # <repo>/harness/core -> <repo>

DEFAULTS = {
    "paths": {
        "assets": "assets",
        "envs": "assets/envs",
        "models": "assets/models",
        "repositories": "assets/repositories",
        "corpus": "samples",
        "workspace": "benchmark",
        "data": "data",
        "hf_cache": "/hf-cache",
    },
    "render": {"dpi_primary": 300, "dpi_secondary": 150},
    "selection": {"mode": "all", "max_pages": 25, "quotas": {}},
    "run": {"profile": "balanced", "system_timeout_s": 3600,
            "continue_on_error": True, "reuse_existing": True},
    "report": {"template": "viewer.html", "image_width": 900, "jpeg_quality": 72},
    "server": {"host": "0.0.0.0", "port": 8080, "max_upload_mb": 200,
               "concurrency": 1, "retain_jobs": 50},
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(default, raw):
    """Environment variables are strings; give them the type the default has."""
    if isinstance(default, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


ROOT = os.path.abspath(os.environ.get("DLA_ROOT", DEFAULT_ROOT))
HARNESS = os.path.join(ROOT, "harness")
CONFIG_PATH = os.environ.get("DLA_CONFIG", os.path.join(ROOT, "config", "dla.yaml"))


def _load():
    cfg = DEFAULTS
    if os.path.exists(CONFIG_PATH):
        try:
            import yaml
            with open(CONFIG_PATH, encoding="utf8") as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
        except Exception as e:                       # never fail closed on config
            sys.stderr.write(f"[paths] could not read {CONFIG_PATH}: {e}\n")
    # Environment overrides.  `paths.*` uses the short name (DLA_WORKSPACE);
    # every other section uses DLA_<SECTION>_<KEY>.
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, default in values.items():
            names = [f"DLA_{section.upper()}_{key.upper()}"]
            if section == "paths":
                names.insert(0, f"DLA_{key.upper()}")
            for name in names:
                if name in os.environ:
                    values[key] = _coerce(default, os.environ[name])
                    break
    return cfg


CONFIG = _load()


def get(section, key, fallback=None):
    return (CONFIG.get(section) or {}).get(key, fallback)


def resolve(p):
    """Absolute path for a config value, relative to the repository root."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ROOT, p))


# --- shared assets: expensive to create, identical for every job -----------
ASSETS = resolve(get("paths", "assets"))
ENVS = resolve(get("paths", "envs"))
MODELS = resolve(get("paths", "models"))
REPOS = resolve(get("paths", "repositories"))
HF_CACHE = resolve(get("paths", "hf_cache"))

# --- the corpus and the active workspace ----------------------------------
CORPUS = resolve(get("paths", "corpus"))
WORKSPACE = resolve(get("paths", "workspace"))
DATA = resolve(get("paths", "data"))
UPLOADS = os.path.join(DATA, "uploads")
JOBS = os.path.join(DATA, "jobs")

REGISTRY = os.path.join(HARNESS, "registry.yaml")
PROFILES = os.path.join(ROOT, "config", "profiles")
REPORT_ASSETS = os.path.join(HARNESS, "report")

# Sub-directories of a workspace.  A job workspace has exactly this shape, so
# every stage works unchanged whether it is pointed at `benchmark/` or at
# `data/jobs/<id>/`.
WORKSPACE_DIRS = ("inventory", "working", "raw_outputs", "normalized_outputs",
                  "visualizations", "metrics", "reports", "logs", "environments")


def workspace(sub="", ws=None):
    return os.path.join(ws or WORKSPACE, sub) if sub else (ws or WORKSPACE)


def ensure_workspace(ws=None):
    ws = ws or WORKSPACE
    for d in WORKSPACE_DIRS:
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    os.makedirs(os.path.join(ws, "working", "pages_300dpi"), exist_ok=True)
    os.makedirs(os.path.join(ws, "working", "pages_150dpi"), exist_ok=True)
    os.makedirs(os.path.join(ws, "working", "pages_pdf"), exist_ok=True)
    return ws


def env_python(env_name):
    return os.path.join(ENVS, env_name, "bin", "python")


def harness_python():
    """The interpreter that runs the harness's own stages.

    Prefers the dedicated `harness` virtualenv, and falls back to the current
    interpreter. The container installs the harness's dependencies system-wide
    precisely so that a freshly cloned repository can ingest a PDF, build the
    reference and render a report before any model environment has been built —
    the UI then reports which models are unavailable instead of refusing to
    start. Model adapters never take this fallback: running one in the wrong
    environment is exactly the failure the per-system venvs exist to prevent.
    """
    p = env_python("harness")
    return p if os.path.exists(p) else sys.executable


def model_dir(name):
    return os.path.join(MODELS, name)


def repo_dir(name):
    return os.path.join(REPOS, name)


def load_profile(name=None):
    """Return the set of system ids a profile selects, or None for 'everything'.

    A profile is a YAML file with `include:` (list of ids, or the string "all")
    and an optional `exclude:` list.  Unknown ids are reported rather than
    silently dropped — a typo in a profile should not quietly shrink a run.
    """
    import yaml
    name = name or get("run", "profile", "balanced")
    if name in ("all", "*", None):
        return None
    path = os.path.join(PROFILES, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such profile: {name} ({path})")
    with open(path, encoding="utf8") as f:
        spec = yaml.safe_load(f) or {}
    with open(REGISTRY, encoding="utf8") as f:
        known = {s["id"] for s in (yaml.safe_load(f) or {}).get("systems", [])}
    inc = spec.get("include", "all")
    ids = set(known) if inc in ("all", "*") else set(inc)
    unknown = ids - known
    if unknown:
        sys.stderr.write(f"[paths] profile '{name}' lists unknown systems: "
                         f"{sorted(unknown)}\n")
    ids &= known
    ids -= set(spec.get("exclude") or [])
    return ids


def describe():
    return {
        "root": ROOT, "config": CONFIG_PATH if os.path.exists(CONFIG_PATH) else None,
        "assets": ASSETS, "envs": ENVS, "models": MODELS, "repositories": REPOS,
        "corpus": CORPUS, "workspace": WORKSPACE, "data": DATA,
        "hf_cache": HF_CACHE, "profile": get("run", "profile"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"paths": describe(), "config": CONFIG}, indent=2))
