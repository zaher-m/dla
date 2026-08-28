#!/usr/bin/env python3
"""Emit benchmark/benchmark_manifest.json — the machine-readable record of
everything needed to reproduce this benchmark.
"""
import hashlib, json, os, platform, subprocess, sys
import yaml

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
REPOS = paths.REPOS
ENVS = paths.ENVS


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_state():
    out = {}
    for name in sorted(os.listdir(REPOS)):
        p = os.path.join(REPOS, name)
        if not os.path.isdir(os.path.join(p, ".git")):
            continue
        out[name] = {
            "commit": sh("git rev-parse HEAD", p),
            "describe": sh("git describe --tags --always", p),
            "branch": sh("git rev-parse --abbrev-ref HEAD", p),
            "remote": sh("git config --get remote.origin.url", p),
            "committed_utc": sh("git log -1 --format=%cI", p),
        }
    return out


def env_state():
    envs = {}
    d = os.path.join(BENCH, "environments")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".runtime.json"):
                try:
                    envs[f.split(".")[0]] = json.load(open(os.path.join(d, f)))
                except Exception:
                    pass
    return envs


def main():
    reg = yaml.safe_load(open(os.path.join(ROOT, "harness", "registry.yaml")))["systems"]
    runs = {}
    for s in reg:
        man = os.path.join(BENCH, "normalized_outputs", s["id"], "_run.json")
        if os.path.exists(man):
            m = json.load(open(man))
            runs[s["id"]] = {k: m.get(k) for k in
                             ("status", "n_pages", "n_ok", "n_failed", "model",
                              "model_load_s", "config", "resources", "torch_env",
                              "wall_s", "exit_code", "python", "torch", "cuda")}
        else:
            runs[s["id"]] = {"status": "not_run"}

    inv_path = os.path.join(BENCH, "inventory", "corpus_inventory.json")
    inv = {}
    if os.path.exists(inv_path):
        with open(inv_path, encoding="utf8") as f:
            inv = json.load(f)

    docs = {}
    corpus = paths.CORPUS
    for f in sorted(os.listdir(corpus)) if os.path.isdir(corpus) else []:
        p = os.path.join(corpus, f)
        if f.lower().endswith(".pdf"):
            docs[f] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}

    man = {
        "benchmark": "document layout analysis, side-by-side evaluation",
        "schema_version": "1.1",
        "generated_utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
        "scope": "layout analysis only (region localisation, class, shape, order). "
                 "No OCR, table-content, formula-recognition or conversion quality is scored.",
        "ground_truth": {
            "human_annotations": False,
            "label": "QUANTITATIVE-GEOMETRY / NO-GROUND-TRUTH-CLASSES",
            "reference": "PDF Structural Reference derived from the source PDFs' content streams "
                         "(harness/core/reference.py). Objective for localisation; carries no class labels.",
            "class_assessment": "cross-system consensus (harness/core/consensus.py) + visual review",
        },
        "host": {
            "uname": sh("uname -a"), "arch": platform.machine(),
            "cpu": sh("lscpu | grep -E 'Model name|^CPU\\(s\\)' | head -2"),
            "ram_gb": sh("free -g | awk 'NR==2{print $2}'"),
            "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
            "nvcc": sh("nvcc --version | tail -2 | head -1"),
            "docker": sh("docker --version"),
        },
        "container": {
            "image": "dla-bench:1.0",
            "base": "nvcr.io/nvidia/vllm:26.02-py3 (linux/arm64, CUDA 13.1, Python 3.12.3)",
            "dockerfile": "docker/Dockerfile",
            "compose": "docker/compose.yaml",
            "rationale": "the base image already carries an aarch64 CUDA build of PyTorch 2.11 + "
                         "torchvision so no adapter re-downloads a CUDA wheel",
            "mounts": {"/work": "project root", "/hf-cache": "/path/to/hf-cache (shared)"},
        },
        "critical_environment_finding": {
            "id": "cudnn-sm121-wrong-convolutions",
            "summary": "cuDNN returns silently incorrect convolution results for several detection "
                       "backbones on some GPUs with the container's CUDA stack.",
            "affected_observed": ["docling.heron", "docling.egret_large", "docling.egret_xlarge",
                                  "mineru.pipeline"],
            "mitigation": "harness/core/torch_env.py disables the cuDNN backend for every adapter; "
                          "override with DLA_ALLOW_CUDNN=1 to reproduce the fault.",
            "validation": "harness/core/validate_device.py compares forced-CPU and GPU outputs "
                          "box-for-box; results in benchmark/metrics/device_agreement.json",
        },
        "input": {"documents": docs,
                  "pages_total": sum(d.get("pages", 0) for d in inv.get("documents", []))
                                 if isinstance(inv, dict) else None,
                  "corpus_dir": os.path.relpath(corpus, ROOT),
                  "selected_pages": os.path.relpath(
                      os.path.join(BENCH, "inventory", "selected_pages.json"), ROOT),
                  "render_dpi": 300, "render_tool": "PyMuPDF get_pixmap(dpi=300, RGB)"},
        "repositories": repo_state(),
        "environments": env_state(),
        "systems": {s["id"]: {"repo": s["repo"], "display": s["display"], "env": s["env"],
                              "adapter": s["adapter"], "taxonomy": s["taxonomy"],
                              "input": s.get("input"), "config": s.get("config")} for s in reg},
        "runs": runs,
        "artifacts": {
            "inventory": "benchmark/inventory/",
            "raw_outputs": "benchmark/raw_outputs/<system>/<page>.raw.json",
            "normalized_outputs": "benchmark/normalized_outputs/<system>/<page>.json",
            "visualizations": "benchmark/visualizations/<page>/",
            "metrics": "benchmark/metrics/",
            "interactive_report": "benchmark/reports/index.html",
            "final_report": "benchmark/reports/FINAL_REPORT.md",
        },
        "commands": {
            "build_image": "make build",
            "setup_all_envs": "make setup",
            "start_ui": "make up            # http://localhost:8080",
            "shell": "make shell",
            "list_systems": "make list",
            "run_reference_benchmark": "make bench PROFILE=full",
            "one_stage": "make stage STAGE=metrics",
            "cli_analyse_a_pdf": "python -m core.pipeline --input <file.pdf> --profile balanced",
        },
        "environment_variables": {
            "HF_HOME": paths.HF_CACHE, "HUGGINGFACE_HUB_CACHE": os.path.join(paths.HF_CACHE, "hub"),
            "DLA_ROOT": ROOT, "DLA_WORKSPACE": BENCH, "PYTHONPATH": paths.HARNESS,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
            "DLA_ALLOW_CUDNN": os.environ.get("DLA_ALLOW_CUDNN", "unset (guard active)"),
        },
        "paths": paths.describe(),
    }
    p = os.path.join(BENCH, "benchmark_manifest.json")
    json.dump(man, open(p, "w"), indent=1, ensure_ascii=False)
    print("wrote", p, f"{os.path.getsize(p)/1024:.0f} KB")


if __name__ == "__main__":
    main()
