#!/usr/bin/env python3
"""Dispatch each registered system to its own virtualenv, over one page set.

Every benchmarked system has dependencies incompatible with most of the others,
so none of them are importable from here.  Each one runs as a separate process
under ``assets/envs/<env>/bin/python``, receives a JSON job file describing the
pages and its configuration, and writes normalised results into the workspace.
The runner never imports a model; it starts processes and records what
happened.  That is what keeps a dozen mutually hostile dependency sets in one
repository.

    python -m core.runner --list
    python -m core.runner --profile fast
    python -m core.runner --systems docling.heron surya.v2_layout --force
    python -m core.runner --profile full --workspace data/jobs/<id>
"""
import argparse, json, os, subprocess, sys, time
import yaml

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402

ROOT = paths.ROOT
HARNESS = paths.HARNESS
REG = paths.REGISTRY


def load_registry():
    with open(REG, encoding="utf8") as f:
        return yaml.safe_load(f)["systems"]


def load_pages(ws):
    """Page records for the workspace, with every input kind resolved to a path."""
    with open(os.path.join(ws, "inventory", "selected_pages.json"), encoding="utf8") as f:
        pages = json.load(f)
    work = os.path.join(ws, "working")
    for p in pages:
        p["image_300dpi"] = os.path.join(work, "pages_300dpi", f"{p['page_id']}.png")
        p["image_150dpi"] = os.path.join(work, "pages_150dpi", f"{p['page_id']}.png")
        p["page_pdf"] = os.path.join(work, "pages_pdf", f"{p['page_id']}.pdf")
    return pages


def select(reg, ids=None, profile=None):
    """Which systems to run: explicit ids win, then a profile, then everything."""
    if ids:
        wanted = set(ids)
        known = {s["id"] for s in reg}
        missing = wanted - known
        if missing:
            raise SystemExit(f"unknown system id(s): {sorted(missing)}")
        return [s for s in reg if s["id"] in wanted]
    allowed = paths.load_profile(profile)
    if allowed is None:
        return list(reg)
    return [s for s in reg if s["id"] in allowed]


def run_system(sysdef, pages, ws, limit=None, force=False, timeout=None, quiet=False):
    rid = sysdef["id"]
    norm_dir = os.path.join(ws, "normalized_outputs", rid)
    raw_dir = os.path.join(ws, "raw_outputs", rid)
    manifest = os.path.join(norm_dir, "_run.json")
    if os.path.exists(manifest) and not force:
        if not quiet:
            print(f"[skip] {rid} already has results ({manifest})")
        with open(manifest, encoding="utf8") as f:
            return json.load(f)

    py = paths.env_python(sysdef["env"])
    if not os.path.exists(py):
        rec = {"run_id": rid, "status": "env_missing",
               "note": f"virtualenv '{sysdef['env']}' not present at {py}. "
                       f"Run: bash harness/setup/{sysdef['env']}.sh"}
        os.makedirs(norm_dir, exist_ok=True)
        with open(manifest, "w", encoding="utf8") as f:
            json.dump(rec, f, indent=1)
        print(f"[env-missing] {rid}")
        return rec

    sel = pages if limit is None else pages[:limit]
    kind = sysdef.get("input", "image_300dpi")
    job = {"run_id": rid, "system": sysdef["repo"], "display": sysdef["display"],
           "taxonomy": sysdef["taxonomy"], "config": sysdef.get("config") or {},
           "input_kind": kind,
           "raw_dir": raw_dir, "norm_dir": norm_dir,
           "root": ROOT, "bench": ws, "workspace": ws,
           "pages": [{**p, "input_path": p[kind]} for p in sel]}
    logs = os.path.join(ws, "logs")
    os.makedirs(logs, exist_ok=True)
    jobf = os.path.join(logs, f"job_{rid}.json")
    with open(jobf, "w", encoding="utf8") as f:
        json.dump(job, f, indent=1)

    log = os.path.join(logs, f"run_{rid}.log")
    cmd = [py, os.path.join(HARNESS, "adapters", sysdef["adapter"] + ".py"), "--job", jobf]
    env = dict(os.environ, PYTHONPATH=HARNESS, DLA_ROOT=ROOT, DLA_WORKSPACE=ws)
    if not quiet:
        print(f"[run] {rid}")
    t0 = time.time()
    timed_out = False
    with open(log, "w", encoding="utf8") as lf:
        lf.write(f"$ {' '.join(cmd)}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                  timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, -1
            lf.write(f"\n*** timed out after {timeout}s ***\n")
    dt = time.time() - t0

    if os.path.exists(manifest) and not timed_out:
        with open(manifest, encoding="utf8") as f:
            man = json.load(f)
        man["wall_s"] = round(dt, 2)
        man["exit_code"] = rc
        with open(manifest, "w", encoding="utf8") as f:
            json.dump(man, f, indent=1)
        print(f"[done] {rid} rc={rc} ok={man.get('n_ok')} "
              f"failed={man.get('n_failed')} {dt:.1f}s")
        return man

    rec = {"run_id": rid, "status": "timed_out" if timed_out else "crashed",
           "exit_code": rc, "wall_s": round(dt, 2), "note": f"see {log}"}
    os.makedirs(norm_dir, exist_ok=True)
    with open(manifest, "w", encoding="utf8") as f:
        json.dump(rec, f, indent=1)
    print(f"[{rec['status']}] {rid} rc={rc} — see {log}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", default=None,
                    help="explicit system ids; overrides --profile")
    ap.add_argument("--profile", default=None,
                    help="named profile from config/profiles (default: run.profile)")
    ap.add_argument("--all", action="store_true", help="every registered system")
    ap.add_argument("--list", action="store_true", help="show the registry and env status")
    ap.add_argument("--workspace", default=None,
                    help="output workspace (default: paths.workspace)")
    ap.add_argument("--limit", type=int, default=None, help="first N pages only")
    ap.add_argument("--force", action="store_true",
                    help="re-run systems that already have results")
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-system wall-clock ceiling, seconds")
    a = ap.parse_args()

    reg = load_registry()
    if a.list:
        allowed = None
        if a.profile:
            try:
                allowed = paths.load_profile(a.profile)
            except FileNotFoundError as e:
                raise SystemExit(str(e))
        for s in reg:
            mark = "✓" if os.path.exists(paths.env_python(s["env"])) else "·"
            sel = "" if allowed is None else ("  " if s["id"] in allowed else " -")
            print(f"{mark}{sel} {s['id']:38s} env={s['env']:12s} {s['display']}")
        return

    ws = paths.ensure_workspace(paths.resolve(a.workspace) if a.workspace else None)
    todo = list(reg) if a.all else select(reg, a.systems, a.profile)
    if not todo:
        raise SystemExit("nothing selected")
    timeout = a.timeout if a.timeout is not None else paths.get("run", "system_timeout_s")
    pages = load_pages(ws)
    print(f"[runner] workspace={ws}  systems={len(todo)}  pages={len(pages)}")

    summary = []
    for s in todo:
        try:
            summary.append(run_system(s, pages, ws, a.limit, a.force, timeout))
        except Exception as e:              # a broken adapter must not stop the batch
            if not paths.get("run", "continue_on_error", True):
                raise
            print(f"[error] {s['id']}: {e}")
            summary.append({"run_id": s["id"], "status": "error", "note": str(e)})
    print(json.dumps([{k: v.get(k) for k in ("run_id", "status", "n_ok", "n_failed", "wall_s")}
                      for v in summary], indent=1))


if __name__ == "__main__":
    main()
