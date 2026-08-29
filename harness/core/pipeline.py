#!/usr/bin/env python3
"""End-to-end orchestration: a PDF in, a self-contained report out.

The stages below are the whole product. Each one is a separate process running
`python -m core.<module>` inside the harness virtualenv, with `DLA_WORKSPACE`
pointing at this job's directory. Subprocesses rather than imports, for three
reasons: a stage that segfaults cannot take the web server with it, each stage
gets its own log file, and every stage stays runnable by hand exactly as the
pipeline runs it — which is what makes a failure reproducible.

Progress is written to `<workspace>/status.json` after every state change, so a
UI can poll one small file instead of holding a connection open, and a job whose
server restarted mid-run still reports what it had reached.

    python -m core.pipeline --input report.pdf --profile fast
    python -m core.pipeline --workspace data/jobs/abc --resume
"""
import argparse, json, os, re, shutil, subprocess, sys, time, uuid
from datetime import datetime, timezone

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# name, module, extra argv, required
STAGES = [
    ("inventory", "core.inventory", [], True,
     "Characterise every page: geometry, text layer, script, images, ruling."),
    ("select", "core.select_pages", [], True,
     "Choose pages and render them once at 300 and 150 dpi."),
    ("reference", "core.reference", [], False,
     "Build the PDF Structural Reference from the file's own content stream."),
    ("run", "core.runner", [], True,
     "Run every selected layout system in its own environment."),
    ("metrics", "core.metrics", [], False,
     "Geometric agreement between each system and the reference."),
    ("consensus", "core.consensus", [], False,
     "Cross-system agreement, deduplicated by repository family."),
    ("evidence", "core.class_evidence", [], False,
     "Per-class leaders and whether they are separable."),
    ("ensemble", "core.ensemble", [], False,
     "Whether routing by class would beat any single system."),
    ("ratings", "core.ratings", [], False,
     "Rubric scores derived from the measured quantities."),
    ("manifest", "core.manifest", [], False,
     "Record what ran: commits, checkpoints, configs, environments, checksums."),
    ("package", "core.package_report", [], True,
     "Assemble the report data bundle: pages, regions, metrics."),
    ("report", "core.build_report", [], True,
     "Write the self-contained HTML report."),
]

# Stages whose module parses a --workspace flag.  The rest read DLA_WORKSPACE
# from the environment, which is set for every stage either way; the flag exists
# so each command in a job's logs can be pasted into a shell and reproduced.
WS_FLAG = {"inventory", "select", "reference", "run", "package", "report"}
CORPUS_FLAG = {"inventory", "select", "reference"}


class Job:
    """One analysis run: its workspace, its status file, its stages."""

    def __init__(self, ws, job_id=None, meta=None):
        self.ws = paths.ensure_workspace(ws)
        self.id = job_id or os.path.basename(ws.rstrip("/"))
        self.path = os.path.join(self.ws, "status.json")
        self.state = self._read() or {
            "job_id": self.id, "state": "pending", "created": now(),
            "workspace": self.ws, "stages": [
                {"name": n, "state": "pending", "description": d}
                for n, _m, _a, _r, d in STAGES],
            "systems": {"total": 0, "done": 0, "current": None, "results": []},
            "error": None, "report": None,
        }
        if meta:
            self.state.setdefault("meta", {}).update(meta)
        self.save()

    def _read(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf8") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.path)          # atomic: a poller never sees half a file

    def stage(self, name):
        for s in self.state["stages"]:
            if s["name"] == name:
                return s
        raise KeyError(name)

    def set_stage(self, name, **kw):
        self.stage(name).update(kw)
        self.save()


def stage_argv(name, module, extra, ws, corpus, profile, template, limit):
    argv = [paths.harness_python(), "-m", module]
    if name in WS_FLAG:
        argv += ["--workspace", ws]
    if name in CORPUS_FLAG:
        argv += ["--corpus", corpus]
    if name == "run":
        if profile:
            argv += ["--profile", profile]
        if limit:
            argv += ["--limit", str(limit)]
    if name == "report" and template:
        argv += ["--template", template]
    return argv + list(extra)


RUN_LINE = re.compile(r"^\[(run|done|skip|env-missing|crashed|timed_out|error)\]\s+(\S+)")


def run_stage(job, name, module, extra, required, corpus, profile, template, limit):
    ws = job.ws
    log_path = os.path.join(ws, "logs", f"stage_{name}.log")
    argv = stage_argv(name, module, extra, ws, corpus, profile, template, limit)
    env = dict(os.environ,
               PYTHONPATH=paths.HARNESS, DLA_ROOT=paths.ROOT, DLA_WORKSPACE=ws,
               PYTHONUNBUFFERED="1")
    job.set_stage(name, state="running", started=now(), log=os.path.relpath(log_path, ws))
    t0 = time.time()
    rc = 0
    with open(log_path, "w", encoding="utf8") as lf:
        lf.write("$ " + " ".join(argv) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, bufsize=1)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            if name == "run":
                m = RUN_LINE.match(line)
                if m:
                    verb, sid = m.group(1), m.group(2)
                    sysinfo = job.state["systems"]
                    if verb == "run":
                        sysinfo["current"] = sid
                    else:
                        sysinfo["current"] = None
                        sysinfo["done"] += 1
                        sysinfo["results"].append({"system": sid, "outcome": verb})
                    job.save()
                elif line.startswith("[runner]"):
                    m2 = re.search(r"systems=(\d+)\s+pages=(\d+)", line)
                    if m2:
                        job.state["systems"]["total"] = int(m2.group(1))
                        job.state["pages"] = int(m2.group(2))
                        job.save()
        rc = proc.wait()
    dt = round(time.time() - t0, 1)

    if rc == 0:
        job.set_stage(name, state="done", ended=now(), seconds=dt)
        return True
    tail = ""
    try:
        with open(log_path, encoding="utf8") as f:
            tail = "".join(f.readlines()[-12:]).strip()
    except Exception:
        pass
    job.set_stage(name, state="failed", ended=now(), seconds=dt,
                  exit_code=rc, error=tail[-2000:])
    if required:
        job.state["state"] = "failed"
        job.state["error"] = f"stage '{name}' failed (exit {rc}). See {log_path}"
        job.save()
        return False
    # An optional stage that fails degrades the report instead of losing the run:
    # a scanned PDF has no text layer, so the reference and every metric derived
    # from it are legitimately unavailable, and the comparison is still worth
    # having.
    job.set_stage(name, state="skipped", note="optional stage failed; continuing")
    return True


def ingest(ws, inputs):
    """Copy the input PDFs into the job so it is self-describing forever after."""
    dst = os.path.join(ws, "input")
    os.makedirs(dst, exist_ok=True)
    kept = []
    for src in inputs:
        if os.path.isdir(src):
            for fn in sorted(os.listdir(src)):
                if fn.lower().endswith(".pdf"):
                    shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                    kept.append(fn)
            continue
        if not src.lower().endswith(".pdf"):
            raise SystemExit(f"not a PDF: {src}")
        fn = os.path.basename(src)
        if os.path.abspath(src) != os.path.abspath(os.path.join(dst, fn)):
            shutil.copy2(src, os.path.join(dst, fn))
        kept.append(fn)
    if not kept:
        raise SystemExit("no PDF inputs")
    return kept


def run(ws, inputs=None, profile=None, template=None, limit=None, job_id=None,
        resume=False, corpus=None):
    profile = profile or paths.get("run", "profile", "balanced")
    template = template or paths.get("report", "template", "viewer.html")
    job = Job(ws, job_id, meta={"profile": profile, "template": template})

    if inputs:
        job.state["meta"]["documents"] = ingest(job.ws, inputs)
    # A job analyses the documents it was given; a corpus run analyses a
    # directory. Both go through the same stages.
    if corpus is None:
        inp = os.path.join(job.ws, "input")
        corpus = inp if os.path.isdir(inp) and os.listdir(inp) else paths.CORPUS
    job.state["meta"]["corpus"] = corpus
    job.state["state"] = "running"
    job.state["started"] = now()
    job.save()

    for name, module, extra, required, _desc in STAGES:
        st = job.stage(name)
        if resume and st.get("state") == "done":
            continue
        ok = run_stage(job, name, module, extra, required, corpus, profile, template, limit)
        if not ok:
            return job.state

    report = os.path.join(job.ws, "reports", "index.html")
    job.state["report"] = os.path.relpath(report, job.ws) if os.path.exists(report) else None
    job.state["state"] = "done"
    job.state["ended"] = now()
    job.save()
    return job.state


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="*", default=None,
                    help="PDF file(s) or a directory of PDFs")
    ap.add_argument("--workspace", default=None,
                    help="job workspace (default: a new one under data/jobs)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--template", default=None, help="report body template")
    ap.add_argument("--limit", type=int, default=None, help="first N pages only")
    ap.add_argument("--corpus", default=None,
                    help="analyse this directory of PDFs instead of the job's own input/")
    ap.add_argument("--resume", action="store_true",
                    help="skip stages already marked done")
    a = ap.parse_args()

    if a.workspace:
        ws = paths.resolve(a.workspace)
    else:
        ws = os.path.join(paths.JOBS, datetime.now().strftime("%Y%m%d-%H%M%S-")
                          + uuid.uuid4().hex[:6])
    if not (a.input or a.corpus or a.resume or os.path.isdir(os.path.join(ws, "input"))):
        raise SystemExit("give --input <pdf...> or --corpus <dir>, or --resume an existing job")

    state = run(ws, a.input, a.profile, a.template, a.limit, resume=a.resume,
                corpus=paths.resolve(a.corpus) if a.corpus else None)
    print(json.dumps({k: state.get(k) for k in
                      ("job_id", "state", "workspace", "report", "error")}, indent=1))
    raise SystemExit(0 if state.get("state") == "done" else 1)


if __name__ == "__main__":
    main()
