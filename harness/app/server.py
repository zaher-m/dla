#!/usr/bin/env python3
"""The web application: upload a PDF, watch it run, open the report.

Deliberately small. It is an HTTP wrapper around ``core.pipeline`` and owns
nothing the pipeline does not: no database, no queue broker, no session state.
A job *is* a directory under ``data/jobs``, its status *is* ``status.json``, and
its result *is* ``reports/index.html``. Restarting the server loses nothing and
a job directory can be copied to another machine and still open.

Concurrency is one job at a time by default (``server.concurrency``), because a
dozen models contend for a single GPU and running two jobs concurrently makes
both slower and the timings meaningless. Extra submissions queue.

    python -m app.server
    DLA_SERVER_PORT=9000 python -m app.server
"""
import json, os, queue, re, shutil, sys, threading, time, uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths            # noqa: E402
from core import pipeline         # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BYTES = int(paths.get("server", "max_upload_mb", 200)) * 1024 * 1024
JOB_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

app = FastAPI(title="DLA Suite", docs_url="/api/docs", openapi_url="/api/openapi.json")

_work = queue.Queue()
_lock = threading.Lock()


# --------------------------------------------------------------------------
# job helpers
# --------------------------------------------------------------------------
def job_dir(job_id):
    """Resolve a job id to its directory, refusing anything that is not one.

    Job ids come from URLs, so the pattern check is the whole defence against
    a path-traversal read of the filesystem.  Nothing else in this file joins
    user input onto a path.
    """
    if not JOB_ID.match(job_id or ""):
        raise HTTPException(400, "malformed job id")
    d = os.path.join(paths.JOBS, job_id)
    if not os.path.isdir(d):
        raise HTTPException(404, "no such job")
    return d


def read_status(d):
    p = os.path.join(d, "status.json")
    if not os.path.exists(p):
        return {"job_id": os.path.basename(d), "state": "unknown"}
    try:
        with open(p, encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {"job_id": os.path.basename(d), "state": "unreadable"}


def profile_ready(wanted):
    """How many systems in a profile actually have an environment on disk."""
    import yaml
    with open(paths.REGISTRY, encoding="utf8") as f:
        reg = yaml.safe_load(f)["systems"]
    return sum(1 for s in reg
               if (wanted is None or s["id"] in wanted)
               and os.path.exists(paths.env_python(s["env"])))


def summarise(st):
    stages = st.get("stages") or []
    done = sum(1 for s in stages if s.get("state") in ("done", "skipped"))
    sysinfo = st.get("systems") or {}
    return {
        "job_id": st.get("job_id"), "state": st.get("state"),
        "created": st.get("created"), "ended": st.get("ended"),
        "documents": (st.get("meta") or {}).get("documents") or [],
        "profile": (st.get("meta") or {}).get("profile"),
        "pages": st.get("pages"),
        "stage_done": done, "stage_total": len(stages),
        "systems_done": sysinfo.get("done", 0), "systems_total": sysinfo.get("total", 0),
        "current": sysinfo.get("current"),
        "error": st.get("error"),
        "has_report": bool(st.get("report")),
    }


# --------------------------------------------------------------------------
# the single worker
# --------------------------------------------------------------------------
def worker():
    while True:
        task = _work.get()
        if task is None:
            return
        ws, profile, template, limit = task
        try:
            pipeline.run(ws, None, profile, template, limit)
        except Exception as e:                 # never let one job kill the worker
            st = read_status(ws)
            st["state"] = "failed"
            st["error"] = f"{type(e).__name__}: {e}"
            with open(os.path.join(ws, "status.json"), "w", encoding="utf8") as f:
                json.dump(st, f, indent=1)
        finally:
            _work.task_done()


for _ in range(max(1, int(paths.get("server", "concurrency", 1)))):
    threading.Thread(target=worker, daemon=True).start()


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/config")
def api_config():
    """What this deployment can do — the UI builds itself from this."""
    profiles = []
    pdir = paths.PROFILES
    if os.path.isdir(pdir):
        import yaml
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".yaml"):
                continue
            with open(os.path.join(pdir, fn), encoding="utf8") as f:
                spec = yaml.safe_load(f) or {}
            name = fn[:-5]
            try:
                n = len(paths.load_profile(name) or [])
            except Exception:
                n = None
            try:
                ready = profile_ready(paths.load_profile(name))
            except Exception:
                ready = None
            profiles.append({"name": name, "systems": n, "ready": ready,
                             "description": (spec.get("description") or "").strip()})
    ready = []
    missing = []
    import yaml
    with open(paths.REGISTRY, encoding="utf8") as f:
        reg = yaml.safe_load(f)["systems"]
    for s in reg:
        (ready if os.path.exists(paths.env_python(s["env"])) else missing).append(s["id"])
    return {
        "profiles": profiles,
        "default_profile": paths.get("run", "profile", "balanced"),
        "max_upload_mb": int(paths.get("server", "max_upload_mb", 200)),
        "max_pages": int(paths.get("selection", "max_pages", 25)),
        "systems_ready": len(ready), "systems_total": len(reg),
        "systems_missing_env": missing,
    }


@app.get("/api/jobs")
def api_jobs(limit: int = 50):
    if not os.path.isdir(paths.JOBS):
        return []
    ids = sorted((d for d in os.listdir(paths.JOBS)
                  if os.path.isdir(os.path.join(paths.JOBS, d))), reverse=True)
    return [summarise(read_status(os.path.join(paths.JOBS, i))) for i in ids[:limit]]


@app.post("/api/jobs")
async def api_create(file: UploadFile = File(...), profile: str = Form(None),
                     max_pages: int = Form(None)):
    name = os.path.basename(file.filename or "document.pdf")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")

    profile = profile or paths.get("run", "profile", "balanced")
    try:
        wanted = paths.load_profile(profile)
    except FileNotFoundError:
        raise HTTPException(400, f"unknown profile: {profile}")

    # Refuse rather than produce an empty report. A fresh clone has no model
    # environments, and a job that "succeeds" with nothing in it is a worse
    # answer than a clear refusal.
    if not profile_ready(wanted):
        raise HTTPException(409,
                            f"no system in the '{profile}' profile has an environment built. "
                            f"Run `make setup SETUP_PROFILE={profile}` first.")

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    ws = paths.ensure_workspace(os.path.join(paths.JOBS, job_id))
    os.makedirs(os.path.join(ws, "input"), exist_ok=True)
    dst = os.path.join(ws, "input", name)

    # Stream to disk with a hard ceiling rather than reading the body into
    # memory: an unbounded upload is the one denial-of-service this endpoint
    # would otherwise hand out for free.
    written = 0
    with open(dst, "wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_BYTES:
                out.close()
                shutil.rmtree(ws, ignore_errors=True)
                raise HTTPException(413, f"file exceeds {MAX_BYTES // 2**20} MB")
            out.write(chunk)
    if written == 0:
        shutil.rmtree(ws, ignore_errors=True)
        raise HTTPException(400, "empty upload")

    # Reject a file that is not actually a PDF before it reaches the pipeline,
    # so the failure is a clear 400 rather than a stage crash three steps later.
    try:
        import fitz
        with fitz.open(dst) as doc:
            n_pages = doc.page_count
            if n_pages < 1:
                raise ValueError("no pages")
    except Exception as e:
        shutil.rmtree(ws, ignore_errors=True)
        raise HTTPException(400, f"could not open as PDF: {e}")

    job = pipeline.Job(ws, job_id, meta={"profile": profile, "documents": [name],
                                         "n_pages_input": n_pages,
                                         "queued": pipeline.now()})
    job.state["state"] = "queued"
    job.save()
    _work.put((ws, profile, None, max_pages or None))
    return JSONResponse({"job_id": job_id, "queued": _work.qsize(),
                         "n_pages_input": n_pages}, status_code=202)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    return read_status(job_dir(job_id))


@app.get("/api/jobs/{job_id}/log/{stage}")
def api_log(job_id: str, stage: str):
    if not re.fullmatch(r"[a-z_]+", stage):
        raise HTTPException(400, "bad stage name")
    p = os.path.join(job_dir(job_id), "logs", f"stage_{stage}.log")
    if not os.path.exists(p):
        raise HTTPException(404, "no log for that stage")
    with open(p, encoding="utf8", errors="replace") as f:
        return PlainTextResponse(f.read()[-200_000:])


@app.get("/api/jobs/{job_id}/report")
def api_report(job_id: str):
    p = os.path.join(job_dir(job_id), "reports", "index.html")
    if not os.path.exists(p):
        raise HTTPException(404, "report not built yet")
    return FileResponse(p, media_type="text/html")


@app.get("/api/jobs/{job_id}/bundle")
def api_bundle(job_id: str):
    """The report's data, for anyone who wants the numbers without the HTML."""
    p = os.path.join(job_dir(job_id), "reports", "report_data.json")
    if not os.path.exists(p):
        raise HTTPException(404, "not built yet")
    return FileResponse(p, media_type="application/json",
                        filename=f"{job_id}-report-data.json")


@app.delete("/api/jobs/{job_id}")
def api_delete(job_id: str):
    d = job_dir(job_id)
    st = read_status(d)
    if st.get("state") in ("running", "queued"):
        raise HTTPException(409, "job is still running")
    shutil.rmtree(d, ignore_errors=True)
    return {"deleted": job_id}


@app.get("/api/health")
def api_health():
    return {"ok": True, "queue": _work.qsize(), "paths": paths.describe()}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html"), encoding="utf8") as f:
        return f.read()


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    job_dir(job_id)
    with open(os.path.join(STATIC, "index.html"), encoding="utf8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    host = paths.get("server", "host", "0.0.0.0")
    port = int(paths.get("server", "port", 8080))
    print(f"DLA Suite on http://{host}:{port}   workspace root={paths.JOBS}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
