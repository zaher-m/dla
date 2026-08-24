"""Shared plumbing for every benchmark adapter.

An adapter is a small script executed by the runner *inside its own
virtualenv*.  It receives a job JSON, loads its model once, warms up, then
times every page with the same four-phase breakdown so systems stay
comparable:

    model_load | preprocess | inference | postprocess

Resource use is captured as peak CUDA allocation/reservation (torch, when
available) and peak process RSS.
"""
import argparse, json, os, resource, sys, time, traceback
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.taxonomy import map_class          # noqa: E402
from core import schema                       # noqa: E402
from core import torch_env                    # noqa: E402

# Applied before any model is constructed, for every adapter.
TORCH_ENV = torch_env.configure()


def parse_job():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    with open(a.job) as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.acc = {}
    @contextmanager
    def phase(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - t0)
    def pop(self):
        out = {k: round(v, 5) for k, v in self.acc.items()}
        out["total_s"] = round(sum(self.acc.values()), 5)
        self.acc = {}
        return out


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def cuda_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def resources():
    r = {"peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    try:
        import torch
        if torch.cuda.is_available():
            r["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            r["cuda_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            r["device"] = torch.cuda.get_device_name(0)
        else:
            r["device"] = "cpu"
    except Exception:
        r.setdefault("device", "cpu")
    return r


def build_regions(taxonomy, raw_items):
    """raw_items: iterable of dicts with source_class, bbox, [confidence,
    polygon, mask, reading_order, extra]."""
    out = []
    for i, it in enumerate(raw_items, 1):
        canon, conf, _note = map_class(taxonomy, it["source_class"])
        out.append(schema.make_region(
            i, canon, it["source_class"], it["bbox"],
            confidence=it.get("confidence"), polygon=it.get("polygon"),
            mask=it.get("mask"), mapping_confidence=conf,
            reading_order=it.get("reading_order"), extra=it.get("extra")))
    return out


class AdapterRun:
    """Context manager writing the per-run manifest and per-page results."""

    def __init__(self, job):
        self.job = job
        self.run_id = job["run_id"]
        self.system = job["system"]
        self.taxonomy = job["taxonomy"]
        self.cfg = job.get("config") or {}
        self.raw_dir = job["raw_dir"]
        self.norm_dir = job["norm_dir"]
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.norm_dir, exist_ok=True)
        self.pages = job["pages"]
        self.model_info = {}
        self.model_load_s = None
        self.page_records = []
        self.failures = []

    def set_model_info(self, **kw):
        self.model_info.update(kw)

    def emit(self, page, regions, timing, raw=None, meta=None):
        res = schema.make_page_result(self.run_id, self.system, page, regions,
                                      timing, resources(), meta)
        schema.write(os.path.join(self.norm_dir, f"{page['page_id']}.json"), res)
        if raw is not None:
            schema.write(os.path.join(self.raw_dir, f"{page['page_id']}.raw.json"), raw)
        self.page_records.append({"page_id": page["page_id"], "n_regions": len(regions),
                                  "timing": timing})

    def fail(self, page, exc):
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.failures.append({"page_id": page["page_id"], "error": str(exc), "traceback": tb})
        sys.stderr.write(f"[{self.run_id}] FAILED {page['page_id']}: {exc}\n{tb}\n")

    def finish(self, status="ok", note=""):
        # a run where every page failed is not a successful run
        if status == "ok" and self.failures and not self.page_records:
            status = "failed"
        man = {"run_id": self.run_id, "system": self.system, "taxonomy": self.taxonomy,
               "config": self.cfg, "model": self.model_info,
               "model_load_s": self.model_load_s,
               "n_pages": len(self.pages), "n_ok": len(self.page_records),
               "n_failed": len(self.failures),
               "pages": self.page_records, "failures": self.failures,
               "resources": resources(), "status": status, "note": note,
               "torch_env": TORCH_ENV,
               "python": sys.version.split()[0]}
        try:
            import torch
            man["torch"] = torch.__version__
            man["cuda"] = torch.version.cuda
        except Exception:
            pass
        schema.write(os.path.join(self.norm_dir, "_run.json"), man)
        print(json.dumps({"run_id": self.run_id, "ok": len(self.page_records),
                          "failed": len(self.failures), "status": status}))
        return man
