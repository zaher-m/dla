#!/usr/bin/env python3
"""GPU-vs-CPU agreement gate.

Runs every torch-based system on a probe page twice, once with the device
forced to CPU and once on GPU, and flags any system whose boxes disagree beyond
a tolerance. Catches silently wrong kernels, which otherwise look like a weak
model.
"""
import argparse, json, os, subprocess, sys, tempfile, shutil

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
HARNESS = os.path.join(ROOT, "harness")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare(a_regions, b_regions, thr=0.7):
    """Symmetric greedy match; returns agreement metrics."""
    used = set(); matched = 0; cls_match = 0
    for ra in a_regions:
        best, bi = 0.0, None
        for i, rb in enumerate(b_regions):
            if i in used:
                continue
            v = iou(ra["bbox"], rb["bbox"])
            if v > best:
                best, bi = v, i
        if bi is not None and best >= thr:
            used.add(bi); matched += 1
            if ra["class"] == b_regions[bi]["class"]:
                cls_match += 1
    n = max(len(a_regions), len(b_regions), 1)
    return {"n_gpu": len(a_regions), "n_cpu": len(b_regions),
            "matched": matched, "class_agree": cls_match,
            "box_agreement": round(matched / n, 4),
            "class_agreement": round(cls_match / n, 4)}


def run_probe(sysdef, pages, device, outdir):
    py = os.path.join(BENCH, "envs", sysdef["env"], "bin", "python")
    job = {"run_id": sysdef["id"] + f".__probe_{device}", "system": sysdef["repo"],
           "display": sysdef["display"], "taxonomy": sysdef["taxonomy"],
           "config": sysdef.get("config") or {}, "input_kind": sysdef.get("input", "image_300dpi"),
           "raw_dir": os.path.join(outdir, "raw"), "norm_dir": os.path.join(outdir, "norm"),
           "root": ROOT, "bench": BENCH,
           "pages": [{**p, "input_path": p[sysdef.get("input", "image_300dpi")]} for p in pages]}
    jf = os.path.join(outdir, "job.json")
    os.makedirs(outdir, exist_ok=True)
    json.dump(job, open(jf, "w"))
    env = dict(os.environ, PYTHONPATH=HARNESS, DLA_ROOT=ROOT)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    log = os.path.join(outdir, "log.txt")
    with open(log, "w") as lf:
        subprocess.run([py, os.path.join(HARNESS, "adapters", sysdef["adapter"] + ".py"),
                        "--job", jf], stdout=lf, stderr=subprocess.STDOUT, env=env)
    res = {}
    nd = job["norm_dir"]
    if os.path.isdir(nd):
        for f in os.listdir(nd):
            if not f.startswith("_"):
                res[f[:-5]] = json.load(open(os.path.join(nd, f)))
    return res


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--pages", nargs="*", default=["page_001", "page_016"])
    a = ap.parse_args()
    reg = yaml.safe_load(open(os.path.join(HARNESS, "registry.yaml")))["systems"]
    allp = json.load(open(os.path.join(BENCH, "inventory", "selected_pages.json")))
    for p in allp:
        p["image_300dpi"] = os.path.join(BENCH, "working", "pages_300dpi", f"{p['page_id']}.png")
        p["image_150dpi"] = os.path.join(BENCH, "working", "pages_150dpi", f"{p['page_id']}.png")
        p["page_pdf"] = os.path.join(BENCH, "working", "pages_pdf", f"{p['page_id']}.pdf")
    probe = [p for p in allp if p["page_id"] in a.pages]

    out = {}
    for s in reg:
        if a.systems and s["id"] not in a.systems:
            continue
        py = os.path.join(BENCH, "envs", s["env"], "bin", "python")
        if not os.path.exists(py):
            continue
        tmp = os.path.join(BENCH, "working", "device_probe", s["id"])
        shutil.rmtree(tmp, ignore_errors=True)
        g = run_probe(s, probe, "gpu", os.path.join(tmp, "gpu"))
        c = run_probe(s, probe, "cpu", os.path.join(tmp, "cpu"))
        per = {}
        for pid in sorted(set(g) & set(c)):
            per[pid] = compare(g[pid]["regions"], c[pid]["regions"])
        agree = (sum(v["box_agreement"] for v in per.values()) / len(per)) if per else None
        out[s["id"]] = {"pages": per, "mean_box_agreement": None if agree is None else round(agree, 4),
                        "verdict": ("ok" if agree is not None and agree >= 0.85 else
                                    "MISMATCH" if agree is not None else "no_result")}
        print(f"{s['id']:34s} agreement={out[s['id']]['mean_box_agreement']} -> {out[s['id']]['verdict']}")
    os.makedirs(os.path.join(BENCH, "metrics"), exist_ok=True)
    json.dump(out, open(os.path.join(BENCH, "metrics", "device_agreement.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
