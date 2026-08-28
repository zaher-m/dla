#!/usr/bin/env python3
"""Eynollah adapter — pixelwise segmentation -> PAGE-XML regions.

Eynollah is the only benchmarked system that performs true *pixelwise*
page segmentation, so its PAGE-XML polygons (not just boxes) are preserved.
Run with the flags that matter for this corpus: right-to-left reading order,
full layout (headers / marginalia / drop capitals), and table detection.
"""
import os, sys, glob, subprocess, shutil, time, re
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, resources

NS = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def parse_pagexml(path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"pc": root.tag.split("}")[0].strip("{")}
    out, order = [], {}
    ro = root.find(".//pc:ReadingOrder", ns)
    if ro is not None:
        for i, rr in enumerate(ro.iter()):
            rid = rr.get("regionRef")
            if rid:
                order[rid] = int(rr.get("index", i))
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if not tag.endswith("Region"):
            continue
        coords = el.find("pc:Coords", ns)
        if coords is None or not coords.get("points"):
            continue
        pts = [[float(v) for v in p.split(",")] for p in coords.get("points").split()]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        src = el.get("type") or tag              # e.g. TextRegion type="paragraph"
        out.append({"source_class": src, "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "polygon": pts, "confidence": None,
                    "reading_order": order.get(el.get("id")),
                    "extra": {"region_tag": tag, "id": el.get("id")}})
    return out


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    bench = job["bench"]
    models = os.path.join(bench, "models", "eynollah", "models_eynollah")
    workdir = os.path.join(bench, "working", "eynollah")
    os.makedirs(workdir, exist_ok=True)

    flags = []
    if cfg.get("right_to_left", True):            flags.append("--right2left")
    if cfg.get("full_layout", True):              flags.append("--full-layout")
    if cfg.get("tables", True):                   flags.append("--tables")
    if cfg.get("reading_order_machine_based", True): flags.append("--reading_order_machine_based")
    if cfg.get("curved_line"):                    flags.append("--curved-line")
    if cfg.get("allow_scaling"):                  flags.append("--allow_scaling")

    run.set_model_info(models_dir=models, source="Zenodo 21381102 models_inference_layout_v0_9_1",
                       framework="onnxruntime (CPU: no aarch64 onnxruntime-gpu wheel)",
                       device="cpu", flags=flags, output="PAGE-XML polygons + reading order",
                       segmentation="pixelwise semantic segmentation")
    run.model_load_s = None      # eynollah re-loads models per CLI invocation

    t = Timer()
    for page in job["pages"]:
        pid = page["page_id"]
        indir = os.path.join(workdir, pid, "in"); outdir = os.path.join(workdir, pid, "out")
        os.makedirs(indir, exist_ok=True); os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(indir, f"{pid}.png")
        if not os.path.exists(dst):
            shutil.copy(page["input_path"], dst)
        driver = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_eynollah_driver.py")
        cmd = [sys.executable, driver, "-m", models, "layout", "-di", indir,
               "-o", outdir, "--overwrite"] + flags
        try:
            with t.phase("inference"):
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                   env=dict(os.environ, OMP_NUM_THREADS="8"))
            with t.phase("postprocess"):
                xmls = glob.glob(os.path.join(outdir, "*.xml"))
                if not xmls:
                    raise RuntimeError(f"no PAGE-XML produced; rc={p.returncode}\n"
                                       f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
                items = parse_pagexml(xmls[0])
                regions = build_regions(run.taxonomy, items)
                raw = {"pagexml": open(xmls[0]).read(), "cmd": cmd, "returncode": p.returncode}
            run.emit(page, regions, t.pop(), raw=raw, meta={"flags": flags})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
