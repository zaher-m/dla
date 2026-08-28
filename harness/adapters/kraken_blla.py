#!/usr/bin/env python3
"""kraken blla adapter: neural baseline and region segmentation.

`blla.segment` returns two things and they are reported as two separate
systems, selected by `emit`:

  lines    one polygon per detected text line, in reading order. This is
           line-level granularity, not regions.
  regions  the model's region head, verbatim — no area filter, no merging.

`text_direction` is a real parameter of the segmenter and is exposed so its
effect can be measured rather than assumed.

Coordinates are checked against the page size on every page: kraken returns
original-image pixels today and a change would corrupt every metric silently.
Per-line polygonisation failures are counted into meta rather than dropped.
"""
import os, sys, logging, hashlib
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset


class _WarnCounter(logging.Handler):
    """Count kraken's per-line polygonisation failures instead of losing them."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.n = 0
    def emit(self, record):
        if "Polygonizer failed" in record.getMessage():
            self.n += 1


def poly_bbox(poly):
    a = np.asarray(poly, dtype=float)
    return [float(a[:, 0].min()), float(a[:, 1].min()),
            float(a[:, 0].max()), float(a[:, 1].max())]


def thin(poly, max_pts=60):
    a = np.asarray(poly, dtype=float)
    if len(a) > max_pts:
        a = a[np.linspace(0, len(a) - 1, max_pts).astype(int)]
    return [[float(x), float(y)] for x, y in a]


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    emit = cfg.get("emit", "lines")                 # "lines" | "regions"
    direction = cfg.get("text_direction", "horizontal-rl")

    counter = _WarnCounter()
    logging.getLogger("kraken").addHandler(counter)

    t = Timer()
    with t.phase("model_load"):
        import torch
        from PIL import Image
        from importlib.metadata import version as _pkgver
        from importlib import resources
        from kraken import blla
        from kraken.lib import vgsl
        # The default model is the `blla.mlmodel` shipped inside the kraken
        # wheel, so the provenance is the release itself; load it explicitly so
        # its path and checksum can be recorded.
        mpath = str(resources.files("kraken").joinpath("blla.mlmodel"))
        model = vgsl.TorchVGSLModel.load_model(mpath)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        im0 = Image.open(job["pages"][0]["input_path"]).convert("RGB")
        blla.segment(im0, text_direction=direction, model=model, device=device)
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    with open(mpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    run.set_model_info(
        model="kraken bundled blla.mlmodel", local_path=mpath, sha256=sha,
        framework=f"kraken {_pkgver('kraken')} (VGSL/CoreML container) on torch 2.11",
        architecture="blla baseline segmenter: U-net style VGSL network, baseline + region heads",
        output=("line baselines and polygonal line environments, in reading order"
                if emit == "lines" else "region head output, verbatim"),
        text_direction=direction, emit=emit, device=device,
        granularity=("line-level: each region is one text line, not a text block"
                     if emit == "lines" else "region-level"),
        provenance=("model file shipped inside the kraken PyPI wheel "
                    "(mittagessen/kraken, Apache-2.0); no separate download"),
        deviations=[
            "coremltools has no linux-aarch64 wheel and was built from the 8.3.0 sdist; "
            "kraken imports it at module scope to read its own .mlmodel container",
        ])

    for page in job["pages"]:
        try:
            cuda_reset()
            counter.n = 0
            with t.phase("preprocess"):
                from PIL import Image
                im = Image.open(page["input_path"]).convert("RGB")
                W, H = im.size
            with t.phase("inference"):
                seg = blla.segment(im, text_direction=direction, model=model, device=device)
                cuda_sync()
            with t.phase("postprocess"):
                n_raw_regions = sum(len(v) for v in (seg.regions or {}).values())
                items = []
                if emit == "lines":
                    for order, ln in enumerate(seg.lines, 1):
                        if not ln.boundary:
                            continue
                        tags = (ln.tags or {}).get("type") or [{}]
                        src = str(tags[0].get("type", "default"))
                        items.append({"source_class": f"line:{src}",
                                      "bbox": poly_bbox(ln.boundary),
                                      "polygon": thin(ln.boundary),
                                      "reading_order": order,
                                      "extra": {"baseline": [[float(x), float(y)]
                                                             for x, y in ln.baseline]}})
                else:
                    for rtype, regs in (seg.regions or {}).items():
                        for r in regs:
                            if not r.boundary:
                                continue
                            items.append({"source_class": f"region:{rtype}",
                                          "bbox": poly_bbox(r.boundary),
                                          "polygon": thin(r.boundary)})
                # Guard the coordinate convention: kraken returns original-image
                # pixels today, and a silent change would corrupt every metric.
                if items:
                    xs = max(i["bbox"][2] for i in items)
                    ys = max(i["bbox"][3] for i in items)
                    if xs > W * 1.05 or ys > H * 1.05:
                        raise RuntimeError(
                            f"kraken returned coordinates outside the page: "
                            f"max ({xs:.0f},{ys:.0f}) vs page ({W},{H})")
                regions = build_regions(run.taxonomy, items)
                raw = {"n_lines": len(seg.lines), "n_raw_regions": n_raw_regions,
                       "region_types": {k: len(v) for k, v in (seg.regions or {}).items()},
                       "text_direction": seg.text_direction,
                       "line_orders": len(seg.line_orders or [])}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"emit": emit, "text_direction": direction,
                           "n_lines": len(seg.lines), "n_raw_regions": n_raw_regions,
                           "polygonizer_warnings": counter.n})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
