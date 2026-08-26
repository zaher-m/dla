#!/usr/bin/env python3
"""PDF-Extract-Kit layout adapter — DocLayout-YOLO (DocStructBench checkpoint).

Uses the repo's own class map and default inference parameters
(configs/layout_detection.yaml).  The checkpoint is the officially published
DocStructBench model, reused from the local Hugging Face cache when present.
"""
import os, sys
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset

ID_TO_NAMES = {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption',
               5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula',
               9: 'formula_caption'}
REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    t = Timer()
    with t.phase("model_load"):
        from huggingface_hub import hf_hub_download
        import torch
        wpath = hf_hub_download(REPO, FILE)
        try:
            from doclayout_yolo import YOLOv10 as M
            backend = "doclayout_yolo.YOLOv10"
        except Exception:
            from ultralytics import YOLO as M
            backend = "ultralytics.YOLO"
        model = M(wpath)
        dev = 0 if torch.cuda.is_available() else "cpu"
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(repo_id=REPO, weights=FILE, local_path=wpath, labels=ID_TO_NAMES,
                       framework=backend)

    model.predict(job["pages"][0]["input_path"], imgsz=cfg["img_size"],
                  conf=cfg["conf_thres"], iou=cfg["iou_thres"], verbose=False, device=dev)
    cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("inference"):
                r = model.predict(page["input_path"], imgsz=cfg["img_size"], conf=cfg["conf_thres"],
                                  iou=cfg["iou_thres"], verbose=False, device=dev)[0]
                cuda_sync()
            with t.phase("postprocess"):
                b = r.boxes
                xyxy = b.xyxy.cpu().tolist(); cls = b.cls.cpu().tolist(); conf = b.conf.cpu().tolist()
                raw = {"boxes": xyxy, "classes": cls, "scores": conf,
                       "id_to_names": ID_TO_NAMES, "orig_shape": list(r.orig_shape)}
                items = [{"source_class": ID_TO_NAMES[int(c)], "bbox": bb, "confidence": s}
                         for bb, c, s in zip(xyxy, cls, conf)]
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw=raw)
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
