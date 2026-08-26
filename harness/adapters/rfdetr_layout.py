#!/usr/bin/env python3
"""RF-DETR-DocLayout adapter — neka-nat/rfdetr-doclayout, DocLayNet-trained.

NOTE: this file must NOT be named `rfdetr_doclayout.py`.  Adapters run with their
own directory as sys.path[0], so that name would shadow the installed package and
`import rfdetr_doclayout.rfdetr` would resolve to this file instead.

A small MIT-licensed repo that fine-tunes Roboflow's RF-DETR on DocLayNet and
publishes a single ONNX checkpoint (`neka-nat/rfdetr-doclaynet-onnx`).  Inference
uses the package's own `RfDetrDoclayout` class, so preprocessing (576x576 resize,
ImageNet normalisation) and post-processing (sigmoid, top-k, cxcywh -> xyxy,
confidence filter) are the author's code, not a reimplementation.

Two facts about the *published* checkpoint, established by inspecting the graph
rather than the README:

  * It has exactly two outputs, `pred_boxes (1,300,4)` and `pred_logits (1,300,11)`.
    There is **no mask head** — `predict()` returns `masks=None`.  The repository's
    description of returning masks applies to the segmentation variant of RF-DETR,
    not to this release.
  * The input is fixed at **576x576**.  A 300 dpi A4 page is 2481x3508, so the
    page is downsampled ~4.3x before the model sees it; that is a real constraint
    on small-text recall, not a tuning choice, and it is recorded here.

Class ids are 0-based DocLayNet order (Caption, Footnote, Formula, List-item,
Page-footer, Page-header, Picture, Section-header, Table, Text, Title) — the same
order the DocLayout-YOLO DocLayNet checkpoints embed, which corroborates it.
"""
import os, sys
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions

DOCLAYNET = ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
             "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
REPO, FILE = "neka-nat/rfdetr-doclaynet-onnx", "checkpoint_best_total.onnx"


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    conf = float(cfg.get("conf_thres", 0.5))
    maxb = int(cfg.get("max_boxes", 300))

    t = Timer()
    with t.phase("model_load"):
        from huggingface_hub import hf_hub_download
        from rfdetr_doclayout.rfdetr import RfDetrDoclayout
        wpath = hf_hub_download(REPO, FILE)
        model = RfDetrDoclayout(onnx_model_path=wpath)
        nout = len(model.ort_session.get_outputs())
        model.predict(job["pages"][0]["input_path"], confidence_threshold=conf,
                      max_number_boxes=maxb)                      # warm-up
    run.model_load_s = t.pop()["total_s"]

    run.set_model_info(
        repo_id=REPO, weights=FILE, local_path=wpath, labels=DOCLAYNET,
        framework="onnxruntime (rfdetr-doclayout 0.1.0)",
        architecture="RF-DETR (LW-DETR head), fine-tuned on DocLayNet",
        device="cpu", input_size=[model.input_width, model.input_height],
        conf_thres=conf, max_boxes=maxb,
        masks_available=(nout == 3),
        note=("published ONNX exposes pred_boxes + pred_logits only; no mask head "
              "despite the repository's description" if nout != 3 else None))

    for page in job["pages"]:
        try:
            with t.phase("inference"):
                scores, labels, boxes, masks = model.predict(
                    page["input_path"], confidence_threshold=conf, max_number_boxes=maxb)
            with t.phase("postprocess"):
                items = []
                for sc, lb, bx in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
                    idx = int(lb)
                    items.append({
                        "source_class": DOCLAYNET[idx] if 0 <= idx < len(DOCLAYNET) else str(idx),
                        "bbox": [float(v) for v in bx], "confidence": float(sc)})
                regions = build_regions(run.taxonomy, items)
                raw = {"scores": scores.tolist(), "labels": [int(v) for v in labels.tolist()],
                       "boxes": boxes.tolist(), "names": DOCLAYNET,
                       "has_masks": masks is not None}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"conf": conf, "input_size": model.input_width})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
