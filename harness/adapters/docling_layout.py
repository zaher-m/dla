#!/usr/bin/env python3
"""Docling layout adapter — layout stage in isolation.

Uses Docling's own object-detection engine (the exact component its PDF
pipeline calls for layout) so that no document-conversion/OCR work is included
in the timings or the scored output.
"""
import os, sys, json
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset

PRESETS = {
    "layout_heron_default": ("docling-project/docling-layout-heron", "RT-DETR ResNet-50"),
    "layout_heron_101":     ("docling-project/docling-layout-heron-101", "RT-DETR ResNet-101"),
    "layout_egret_medium":  ("docling-project/docling-layout-egret-medium", "D-FINE medium"),
    "layout_egret_large":   ("docling-project/docling-layout-egret-large", "D-FINE large"),
    "layout_egret_xlarge":  ("docling-project/docling-layout-egret-xlarge", "D-FINE xlarge"),
}


def main():
    job = parse_job()
    run = AdapterRun(job)
    preset = run.cfg.get("preset", "layout_heron_default")
    repo_id, arch = PRESETS[preset]

    t = Timer()
    with t.phase("model_load"):
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
        from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec
        from docling.datamodel.object_detection_engine_options import (
            TransformersObjectDetectionEngineOptions)
        from docling.models.inference_engines.object_detection import (
            create_object_detection_engine, ObjectDetectionEngineInput)

        import torch
        acc = AcceleratorOptions(
            device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU)
        engine = create_object_detection_engine(
            options=TransformersObjectDetectionEngineOptions(),
            model_spec=ObjectDetectionModelSpec(name=preset, repo_id=repo_id, revision="main"),
            accelerator_options=acc)
        engine.initialize()
        labels = engine.get_label_mapping()
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(repo_id=repo_id, preset=preset, architecture=arch,
                       revision="main", labels=labels, framework="pytorch/transformers")

    # warm-up (excluded from timings)
    warm = Image.open(job["pages"][0]["input_path"]).convert("RGB")
    engine.predict_batch([ObjectDetectionEngineInput(image=warm, metadata={})])
    cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = Image.open(page["input_path"]).convert("RGB")
                inp = [ObjectDetectionEngineInput(image=img, metadata={"page_no": page["page"]})]
            with t.phase("inference"):
                out = engine.predict_batch(inp)[0]
                cuda_sync()
            with t.phase("postprocess"):
                raw = {"label_ids": out.label_ids, "scores": out.scores,
                       "bboxes": out.bboxes, "label_mapping": labels,
                       "image_size": list(img.size)}
                items = [{"source_class": labels.get(int(lid), str(lid)),
                          "bbox": bb, "confidence": sc}
                         for lid, sc, bb in zip(out.label_ids, out.scores, out.bboxes)]
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"preset": preset, "repo_id": repo_id})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
