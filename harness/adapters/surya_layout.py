#!/usr/bin/env python3
"""Surya 2 layout adapter — VLM layout + reading order.

Surya 2's `LayoutPredictor` issues only the layout prompt (no OCR decoding) and
gets its tokens from a `SuryaInferenceManager` backend.  Upstream ships two
backends, both of which spawn a server this container cannot host (see
`_surya_tf_backend.py` for the full reasoning and the one documented deviation:
no guided decoding).  Config `backend: transformers` runs the same checkpoint
locally through transformers; `backend: vllm`/`llamacpp` still take surya's own
path, including `SURYA_INFERENCE_URL` for an external server.

Everything after token generation — `parse_layout`, `LAYOUT_PRED_RELABEL`,
`denorm_bbox`, the blank-region filter, reading order — is surya's own code.
"""
import os, sys
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    backend = cfg.get("backend", "transformers")
    os.environ.setdefault("SURYA_INFERENCE_KEEP_ALIVE", "true")

    t = Timer()
    with t.phase("model_load"):
        from surya.settings import settings
        if backend == "transformers":
            # transformers has no constrained decoding; unguided is a supported
            # surya mode but it is a deviation from the vllm default, recorded
            # in model info and in the run report.
            settings.SURYA_GUIDED_LAYOUT = False
        if cfg.get("max_tokens_layout"):
            settings.SURYA_MAX_TOKENS_LAYOUT = int(cfg["max_tokens_layout"])

        from surya.layout import LayoutPredictor
        ckpt = cfg.get("checkpoint", settings.SURYA_MODEL_CHECKPOINT)

        if backend == "transformers":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from _surya_tf_backend import build_manager
            mgr = build_manager(ckpt, dtype=cfg.get("dtype", "bfloat16"),
                                device=cfg.get("device", "cuda"),
                                attn_implementation=cfg.get("attn_implementation"),
                                max_tokens_default=settings.SURYA_MAX_TOKENS_LAYOUT)
            mgr.backend.start()
            pred = LayoutPredictor(manager=mgr)
        else:
            os.environ["SURYA_INFERENCE_BACKEND"] = backend
            pred = LayoutPredictor()
            mgr = None
            pred([Image.open(job["pages"][0]["input_path"]).convert("RGB")])  # spin up server
    run.model_load_s = t.pop()["total_s"]

    run.set_model_info(
        checkpoint=ckpt, backend=backend, framework="surya-ocr 0.22.1 (VLM)",
        architecture="Qwen3_5ForConditionalGeneration (hybrid linear/full attention)",
        transport=("local transformers (container vLLM 0.15.1 lacks qwen3_5; "
                   "upstream backend needs Docker-in-Docker)"
                   if backend == "transformers" else backend),
        reading_order=True,
        guided_layout=settings.SURYA_GUIDED_LAYOUT,
        guided_layout_note=("DEVIATION: vllm applies LAYOUT_JSON_SCHEMA as guided "
                            "decoding by default; transformers cannot, so output is "
                            "parsed by surya's tolerant parse_layout instead."
                            if backend == "transformers" else None),
        labels=__import__("surya.inference.prompts", fromlist=["LAYOUT_LABEL_SET"]).LAYOUT_LABEL_SET,
        max_tokens_layout=settings.SURYA_MAX_TOKENS_LAYOUT,
        image_dpi=settings.IMAGE_DPI, bbox_scale=settings.BBOX_SCALE,
        dtype=cfg.get("dtype", "bfloat16"))

    parse_failures = 0
    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = Image.open(page["input_path"]).convert("RGB")
            with t.phase("inference"):
                res = pred([img])[0]
                cuda_sync()
            with t.phase("postprocess"):
                if res.error:
                    parse_failures += 1
                items = []
                for b in res.bboxes:
                    poly = [[float(x), float(y)] for x, y in b.polygon] if b.polygon else None
                    items.append({"source_class": b.label,
                                  "bbox": [float(v) for v in b.bbox],
                                  "confidence": (float(b.confidence)
                                                 if b.confidence is not None else None),
                                  "polygon": poly,
                                  "reading_order": b.position,
                                  "extra": {"raw_label": b.raw_label, "count": b.count}})
                regions = build_regions(run.taxonomy, items)
                raw = {"bboxes": [b.model_dump() for b in res.bboxes],
                       "image_bbox": res.image_bbox, "error": res.error,
                       "model_output": res.raw}
            if res.error:
                raise RuntimeError(
                    "layout output did not parse (unguided decoding); "
                    f"raw[:300]={(res.raw or '')[:300]!r}")
            run.emit(page, regions, t.pop(), raw=raw, meta={"backend": backend})
        except Exception as e:
            t.pop(); run.fail(page, e)

    if mgr is not None:
        run.set_model_info(generation_stats=mgr.backend.stats,
                           parse_failures=parse_failures)
    run.finish()


if __name__ == "__main__":
    main()
