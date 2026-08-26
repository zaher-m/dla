#!/usr/bin/env python3
"""Dolphin v2 adapter — stage-1 layout + reading order only.

Dolphin is a two-stage VLM parser.  Only stage 1 ("Parse the reading order of
this document.") is invoked, so element-level OCR/table/formula parsing never
runs and cannot influence the layout score or the timings.
"""
import os, sys, json
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset

REPO = "ByteDance/Dolphin-v2"
PROMPT = "Parse the reading order of this document."


def main():
    job = parse_job()
    run = AdapterRun(job)
    repo_dir = os.path.join(job["bench"], "repositories", "Dolphin")
    sys.path.insert(0, repo_dir)
    t = Timer()
    with t.phase("model_load"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from utils.utils import parse_layout_string, process_coordinates, resize_img
        from qwen_vl_utils import process_vision_info
        local = snapshot_download(REPO)
        processor = AutoProcessor.from_pretrained(local)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(local)
        model.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        model = model.bfloat16() if dev == "cuda" else model.float()
        processor.tokenizer.padding_side = "left"
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(repo_id=REPO, local_path=local, framework="transformers (Qwen2.5-VL)",
                       device=dev, precision="bfloat16" if dev == "cuda" else "float32",
                       prompt=PROMPT, stage="layout only (stage 1 of 2)",
                       reading_order=True,
                       n_params=sum(p.numel() for p in model.parameters()))

    from PIL import Image

    def chat(img):
        msgs = [{"role": "user", "content": [{"type": "image", "image": resize_img(img)},
                                             {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(dev)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=4096, do_sample=False,
                                 num_beams=1, use_cache=True)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    import torch
    chat(Image.open(job["pages"][0]["input_path"]).convert("RGB"))   # warm-up
    cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = Image.open(page["input_path"]).convert("RGB")
            with t.phase("inference"):
                out = chat(img)
                cuda_sync()
            with t.phase("postprocess"):
                parsed = parse_layout_string(out)
                items = []
                for order, (bbox, label, tags) in enumerate(parsed):
                    x1, y1, x2, y2 = process_coordinates(bbox, img)
                    items.append({"source_class": label, "bbox": [x1, y1, x2, y2],
                                  "confidence": None, "reading_order": order,
                                  "extra": {"tags": tags} if tags else None})
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw={"model_output": out},
                     meta={"prompt": PROMPT})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
