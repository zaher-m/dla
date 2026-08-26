#!/usr/bin/env python3
"""A local-transformers Backend for Surya 2's VLM inference manager.

Why this exists
---------------
Surya 2 replaced its encoder-decoder layout model with a VLM and speaks to it
only over an OpenAI-compatible HTTP endpoint.  Its two shipped backends both
spawn a server:

  * `vllm`     — `docker run vllm/vllm-openai:v0.20.1`, i.e. Docker-in-Docker,
                 which is not available inside the benchmark container; and
  * `llamacpp` — needs a GGUF conversion that upstream does not publish.

Pointing `SURYA_INFERENCE_URL` at a vLLM server started inside the benchmark
container does not work either: the checkpoint is `Qwen3_5ForConditionalGeneration`
(`model_type: qwen3_5`, hybrid linear/full attention + MTP) and the container's
vLLM 0.15.1 does not carry that architecture — `ModelRegistry.get_supported_archs()`
returns False for it.

The surya venv's transformers 5.16.1 *does* implement `qwen3_5` natively, so this
backend runs the same checkpoint locally and satisfies the same `Backend`
contract.  Only the token-generation transport differs: prompt selection,
`scale_to_fit` preprocessing, message construction, the retry-on-repetition
policy, sequence confidence, and every downstream parser/relabel/blank-filter
step remain surya's own code, untouched.

Documented deviation
--------------------
vLLM applies `LAYOUT_JSON_SCHEMA` as guided decoding (`SURYA_GUIDED_LAYOUT=True`
by default).  transformers has no equivalent constrained-decoding hook, so this
path generates unguided.  `SURYA_GUIDED_LAYOUT` is a first-class surya setting
and unguided is a supported mode, but it is a real difference: malformed JSON
that the server would have made impossible is instead caught by surya's tolerant
`parse_layout`, and any page that still fails to parse is recorded as a failure
rather than silently dropped.  The per-page parse-failure count is reported.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import torch

from surya.inference.backends.base import Backend, ServerHandle
from surya.inference.prompts import PROMPT_MAPPING
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.inference.util import detect_repeat_token, scale_to_fit


class TransformersBackend(Backend):
    """Runs the surya-2 VLM in-process with transformers instead of a server."""

    name = "transformers"

    # Mirrors chat_completions_batch()'s client-side defaults.
    TEMPERATURE = 0.0
    TOP_P = 0.1
    MAX_RETRIES = 3

    def __init__(self, checkpoint: str, dtype: str = "bfloat16",
                 device: str = "cuda", attn_implementation: Optional[str] = None,
                 max_tokens_default: int = 2048):
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.device = device
        self.attn_implementation = attn_implementation
        self.max_tokens_default = max_tokens_default
        self.model = None
        self.processor = None
        self.stats = {"generated": 0, "retries": 0, "repeat_retries": 0, "errors": 0}

    # ---- Backend contract -------------------------------------------------
    def capacity(self) -> int:
        return 1  # one local model, one page at a time

    def start(self) -> ServerHandle:
        if self.model is not None:
            return self._handle
        from transformers import AutoProcessor, AutoModelForImageTextToText

        proc_kwargs = {}
        # Same pixel budget the vllm backend passes as --mm-processor-kwargs.
        for k, v in (("min_pixels", 3136), ("max_pixels", 6291456)):
            proc_kwargs[k] = v
        try:
            self.processor = AutoProcessor.from_pretrained(self.checkpoint, **proc_kwargs)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(self.checkpoint)

        kwargs = {"dtype": getattr(torch, self.dtype)}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(self.checkpoint, **kwargs)
        self.model.to(self.device).eval()

        self._handle = ServerHandle(base_url="local://transformers",
                                    model_name=self.checkpoint, spawned_by_us=False)
        return self._handle

    def stop(self) -> None:
        self.model = None
        self.processor = None

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        if self.model is None:
            self.start()
        return [self._one(item) for item in batch]

    # ---- generation -------------------------------------------------------
    def _one(self, item: BatchInputItem) -> BatchOutputItem:
        prompt = item.prompt or PROMPT_MAPPING[item.prompt_type]
        image = scale_to_fit(item.image)
        max_tokens = item.max_tokens or self.max_tokens_default
        temp = item.temperature if item.temperature is not None else self.TEMPERATURE
        top_p = item.top_p if item.top_p is not None else self.TOP_P

        raw, ntok, mean_p, err = self._generate_once(image, prompt, max_tokens, temp, top_p)
        retries = 0
        while self._should_retry(raw, err, retries):
            self.stats["retries"] += 1
            if not err:
                self.stats["repeat_retries"] += 1
            retries += 1
            # Same escalation ladder as surya's chat_completions_batch.
            retry_temp = min(temp + 0.2 * retries, 0.8)
            retry_top_p = 0.95 if not err else top_p
            raw, ntok, mean_p, err = self._generate_once(
                image, prompt, max_tokens, retry_temp, retry_top_p)
        self.stats["generated"] += 1
        if err:
            self.stats["errors"] += 1
        return BatchOutputItem(raw=raw, token_count=ntok, error=err,
                               mean_token_prob=mean_p, logprobs=None,
                               metadata=dict(item.metadata, retries=retries))

    def _should_retry(self, raw: str, err: bool, retries: int) -> bool:
        if retries >= self.MAX_RETRIES:
            return False
        if err:
            return True
        return bool(detect_repeat_token(raw)) or (
            len(raw) > 50 and bool(detect_repeat_token(raw, cut_from_end=50)))

    def _generate_once(self, image, prompt, max_tokens, temperature, top_p):
        try:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            gen_kwargs = dict(max_new_tokens=max_tokens,
                              return_dict_in_generate=True, output_scores=True)
            # The checkpoint's config carries text_config.eos_token_id=248044, which
            # is outside its own 65,425-token vocabulary; its generation_config and
            # tokenizer both say <|im_end|> (id 2). Pin both explicitly so a future
            # transformers release preferring the model config cannot make every page
            # run to max_new_tokens.
            tok = self.processor.tokenizer
            if tok.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tok.eos_token_id
            if tok.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = tok.pad_token_id
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=float(temperature),
                                  top_p=float(top_p))
            else:
                gen_kwargs.update(do_sample=False)
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)

            in_len = inputs["input_ids"].shape[1]
            seq = out.sequences[0][in_len:]
            raw = self.processor.decode(seq, skip_special_tokens=True)
            mean_p = self._mean_token_prob(out)
            return raw, int(seq.shape[0]), mean_p, False
        except Exception as e:  # mirrors the client's warn-and-mark-error path
            print(f"[surya-transformers] generation error: {e!r}", flush=True)
            return "", 0, None, True

    def _mean_token_prob(self, out) -> Optional[float]:
        """exp(logprob) averaged over generated tokens — surya uses this as the
        page's LayoutBox.confidence, so it must mean the same thing as the
        OpenAI `logprobs` path."""
        try:
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True)
            lp = scores[0]
            lp = lp[torch.isfinite(lp)]
            if lp.numel() == 0:
                return None
            return float(torch.exp(lp.float()).mean().item())
        except Exception:
            return None


def build_manager(checkpoint: str, **kw):
    """A SuryaInferenceManager whose backend is this one, for LayoutPredictor."""
    from surya.inference import SuryaInferenceManager
    mgr = SuryaInferenceManager.__new__(SuryaInferenceManager)
    mgr.method = "transformers"
    mgr.backend = TransformersBackend(checkpoint, **kw)
    return mgr
