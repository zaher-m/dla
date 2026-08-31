# Adding a model

Three files, no image rebuild. Each step is testable on its own.

## 1. Environment — `harness/setup/<env>.sh`

```bash
#!/usr/bin/env bash
# What this model is, and any deviation from upstream's published install.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv mymodel)

clone someorg/their-repo
pipi "$PY" "their-package==1.2.3" "opencv-python-headless"

mkdir -p "$MODELS/mymodel"
curl -L -o "$MODELS/mymodel/weights.pth" https://example.org/weights.pth

assert_torch "$PY" mymodel
record_env  "$PY" mymodel
```

`_common.sh` provides `$ENVS`, `$MODELS`, `$REPOS`, `$BENCH` (resolved from `config/dla.yaml`) and:

| Helper | Purpose |
|---|---|
| `mkenv <name>` | venv with `--system-site-packages`, so the container's CUDA torch counts as satisfied |
| `pipi <py> ...` | `pip install` that never lets a resolver replace the inherited torch |
| `clone <org/repo>` | idempotent shallow clone into `assets/repositories` |
| `assert_torch` | warns if the env's torch diverged from the system one |
| `record_env` | writes `pip freeze` and a runtime fingerprint into the workspace |

Deviations from upstream belong in a comment: pins that cannot be installed, kernels that need
patching, undeclared imports.

```bash
make setup-env ENV=mymodel
```

## 2. Adapter — `harness/adapters/<adapter>.py`

The runner executes this with that environment's interpreter. It receives a job JSON, loads the model
once, and emits regions per page.

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

REPO = paths.repo_dir("their-repo")
MODELS = paths.model_dir("mymodel")


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    t = Timer()

    with t.phase("model_load"):
        sys.path.insert(0, REPO)
        from their_package import Detector
        model = Detector(os.path.join(MODELS, cfg["weights"]))
    run.model_load_s = t.pop()["total_s"]

    run.set_model_info(
        weights=cfg["weights"], architecture="...", training_set="...",
        framework="...", device=device, labels=model.class_names,
        provenance="where the checkpoint came from",
        deviations=["anything that differs from upstream's inference path"])

    model.predict(job["pages"][0]["input_path"])          # warm-up, outside timing

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("inference"):
                out = model.predict(page["input_path"])
                cuda_sync()
            with t.phase("postprocess"):
                items = [{"source_class": d.label,
                          "bbox": [d.x1, d.y1, d.x2, d.y2],
                          "confidence": d.score,
                          "polygon": d.polygon}
                         for d in out]
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw={})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
```

Three requirements:

1. Import `core.adapter_base` before constructing a model. It applies the runtime guards.
2. Emit coordinates in the page's 300 dpi pixel space. If the model resizes internally, scale back —
   a scale error is indistinguishable from a bad model.
3. Fill in `deviations` and `provenance`.

## 3. Registry entry — `harness/registry.yaml`

```yaml
  - id: myrepo.mymodel
    repo: their-repo
    display: "Their Model · ResNet-50 (DocLayNet)"
    env: mymodel
    adapter: mymodel_layout
    taxonomy: doclaynet
    input: image_300dpi          # image_300dpi | image_150dpi | page_pdf
    config: {weights: weights.pth, threshold: 0.3}
```

`repo` is the consensus family key: configurations sharing it vote once. Splitting them lets one
project out-vote the field.

## 4. New class names — `harness/core/taxonomy.py`

```python
    "their_taxonomy": {
        "TextBlock": ("text",    "exact", ""),
        "Head":      ("heading", "approximate",
                      "Covers section headings and document titles alike."),
        "Misc":      ("other",   "ambiguous", "Upstream catch-all."),
    },
```

Map from the dataset's own definitions rather than from what the English word suggests — D4LA's
`Footer`, for instance, is documented as the footnote of the document, not a page footer. Mark each
mapping `exact`, `approximate` or `ambiguous`; a class with no entry is reported as `unmapped`.

## 5. Run it

```bash
make list
make shell
  python -m core.runner --systems myrepo.mymodel --limit 2 --force
make analyse FILE=samples/<file>.pdf PROFILE=full
```

Look at the overlays before reading any number. A model that scores well and draws nonsense is a
mapping or scaling bug. Add the id to a file in `config/profiles/` to include it in a profile.
