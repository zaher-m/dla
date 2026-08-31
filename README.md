# DLA Suite

Run and compare 58 document layout analysis configurations across 23 model repositories over your documents and evaluate thier results.

Each page is rendered once at 300 dpi and passed to every selected model in its own isolated
environment. The output is a single self-contained HTML file: every model's regions drawn around
your page, plus a side-by-side view of any model against the page.

```bash
make build     # container image
make setup     # environments + weights
make up        # http://localhost:8080
```

Upload a PDF in the page that opens.

<p align="center">
  <img src="docs/orbit.svg" alt="Sixteen layout models arranged around one page, each panel showing that model's predicted regions" width="100%">
</p>

<p align="center">
  <sub>One page of a real financial report through the <code>balanced</code> profile. The page sits at the
  centre; every model holds a fixed angular position so you can track it across pages.</sub>
</p>

## Why it exists

Layout models are trained on different datasets with different class vocabularies, and published
benchmarks rarely tell you how any of them behave on *your* documents. Running them yourself is
awkward because their dependencies conflict — Docling wants current `transformers`, PDF-Extract-Kit's
vendored LayoutLMv3 wants an old one, detectron2 pins `iopath<0.1.10`, RoDLA needs mmcv 1.x, M2Doc
needs mmcv 2.x. This puts each one in its own virtualenv behind a common adapter interface, so
comparing them is a config change instead of a week of dependency work.

## Requirements

Docker with Compose. An NVIDIA GPU for most models; the `cpu` profile runs without one. About 15 GB
of disk for the `balanced` profile, 45 GB for `full`.

## Usage

```bash
cp .env.example .env                      # port, profile, page cap, HF cache location
make setup SETUP_PROFILE=fast             # smaller install
make analyse FILE=doc.pdf PROFILE=fast    # no UI, any path
make corpus DIR=pdfs/ WORKSPACE=out       # a whole directory into one workspace
make list                                 # what is registered and what is installed
make doctor                               # resolved paths, GPU visibility
make test                                 # end-to-end smoke test
```

Inside the container:

```bash
python -m core.pipeline --input doc.pdf --profile balanced
python -m core.pipeline --workspace data/jobs/<id> --resume
python -m core.runner --systems docling.heron surya.v2_layout --force
```

HTTP API at `/api/docs`: `POST /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/report`,
`GET /api/jobs/{id}/bundle`.

## Models

58 configurations across 23 repositories. `harness/registry.yaml` is the source of truth; a model is
one setup script, one adapter and one registry entry.

| Repository | Configs | What it contributes |
|---|---|---|
| [Docling](https://github.com/docling-project/docling) | 3 | RT-DETR layout, three presets, 11-class DocLayNet vocabulary |
| [Surya](https://github.com/datalab-to/surya) | 3 | Fast detector plus a VLM layout model, with reading order |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 4 | PP-DocLayout family; V3 emits instance masks |
| [MinerU](https://github.com/opendatalab/MinerU) | 2 | Pipeline backend with reading order; VLM backend registered |
| [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) | 4 | DocLayNet / D4LA / DocStructBench, with and without DocSynth300K pretraining |
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) | 2 | DocLayout-YOLO and LayoutLMv3 as packaged upstream |
| [yolo-doclaynet](https://github.com/ppaanngggg/yolo-doclaynet) | 3 | YOLOv8-X / v12-L / YOLO26-L on DocLayNet |
| [Armaggheddon/yolo-document-layout](https://huggingface.co/Armaggheddon/yolo26-document-layout) | 2 | YOLO11-M and YOLO26-M on DocLayNet v1.2 — same data, two architectures |
| [YOLOv11-Document-Layout-Analysis](https://github.com/moured/YOLOv11-Document-Layout-Analysis) | 1 | A third independent DocLayNet YOLO |
| [RapidLayout](https://github.com/RapidAI/RapidLayout) | 5 | ONNX PicoDet on CDLA and PubLayNet, plus [360LayoutAnalysis](https://github.com/360AILAB-NLP/360LayoutAnalysis) YOLOv8n |
| [Layout-Parser](https://github.com/Layout-Parser/layout-parser) | 3 | detectron2 baselines on PubLayNet and PRImA |
| [unilm](https://github.com/microsoft/unilm) | 2 | DiT and LayoutLMv3 on PubLayNet |
| [VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery) | 4 | Vision Grid Transformer; consumes the PDF text layer |
| [M2Doc](https://github.com/johnning2333/M2Doc) | 4 | DINO-4scale with multilingual text-line fusion |
| [RoDLA](https://github.com/yufanchen96/RoDLA) | 1 | InternImage-XL with DCNv3 plus a DINO head |
| [SwinDocSegmenter](https://github.com/ayanban011/SwinDocSegmenter) | 1 | Swin-L + MaskDINO, instance masks |
| [DocSAM](https://github.com/xhli-git/DocSAM) | 5 | Mask2Former with class names as prompts, instance masks |
| [ndl_layout](https://github.com/ndl-lab/ndl_layout) | 2 | Cascade Mask R-CNN trained on Japanese material, instance masks |
| [kraken](https://github.com/mittagessen/kraken) | 3 | Baseline segmentation with an explicit text-direction setting |
| [RF-DETR-DocLayout](https://github.com/roboflow/rf-detr) | 1 | RF-DETR on DocLayNet, ONNX |
| [Dolphin](https://github.com/bytedance/Dolphin) | 1 | Stage-1 layout of a parsing VLM |
| [Eynollah](https://github.com/qurator-spk/eynollah) | 1 | Pixelwise segmentation to PAGE-XML |
| [olmOCR](https://github.com/allenai/olmocr) | 1 | Anchor/layout output of a parsing VLM |

Weights come from each project's own release. Licences differ and several are stricter than this
repository — the Ultralytics-derived YOLO checkpoints are AGPL-3.0, some parsers ship custom terms.
Check the licence of anything you deploy.

## Profiles

A profile is a list of system ids in `config/profiles/`.

| Profile | Models | Use |
|---|---|---|
| `fast` | 5 | Quick look, one model per family |
| `balanced` | 16 | Default. Every competitive family plus masks, reading order and RTL segmentation |
| `full` | 54 | Everything evaluable. Minutes per page |
| `cpu` | 7 | No GPU required |
| `segmentation` | 6 | Only models that emit instance masks |

## How it works

Twelve stages, each a separate process running `python -m core.<stage>` with `DLA_WORKSPACE` set to
the job directory:

`inventory` → `select` → `reference` → `run` → `metrics` → `consensus` → `evidence` → `ensemble` →
`ratings` → `manifest` → `package` → `report`

A job is a directory under `data/jobs/`. Its status is `status.json`, its result is
`reports/index.html`, and its inputs and intermediate outputs sit beside them. No database, no queue
broker: restarting the server loses nothing.

Models never share a process with the harness. The runner writes a job JSON, starts
`assets/envs/<env>/bin/python harness/adapters/<adapter>.py`, and records the outcome. A model that
crashes is recorded as `crashed`, `timed_out` or `env_missing`; the batch continues.

**Metrics.** There is no human ground truth for an arbitrary PDF, so nothing here is mAP. For
born-digital pages the content stream states exactly where glyphs, images, vector art and ruling
lines are, and that is used as a geometric reference: text recall, text precision, spill, line
capture, figure IoU, column bleed. Scanned pages have no text layer, so those columns are empty and
the visual comparison stands on its own.

**Consensus** is deduplicated by repository: three configurations of one project vote once, keyed on
the registry's `repo` field.

More detail in [docs/architecture.md](docs/architecture.md).

## Configuration

`config/dla.yaml`. Every key has an environment override:

| Key | Variable |
|---|---|
| `paths.workspace` | `DLA_WORKSPACE` |
| `run.profile` | `DLA_RUN_PROFILE` |
| `selection.max_pages` | `DLA_SELECTION_MAX_PAGES` |
| `server.port` | `DLA_SERVER_PORT` |
| `report.template` | `DLA_REPORT_TEMPLATE` |

`DLA_CONFIG` points at a different YAML file entirely.

## Adding a model

Three files, no image rebuild:

1. `harness/setup/<env>.sh` — create the virtualenv and fetch weights
2. `harness/adapters/<adapter>.py` — load the model, emit regions per page
3. an entry in `harness/registry.yaml`

Add a mapping table to `harness/core/taxonomy.py` if the class names are new. See
[docs/adding-a-model.md](docs/adding-a-model.md).

## Layout

```
config/            dla.yaml and run profiles
docker/            Dockerfile, compose, entrypoint
docs/              architecture, adding a model
harness/core/      pipeline, ingestion, reference, metrics, report generation
harness/adapters/  one per model family
harness/setup/     one per environment
harness/report/    report shell and viewer
harness/app/       FastAPI service and UI
samples/           gitignored: put input PDFs here
assets/            gitignored: virtualenvs, weights, cloned repositories
data/              gitignored: uploads and job workspaces
```

## Licence

Apache-2.0. Model weights are not covered by it; see the table above.
