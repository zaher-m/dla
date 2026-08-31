# Architecture

The design follows from one constraint: the models being compared cannot share a Python environment.
Docling wants current `transformers`, PDF-Extract-Kit's vendored LayoutLMv3 wants an old one,
detectron2 pins `iopath<0.1.10`, RoDLA needs mmcv 1.x while M2Doc needs mmcv 2.x, kraken needs a
coremltools with no wheel for many platforms. Resolving that into one environment either fails or
silently downgrades something.

So the harness never imports a model.

```
             ┌──────────────────────────────────────────────┐
  PDF ──────►│  core.pipeline          (harness env)        │
             │    inventory → select → reference            │
             │    → run → metrics … → package → report      │
             └───────────────┬──────────────────────────────┘
                             │ one subprocess per model
             ┌───────────────▼──────────────────────────────┐
             │  assets/envs/<env>/bin/python                 │
             │    harness/adapters/<adapter>.py --job x.json │
             │      loads the model, writes normalised JSON  │
             └──────────────────────────────────────────────┘
```

The runner starts processes and records what happened. Adding an incompatible repository costs one
setup script, not a dependency negotiation.

## Three trees

| Tree | Contents | Lifetime |
|---|---|---|
| `assets/` | virtualenvs, weights, cloned repositories | shared by every job, expensive to build, gitignored |
| workspace | renders, raw and normalised outputs, metrics, report | one per job under `data/jobs/`, cheap to delete |
| repository | code, config, samples | tracked |

`harness/core/paths.py` is the only module that resolves a path. It reads `config/dla.yaml`, applies
`DLA_*` overrides, and exposes absolute paths. A workspace is a shape — `inventory/ working/
raw_outputs/ normalized_outputs/ metrics/ reports/ logs/` — and every stage takes `DLA_WORKSPACE`, so
the same code runs a single upload or a whole corpus with no branching.

## PDF structural reference

There is no human ground truth for an arbitrary PDF. But a born-digital PDF is not an image: its
content stream states where every glyph, image, vector path and ruling line sits.

`core.reference` extracts that per page — text lines split into body, in-graphic and in-table; image
and vector extents; horizontal and vertical rules; grid candidates; column bands and gutters — scaled
into the same 300 dpi pixel space the models see.

This is not a layout annotation. It answers *did this predicted text region contain text?* and *does
this region straddle two columns?* exactly. It cannot say whether a region should have been
`heading` or `title`; consensus and visual review carry that.

One trap worth knowing before editing that file: PyMuPDF renders a page with its `/Rotate` applied
but reports geometry in unrotated space. Scaling unrotated coordinates by the rotated rectangle puts
the reference 90° out of register on landscape pages. `page.rotation_matrix` is applied before
scaling and is the identity when `/Rotate` is 0.

## Normalised output

One JSON file per page per model (`core/schema.py`):

```json
{"page_id": "page_001", "width": 2481, "height": 3508,
 "regions": [{"id": 1, "class": "text", "source_class": "Text",
              "bbox": [x1, y1, x2, y2], "confidence": 0.98,
              "polygon": [[x, y]], "mapping_confidence": "exact",
              "reading_order": 3}],
 "timing": {"preprocess": 0, "inference": 0, "postprocess": 0},
 "resources": {"cuda_peak_alloc_mb": 0, "peak_rss_mb": 0}}
```

`class` is canonical, `source_class` is always kept so a questionable mapping can be audited later.
`mapping_confidence` is `exact`, `approximate`, `ambiguous` or `unmapped`; an unknown class is
reported as unmapped rather than bucketed into `other`. Timing uses the same four-phase timer for
every model.

## Consensus

With no ground truth, cross-model agreement is the available signal about class labels — but only if
the voters are independent. Three Docling presets are not three opinions, so `core.consensus` keys
voting on the registry's `repo` field: one repository, one vote.

Per model that yields consensus recall, class agreement, the share of regions corroborated by other
repositories, and a solo rate. High recall with a high solo rate means regions nobody else sees.

## Statistics

A small page set cannot separate two good models by a couple of points, so comparisons use a paired
per-page two-sided sign test rather than a table ordering, and the report marks the group no test
separates from the leader.

Single-class models are excluded from the choice of leader: both headline metrics ask what share of
reference body-text lines falls inside a predicted text-or-table region, and a model with one output
class maximises that by construction. They are detected from the data (`class_diversity <= 1` on
every page), still ranked and shown, just not used as the baseline.

## Report

`core.package_report` builds one JSON bundle: page images as downscaled JPEG data URIs, region
geometry as compact numeric arrays, metrics, consensus, taxonomy mapping. One image per page serves
every model's overlay, drawn as SVG in the browser, rather than one image per (page × model).

`core.build_report` composes `head.html` + `style.css` + a body template + the bundle + `app.js` into
a single file. `--template` selects the body; `viewer.html` ships. `app.js` reads
`window.NARRATIVE` for optional prose blocks and checks each host element exists, so a template can
omit any section.

Orbit positions are fixed and append-only, persisted in `reports/orbit_slot_order.json`. A model
added later takes the next free outer slot; existing models do not move.

## Failure handling

| Where | Result |
|---|---|
| one page in a model | traceback stored in that run's manifest, remaining pages continue |
| one model in a job | `crashed` / `timed_out` / `env_missing` with a log path, batch continues |
| an optional stage | job continues without that analysis — a scanned PDF has no reference, so no geometric metrics |
| a required stage | job stops, `status.json` carries the exit code and the last log lines |

Nothing is substituted or estimated. A model that could not run is reported as not having run.

## Runtime guards

`core/torch_env.py` is imported by `core/adapter_base` before any model is constructed, and every
adapter imports `adapter_base` first.

Some GPU and driver combinations return wrong convolution results from cuDNN for these backbones
without raising — the output is degenerate in a way that looks like a weak model. The guard disables
the cuDNN backend; `DLA_ALLOW_CUDNN=1` turns it back on. The guard state is recorded in every run
manifest. `core/validate_device.py` compares forced-CPU against GPU output box-for-box on probe pages
and flags disagreement.
