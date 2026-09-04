# validation

Per-page verification for document layout output: accept, escalate to a human, or defer. No annotated data, no second model, no GPU.

## Why

Layout quality is easy to judge by eye and hard to judge at scale. A pipeline ingesting a corpus into a vector index, an object store and a table database makes one decision per page, thousands of times over, and a bad decision does not look like a lower score. It looks like a paragraph absent from the index, a table whose rows arrive interleaved, or a figure written into the text store. Those failures are silent: nothing raises, nothing logs, and a corpus-level recall average hides all of them.

Verifying that with a held-out set means human labels, which do not exist for an arbitrary PDF and cost weeks to create. This package takes the other route.

For a born-digital PDF the characters are already in the file, so a layout stage contributes exactly three things: which glyph lines group together, in what order those groups are read, and which store each group goes to. The PDF's own content stream states where every glyph, image, vector drawing and ruling line sits, which makes all three checkable — exactly, per page, before an OCR engine has even been chosen.

That reference is not ground truth. It is a deterministic reading of the file, wrong in known ways (see [Known gaps](#known-gaps)). What it is instead is a reference that exists for every page at no cost, needs no annotation, and never disagrees with itself. Every check here is a question about a specific way downstream data gets corrupted, not a similarity score between two layouts.

## Flow

```
signals.py    what the page's content stream says: glyphs, ink, direction, encoding
router.py     -> page_kind, psr_trust, direction.  Derived, never passed in
reference.py  the PDF Structural Reference: lines, blocks, graphics, grids, columns
lines.py      reading lines rebuilt from the reference's boxes
assemble.py   layout + reference -> the blocks, order and buckets a consumer receives
checks.py     34 checks -> findings, each naming regions and saying what is wrong
decide.py     deterministic veto, then a score -> accept | escalate | defer | reject
stage.py      the same over a workspace, writing decisions, queue, backlog, summary
```

`compare.py`, `psr_layout.py`, `document.py`, `tables.py` and `buckets.py` support these.
`evaluate.py` reports how often each check fires over a corpus, and `inspect.py` renders findings as
contact sheets — every check here was verified by looking at its findings as images, and of the
eighteen examined that way, four were right and fourteen were wrong.

## Decision

```
psr_trust unusable          -> defer     (or escalate; policy.unusable)
any blocking check fired    -> escalate  with that finding
a blocking check could not run -> escalate  "incomplete evidence"
risk >= t_reject            -> reject    (both thresholds null by default)
risk >= t_escalate          -> escalate
otherwise                   -> accept
```

The veto is deterministic and label-free. The score is a hand-weighted scorecard whose thresholds are
`null`, so today it orders the queue and decides nothing — an invented weight sum is not grounds for
holding back a page. When labels exist a calibrated model replaces `Scorecard` behind the same
interface and nothing else changes.

Severity is resolved from policy at decision time, not from the check. A check says what the defect
is; `config/checks.yaml` under `policy:` says what the pipeline does about it. Whether DISCARD
regions are archived or dropped moves C6-05 and C6-06 between MAJOR and BLOCK, which on a 120-page
sample is 29 pages and 2-5 points of escalation.

## Escalation

One page, one task. Findings are grouped first, so four checks firing on one defect produce one job.

| Task | Raised by | Reviewer does |
|---|---|---|
| E1 | C1 coverage | draw a box round the missed content, or confirm it is not content |
| E2 | C2, C3, C5 | split or merge a region, or confirm |
| E3 | C4 order | reorder, or confirm |
| E4 | C6 buckets | pick the right store from four buttons |
| E5 | committee disagreement | choose A, B, or neither — not built, needs a committee |
| E6 | C7, C8, or unverifiable | annotate the page |
| E7 | random audit | confirm or correct — the unbiased error estimate, not built |

## Checks

| Check | Sev | |
|---|---|---|
| `C1-01` | BLOCK | Body glyph lines that no region covers |
| `C1-02` | BLOCK | Glyph ink area no region covers — catches one big miss C1-01 dilutes |
| `C1-03` | BLOCK | Vertically contiguous run of orphans: a missed block, not stray glyphs |
| `C1-04` | BLOCK | Orphans inside a real text column, as opposed to page margins |
| `C1-05` | BLOCK | No text region at all on a page that plainly has text |
| `C1-06` | MAJOR | Uncovered lines in the footnote band at the foot of the page |
| `C1-07` | MAJOR | A real graphic with no MEDIA region on it |
| `C2-01` | MAJOR | Region boundaries cutting through glyph lines |
| `C2-06` | ADV | A text region holding no glyphs at all |
| `C3-01` | ADV | A text region straddling a whitespace corridor between columns |
| `C3-02` | BLOCK | A text region overlapping two column bands |
| `C3-05` | MAJOR | One column detected, the other silently ignored |
| `C4-02` | BLOCK | Within one column, blocks are not read top to bottom |
| `C4-03` | BLOCK | Reading order ping-ponging between column bands |
| `C4-07` | MAJOR | The reading order disagrees with the order the page implies |
| `C4-08` | BLOCK | A table or figure is read interleaved with the text around it |
| `C4-09` | ADV | The stream jumps back up the page without changing column |
| `C4-10` | MAJOR | On an RTL page, blocks sharing a row are read left to right |
| `C5-01` | MAJOR | Two same-bucket regions overlapping substantially |
| `C5-03` | MAJOR | A glyph line covered by two text regions is stored twice |
| `C6-01` | BLOCK | A ruled table with no table region on it |
| `C6-02` | MAJOR | A table region with nothing about the page to support it |
| `C6-03` | ADV | A figure region that is really text: content sent to object storage |
| `C6-05` | BLOCK | Body content classified as running furniture and dropped |
| `C6-06` | BLOCK | Wholesale deletion: too much of the page routed to DISCARD |
| `C7-01` | BLOCK | A region outside the page: a coordinate-space or rescaling bug |
| `C7-03` | ADV | A ruling line detected as a text region |
| `C7-04` | BLOCK | One region swallowing a multi-column page |
| `C7-05` | ADV | Gross under-detection against the page's own ink extent |
| `C8-01` | MAJOR | A running header the document almost always has is missing |
| `C8-02` | ADV | The column count differs from the rest of the document |
| `C8-03` | ADV | The region count is a gross outlier against the document's spread |
| `C8-05` | MAJOR | The class vocabulary collapsed on this page alone |
| `C8-06` | ADV | The body font size differs sharply from the rest of the document |

Severities and every threshold live in [`config/checks.yaml`](../../config/checks.yaml), annotated
with the corpus each was fitted on and a verification status per check. `DLA_CHECKS_CONFIG` points
at a different file.

## Tests

```bash
python -m validation.selftest      # or: make validate-selftest
```

Eight cases, each running the whole chain against a PDF written in memory: a clean layout is
accepted; a missing column, a missing page, text boxed as a figure and body marked as a running
header all escalate as E1; a page with no text layer defers; the discard policy moves a severity.
The standalone image runs this at build time, so a change that breaks the chain — or that gives the
package a dependency on `core` — fails the build.

## Known gaps

**Escalation is a review cost, not an error rate.** The thresholds bound how often a check fires.
How often an accepted page is nonetheless wrong is a false-accept rate and cannot be measured
without annotated pages. Nothing here should be read as a confidence figure.

**A page column and a table column look the same.** `page_columns` merges bands whose sides share
rows, because a financial table's columns otherwise register as page columns and every column check
fires on every table. On the fitted corpus 12 of 14 such pages were tables, so the merge is usually
right — but a genuine two-column page is then merged too, and one region spanning both columns is
accepted although its lines are read left-right-left-right. `validation.selftest` prints this case.
Ruling lines do not separate the two (1 of 13 collapses sits in a ruled grid) and neither do
gutters. Separating them needs annotated pages.

**A shaded table registers as a graphic.** `core.reference` clusters filled vector drawings, and a
table with coloured cell fills is a cluster of filled drawings, so its text moves out of the body
lines. 6-8% of glyph lines on 9-34% of pages. Density does not separate it from a chart: p50 0.27
against 0.28, and the best combined rule caught 6 of 7 tables but also 11 of 34 charts. Compensated
locally in `tables.py` and `figure_areas`, documented rather than patched.

**Reading order is derived, not given.** No system measured on this corpus emits one, so
`assemble.derive_order` supplies it and the order checks are partly measuring that. Each was run
against regions taken from the PDF's own geometry, where a violation can only be the derivation:
C4-08 separates (11.2% against 2.0%) and blocks; C4-09 fires more often on correct regions and is
advisory; C4-10 is inconclusive and does not block.

**Scanned pages are out of scope.** They are deferred, listed in `deferred.json`, and set
`policy.unusable: escalate` to queue them for a human instead. Note that most deferred pages on the
fitted corpus were not scans but born-digital files whose glyphs were converted to vector outlines.

## Not built

The committee (E5) needs measured error decorrelation between systems, which needs labels. So does
the random audit (E7), the calibrated risk model, and any statement of a false-accept rate. Those
are the next phase, and they start when annotated pages exist.
