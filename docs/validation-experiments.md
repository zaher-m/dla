# Validation experiments

Results that changed the framework, with the method to reproduce them. Each entry
records what was measured, what it showed, and what was done about it — including
where the answer was "nothing, and here is why".

---

## E1. Defect injection: what does the gate fail to catch?

Every check was verified by rendering its findings, which answers *is what this
check reports real*. It does not answer the complementary question: what happens
on a page that is broken and no check fires. A corpus does not come labelled with
its own failures, so the failures were manufactured.

```bash
python -m validation.sensitivity --workspace data/sample120 --corpus data/corpus_flat
python -m validation.sensitivity --workspace data/sample120 --corpus data/corpus_flat --source model
```

Thirteen defect types at graded intensities, each a failure mode seen in real
detector output on this corpus. Injected into two independent baselines: the PSR
reference layout (98 pages) and real detector output (401 system-pages).
Detection is reported as lift over the unmutated baseline, since the baseline
itself trips checks.

### Caught well

| Defect | Lift (PSR baseline) |
|---|---|
| regions dropped, 25% / 50% / all | +63% / +75% / +80% |
| largest region dropped, 1 / 3 | +30% / +50% |
| boxes shrunk 10% / 25% / 40% | +25% / +51% / +78% |
| boxes grown 15% / 35% | +16% / +52% |
| every class changed, to figure / header / table | +78% / +78% / +70% |
| reading order shuffled | +61% |
| all boxes offset 2% / 5% of the page | +44% / +61% |

### Blind spots

| Defect | Lift (PSR) | Lift (model) |
|---|---|---|
| every region duplicated | **+0.0%** | **+0.0%** |
| 30% of regions duplicated | −0.5% | −0.2% |
| 30% of regions reclassified as `table` | **−0.2%** | **+0.1%** |
| 30% reclassified as `figure` / `header` | +4.1% / +4.7% | +7.7% / +7.3% |
| adjacent regions merged vertically, 50% | +3.2% | +4.7% |

Two baselines, measured independently, agree. Note also that
`merge_horizontal` detection *falls* as more of the page is merged (+24.6% at
half, +11.1% at all): once every column is merged the page looks uniformly
single-column and the column checks stop having anything to compare.

### What was done

**Duplication now blocks (C5-04).** Writing a page to its store twice is
corruption under any downstream arrangement, so unlike C5-01 and C5-03 this is
not a policy question. Detection went 0.0% → **+83.7%** at full duplication and
+52.1% at 30%.

**Partial bucket misroute is left open.** Reclassifying a third of a page's
regions as tables is invisible, and the check that should catch it (C6-02) was
cut back to its structural half after it flagged five unmistakable tables:
geometry does not separate a real table from text called a table. See E2.

**Vertical merge is left open, deliberately.** A heading absorbed into the
paragraph below loses structure but loses no content, keeps the order, and
routes to the same store. For a chunked vector index it is arguably not a defect
at all. Recorded rather than fixed, pending a downstream that cares.

---

## E2. Duplicate detections in production output

E1 asked whether duplication *would* be caught. C5-04 then found that it is not
hypothetical.

| System | Exact-position repeats | Pages affected |
|---|---|---|
| `docling.heron` | **901 / 2984 regions (30.2%)** | 112 / 119 |
| `dly.docstructbench_1280` | 88 / 1397 (6.3%) | 51 / 119 |
| `paddleocr.pp_doclayoutv2` | 23 / 1320 (1.7%) | 19 / 120 |
| `ndl.layout` | 2 / 4624 (0.0%) | 2 / 119 |

Docling's raw output carries the cause: its RT-DETR emits two detections per box
with different labels and different confidences — `list_item` 0.9 alongside
`text` 0.427, `list_item` 0.473 alongside `text` 0.856 — and nothing suppresses
the loser. Class-agnostic NMS is missing. Both labels map to the TEXT bucket, so
a downstream index receives every such paragraph twice.

No metric in the benchmark detects this. Duplicates do not reduce recall, they
sit on real text so precision is unaffected, and they do not move a single
geometric measure. It is visible only to a check that asks the question directly.

Left in place rather than suppressed during normalisation: the benchmark's job is
to report what a model emits. Whether the ingestion path should apply
class-agnostic NMS before writing is a separate decision, and it would move every
metric in the report.

---

## E3. Order invariants against the PDF's own geometry

Order checks run on a reading order this pipeline derives, because no system
measured on this corpus emits one. To find out whether they measure the layout or
the derivation, each was run against regions taken from the PDF's own geometry,
where a violation can only be the derivation's fault.

| Check | On a model | On PSR regions | Verdict |
|---|---|---|---|
| C4-08 | 11.2% | 2.0% | attributable to the layout — blocks |
| C4-09 | 9.2% | 16.3% | fires *more* on correct regions — advisory |
| C4-10 | 3.1% | 4.1% | inconclusive — feature, does not block |

The control also found a defect in the pipeline's own reading order:
`derive_order` sorted regions by left edge regardless of text direction, so
side-by-side blocks on an Arabic page came out mirrored, and it sorted rows on
exact top edge, so a point of jitter between two cells discarded the direction
key. C4-10 on the reference layout went from 18.4% to 4.1% once both were fixed.

---

## E5. A language model as a reading-order oracle

Every other order check compares one derivation against another or tests an
order's internal consistency. None reads the words. But for a born-digital page
the words are there, and a correct order joins them into running text while a
wrong one splices unrelated fragments.

A character 5-gram plus a word bigram, trained on the corpus's own text — no
external model, no download, Arabic-capable, CPU only. Trained on **within-line
text only**, so it never sees a line-to-line transition and cannot have memorised
an ordering; scoring then asks a question it was not trained on.

```bash
python -m validation.orderlm --workspace data/sample120 --corpus data/corpus_flat
```

**A methodological trap, worth recording.** The first version scored a 12-character
window either side of each junction and separated a true order from a shuffled
one at 72% — barely useful. The reason was that most n-grams in that window sit
*inside* a line, so the score mostly measured "is this Arabic", identically for
any pairing. Scoring only the n-grams that straddle the boundary, plus the word
bigram across it, is what makes the measurement about order at all.

**Accuracy**, choosing the true order over a shuffled one, 994 corpus pages:

| Pages | vs shuffled | vs reversed |
|---|---|---|
| with prose (prosiness ≥ 0.45) | 88% | 93% |
| mixed (0.15–0.45) | 94% | 92% |
| table-like (< 0.15) | 80% | 77% |
| all | 85% | 83% |

The split is the result. A junction between two numeric table cells carries no
language, so the oracle is strong where there is prose and weak where there is
not — and the median page of this corpus has prosiness 0.13. It detects gross
corruption; on a local swap of two adjacent three-line blocks it is at chance
(47%), because only two junctions change.

**What could not be concluded.** On 261 pages where two column bands are
detected, column-major order scores higher than row-major 82% of the time. That
*cannot* be read as accuracy: if a page is really a table then row-major is the
correct order, and which of the two a page is remains exactly the open question
of E4. The number is reported so nobody re-derives it and mistakes it for a
validation.

**What was done.** C4-11, advisory, gated on `prosiness >= 0.15` and 12 lines.
Nothing at 85% may block a page.

**And an unexpected positive.** On all 42 sampled pages where the check can
speak, the pipeline's derived order scores *better* than the plain top-to-bottom
alternative — median −0.30, best case +0.13 against a margin of 0.5. That is the
first evidence that `assemble.derive_order` is sound which does not come from
comparing it to another derivation. C4-11 firing on nothing is the finding.

---

## E6. A learned score for "is this actually a table"

E1 found that reclassifying a third of a page's regions as tables is invisible
(−0.2% lift), and E4 records two thresholds that failed at the same question.
Both failures were of one quantity with one cut point, and the question is
instant for a human eye. That is the shape of a problem whose decision boundary
no single coordinate has.

**Why not a convolutional model.** The shipped package is 314 MB with four
dependencies and no GPU, and this machine's cuDNN returns bad convolutions on its
architecture in any case. So: fourteen features computed from the PDF's own
geometry, logistic regression, weights in `config/table_model.json` (992 bytes),
inference a dot product in numpy. The deployable property is worth more than the
last few points of accuracy.

**Two independent weak-label sources, never mixed.**

- *A, geometry*: a region spanned by ≥3 horizontal and ≥2 vertical rules is a
  table; one with no rules, long words and a single left edge is not. 57
  regions, 7 documents. High precision, blind to unruled tables by construction.
- *B, consensus*: three or more systems agreeing on a region's class. 677
  regions, 20 documents, 9% positive. Independent of A — it comes from models,
  not from the PDF's strokes.

Trained on B, **held out by document**, not by region: two regions of one page
share a template, a font and a generator, and splitting them across a fold is how
a model scores well on paper and fails on the next report.

| | |
|---|---|
| out-of-fold AUC, grouped by document | **0.929** |
| AUC on the independent geometric set | 1.000 |
| mean score, ruled tables / rule-free prose | 0.86 / 0.00 |

The geometric set is easy, but it is what rules out the model having merely
memorised the consensus.

**Only the negative direction is usable.**

| Operating point | Result |
|---|---|
| score < 0.05, "certainly not a table" | wrong **0.7%** of the time |
| score ≥ 0.98, "a table nothing labelled" | wrong 15% of the time |

So C6-02 consults the model below 0.05 and the reverse use is not shipped, which
is the same conclusion C6-01's history reached by another route.

**A hypothesis that was wrong.** The feature expected to carry the result was
factorisation — a table's lines should resolve into r rows by c columns with
r·c near the line count, where prose gives r×1. It came out with a *negative*
weight. The dominant feature is `left_edges`, the number of distinct positions
lines start at: a table has one per column, prose has one.

**Effect.** C6-02 promoted to blocking on evidence from both directions: it
reports nothing on 424 real (system, page) pairs across four systems, and firing
on nothing is also what an inert check does — so it was measured under injected
defect, where it fires on 62% of pages against an 11% baseline. The E1 blind spot
closes from −0.2% to **+49.6%**, with no change to any system's real block rate.

**Limits.** Trained on Arabic financial and statistical reports. The consensus
labels come from four systems that also produce the output this check judges,
which is partly circular — mitigated by using only the extreme negative end,
where the geometric test set agrees. Retrain for a different corpus.

---

## E4. Negative results

Kept because a documented dead end is cheaper than repeating it.

**Text density does not separate a shaded table from a chart.** `graphic_areas`
clusters filled vector drawings and a table with coloured cell fills is such a
cluster. Density p50 0.27 for tables against 0.28 for charts; the best combined
rule caught 6 of 7 tables but also 11 of 34 charts. A density filter applied in
the PSR deleted genuine charts from 8 benchmark pages and was reverted.

**Ruling lines do not separate a page column from a table column.** Of 13 pages
where two detected bands are merged into one, 1 sits inside a ruled grid.
Gutters do not separate them either — 10 of the 13 have one.

**Coverage thresholds were measuring the wrong unit.** PyMuPDF's line grouping is
one box per glyph on much of this corpus: 44% of body lines are taller than wide,
and one page carries 58 "lines" holding 225 characters. Separately, 51% of what
the coverage family called lost content held no visible character — whitespace
boxes. Rebuilding on reading lines took C1-03 from 17.0% to 2.1% and the best
system's escalation rate from 34.2% to 15.8%, with the count thresholds
*tightened* rather than loosened.

---

## Open, and why

**Partial bucket misroute (E1).** Needs a real answer to "is this region actually
a table", which geometry has now failed at twice. The natural next step is a
learned classifier over the rendered crop, which is also the natural answer to
the page-column/table-column ambiguity in E4. Both are instant judgements for a
human eye and neither has yielded to a threshold.

**Reading order has an unused oracle.** For born-digital pages the text is
available, and a correct order produces fluent language where a wrong one
produces interleaved fragments. A character n-gram model trained on the corpus
itself — no external model, works for Arabic, CPU only — would let a reading
order be scored on its own likelihood rather than against another derivation.
C4 is the largest finding family and currently the largest escalation driver.

**False-accept rate.** Still needs verdicts from the audit stratum. Nothing in
this document is a false-accept rate.
