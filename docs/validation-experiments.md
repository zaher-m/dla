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

## E7. Repairing the reference: figure, or the document's own content?

E4 records that text density does not separate a shaded table from a chart, and
E6 built a learned table score. Neither addresses the defect. Rendering every
graphic cluster on the corpus that holds four or more glyph lines -- 118 of
them, adjudicated from the image before any score was joined -- shows the
question was posed wrongly:

| What the cluster really is | | Holding |
|---|---|---|
| a genuine figure: chart, pie, logo | 51 | 1787 lines |
| a table: shaded, banded, or a header row | 50 | |
| prose: a heading, a source note, a framed paragraph, a form | 16 | 2297 lines together |
| blank | 1 | |

**56% are not figures**, and a third of those are not tables either. That is why
both earlier attempts failed: the table score puts prose at "not a table"
exactly where it puts a chart (median 0.001 against 0.003), so it cannot see two
thirds of the damage. The question is figure against content.

Everything in those 66 clusters is outside `body_text_lines` -- **8.0% of all
glyph lines, on 17% of pages** -- so the coverage family, five of whose checks
block, cannot see any of it.

**What separates them is what the cluster is drawn from**, which the content
stream states and the reference now records as six numbers per cluster
(`graphic_shape`, purely additive). A chart is built from curves and overlapping
fills and labelled in type smaller than the body; a shaded table or a tinted
paragraph is built from plain rectangles and holds body-sized text in rows.

Eight features, logistic regression, `config/graphic_model.json` at 595 bytes.
No feature selection: the eight are fixed, so nothing leaks into the estimate.

| | |
|---|---|
| out-of-fold AUC, grouped by document (13 documents) | **0.977** |
| against a system-consensus label that never saw the adjudication | 0.987 |
| single best feature alone, same protocol | 0.857 |

The last row is the point. E4's conclusion was that no threshold does this, and
that survives: the best single coordinate reaches 0.857 out of fold where eight
together reach 0.977.

| Floor | Content recovered | Figures miscalled | Lines recovered | Lines wrongly moved |
|---|---|---|---|---|
| 0.50 | 62/66 | 6/51 | 1937 | 245 |
| **0.80** | **59/66** | **2/51** | **1875** | **67** |
| 0.90 | 35/66 | 0/51 | 1153 | 0 |

0.80 ships: 28 recovered lines per line wrongly moved.

**Effect on real output**, 424 (system, page) pairs, every change verified by
rendering the findings that moved:

| | HEAD | repaired | |
|---|---|---|---|
| C1-07 a graphic with no media region | 34 | 6 | shaded tables charged for not being images |
| C2-01 boundaries cutting lines | 37 | 21 | rate over a denominator that now holds the whole page |
| C4-08 a unit read interleaved | 51 | 40 | 13 removed, 2 added, all rendered |
| C1-05 no text region | 4 | 3 | |

Escalation falls 3.3 points for the best system and 5.8 for the worst, and every
removal that was rendered was a false positive.

**Two defects this surfaced in checks that were passing.**

`C1-05` fired whenever a page had no text-bucket region, counted over every line
on the page. A page that is one large table, correctly boxed as a table, has no
text region and does not need one -- the check was charging systems for being
right, unseen while a shaded table's cells sat outside the body. Counting only
*uncovered* lines instead gives the defect back: when every region is relabelled
`figure` the lines are still covered, and detection of that fell from 94% to
31% of injected pages. It now counts lines that no text **or table** region
covers, which is right in both directions.

`C4-08` took its structural units from `figure_areas`, which it shares with
C1-07, where excluding tables is correct and here is not. Extending it to every
content cluster was tried and rendered: a banded table is clustered one column
at a time, and a column's cells are *meant* to arrive one per row -- 36 findings,
all on tables read correctly. Unioning the columns back into whole tables was
tried too and is worse, reaching across both panels of a two-panel page and
taking baseline firing from 1 page to 7. What ships is the clusters that stand
alone, by the row-sharing test `page_columns` already uses.

**What it did not do.** Escalation did not rise anywhere. The systems on this
corpus do detect the shaded tables, so the recovered lines are already covered
and no new miss appears. The blindness was real but unexploited here, which the
injection sweep confirms rather than contradicts: `drop_largest` detection rises
3 points and `drop_region` 1, because the coverage family can now see content it
previously could not. Two rows fall -- `merge_horizontal` at full intensity from
+6.6% to +0.1% -- and that lift came from the same table-column mechanism that
produced the 13 false positives. It is not worth buying back that way.

**Not applied to the shared reference.** `core.metrics` scores text recall
against `body_text_lines`, so repairing the reference itself moves published
benchmark numbers. Measured, it would add 15.2% to the recall denominator on one
workspace and 3.2% on the other, and move every system's line coverage **up** by
0.3-1.4 points without changing the order. That is a product decision, not a
validation one, so validation repairs its own view and the reference is left
saying exactly what the file says.

**Limits.** 118 hand labels over 13 documents of Arabic financial and
statistical reports; retrain elsewhere. Two errors at 0.80 are clusters holding
both a table and a chart, where no single label is right -- a clustering
granularity problem, not a scoring one.

---

## E4. Negative results

Kept because a documented dead end is cheaper than repeating it.

**Text density does not separate a shaded table from a chart.** `graphic_areas`
clusters filled vector drawings and a table with coloured cell fills is such a
cluster. Density p50 0.27 for tables against 0.28 for charts; the best combined
rule caught 6 of 7 tables but also 11 of 34 charts. A density filter applied in
the PSR deleted genuine charts from 8 benchmark pages and was reverted. Still
true: density reaches 0.693 as a single feature in E7, against 0.977 for the
eight together. What was wrong was the question -- a third of the clusters at
issue are prose, not tables.

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

**A page column and a table column still look the same.** E7 answers the other
half of E4 and not this one: it says whether a filled cluster holds content, not
whether two bands of text are one table or two columns. The clusters carry no
information on a page whose columns are plain text.

**Mixed clusters.** `cluster_boxes` glues a table to the chart beneath it, and
one label is then wrong whichever way it goes. Both of E7's errors at the
shipped floor are this. It is a granularity problem in the clustering, not a
scoring one, and splitting a cluster is a different experiment.

**Partial bucket misroute (E1).** `class_to_figure` and `class_to_header` at 30%
still show +3-4%. C6-03, C6-05 and C6-06 all detect them; none blocks under
`policy.discard: archive`. That is a policy question, like discard itself, and
not a detection gap.

**False-accept rate.** Still needs verdicts from the audit stratum. Nothing in
this document is a false-accept rate.
