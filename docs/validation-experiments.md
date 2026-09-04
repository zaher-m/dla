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
