#!/usr/bin/env python3
"""End-to-end self-test on pages built here, so it needs no corpus.

    python -m validation.selftest

Every case runs the whole chain -- page signals, routing, the structural
reference, line reconstruction, assembly, the checks, the decision -- against a
PDF written in memory.  That is the point: `decide._selftest` covers the
decision logic on synthetic findings and would keep passing if the checks
stopped reading PDFs at all.

The last section is not assertions.  It exercises a case the framework is known
to get wrong and prints what it does, because a test that asserted the wrong
answer would cement it.
"""
import sys

import fitz

from validation import audit, checks, decide, orderlm, verdicts
from validation.api import decide_page

DPI_SCALE = 300 / 72.0
LOREM = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
         "tempor incididunt ut labore et dolore magna aliqua. ") * 6


def _region(x0, y0, x1, y1, cls="text"):
    """A region in the pixel space of a 300 dpi render, as a model emits."""
    s = DPI_SCALE
    return {"bbox": [x0 * s, y0 * s, x1 * s, y1 * s], "class": cls}


def two_column(path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 60, 280, 780), LOREM, fontsize=9)
    page.insert_textbox(fitz.Rect(320, 60, 545, 780), LOREM, fontsize=9)
    doc.save(path)
    doc.close()


def empty(path):
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(path)
    doc.close()


LEFT, RIGHT = _region(50, 60, 280, 780), _region(320, 60, 545, 780)


def run(tmp):
    good, blank = tmp + "/two_column.pdf", tmp + "/empty.pdf"
    two_column(good)
    empty(blank)
    ok = []

    def case(name, pdf, layout, want, task=None, fires=()):
        d = decide_page(pdf, 1, layout)
        got = d["decision"]
        ids = {f["id"] for f in d["findings"]}
        assert got == want, f"{name}: expected {want}, got {got} ({d['reason']})"
        assert task is None or d["task"] == task, \
            f"{name}: expected task {task}, got {d['task']}"
        for c in fires:
            assert c in ids, f"{name}: expected {c} to fire, got {sorted(ids)}"
        ok.append(name)
        return d

    # A layout that boxes what is there is accepted, and nothing fires.
    case("clean layout", good, [LEFT, RIGHT], "accept")

    # Content the model never emitted is the error that matters most, and it is
    # caught whether some of the page is missing or all of it.
    case("one column missing", good, [LEFT], "escalate", "E1", ("C1-04",))
    case("no regions at all", good, [], "escalate", "E1", ("C1-04", "C1-05"))

    # Text routed to the wrong store never reaches the text index, even though
    # every glyph is inside a region.
    case("text boxed as a figure", good,
         [_region(50, 60, 280, 780, "figure"), _region(320, 60, 545, 780, "figure")],
         "escalate", "E1", ("C1-05",))
    case("body marked as running header", good,
         [_region(50, 60, 280, 780, "header"), _region(320, 60, 545, 780, "header")],
         "escalate", "E1", ("C1-05", "C6-06"))

    # A page this pipeline cannot read is parked, not queued: a reviewer has
    # nothing to act on and it is not an error against the model.
    d = case("page with no text layer", blank, [], "defer")
    assert d["task"] is None and d["page_kind"] == "scanned", d

    # Policy drives severity, not the check.  Same page, same findings.
    pol = decide.load_policy()
    a = decide_page(good, 1,
                    [_region(50, 60, 280, 780, "header"), RIGHT],
                    policy={**pol, "discard": "archive"})
    b = decide_page(good, 1,
                    [_region(50, 60, 280, 780, "header"), RIGHT],
                    policy={**pol, "discard": "drop"})
    sev = lambda d, c: next((f["severity"] for f in d["findings"] if f["id"] == c), None)
    assert sev(a, "C6-06") in (None, "MAJOR") and sev(b, "C6-06") in (None, "BLOCK"), \
        f"discard policy did not change severity: {sev(a,'C6-06')} vs {sev(b,'C6-06')}"
    ok.append("discard policy changes severity")

    decide._selftest()
    ok.append("decision logic")

    # The audit stratum has to survive the corpus growing, or a sample gathered
    # over months is several unrelated samples.
    ids = [f"page_{i:04d}" for i in range(400)]
    more = ids + [f"page_{i:04d}" for i in range(400, 4000)]
    pick = lambda xs, sys_: {p for p in xs if audit.sampled(p, sys_, 0.015, 7)}
    assert pick(ids, "a") == pick(more, "a") & set(ids), \
        "audit membership moved when the corpus grew"
    assert pick(ids, "a") != pick(ids, "b"), "audit stratum is not per system"
    n = 40000
    hit = sum(audit.sampled(f"p{i}", "s", 0.02, 3) for i in range(n))
    assert 0.017 < hit / n < 0.023, f"audit rate is {hit/n:.4f}, wanted 0.02"
    ok.append("audit sampling is stable and on rate")

    # A zero-width interval at zero errors is how a quality programme convinces
    # itself of a rate it has not measured.
    lo, hi = verdicts.wilson(0, 50)
    assert lo == 0.0 and 0.02 < hi < 0.15, f"wilson(0,50) = {lo},{hi}"
    e = verdicts.estimate(
        [{"page_id": "a", "system": "s", "frame": "audit", "outcome": "confirm"},
         {"page_id": "b", "system": "s", "frame": "escalation", "outcome": "correct",
          "regions": [{"bbox": [0, 0, 1, 1]}]}])
    assert e["audit_n"] == 1 and e["escalation_n"] == 1, e
    assert e["false_accept_rate"] == 0.0, "escalation labels leaked into the estimate"
    for bad in ({"page_id": "a", "system": "s", "frame": "both", "outcome": "confirm"},
                {"page_id": "a", "system": "s", "frame": "audit", "outcome": "correct"},
                {"page_id": "a", "frame": "audit", "outcome": "confirm"}):
        try:
            verdicts.validate(bad)
        except ValueError:
            continue
        raise AssertionError(f"verdict should have been rejected: {bad}")
    ok.append("verdicts keep their frames apart")

    # The order model must prefer real text to spliced text, and must be trained
    # on within-line text only -- if it ever sees a junction it has memorised an
    # ordering and the score means nothing.
    sents = ["the quarterly report shows a rise in net foreign assets",
             "the balance of payments recorded a surplus this year",
             "domestic liquidity grew by seven percent over the period",
             "interest rates remained unchanged during the second quarter"]
    m = orderlm.train(sents * 40)
    a = orderlm.score(m, ["the quarterly report shows a", "rise in net foreign assets"])
    b = orderlm.score(m, ["the quarterly report shows a", "surplus this year"])
    assert a is not None and b is not None and a > b, \
        f"the model does not prefer the real continuation: {a} vs {b}"
    assert orderlm.prosiness(["1.2 3.4 5.6", "7.8 9.0"]) == 0.0
    assert orderlm.prosiness(sents) == 1.0
    ok.append("order model prefers real continuations")
    return ok, good


def known_gaps(pdf):
    """Printed, never asserted.  Asserting a wrong answer preserves it."""
    d = decide_page(pdf, 1, [_region(50, 60, 545, 780)])
    print("\nknown gap -- one region spanning both columns:")
    print(f"  decided {d['decision']}, findings {[f['id'] for f in d['findings']]}")
    print("  every glyph is inside a region, so coverage is satisfied, but the")
    print("  lines are then read left-right-left-right and the text is scrambled.")
    print("  `page_columns` merges the two bands because a table's columns and a")
    print("  page's columns are both a whitespace corridor with text either side;")
    print("  on the fitted corpus 12 of 14 such pages were tables, so the merge is")
    print("  usually right.  Separating the two needs annotated pages.")


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ok, pdf = run(tmp)
        for name in ok:
            print(f"  ok  {name}")
        print(f"\nvalidation selftest: {len(ok)} cases passed")
        known_gaps(pdf)


if __name__ == "__main__":
    main()
