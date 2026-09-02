#!/usr/bin/env python3
"""Profile a corpus: extract C0 signals for every page and summarise them.

This is how thresholds get fitted.  Running it over a real corpus turns the
guessed numbers in `config/checks.yaml` into percentiles of an observed
distribution, and shows immediately when a check would fire on most of the
corpus -- which is the usual way a plausible threshold turns out to be wrong.

    python -m validation.profile --corpus samples/CBE --out data/validation
"""
import argparse, json, os, sys, time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import signals, router  # noqa: E402

PCTS = (1, 5, 25, 50, 75, 95, 99)
# Reported as distributions rather than means: every one of these is a ratio
# whose tail is the interesting part.
NUMERIC = ("n_chars", "n_lines", "n_spans", "n_images", "n_drawings",
           "glyph_area_frac", "image_area_frac", "big_image_area_frac",
           "lines_in_big_image_frac", "median_font_pt", "median_line_h_pt",
           "median_chars_per_span", "rtl_cp_frac", "pua_cp_frac",
           "undecoded_cp_frac", "lines_outside_crop_frac", "invisible_frac",
           "white_frac", "type3_frac", "no_tounicode_frac", "n_fonts")


def collect(corpus, max_pages=None, recursive=False):
    if recursive:
        pdfs = sorted(os.path.relpath(os.path.join(r, f), corpus)
                      for r, _, fs in os.walk(corpus) for f in fs
                      if f.lower().endswith(".pdf"))
    else:
        pdfs = sorted(f for f in os.listdir(corpus) if f.lower().endswith(".pdf"))
    t = router.load_thresholds()
    rows = []
    for name in pdfs:
        t0 = time.time()
        try:
            pages = signals.doc_signals(os.path.join(corpus, name), max_pages)
        except Exception as e:
            print(f"  !! {name[:58]}: {type(e).__name__}: {e}")
            continue
        ctx = router.document_context(pages, t)
        for s in pages:
            r = router.route(s, t, ctx)
            s = dict(s)
            s["doc"] = name
            s["page_kind"] = r["page_kind"]
            s["psr_trust"] = r["psr_trust"]
            s["direction"] = r["direction"]
            s["c0"] = [x["id"] for x in r["findings"]]
            s["doc_direction"] = ctx["direction"]
            rows.append(s)
        kinds = Counter(x["page_kind"] for x in rows[-len(pages):])
        print(f"  {len(pages):4d}p {time.time()-t0:5.1f}s  "
              f"{dict(kinds)}  {name[:52]}")
    return rows


def summarise(rows):
    out = {"n_pages": len(rows), "n_docs": len({r['doc'] for r in rows})}
    out["page_kind"] = dict(Counter(r["page_kind"] for r in rows))
    out["psr_trust"] = dict(Counter(r["psr_trust"] for r in rows))
    out["direction"] = dict(Counter(r["direction"] for r in rows))
    out["c0_fire_rate"] = {k: round(v / len(rows), 4) for k, v in
                           Counter(c for r in rows for c in r["c0"]).most_common()}
    out["pages_with_any_c0"] = round(
        sum(1 for r in rows if r["c0"]) / len(rows), 4)
    # Which documents a check implicates is what makes it actionable.
    affected = {}
    for r in rows:
        for c in r["c0"]:
            affected.setdefault(c, Counter())[r["doc"]] += 1
    out["c0_docs"] = {k: dict(v.most_common(6)) for k, v in affected.items()}
    dist = {}
    for k in NUMERIC:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if v:
            dist[k] = {f"p{p}": round(float(np.percentile(v, p)), 4) for p in PCTS}
    out["distributions"] = dist
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="data/validation")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="cap pages per document (smoke tests)")
    ap.add_argument("--recursive", action="store_true")
    a = ap.parse_args()

    print(f"profiling {a.corpus}")
    rows = collect(a.corpus, a.max_pages, a.recursive)
    if not rows:
        sys.exit("no pages profiled")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "corpus_signals.json"), "w") as f:
        json.dump(rows, f)
    summary = summarise(rows)
    with open(os.path.join(a.out, "corpus_profile.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n{summary['n_pages']} pages, {summary['n_docs']} documents")
    print("page_kind ", summary["page_kind"])
    print("psr_trust ", summary["psr_trust"])
    print("direction ", summary["direction"])
    print(f"pages firing any C0 check: {summary['pages_with_any_c0']:.1%}")
    for k, v in summary["c0_fire_rate"].items():
        print(f"  {k}  {v:.2%}")
    print(f"\nwrote {a.out}/corpus_profile.json")


if __name__ == "__main__":
    main()
