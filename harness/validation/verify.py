#!/usr/bin/env python3
"""Phase 0 exit test: compare two layouts for one page and say what differs.

    python -m validation.verify --pdf doc.pdf --page 4 --layout out.json
    python -m validation.verify --workspace benchmark --page-id page_004 \
                                --system docling.heron

With no `--layout` the prediction is compared against the layout the PDF itself
implies, which is what makes this runnable today: no OCR engine, no annotated
pages, no model beyond the one being checked.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import assemble, compare, psr_layout, router, signals  # noqa: E402


def _psr_for(pdf, page, direction=None):
    import fitz
    from validation import psr as psrmod
    doc = fitz.open(pdf)
    try:
        sigs = [signals.page_signals(doc, i) for i in range(doc.page_count)]
        ctx = router.document_context(sigs)
        r = router.route(sigs[page - 1], ctx=ctx)
        return psrmod.page_psr(doc, page - 1), r, (direction or r["direction"])
    finally:
        doc.close()


def run(psr_page, regions, direction, route_info=None):
    ref_regions, meta = psr_layout.build(psr_page)
    ref = assemble.assemble(ref_regions, psr_page, direction)
    pred = assemble.assemble(regions, psr_page, direction)
    c = compare.compare(pred, ref, psr=psr_page)
    v = compare.verdict(c)
    return {"route": route_info, "reference": meta, "compare": c, "verdict": v}


def line(res):
    c, v = res["compare"], res["verdict"]
    return (f"grouping P={c['grouping_precision']} R={c['grouping_recall']}, "
            f"orphans {c['orphan_rate']:.1%}, cross-merge {c.get('cross_merge_rate', 0):.1%}, "
            f"order tau={c['order_tau']}, buckets {c['bucket_confusion'] or 'clean'} "
            f"-> {'TIER 1' if v['tier1'] else 'pass'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--layout", help="normalized layout JSON for that page")
    ap.add_argument("--workspace", help="use a benchmark workspace instead of a PDF")
    ap.add_argument("--page-id")
    ap.add_argument("--system")
    ap.add_argument("--direction", choices=("ltr", "rtl"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.workspace:
        ref = json.load(open(os.path.join(
            a.workspace, "inventory", "pdf_structural_reference.json")))
        psr_page = ref[a.page_id]
        d = json.load(open(os.path.join(a.workspace, "normalized_outputs",
                                        a.system, a.page_id + ".json")))
        res = run(psr_page, d["regions"], a.direction or "rtl")
    else:
        psr_page, route_info, direction = _psr_for(a.pdf, a.page, a.direction)
        if route_info["psr_trust"] == "unusable":
            print(f"page {a.page}: {route_info['page_kind']}, PSR unusable -- "
                  f"{route_info['findings'][0]['message'] if route_info['findings'] else ''}")
            print("cannot verify: escalate")
            return
        regions = json.load(open(a.layout))["regions"] if a.layout else []
        res = run(psr_page, regions, direction, route_info)

    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(line(res))
    for why in res["verdict"]["reasons"]:
        print("  -", why)


if __name__ == "__main__":
    main()
