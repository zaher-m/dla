#!/usr/bin/env python3
"""One page in, one decision out.  The entry point other software calls.

    from validation.api import decide_page
    d = decide_page("report.pdf", 4, regions)
    d["decision"]        # accept | escalate | defer | reject
    d["task"]            # the reviewer task, when escalated
    d["findings"]        # what fired, in sentences, with region indices

    python -m validation.api --pdf report.pdf --page 4 --layout regions.json

`regions` is a list of {"bbox": [x0, y0, x1, y1], "class": "...", optionally
"reading_order"} in the pixel space of the page rendered at `dpi` -- the same
space the layout model was given.  Getting that wrong is the one way to misuse
this function, so the render size is derived here from the same `get_pixmap`
call the pipeline uses rather than being passed in.

`validation.stage` is the batch form of this, over a workspace.  This form
holds no state, touches no workspace and needs no benchmark, which is what
makes it usable from a service, a notebook or another repository.

Text direction is a property of a document, not a page: a page of numeric
tables inside an Arabic report is 8% Arabic characters and still reads right to
left.  So `decide_page` reads the whole file to establish it.  Deciding many
pages of one document, call `decide_document`, which pays that cost once.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from validation import (assemble, checks, decide as decidemod,  # noqa: E402
                        psr as psrmod, router, signals)
from validation.stage import SCHEMA  # noqa: E402


def _open(pdf):
    return pdf if isinstance(pdf, fitz.Document) else fitz.open(pdf)


def decide_document(pdf, layouts, policy=None, dpi=psrmod.DPI):
    """Decide several pages of one document.

    `layouts` maps 1-based page number -> list of regions.  Document context --
    text direction, and nothing else -- is established once over every page of
    the file, including the ones not being decided.
    """
    doc, close = _open(pdf), not isinstance(pdf, fitz.Document)
    try:
        t = router.load_thresholds()
        sigs = [signals.page_signals(doc, i) for i in range(doc.page_count)]
        ctx = router.document_context(sigs, t)
        p = policy or decidemod.load_policy()
        score = decidemod.scorer(p)
        out = {}
        for pno, regions in layouts.items():
            route = router.route(sigs[pno - 1], t, ctx)
            if route["psr_trust"] == "unusable":
                # No reference, so no check can run.  The route decides alone,
                # and the record says that rather than implying a clean page.
                res = {"findings": [], "skipped": [], "inapplicable": [],
                       "families": []}
                psr = None
            else:
                psr = psrmod.page_psr(doc, pno - 1, dpi)
                stream = assemble.assemble(regions, psr,
                                           direction=route["direction"])
                res = checks.run(regions, psr, stream, route)
            d = decidemod.decide(res, route, policy=p, score=score)
            out[pno] = {"schema": SCHEMA, "page": pno,
                        "page_size": [psr["width"], psr["height"]] if psr else None,
                        **d}
        return out
    finally:
        if close:
            doc.close()


def decide_page(pdf, page, regions, policy=None, dpi=psrmod.DPI):
    """Decide one page.  `page` is 1-based."""
    return decide_document(pdf, {page: regions}, policy, dpi)[page]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, required=True, help="1-based")
    ap.add_argument("--layout", required=True,
                    help="JSON: a list of regions, or {'regions': [...]}")
    ap.add_argument("--dpi", type=int, default=psrmod.DPI)
    ap.add_argument("--json", action="store_true", help="print the whole record")
    a = ap.parse_args()
    with open(a.layout, encoding="utf8") as f:
        got = json.load(f)
    regions = got["regions"] if isinstance(got, dict) else got
    d = decide_page(a.pdf, a.page, regions, dpi=a.dpi)
    if a.json:
        print(json.dumps(d, indent=1, ensure_ascii=False))
        return
    print(f"{d['decision'].upper()}"
          + (f" -> {d['task']}" if d["task"] else "")
          + f"  ({d['page_kind']}, psr {d['psr_trust']}, {d['direction']})")
    print(f"  {d['reason']}")
    for f_ in d["findings"]:
        print(f"  {f_['severity']:6s} {f_['id']}  {f_['message']}")


if __name__ == "__main__":
    main()
