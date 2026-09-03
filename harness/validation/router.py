#!/usr/bin/env python3
"""Page routing and PSR-trust verdict -- check family C0.

This runs before every other check and decides two things:

  page_kind   born_digital / hybrid / scanned
  psr_trust   full / degraded / unusable

Nothing downstream may assume the PDF is a usable reference until this says so.
The distinction that matters most is between a check that *passed* and a check
that *could not run*: a page with no text layer has not passed the coverage
checks, it has escaped them, and recording that as a pass is the most direct
route to a silent failure in the whole framework.  Callers get an explicit
`available` set and are expected to mark everything else `n/a`.
"""
import os

import yaml

DEFAULTS = {
    "min_chars": 1,
    "sparse_max_lines": 5,
    "sparse_min_ink": 0.15,
    "invisible_frac": 0.30,
    "ocr_lines_in_image": 0.60,
    "ocr_image_area": 0.50,
    "undecodable_frac": 0.20,
    "frag_max_chars_per_span": 1,
    "frag_min_spans": 200,
    "outline_vector_ink": 0.20,
    "outline_max_glyph": 0.05,
    "rtl_frac": 0.20,
    "doc_rtl_aggregate": 0.05,
    "doc_rtl_max_page": 0.50,
    "doc_min_chars": 200,
    "clipped_frac": 0.05,
}

# Which check families each page kind can support.  A scanned page has no
# content stream to compare against, so nothing geometric applies.
AVAILABLE = {
    "born_digital": {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"},
    "hybrid":       {"C2", "C4", "C5", "C7", "C8"},
    "scanned":      {"C5", "C7", "C8"},
}


def load_thresholds(path=None):
    path = path or os.environ.get("DLA_CHECKS_CONFIG")
    if not path:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, "config", "checks.yaml")
    t = dict(DEFAULTS)
    try:
        with open(path, encoding="utf8") as f:
            t.update((yaml.safe_load(f) or {}).get("c0") or {})
    except FileNotFoundError:
        pass
    return t


def document_context(sigs, t=None):
    """Document-wide facts that a single page cannot establish.

    Text direction is the one that matters.  Deciding it per page misroutes
    every numeric table in an Arabic report -- one statistical bulletin in the
    reference corpus is 8% Arabic characters per page and entirely right to
    left.  A document with almost no text at all cannot vote either way, and
    says `unknown` rather than guessing `ltr`, because asserting a direction for
    a scanned Arabic document would invert its whole reading order.
    """
    t = t or load_thresholds()
    total = sum(s["n_chars"] for s in sigs)
    rtl = sum(s["n_chars"] * s["rtl_cp_frac"] for s in sigs)
    mx = max((s["rtl_cp_frac"] for s in sigs), default=0.0)
    agg = (rtl / total) if total else 0.0
    if total < t["doc_min_chars"]:
        direction = "unknown"
    elif agg > t["doc_rtl_aggregate"] or mx > t["doc_rtl_max_page"]:
        direction = "rtl"
    else:
        direction = "ltr"
    return {"direction": direction, "rtl_aggregate": round(agg, 4),
            "rtl_max_page": round(mx, 4), "n_chars": total, "n_pages": len(sigs)}


def route(sig, t=None, ctx=None):
    """Signals for one page -> routing decision, trust verdict and C0 findings.

    `ctx` comes from `document_context` and supplies the direction.  Without it
    the page falls back to its own character mix, which is only safe for a
    single-page document.
    """
    t = t or load_thresholds()
    f = []

    def fire(cid, msg, **kw):
        f.append(dict(id=cid, message=msg, **kw))

    no_text = sig["n_chars"] < t["min_chars"]
    if no_text:
        fire("C0-01", "no text layer: the page carries no glyphs")

    ink = max(sig["image_area_frac"], sig["glyph_area_frac"])
    sparse = (not no_text and sig["n_lines"] < t["sparse_max_lines"]
              and ink > t["sparse_min_ink"])
    if sparse:
        fire("C0-02", f"text layer too thin to be the page: {sig['n_lines']} lines "
                      f"over {ink:.0%} ink coverage")

    hidden = sig.get("invisible_frac", 0.0) + sig.get("white_frac", 0.0)
    if hidden > t["invisible_frac"]:
        fire("C0-03", f"{hidden:.0%} of characters are painted invisibly or in the "
                      f"background colour and no detector can see them")

    ocr_layer = (sig["lines_in_big_image_frac"] > t["ocr_lines_in_image"]
                 and sig["big_image_area_frac"] > t["ocr_image_area"])
    if ocr_layer:
        fire("C0-04", f"{sig['lines_in_big_image_frac']:.0%} of lines sit inside a "
                      f"full-page image: this text is another model's OCR output")

    # Text converted to curves.  There is no missing text layer and no image, so
    # every other C0 test passes it as born-digital, and the reference then
    # describes a page of 3 lines while the render is dense with words.  Four
    # such pages in a 120-page sample carried 96 characters each -- a header and
    # a footer -- against 229-654 vector drawings.
    outlined = (sig.get("vector_area_frac", 0.0) > t["outline_vector_ink"]
                and sig["glyph_area_frac"] < t["outline_max_glyph"])
    if outlined:
        fire("C0-10", f"the page's text is drawn as vector outlines: "
                      f"{sig['n_drawings']} drawings cover "
                      f"{sig['vector_area_frac']:.0%} of the page against "
                      f"{sig['glyph_area_frac']:.0%} of glyphs")

    if sig["rotation"] % 90 != 0:
        fire("C0-05", f"page rotation {sig['rotation']} is not a right angle")

    pua = sig.get("pua_cp_frac", 0.0)
    und = sig.get("undecoded_cp_frac", 0.0)
    bad = pua + und
    if bad > t["undecodable_frac"]:
        # These two mean different things.  A private-use codepoint is a font
        # with a custom encoding and no usable ToUnicode map: the geometry is
        # perfect and the characters are unrecoverable, so the page needs OCR
        # despite being born-digital.  A replacement codepoint is a decode that
        # failed outright.
        why = ("a custom font encoding with no character map"
               if pua >= und else "characters that failed to decode")
        fire("C0-06", f"{bad:.0%} of characters are unusable -- {why}; "
                      f"geometry is fine, text is not")

    frag = (sig["median_chars_per_span"] <= t["frag_max_chars_per_span"]
            and sig["n_spans"] > t["frag_min_spans"])
    if frag:
        fire("C0-07", f"{sig['n_spans']} spans averaging "
                      f"{sig['median_chars_per_span']} character: line grouping "
                      f"cannot be trusted")

    if sig["lines_outside_crop_frac"] > t["clipped_frac"]:
        fire("C0-09", f"{sig['lines_outside_crop_frac']:.0%} of lines fall outside "
                      f"the crop box and never render")

    if no_text:
        kind = "scanned"
    elif sparse or ocr_layer or outlined:
        kind = "hybrid"
    else:
        kind = "born_digital"

    if kind == "scanned" or hidden > 0.90 or outlined:
        trust = "unusable"
    elif kind == "hybrid" or frag or bad > t["undecodable_frac"] \
            or sig["lines_outside_crop_frac"] > t["clipped_frac"]:
        trust = "degraded"
    else:
        trust = "full"

    direction = (ctx or {}).get(
        "direction", "rtl" if sig["rtl_cp_frac"] > t["rtl_frac"] else "ltr")
    if direction == "unknown":
        fire("C0-08", "text direction cannot be established: the document "
                      "carries too little text to judge")

    return {
        "page": sig.get("page"),
        "page_kind": kind,
        "psr_trust": trust,
        "direction": direction,
        "available": sorted(AVAILABLE[kind]),
        "findings": f,
    }
