#!/usr/bin/env python3
"""Per-document profiles, for the checks that need more than one page.

Every other check judges a page against the PDF underneath it.  These judge a
page against the rest of its own document, which is the only way to see an
*intermittent* failure: a running header found on forty pages and missed on two,
a column count that changes for one page in the middle of a chapter, a body font
that shifts because the page was rendered at the wrong size.  Nothing page-local
can see any of that, because each of those pages is internally consistent.

The profile is built per (system, document): it describes what that system
usually does with that document, so the comparison is against the system's own
behaviour rather than against an expectation we invented.
"""
from collections import Counter

import numpy as np

from validation.buckets import bucket as to_bucket, DISCARD


def page_facts(regions, psr, bands=None, margin=0.08):
    """The handful of page properties the document checks compare."""
    H = psr["height"]
    buckets = [to_bucket(r.get("class")) for r in regions]
    top, bot = margin * H, (1 - margin) * H
    running = any(b == DISCARD and not (top <= (r["bbox"][1] + r["bbox"][3]) / 2 <= bot)
                  for r, b in zip(regions, buckets))
    return {
        "n_regions": len(regions),
        "n_columns": len(bands if bands is not None else (psr.get("column_bands") or [])),
        "classes": len({r.get("class") for r in regions}),
        "running": bool(running),
        "body_font": psr.get("body_font_px") or 0.0,
        "buckets": Counter(buckets),
    }


def profile(facts):
    """Aggregate per-page facts into what the document usually looks like."""
    if not facts:
        return None
    n = len(facts)
    regions = [f["n_regions"] for f in facts]
    fonts = [f["body_font"] for f in facts if f["body_font"]]
    mix = Counter()
    for f in facts:
        tot = sum(f["buckets"].values()) or 1
        for k, v in f["buckets"].items():
            mix[k] += v / tot
    return {
        "n_pages": n,
        "running_rate": sum(1 for f in facts if f["running"]) / n,
        "columns_mode": Counter(f["n_columns"] for f in facts).most_common(1)[0][0],
        "classes_median": float(np.median([f["classes"] for f in facts])),
        "regions_mean": float(np.mean(regions)),
        "regions_std": float(np.std(regions)),
        "font_mode": float(np.median(fonts)) if fonts else 0.0,
        "bucket_mix": {k: v / n for k, v in mix.items()},
    }
