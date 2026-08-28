#!/usr/bin/env python3
"""Observed-quality ratings (1-5) on the supplied samples.

The spec asks for a per-dimension rating table based on *observed outputs*.
To keep that reproducible rather than impressionistic, every rating is produced
by a fixed rubric applied to a measured quantity.  The rubric and the driving
quantity are emitted alongside each score so a reader can audit any cell.

  5 excellent · 4 very good · 3 acceptable · 2 weak · 1 poor · N/A not supported
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.taxonomy import MAPPINGS

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE

import yaml
met = json.load(open(os.path.join(BENCH, "metrics", "layout_metrics.json")))
TAXONOMY = {s["id"]: s["taxonomy"] for s in
            yaml.safe_load(open(os.path.join(ROOT, "harness", "registry.yaml")))["systems"]}
ev = json.load(open(os.path.join(BENCH, "metrics", "class_evidence.json")))
cons = json.load(open(os.path.join(BENCH, "metrics", "consensus.json")))


def supported(tax, canon):
    """Can this system's own class vocabulary even express the canonical class?

    A PubLayNet detector has no header/footer/caption class at all; scoring it 1
    for "missed the running header" would punish it for a task it was never
    asked to do.  Those cells read N/A instead, per the spec.
    """
    table = MAPPINGS.get(tax, {})
    return any(v[0] == canon for v in table.values())


def band(v, cuts, reverse=False):
    """cuts: four thresholds ascending -> scores 1..5 (or 5..1 when reverse)."""
    if v is None:
        return None
    s = 1 + sum(1 for c in cuts if v >= c)
    return 6 - s if reverse else s


# (name, basis, scorer, canonical classes the dimension depends on)
DIMS = [
    ("Text", "text_or_table_recall (median)",
     lambda r, e, c: band(r.get("text_or_table_recall"), [.35, .60, .75, .85]), ["text"]),
    ("Headings", "heading+title presence rate x consensus class agreement",
     lambda r, e, c: band((e["class_presence_rate"]["heading"] + e["class_presence_rate"]["title"])
                          * (c.get("class_agreement") or 0), [.05, .20, .40, .60]),
     ["heading", "title"]),
    ("Tables", "share of ruled-table pages localised at IoU>=0.5",
     lambda r, e, c: band(e.get("table_hit_rate"), [.05, .20, .40, .60]), ["table"]),
    ("Figures", "figure IoU against PDF graphic areas",
     lambda r, e, c: band(r.get("graphic_iou"), [.12, .25, .38, .47]), ["figure"]),
    ("Captions", "share of pages with a caption region",
     lambda r, e, c: band(e["caption_pages"], [.05, .20, .40, .60]), ["caption"]),
    ("Lists", "share of pages with a list region",
     lambda r, e, c: band(e["class_presence_rate"]["list"], [.02, .10, .25, .45]), ["list"]),
    ("Formulas", "share of pages with a formula region",
     lambda r, e, c: band(e["class_presence_rate"]["formula"], [.02, .10, .25, .45]), ["formula"]),
    ("Headers/footers", "share of pages with header, footer or page-number regions",
     lambda r, e, c: band(e["headerfooter_pages"], [.10, .35, .60, .80]),
     ["header", "footer", "page_number"]),
    ("Columns", "gutter-crossing rate among text regions (lower better)",
     lambda r, e, c: band(r.get("gutter_cross_rate"), [.005, .010, .020, .040], reverse=True), ["text"]),
    ("Boundary quality", "F1(text precision, text recall) minus spill",
     lambda r, e, c: band(_boundary(r), [.20, .35, .48, .58]), ["text"]),
]

def _boundary(r):
    p, rc, sp = r.get("text_precision"), r.get("text_recall"), r.get("text_spill") or 0
    if not p or not rc:
        return None
    return 2 * p * rc / (p + rc) - sp


WEIGHTS = {"Text": 3, "Headings": 1.5, "Tables": 2, "Figures": 2, "Captions": 1,
           "Lists": .5, "Formulas": .5, "Headers/footers": 1.5, "Columns": 2,
           "Boundary quality": 2}


def main():
    out = {}
    for rid, m in met.items():
        if m.get("status") != "ok" or rid not in ev:
            continue
        agg = {k: (v.get("median") if isinstance(v, dict) else v) for k, v in m["aggregate"].items()}
        e = ev[rid]
        c = cons["systems"].get(rid, {})
        tax = m.get("config") and None
        tax = TAXONOMY.get(rid)
        row, why = {}, {}
        # a system that finds almost no text cannot be judged on how it handles
        # columns or boundaries -- there is nothing to observe
        blind = (agg.get("text_or_table_recall") or 0) < 0.20
        for name, basis, fn, needs in DIMS:
            if not any(supported(tax, cnl) for cnl in needs):
                row[name] = None; why[name] = basis + " [class not in this taxonomy]"
                continue
            if blind and name in ("Columns", "Boundary quality", "Headings"):
                row[name] = None
                why[name] = basis + " [not observable: system found <20% of the text]"
                continue
            row[name] = fn(agg, e, c)
            why[name] = basis
        vals = [(WEIGHTS[k], v) for k, v in row.items() if v is not None]
        overall = sum(w * v for w, v in vals) / sum(w for w, _ in vals) if vals else None
        row["Overall layout"] = round(overall, 2) if overall else None
        out[rid] = {"ratings": row, "basis": why}
    json.dump({"rubric": {n: b for n, b, _, _ in DIMS}, "weights": WEIGHTS, "systems": out},
              open(os.path.join(BENCH, "metrics", "ratings.json"), "w"), indent=1)
    names = [n for n, _, _, _ in DIMS] + ["Overall layout"]
    print(f"{'system':34s} " + " ".join(f"{n[:7]:>7s}" for n in names))
    for rid in sorted(out, key=lambda k: -(out[k]["ratings"]["Overall layout"] or 0)):
        r = out[rid]["ratings"]
        print(f"{rid:34s} " + " ".join(f"{(r[n] if r[n] is not None else 'N/A'):>7}" for n in names))


if __name__ == "__main__":
    main()
