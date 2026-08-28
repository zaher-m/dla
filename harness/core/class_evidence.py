#!/usr/bin/env python3
"""Per-class behavioural evidence used to ground the qualitative rating table.

For each system this counts how reliably it emits each canonical class on the
pages where the PDF itself says that structure exists, so the 1-5 ratings in the
report are anchored to observed output rather than repository claims.
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
NORM = os.path.join(BENCH, "normalized_outputs")

ref = json.load(open(os.path.join(BENCH, "inventory", "pdf_structural_reference.json")))
met = json.load(open(os.path.join(BENCH, "metrics", "layout_metrics.json")))

# pages where each structure demonstrably exists, from the PDF itself
has_table = {p for p, r in ref.items() if r["grid_candidates"]}
has_fig = {p for p, r in ref.items() if r["graphic_areas"]}
multicol = {p for p, r in ref.items() if len(r.get("column_bands", [])) >= 2}

out = {}
for rid in sorted(os.listdir(NORM)):
    man = os.path.join(NORM, rid, "_run.json")
    if not os.path.exists(man) or json.load(open(man)).get("status") != "ok":
        continue
    per_page_cls, counts = {}, Counter()
    for f in sorted(os.listdir(os.path.join(NORM, rid))):
        if f.startswith("_"):
            continue
        d = json.load(open(os.path.join(NORM, rid, f)))
        cs = Counter(r["class"] for r in d["regions"])
        per_page_cls[f[:-5]] = cs
        counts.update(cs)
    npg = len(per_page_cls)
    presence = {c: round(sum(1 for v in per_page_cls.values() if v.get(c)) / npg, 3)
                for c in ["text", "title", "heading", "list", "table", "figure", "caption",
                          "formula", "header", "footer", "footnote", "page_number",
                          "sidebar", "separator", "other"]}
    m = met[rid]["pages"]
    def med(key, subset=None):
        v = [m[p][key] for p in m if (subset is None or p in subset)
             and isinstance(m[p].get(key), (int, float))]
        return round(float(np.median(v)), 4) if v else None
    out[rid] = {
        "class_presence_rate": presence,
        "total_class_counts": dict(counts),
        "table_iou_on_table_pages": med("table_iou", has_table),
        "table_recall_on_table_pages": med("table_recall", has_table),
        "table_hit_rate": round(sum(1 for p in has_table
                                    if (m.get(p, {}).get("table_iou") or 0) >= 0.5)
                                / max(len(has_table), 1), 3),
        "table_partial_rate": round(sum(1 for p in has_table
                                        if (m.get(p, {}).get("table_iou") or 0) >= 0.2)
                                    / max(len(has_table), 1), 3),
        "n_table_pages": len(has_table),
        "figure_iou_on_figure_pages": med("graphic_iou", has_fig),
        "text_recall_multicol": med("text_recall", multicol),
        "bleed_multicol": med("column_bleed_rate", multicol),
        "headerfooter_pages": round(sum(1 for v in per_page_cls.values()
                                        if v.get("header") or v.get("footer")
                                        or v.get("page_number")) / npg, 3),
        "caption_pages": round(sum(1 for v in per_page_cls.values() if v.get("caption")) / npg, 3),
        "reading_order": bool(med("has_reading_order")),
        "polygons": bool(med("has_polygons")),
    }
    o = out[rid]
    print(f"{rid:34s} tblHit={o['table_hit_rate']} tblPart={o['table_partial_rate']} "
          f"figIoU={o['figure_iou_on_figure_pages']} "
          f"mcolTxtR={o['text_recall_multicol']} hdrftr={o['headerfooter_pages']} "
          f"cap={o['caption_pages']} ro={int(o['reading_order'])} poly={int(o['polygons'])}")
json.dump(out, open(os.path.join(BENCH, "metrics", "class_evidence.json"), "w"), indent=1)
print("\nwrote metrics/class_evidence.json")
