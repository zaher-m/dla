"""Normalized layout-analysis output schema shared by every adapter.

One JSON document per (system_run, page):

{
  "document": "2-2019_AR.pdf", "page": 4, "page_id": "page_001",
  "width": 2481, "height": 3508,          # pixel space of the 300-dpi render
  "system": "surya", "run_id": "surya.default",
  "regions": [{"id":1,"class":"text","confidence":0.94,"bbox":[x1,y1,x2,y2],
               "polygon":null,"mask":null,"source_class":"Text",
               "mapping_confidence":"exact","reading_order":0}],
  "timing": {...}, "resources": {...}, "meta": {...}
}

All geometry is expressed in the pixel coordinate system of the canonical
300-dpi render so that every system is directly comparable.
"""
import json, os
from datetime import datetime, timezone

SCHEMA_VERSION = "1.1"


def make_region(idx, canonical, source_class, bbox, confidence=None, polygon=None,
                mask=None, mapping_confidence="exact", reading_order=None, extra=None):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    r = {"id": int(idx), "class": canonical, "source_class": source_class,
         "confidence": None if confidence is None else round(float(confidence), 5),
         "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
         "polygon": polygon, "mask": mask,
         "mapping_confidence": mapping_confidence,
         "reading_order": reading_order}
    if extra:
        r["extra"] = extra
    return r


def make_page_result(run_id, system, page_meta, regions, timing, resources, meta=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "system": system,
        "document": page_meta["doc"], "page": page_meta["page"],
        "page_id": page_meta["page_id"],
        "width": page_meta["px_width"], "height": page_meta["px_height"],
        "regions": regions,
        "n_regions": len(regions),
        "timing": timing, "resources": resources, "meta": meta or {},
    }


def scale_regions(regions, sx, sy):
    """Rescale geometry produced at a different resolution into canonical px."""
    for r in regions:
        r["bbox"] = [round(r["bbox"][0] * sx, 2), round(r["bbox"][1] * sy, 2),
                     round(r["bbox"][2] * sx, 2), round(r["bbox"][3] * sy, 2)]
        if r.get("polygon"):
            r["polygon"] = [[round(x * sx, 2), round(y * sy, 2)] for x, y in r["polygon"]]
    return regions


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
