#!/usr/bin/env python3
"""Package everything the interactive report needs into one JSON bundle.

Page images are downscaled JPEGs embedded as data URIs; region geometry is sent
as compact numeric arrays and drawn as SVG in the browser, so one image per page
serves every system overlay instead of one image per (page x system).
"""
import argparse, base64, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import yaml
from core.taxonomy import ALL_CLASSES, COLORS, mapping_rows

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
NORM = os.path.join(BENCH, "normalized_outputs")
# Page images are embedded in the report as data URIs, so these two numbers
# decide its size: a 29-page, 54-system bundle is ~6 MB at these settings.
IMG_W = int(paths.get("report", "image_width", 760))
JPEG_Q = int(paths.get("report", "jpeg_quality", 72))


def img_data_uri(path, width=IMG_W, q=JPEG_Q):
    im = Image.open(path).convert("RGB")
    im = im.resize((width, int(round(im.height * width / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()



SLOT_ORDER_PATH = os.path.join(BENCH, "reports", "orbit_slot_order.json")


def stable_slot_order(ok_ids):
    """Fixed orbit slots that survive a re-run with new systems added.

    The report's orbit gives every model a permanent angular position so a
    reader can track one model across pages and across report versions.  Deriving
    that order from the registry would break the promise the moment a system is
    inserted in the middle of the file: everything after it would rotate to a new
    slot.  So the assignment order is persisted once and only ever appended to —
    new systems take the next free slot, which is what pushes them to the outer
    rings.  Systems that stop being evaluable keep their slot reserved rather than
    letting the ones behind them shift forward.
    """
    try:
        order = json.load(open(SLOT_ORDER_PATH))
        order = [i for i in order if isinstance(i, str)]
    except Exception:
        order = []
    seen = set(order)
    for sid in ok_ids:
        if sid not in seen:
            order.append(sid)
            seen.add(sid)
    os.makedirs(os.path.dirname(SLOT_ORDER_PATH), exist_ok=True)
    json.dump(order, open(SLOT_ORDER_PATH, "w"), indent=1)
    return order


def separability(met, ok_ids, metrics=("text_or_table_recall", "text_recall")):
    """Which systems are actually distinguishable from the leader?

    The report's tables show the median over 29 pages, which is a fair
    description of a system but a bad basis for ranking: a four-page change can
    move a median by several points.  So for each headline metric, rank by the
    per-page mean and run a two-sided sign test on the paired per-page
    differences against the leader.  Systems the test cannot separate at
    p <= 0.05 are a *leading group*, and their order in the table is noise.

    One exclusion, applied to the choice of *leader* only.  Both headline
    metrics ask what fraction of the reference's body-text lines fall inside a
    predicted text-or-table region.  A system with a single output class
    maximises that by construction -- it has nothing else to label anything as,
    so every region it emits is a text region.  kraken's region head is such a
    system, and it wins text recall outright (0.919 mean, ahead of every other
    system on every page where they differ).  Letting it define the leading
    group would make every typed multi-class detector look separable from "the
    leader" for a reason that has nothing to do with layout quality.  So
    single-class systems are ranked and reported like everything else, and
    flagged, but the leader is chosen among systems that actually distinguish
    classes.  `single_class` lists who was excluded and why.
    """
    import math
    out = {}
    for name in metrics:
        S = {}
        for sid in ok_ids:
            pg = (met.get(sid) or {}).get("pages") or {}
            v = {p: d.get(name) for p, d in pg.items() if d.get(name) is not None}
            if v:
                S[sid] = v
        if not S:
            continue
        # A system counts as single-class when it never emits more than one
        # canonical class on any page.  Derived from the data, not hand-listed.
        single = set()
        for sid in S:
            div = [d.get("class_diversity") for d in (met[sid].get("pages") or {}).values()
                   if d.get("class_diversity") is not None]
            if div and max(div) <= 1:
                single.add(sid)
        rank = sorted(S, key=lambda k: -sum(S[k].values()) / len(S[k]))
        lead = next((k for k in rank if k not in single), rank[0])
        ps = {}
        for sid in rank:
            common = sorted(set(S[sid]) & set(S[lead]))
            d = [S[sid][p] - S[lead][p] for p in common]
            pos = sum(1 for x in d if x > 1e-9)
            neg = sum(1 for x in d if x < -1e-9)
            n = pos + neg
            if n == 0:
                ps[sid] = 1.0
            else:
                k = min(pos, neg)
                ps[sid] = min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)
        out[name] = {
            "leader": lead,
            "rank": rank,
            "mean": {k: round(sum(S[k].values()) / len(S[k]), 4) for k in rank},
            "p_vs_leader": {k: round(v, 4) for k, v in ps.items()},
            "group": [k for k in rank if ps[k] > 0.05 and k not in single],
            "single_class": sorted(single),
        }
    return out


def true_device(man):
    """Authoritative device for one run.

    `resources()` reports the CUDA device whenever torch can see one, which is
    true even for adapters that never touch torch -- onnxruntime and
    PaddlePaddle both run on CPU here but live in venvs that inherit the
    container's CUDA torch.  Trust the adapter's own declaration first when it
    says CPU, then fall back to whether the run actually allocated any CUDA
    memory.
    """
    res = man.get("resources") or {}
    mod = man.get("model") or {}
    decl = str(mod.get("device") or "")
    if decl.lower().startswith("cpu"):
        return decl
    if (res.get("cuda_peak_alloc_mb") or 0) > 0:
        return res.get("device") or decl or "\u2014"
    return decl or res.get("device") or "\u2014"


def _load(path, default):
    """Optional analysis products.

    A short job — a handful of pages, a small profile — can legitimately produce
    no ensemble or class-evidence file, because there is nothing to compute from.
    The viewer must still render, so a missing optional product degrades the
    report rather than aborting it.  The three genuinely required inputs
    (pages, metrics, registry) are loaded without a fallback and fail loudly.
    """
    if not os.path.exists(path):
        sys.stderr.write(f"[package_report] optional input missing: {path}\n")
        return default
    with open(path, encoding="utf8") as f:
        return json.load(f)


INCLUDE_NOT_RUN = False


def main():
    global BENCH, NORM
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--include-not-run", action="store_true",
                    help="keep registry entries with no manifest in this workspace")
    a, _ = ap.parse_known_args()
    if a.workspace:
        BENCH = paths.resolve(a.workspace)
        NORM = os.path.join(BENCH, "normalized_outputs")
    global INCLUDE_NOT_RUN
    INCLUDE_NOT_RUN = a.include_not_run

    with open(paths.REGISTRY, encoding="utf8") as f:
        reg = yaml.safe_load(f)["systems"]
    with open(os.path.join(BENCH, "inventory", "selected_pages.json"), encoding="utf8") as f:
        pages = json.load(f)
    with open(os.path.join(BENCH, "metrics", "layout_metrics.json"), encoding="utf8") as f:
        met = json.load(f)
    inv = _load(os.path.join(BENCH, "inventory", "corpus_inventory.json"),
                {"documents": [], "pages": []})
    ref = _load(os.path.join(BENCH, "inventory", "pdf_structural_reference.json"), {})
    cons = _load(os.path.join(BENCH, "metrics", "consensus.json"),
                 {"systems": {}, "per_page": {}, "consensus_class_confusion": {}})
    rat = _load(os.path.join(BENCH, "metrics", "ratings.json"), {"systems": {}, "weights": {}})
    ens = _load(os.path.join(BENCH, "metrics", "ensemble.json"),
                {"per_class": {}, "routing": {}, "per_page_class_counts": {}})
    cev = _load(os.path.join(BENCH, "metrics", "class_evidence.json"), {})
    # Written by the `validate` stage, which is optional and produces nothing on
    # a job with no text layer.  Embedded whole: it is deliberately small, and a
    # report that can show what was accepted is worth more than one that shows
    # only how systems compare to each other.
    val = _load(os.path.join(BENCH, "validation", "summary.json"),
                {"systems": {}, "pages": {}})
    dev = {}
    p = os.path.join(BENCH, "metrics", "device_agreement.json")
    if os.path.exists(p):
        dev = json.load(open(p))

    cls_idx = {c: i for i, c in enumerate(ALL_CLASSES)}

    systems = []
    for s in reg:
        man_p = os.path.join(NORM, s["id"], "_run.json")
        man = json.load(open(man_p)) if os.path.exists(man_p) else {"status": "not_run"}
        m = met.get(s["id"], {})
        systems.append({
            "id": s["id"], "display": s["display"], "repo": s["repo"],
            "taxonomy": s["taxonomy"], "config": s.get("config", {}),
            "status": man.get("status", "not_run"),
            "n_ok": man.get("n_ok"), "n_failed": man.get("n_failed"),
            "model": man.get("model", {}), "model_load_s": man.get("model_load_s"),
            "resources": man.get("resources", {}), "torch_env": man.get("torch_env", {}),
            "device": true_device(man),
            "wall_s": man.get("wall_s"),
            "metrics": m.get("aggregate", {}),
            "consensus": cons["systems"].get(s["id"], {}),
            "device_agreement": dev.get(s["id"], {}).get("mean_box_agreement"),
            "ratings": rat["systems"].get(s["id"], {}).get("ratings", {}),
            "evidence": cev.get(s["id"], {}),
        })

    # Report the systems this workspace actually ran. Registry entries with no
    # manifest here are dropped -- listing models the caller never asked for as
    # "not run" is noise. --include-not-run keeps them.
    if not INCLUDE_NOT_RUN:
        systems = [s for s in systems if s["status"] != "not_run"]

    ok_ids = [s["id"] for s in systems if s["status"] == "ok"]
    slot_order = stable_slot_order(ok_ids)

    out_pages, preds = [], {}
    for pg in pages:
        pid = pg["page_id"]
        img = os.path.join(BENCH, "working", "pages_300dpi", f"{pid}.png")
        r = ref.get(pid, {})
        out_pages.append({
            "id": pid, "doc": pg["doc"], "page": pg["page"], "stratum": pg["stratum"],
            "w": pg["px_width"], "h": pg["px_height"],
            "img": img_data_uri(img),
            "ref": {"n_body_lines": len(r.get("body_text_lines", [])),
                    "n_graphic_lines": len(r.get("graphic_text_lines", [])),
                    "n_table_lines": len(r.get("table_text_lines", [])),
                    "n_graphics": len(r.get("graphic_areas", [])),
                    "n_grids": len(r.get("grid_candidates", [])),
                    "columns": len(r.get("column_bands", [])),
                    "graphic_areas": [[round(v) for v in b] for b in r.get("graphic_areas", [])],
                    "grid_candidates": [[round(v) for v in b] for b in r.get("grid_candidates", [])],
                    "column_bands": [[round(v) for v in b] for b in r.get("column_bands", [])]},
            "metrics": {sid: met.get(sid, {}).get("pages", {}).get(pid, {}) for sid in ok_ids},
        })
        preds[pid] = {}
        for sid in ok_ids:
            f = os.path.join(NORM, sid, f"{pid}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            rows = []
            for rr in d["regions"]:
                b = rr["bbox"]
                rows.append([cls_idx.get(rr["class"], cls_idx["other"]),
                             round(rr["confidence"], 3) if rr["confidence"] is not None else -1,
                             round(b[0]), round(b[1]), round(b[2]), round(b[3]),
                             rr["reading_order"] if rr["reading_order"] is not None else -1,
                             rr["source_class"], rr["mapping_confidence"]])
            preds[pid][sid] = {"r": rows,
                               "t": round(d["timing"].get("inference", 0) * 1000, 1)}

    bundle = {
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "classes": ALL_CLASSES,
        "colors": {c: "#%02x%02x%02x" % COLORS[c] for c in ALL_CLASSES},
        "systems": systems, "pages": out_pages, "predictions": preds,
        "slot_order": slot_order,
        "separability": separability(met, ok_ids),
        "taxonomy_map": mapping_rows(),
        "corpus": {"documents": [{k: v for k, v in d.items() if k != "metadata"}
                                 for d in inv["documents"]],
                   "n_pages_total": len(inv["pages"]),
                   "page_stats": inv["pages"]},
        "consensus_confusion": cons.get("consensus_class_confusion", {}),
        "rubric": rat.get("rubric", {}), "rubric_weights": rat.get("weights", {}),
        "ensemble": {"per_class": ens["per_class"], "routing": ens["routing"],
                     "per_page_class_counts": ens["per_page_class_counts"],
                     "iou_threshold": ens["iou_threshold"], "min_backing": ens["min_backing"]},
        "validation": val,
    }
    outp = os.path.join(BENCH, "reports", "report_data.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(bundle, open(outp, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {outp}  {os.path.getsize(outp)/1e6:.1f} MB  "
          f"systems={len(systems)} ok={len(ok_ids)} pages={len(out_pages)}")


if __name__ == "__main__":
    main()
