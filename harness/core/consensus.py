#!/usr/bin/env python3
"""Cross-system consensus analysis.

With no human ground truth, agreement between independently-trained systems is
the strongest available signal about *class* decisions.  For each page a
consensus region set is built by clustering predictions from all evaluable
systems (IoU linkage), then each system is scored on:

  consensus_recall     share of consensus regions it found
  consensus_precision  share of its regions that are backed by others
  class_agreement      share of its matched regions where it agrees with the
                       consensus class (majority vote over the cluster)
  solo_rate            share of its regions that no other system predicts

A high solo_rate is not automatically wrong -- it can mean the system is the
only one that sees a real region -- so it is reported, never ranked blindly.
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
IOU_T = 0.55

# Several repositories contribute more than one configuration (3 Docling presets,
# 3 PaddleOCR models, 3 YOLO sizes, 3 Layout-Parser checkpoints, 3 Surya 2
# configurations, 4 DocLayout-YOLO training sets).  Counting those as independent
# votes lets one family manufacture its own consensus, so backing is counted in
# distinct repository families, not configurations.
#
# The family key is the registry's `repo` field, not the run-id prefix: DiT and
# LayoutLMv3-PubLayNet are separate run-ids but both ship from microsoft/unilm and
# are both PubLayNet-trained, so their errors are correlated and they must not
# vote twice.
_FAMILY = None


def family(rid):
    global _FAMILY
    if _FAMILY is None:
        import yaml
        _reg = yaml.safe_load(open(os.path.join(ROOT, "harness", "registry.yaml")))["systems"]
        _FAMILY = {s["id"]: str(s["repo"]).lower() for s in _reg}
    return _FAMILY.get(rid, rid.split(".")[0])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cluster(entries):
    """entries: [(system, region)] -> list of clusters (lists of entries)."""
    clusters = []
    for e in entries:
        placed = False
        for c in clusters:
            if any(iou(e[1]["bbox"], m[1]["bbox"]) >= IOU_T for m in c):
                c.append(e); placed = True; break
        if not placed:
            clusters.append([e])
    return clusters


def main():
    runs = {}
    for rid in sorted(os.listdir(NORM)):
        man = os.path.join(NORM, rid, "_run.json")
        if not os.path.exists(man):
            continue
        if json.load(open(man)).get("status") != "ok":
            continue
        runs[rid] = {}
        for f in os.listdir(os.path.join(NORM, rid)):
            if not f.startswith("_"):
                runs[rid][f[:-5]] = json.load(open(os.path.join(NORM, rid, f)))

    pages = sorted({p for r in runs.values() for p in r})
    n_sys = len(runs)
    n_fam = len({family(r) for r in runs})
    stats = {rid: Counter() for rid in runs}
    confusion = defaultdict(Counter)
    per_page = defaultdict(dict)

    for pid in pages:
        entries = [(rid, r) for rid, pp in runs.items() if pid in pp
                   for r in pp[pid]["regions"]]
        cls = cluster(entries)
        # a consensus region is one that at least half the *families* agree exists
        consensus = [c for c in cls
                     if len({family(s) for s, _ in c}) >= max(2, n_fam // 2)]
        cons_class = {id(c): Counter(r["class"] for _, r in c).most_common(1)[0][0]
                      for c in consensus}
        for rid in runs:
            if pid not in runs[rid]:
                continue
            mine = runs[rid][pid]["regions"]
            found = 0; agree = 0
            for c in consensus:
                if any(s == rid for s, _ in c):
                    found += 1
                    mycls = [r["class"] for s, r in c if s == rid][0]
                    if mycls == cons_class[id(c)]:
                        agree += 1
                    confusion[cons_class[id(c)]][mycls] += 1
            backed = 0; solo = 0
            for c in cls:
                members = {s for s, _ in c}
                fams = {family(s) for s, _ in c}
                if rid not in members:
                    continue
                k = sum(1 for s, _ in c if s == rid)
                if len(fams) >= 3:
                    backed += k
                elif len(fams) == 1:
                    solo += k
            stats[rid]["consensus_found"] += found
            stats[rid]["consensus_total"] += len(consensus)
            stats[rid]["class_agree"] += agree
            stats[rid]["backed"] += backed
            stats[rid]["solo"] += solo
            stats[rid]["regions"] += len(mine)
            per_page[rid][pid] = {
                "consensus_recall": round(found / len(consensus), 4) if consensus else None,
                "class_agreement": round(agree / found, 4) if found else None,
                "solo_rate": round(solo / len(mine), 4) if mine else None,
            }

    out = {"iou_threshold": IOU_T, "n_systems": n_sys, "n_families": n_fam,
           "backing_unit": "distinct repository families (a repo's configurations vote once)", "systems": {}, "per_page": per_page,
           "consensus_class_confusion": {k: dict(v) for k, v in confusion.items()}}
    for rid, s in stats.items():
        out["systems"][rid] = {
            "consensus_recall": round(s["consensus_found"] / s["consensus_total"], 4) if s["consensus_total"] else None,
            "class_agreement": round(s["class_agree"] / s["consensus_found"], 4) if s["consensus_found"] else None,
            "consensus_precision": round(s["backed"] / s["regions"], 4) if s["regions"] else None,
            "solo_rate": round(s["solo"] / s["regions"], 4) if s["regions"] else None,
            "n_regions": s["regions"],
        }
        v = out["systems"][rid]
        print(f"{rid:34s} consRecall={v['consensus_recall']} classAgree={v['class_agreement']} "
              f"consPrec={v['consensus_precision']} solo={v['solo_rate']}")
    json.dump(out, open(os.path.join(BENCH, "metrics", "consensus.json"), "w"), indent=1)
    print("\nwrote metrics/consensus.json")


if __name__ == "__main__":
    main()
