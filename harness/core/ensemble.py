#!/usr/bin/env python3
"""Per-class competence and model-routing analysis.

The question this answers is the one an engineering team actually has to settle:
*not* "which model is best overall" but "for each kind of region, whose output
should I take?"  That is how production document pipelines are usually built —
a router or an ensemble over specialists, rather than one monolithic detector.

Method, per canonical class:

  1. Build consensus regions for that class on every page: cluster all systems'
     predictions of that class by IoU, keep clusters that at least `MIN_BACKING`
     distinct systems produced.  This is a majority reference, not ground truth.
  2. Score each system against it: recall (found), precision (its regions of that
     class that are backed), F1.
  3. Report the leader, the margin over second place, and how often the leader
     wins page-by-page — a leader that only wins on average is not a safe route.

Also emits, per page, the raw per-class region counts for every system, so a
reviewer can see the disagreement directly rather than through a summary.
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.taxonomy import ALL_CLASSES

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
B = paths.WORKSPACE
NORM = os.path.join(B, "normalized_outputs")
IOU_T = 0.55
MIN_BACKING = 3          # distinct *families*, not configurations -- see FAMILY below

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
    out = []
    for e in entries:
        for c in out:
            if any(iou(e[1], m[1]) >= IOU_T for m in c):
                c.append(e); break
        else:
            out.append([e])
    return out


def load():
    runs = {}
    for rid in sorted(os.listdir(NORM)):
        man = os.path.join(NORM, rid, "_run.json")
        if not os.path.exists(man) or json.load(open(man)).get("status") != "ok":
            continue
        runs[rid] = {}
        for f in os.listdir(os.path.join(NORM, rid)):
            if not f.startswith("_"):
                runs[rid][f[:-5]] = json.load(open(os.path.join(NORM, rid, f)))
    return runs


def main():
    runs = load()
    systems = sorted(runs)
    pages = sorted({p for r in runs.values() for p in r})

    # ---- raw per-page, per-class counts -------------------------------------
    counts = {p: {s: Counter(r["class"] for r in runs[s][p]["regions"])
                  for s in systems if p in runs[s]} for p in pages}

    # ---- per-class consensus scoring ---------------------------------------
    per_class = {}
    page_win = defaultdict(lambda: defaultdict(int))     # class -> system -> pages won
    page_seen = Counter()
    for k in ALL_CLASSES:
        tp = Counter(); fp = Counter(); tot = 0
        for p in pages:
            ent = [(s, r["bbox"]) for s in systems if p in runs[s]
                   for r in runs[s][p]["regions"] if r["class"] == k]
            if not ent:
                continue
            cls = cluster(ent)
            cons = [c for c in cls if len({family(s) for s, _ in c}) >= MIN_BACKING]
            if not cons:
                continue
            tot += len(cons)
            page_seen[k] += 1
            page_f1 = {}
            for s in systems:
                if p not in runs[s]:
                    continue
                mine = [r for r in runs[s][p]["regions"] if r["class"] == k]
                found = sum(1 for c in cons if any(x == s for x, _ in c))
                backed = sum(1 for c in cls if len({family(x) for x, _ in c}) >= MIN_BACKING
                             and any(x == s for x, _ in c))
                tp[s] += found
                fp[s] += max(0, len(mine) - backed)
                rc = found / len(cons)
                pr = backed / len(mine) if mine else 0.0
                page_f1[s] = 0.0 if (rc + pr) == 0 else 2 * rc * pr / (rc + pr)
            # ties are shared: on 29 pages, declaring a single winner from a
            # 0.001 F1 difference would manufacture a leader that does not exist
            if page_f1 and max(page_f1.values()) > 0:
                top = max(page_f1.values())
                for s2, v2 in page_f1.items():
                    if v2 >= top - 0.01:
                        page_win[k][s2] += 1
        if not tot:
            continue
        rows = {}
        for s in systems:
            rc = tp[s] / tot
            pr = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) else 0.0
            f1 = 0.0 if (rc + pr) == 0 else 2 * rc * pr / (rc + pr)
            rows[s] = {"recall": round(rc, 4), "precision": round(pr, 4),
                       "f1": round(f1, 4), "found": tp[s], "unbacked": fp[s]}
        order = sorted(rows, key=lambda s: -rows[s]["f1"])
        lead, second = order[0], (order[1] if len(order) > 1 else None)
        # every system statistically indistinguishable from the leader on this
        # sample size: this, not the single winner, is the useful output
        top_f1 = rows[lead]["f1"]
        equiv = [s for s in order if rows[s]["f1"] >= top_f1 - 0.02]
        wins = page_win[k]
        nseen = page_seen[k]
        per_class[k] = {
            "n_consensus_regions": tot, "n_pages_with_class": nseen,
            "systems": rows, "ranking": order,
            "leader": lead, "leader_f1": rows[lead]["f1"],
            "runner_up": second, "margin": round(rows[lead]["f1"] - rows[second]["f1"], 4) if second else None,
            "leader_page_win_rate": round(wins.get(lead, 0) / nseen, 3) if nseen else None,
            "page_wins": dict(wins),
            "equivalent_group": equiv,
        }

    # ---- routing proposal ---------------------------------------------------
    # A class is safely routable when its leader is both ahead on aggregate F1
    # and wins on a majority of the pages where the class appears.
    routing = {}
    for k, v in per_class.items():
        eq = v["equivalent_group"]
        if v["n_consensus_regions"] < 8:
            verdict = "insufficient evidence"
            why = (f"only {v['n_consensus_regions']} consensus regions across "
                   f"{v['n_pages_with_class']} pages — too few to choose on")
        elif len(eq) > 1 and v["leader_f1"] >= 0.97:
            verdict = "tied at ceiling"
            why = (f"{len(eq)} systems reach F1 ≥ {v['leader_f1']-0.02:.2f}; pick on cost, "
                   f"not accuracy")
        elif len(eq) > 1:
            verdict = "no separable leader"
            why = (f"{len(eq)} systems within 0.02 F1 of the top ({v['leader_f1']:.3f}); "
                   f"the sample cannot separate them")
        elif (v["leader_page_win_rate"] or 0) < 0.5:
            verdict = "unstable leader"
            why = (f"ahead on aggregate F1 ({v['leader_f1']:.3f}) but top-ranked on only "
                   f"{v['leader_page_win_rate']:.0%} of pages — not a safe route")
        else:
            verdict = "route"
            why = (f"F1 {v['leader_f1']:.3f}, +{v['margin']:.3f} over {v['runner_up']}, "
                   f"top-ranked on {v['leader_page_win_rate']:.0%} of pages")
        routing[k] = {"decision": verdict, "system": v["leader"], "rationale": why,
                      "f1": v["leader_f1"], "margin": v["margin"],
                      "page_win_rate": v["leader_page_win_rate"],
                      "equivalent_group": eq}

    out = {"iou_threshold": IOU_T, "min_backing": MIN_BACKING,
           "backing_unit": "distinct repository families (a repo's multiple configurations vote once)",
           "families": sorted({family(s) for s in systems}),
           "systems": systems, "pages": pages,
           "per_page_class_counts": {p: {s: dict(c) for s, c in d.items()} for p, d in counts.items()},
           "per_class": per_class, "routing": routing}
    json.dump(out, open(os.path.join(B, "metrics", "ensemble.json"), "w"), indent=1)

    print(f"{'class':13s} {'n':>4s} {'leader':32s} {'F1':>6s} {'win%':>5s} {'eq':>3s}  decision")
    for k, v in per_class.items():
        r = routing[k]
        print(f"{k:13s} {v['n_consensus_regions']:4d} {v['leader']:32s} "
              f"{v['leader_f1']:6.3f} {(v['leader_page_win_rate'] or 0)*100:5.0f} "
              f"{len(v['equivalent_group']):3d}  {r['decision']}")
    print("\nwrote metrics/ensemble.json")


if __name__ == "__main__":
    main()
