#!/usr/bin/env python3
"""Render standardized per-system panels and side-by-side comparison sheets."""
import argparse, json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import visualize as V
from core.taxonomy import ALL_CLASSES

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
NORM = os.path.join(BENCH, "normalized_outputs")
VIS = os.path.join(BENCH, "visualizations")


def evaluable_runs():
    runs = []
    for rid in sorted(os.listdir(NORM)):
        man = os.path.join(NORM, rid, "_run.json")
        if not os.path.exists(man):
            continue
        m = json.load(open(man))
        if m.get("status") == "ok" and m.get("n_ok", 0) > 0:
            runs.append((rid, m))
    return runs


def display_names():
    import yaml
    reg = yaml.safe_load(open(os.path.join(ROOT, "harness", "registry.yaml")))["systems"]
    return {s["id"]: s["display"] for s in reg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--panel-w", type=int, default=V.PANEL_W)
    a = ap.parse_args()

    pages = json.load(open(os.path.join(BENCH, "inventory", "selected_pages.json")))
    runs = evaluable_runs()
    disp = display_names()
    print(f"{len(runs)} evaluable runs, {len(pages)} pages")

    V.legend_image(ALL_CLASSES, os.path.join(VIS, "legend.png"))

    for p in pages:
        pid = p["page_id"]
        pdir = os.path.join(VIS, pid)
        img = os.path.join(BENCH, "working", "pages_300dpi", f"{pid}.png")
        panels = [V.render_original(img, os.path.join(pdir, "original.png"),
                                    title=f"Original · {p['doc'][:28]} p{p['page']} · {p['stratum']}",
                                    panel_w=a.panel_w)]
        for rid, man in runs:
            f = os.path.join(NORM, rid, f"{pid}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            n = d["n_regions"]
            ttl = f"{disp.get(rid, rid)} · {n} regions · {d['timing'].get('inference',0)*1000:.0f} ms"
            panels.append(V.render_page(img, d["regions"], os.path.join(pdir, f"{rid}.png"),
                                        title=ttl, panel_w=a.panel_w,
                                        show_order=any(r.get("reading_order") is not None
                                                       for r in d["regions"])))
        # comparison sheets, capped at 6 panels each so nothing becomes unreadable
        per = a.cols * 2
        for i in range(0, len(panels), per):
            chunk = panels[i:i + per]
            if i > 0:
                chunk = [panels[0]] + chunk       # keep the original visible on every sheet
            out = os.path.join(pdir, f"comparison_{i // per + 1}.png")
            V.grid(chunk, out, cols=a.cols)
        V.grid(panels, os.path.join(pdir, "comparison.png"),
               cols=max(3, math.ceil(math.sqrt(len(panels)))))
        print("  ", pid, len(panels), "panels")


if __name__ == "__main__":
    main()
