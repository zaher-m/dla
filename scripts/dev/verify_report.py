#!/usr/bin/env python3
"""Render a built report in a headless browser and assert it is not broken.

The report is a program: it parses a JSON bundle and builds its own DOM. Reading
the HTML tells you nothing about whether it works, so this opens it, watches for
console and page errors, counts the elements that must exist, and checks that no
viewport width produces horizontal scroll.

    python scripts/dev/verify_report.py benchmark/reports/index.html
    python scripts/dev/verify_report.py data/jobs/<id>/reports/index.html --min-satellites 5

Requires the optional `shot` environment: bash harness/setup/shot.sh
"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "harness"))

WIDTHS = (1700, 1100, 720, 390)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--min-satellites", type=int, default=1)
    ap.add_argument("--themes", default="light,dark")
    a = ap.parse_args()
    path = os.path.abspath(a.report)
    if not os.path.exists(path):
        sys.exit(f"no such report: {path}")

    from playwright.sync_api import sync_playwright
    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for scheme in a.themes.split(","):
            page = browser.new_page(viewport={"width": 1600, "height": 1100},
                                    color_scheme=scheme)
            errs = []
            page.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
            page.on("console", lambda m: errs.append(f"console: {m.text}")
                    if m.type == "error" else None)
            page.goto("file://" + path)
            page.wait_for_timeout(6000)

            n = lambda sel: page.eval_on_selector_all(sel, "e=>e.length")
            counts = {"satellites": n(".sat"), "systemCards": n(".syscard"),
                      "metricRows": n("#metricTable tbody tr"),
                      "pageChips": n("#pageStrip *")}
            print(f"[{scheme}] {counts}  title={page.title()!r}")
            if errs:
                failures.append(f"{scheme}: {errs[:5]}")
            if counts["satellites"] < a.min_satellites:
                failures.append(f"{scheme}: only {counts['satellites']} orbit panels")

            for w in WIDTHS:
                page.set_viewport_size({"width": w, "height": 900})
                page.wait_for_timeout(350)
                sw = page.evaluate("document.documentElement.scrollWidth")
                ok = sw <= w + 2
                print(f"    {w}px -> scrollWidth {sw} {'ok' if ok else 'HORIZONTAL SCROLL'}")
                if not ok:
                    failures.append(f"{scheme}: horizontal scroll at {w}px ({sw})")
            page.close()
        browser.close()

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nok")


if __name__ == "__main__":
    main()
