#!/usr/bin/env python3
"""End-to-end check of the web application: upload a PDF, wait, open the report.

This is the test that catches the failures nothing else does — a job that runs
but whose report never loads, a progress bar that never reaches the end, an
upload form that rejects a valid file. It drives a real browser against a
running server.

    python scripts/dev/ui_smoke.py --url http://127.0.0.1:8080 \
        --pdf /work/samples/BODAchievements_2012.pdf --profile fast

Requires the optional `shot` environment and a running server (`make up`).
"""
import argparse, sys, time


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--pdf", default="/work/samples/BODAchievements_2012.pdf")
    ap.add_argument("--profile", default="fast")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the job")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright
    problems = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errs.append(f"console: {m.text}")
              if m.type == "error" else None)

        pg.goto(a.url)
        pg.wait_for_selector("#profiles .opt", timeout=20000)
        names = pg.eval_on_selector_all("#profiles .opt b", "e=>e.map(x=>x.textContent.trim())")
        print("profiles:", names)

        pg.click(f"#profiles .opt:has-text('{a.profile}')")
        pg.set_input_files("#fileInput", a.pdf)
        pg.wait_for_timeout(400)
        if not pg.is_enabled("#goBtn"):
            problems.append("submit button stayed disabled after choosing a file")
        pg.click("#goBtn")
        pg.wait_for_url("**/jobs/**", timeout=30000)
        print("job url:", pg.url)

        deadline = time.time() + a.timeout
        state = ""
        while time.time() < deadline:
            state = pg.inner_text("#jobState").strip().lower()
            if state in ("done", "failed"):
                break
            time.sleep(2)
        print("final state:", state, "|", pg.inner_text("#jobSub"))
        if state != "done":
            problems.append(f"job did not finish cleanly (state={state or 'timeout'})")

        pg.wait_for_timeout(5000)
        src = pg.eval_on_selector("#reportFrame", "e=>e.getAttribute('src')")
        sats = pg.frame_locator("#reportFrame").locator(".sat").count() if src else 0
        print("report:", src, "| orbit panels:", sats)
        if not src:
            problems.append("report iframe was never given a source")
        if sats < 1:
            problems.append("report loaded but drew no orbit panels")

        for w in (1400, 900, 500, 390):
            pg.set_viewport_size({"width": w, "height": 900})
            pg.wait_for_timeout(300)
            sw = pg.evaluate("document.documentElement.scrollWidth")
            print(f"  {w}px -> {sw} {'ok' if sw <= w + 2 else 'HORIZONTAL SCROLL'}")
            if sw > w + 2:
                problems.append(f"horizontal scroll at {w}px")
        if errs:
            problems.append(f"javascript errors: {errs[:5]}")
        b.close()

    if problems:
        print("\nFAILED:")
        for x in problems:
            print("  -", x)
        sys.exit(1)
    print("\nok")


if __name__ == "__main__":
    main()
