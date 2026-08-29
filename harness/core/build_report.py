#!/usr/bin/env python3
"""Assemble a self-contained HTML report: template + data bundle + app.

Output is a single file with no external requests apart from web fonts: page
images are data URIs, the data bundle is inlined as JSON, and the stylesheet
and script are inlined.

`--template` selects the body template under harness/report. `viewer.html` is
the one that ships: the orbit graph plus the side-by-side comparison. Drop in
another template (and an optional narrative script that sets window.NARRATIVE)
to build a different report from the same bundle.

    python -m core.build_report
    python -m core.build_report --workspace data/jobs/<id>
    python -m core.build_report --template mine.html --narrative mine.js
"""
import argparse, json, os, sys

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402

DEFAULT_TEMPLATE = "viewer.html"
DEFAULT_TITLE = "Document Layout Analysis"


def build(ws=None, out=None, title=None, template=None, narrative=None):
    ws = ws or paths.WORKSPACE
    R = paths.REPORT_ASSETS
    template = template or paths.get("report", "template", DEFAULT_TEMPLATE)
    if not os.path.exists(os.path.join(R, template)):
        raise SystemExit(f"no such template: {os.path.join(R, template)}")

    def read(name):
        with open(os.path.join(R, name), encoding="utf8") as f:
            return f.read()

    body = read(template)
    head = read("head.html").replace("__STYLE__", read("style.css"))
    with open(os.path.join(ws, "reports", "report_data.json"), encoding="utf8") as f:
        data = f.read()
    with open(os.path.join(R, "app.js"), encoding="utf8") as f:
        app = f.read()

    # app.js reads window.NARRATIVE for optional prose blocks. Templates that
    # have no such blocks get an empty object rather than a code path.
    narr = "window.NARRATIVE = {};\n"
    if narrative:
        with open(narrative if os.path.isabs(narrative) else os.path.join(R, narrative),
                  encoding="utf8") as f:
            narr = f.read()

    title = title or DEFAULT_TITLE
    html = (head.replace("__TITLE__", title) + body + "\n</body>\n</html>\n")
    html = (html
            .replace("__DATA__", data)
            .replace("__NARRATIVE__", narr)
            .replace("__APP__", app))

    out = out or os.path.join(ws, "reports", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        f.write(html)
    print(f"wrote {out}  {os.path.getsize(out)/2**20:.2f} MB")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--template", default=None, help="body template under harness/report")
    ap.add_argument("--narrative", default=None, help="script that sets window.NARRATIVE")
    a = ap.parse_args()
    build(paths.resolve(a.workspace) if a.workspace else None,
          paths.resolve(a.out) if a.out else None,
          a.title, a.template, a.narrative)


if __name__ == "__main__":
    main()
