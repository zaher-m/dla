#!/usr/bin/env bash
# Optional: a headless browser for the report and UI smoke tests.
#
# Not needed to run an analysis. It exists because the report is a program, not
# a document -- it builds its own DOM from a data bundle -- so "the report is
# fine" is a claim that has to be checked by rendering it, not by reading the
# HTML. scripts/dev/verify_report.py and scripts/dev/ui_smoke.py use this env.
#
# Playwright pins a specific Chromium build per release, and the browser it
# downloads may need system libraries this image does not carry (libatk and
# friends). If the download succeeds but launching fails with a missing .so,
# the browser checks are simply unavailable -- nothing else is affected.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv shot)
pipi "$PY" "playwright"
"$PY" -m playwright install chromium || {
  echo "!! [shot] chromium download failed; browser-based checks unavailable" \
    | tee -a "$LOGS/setup_warnings.log"
}
"$PY" -c "from playwright.sync_api import sync_playwright; print('playwright ok')"
record_env "$PY" shot
