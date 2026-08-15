#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$ROOT/report"
mkdir -p "$REPORT/_build" "$REPORT/_rendered"
rm -f "$REPORT/_build/technical-retrospective.html" "$REPORT/technical-retrospective.pdf"
rm -f "$REPORT/_rendered"/*.png
(
  cd "$REPORT"
  pandoc technical-retrospective.md \
    --standalone \
    --citeproc \
    --toc \
    --toc-depth=2 \
    --metadata toc-title="Contents" \
    --embed-resources \
    --css report.css \
    --metadata title-prefix="OGC 2026 Grand Shipyard" \
    -o _build/technical-retrospective.html
)
wkhtmltopdf \
  --enable-local-file-access \
  --page-size Letter \
  --margin-top 12mm \
  --margin-bottom 14mm \
  --margin-left 13mm \
  --margin-right 13mm \
  --footer-center '[page] / [topage]' \
  --footer-font-size 8 \
  --footer-spacing 5 \
  "$REPORT/_build/technical-retrospective.html" \
  "$REPORT/technical-retrospective.pdf"
pdfinfo "$REPORT/technical-retrospective.pdf" > "$REPORT/_build/pdfinfo.txt"
pdftotext "$REPORT/technical-retrospective.pdf" "$REPORT/_build/technical-retrospective.txt"
pdftoppm -png -r 145 "$REPORT/technical-retrospective.pdf" "$REPORT/_rendered/page" >/dev/null 2>&1
