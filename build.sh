#!/usr/bin/env bash
# Rebuild every figure and the paper, then run all three gates.
#
# latexmk is deliberately not used: it is a Perl script and this host has no Perl.
# `set -e` is deliberately not used either: MiKTeX writes an update nag to stderr and
# returns non-zero on runs that produced a perfectly good PDF, which aborted the script
# halfway with no diagnostic. Each step is checked explicitly instead.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"

echo "== figures =="
python tools/make_figures.py        || { echo "FAILED: make_figures"; exit 1; }
python tools/make_diagrams.py       || { echo "FAILED: make_diagrams"; exit 1; }
python tools/make_result_figures.py || { echo "FAILED: make_result_figures"; exit 1; }

echo "== paper =="
rm -f paper.aux paper.bbl paper.blg paper.log paper.out
pdflatex -interaction=nonstopmode -file-line-error paper.tex > /dev/null 2>&1
bibtex   paper                                              > /dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error paper.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error paper.tex > /dev/null 2>&1
[ -f paper.pdf ] || { echo "FAILED: no paper.pdf produced"; exit 1; }
printf "  errors=%s undefined=%s overfull=%s underfull=%s\n" \
  "$(grep -c '^!' paper.log)" "$(grep -ci 'undefined' paper.log)" \
  "$(grep -c 'Overfull' paper.log)" "$(grep -c 'Underfull' paper.log)"
grep -o "Output written.*" paper.log

echo "== gates =="
python tools/check_tex.py paper.tex | grep -E "RESULT|dangling|uncited"
python tools/check_claims.py        | tail -1
printf "  em dashes in paper.tex: %s\n" "$(grep -c -- '---' paper.tex)"
