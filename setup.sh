#!/usr/bin/env bash
# One-time setup on macOS or Linux. Run from this folder:  ./setup.sh
set -e
cd "$(dirname "$0")"
PY=""
for c in python3.13 python3.12 python3.11 python3.14 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11 or newer is needed. On a Mac: brew install python@3.12 (or get it from python.org)."
  exit 1
fi
echo "Using $($PY --version)"
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python captions.py download || true
echo
echo "Setup done. Next:"
echo "  .venv/bin/python captions.py devices   # pick the microphone"
echo "  .venv/bin/python captions.py check     # say a sentence in English, then in Chinese"
echo "  ./start.command                         # start captioning"
