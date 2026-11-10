#!/usr/bin/env bash
# Start captioning and open the caption screen + control page. Double-click on a Mac.
cd "$(dirname "$0")"
exec .venv/bin/python captions.py --open "$@"
