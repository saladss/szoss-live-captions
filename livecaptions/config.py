"""Load config.toml and merge it over built-in defaults."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("Python 3.11 or newer is needed (for tomllib). See README.md.")

FROZEN = bool(getattr(sys, "frozen", False))
# The folder holding config.toml, models/ and transcripts/: next to the .exe when
# packaged, otherwise the project folder.
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "audio": {"device": "", "channel": 0, "gain": 1.0},
    "segmenter": {
        "vad_threshold": 0.5,
        "end_silence_ms": 600,
        "min_speech_ms": 300,
        "max_caption_s": 12.0,
        "preroll_ms": 300,
        "partial_interval_s": 1.0,
    },
    "asr": {
        "engine": "auto",
        "model": "auto",
        "device": "auto",
        "compute_type": "default",
        "beam_size": 5,
        "language": "auto",
        "prompt_en": "",
        "prompt_zh": "",
    },
    "translation": {
        "order": [],
        "context_lines": 3,
        "timeout_s": 8,
        "topic": "a technical meetup",
        "backends": {},
    },
    "glossary": {},
    "display": {"title": "Live captions", "note": "", "order": ["zh", "en"], "lines": 3},
    "server": {"host": "0.0.0.0", "port": 8765},
    "log": {"folder": "transcripts"},
}


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        path = APP_DIR / "config.toml"
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    else:
        print(f"[config] {path} not found, using defaults")
    cfg["_path"] = str(path)
    return cfg


def resolve_path(p: str) -> Path:
    """Paths in the config are relative to the app folder."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else APP_DIR / path


def save_setting(path: str | os.PathLike, section: str, key: str, value: str) -> bool:
    """Set `key = "value"` inside [section] of a TOML file, keeping comments. Best effort."""
    import json
    import re

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return False
    newline = f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
    start, end = None, len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("["):
            if start is not None:
                end = i
                break
            if s == f"[{section}]":
                start = i
    if start is None:
        return False
    for i in range(start + 1, end):
        if re.match(rf"{re.escape(key)}\s*=", lines[i].strip()):
            lines[i] = newline
            break
    else:
        lines.insert(start + 1, newline)
    try:
        path.write_text("".join(lines), encoding="utf-8")
        return True
    except OSError:
        return False
