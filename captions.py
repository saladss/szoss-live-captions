#!/usr/bin/env python3
"""SZOSS live captions: English <-> Mandarin speech to on-screen captions.

    python captions.py                  start captioning from the microphone
    python captions.py devices          list microphones / audio inputs
    python captions.py check            record a few seconds and test the whole chain
    python captions.py download         fetch the models for offline use (do this before travelling)
    python captions.py --file talk.mp3  rehearse with a recording instead of the mic

Then open http://127.0.0.1:8765 for the caption screen and /control for the operator page.
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import time
import webbrowser

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # Chinese text on Windows consoles
    except Exception:
        pass

from livecaptions import __version__
from livecaptions.config import FROZEN, load_config, resolve_path


def setup_windows_console():
    """Title the window, and turn off Quick Edit: otherwise a stray click in the console
    window freezes all output (and with it the captions) until a key is pressed."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        k32.SetConsoleTitleW("SZOSS Live Captions")
        handle = k32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = wintypes.DWORD()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            k32.SetConsoleMode(handle, (mode.value | 0x0080) & ~0x0040)  # EXTENDED_FLAGS on, QUICK_EDIT off
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live English <-> Mandarin captions from a microphone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("command", nargs="?", default="run", choices=["run", "devices", "check", "download"])
    p.add_argument("--config", help="settings file (default: config.toml next to this script)")
    p.add_argument("--file", help="use an audio/video file instead of the microphone")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed for --file (default 1.0)")
    p.add_argument("--lang", choices=["auto", "en", "zh"], help="starting language mode")
    p.add_argument("--device", help="microphone name (part of it) or number")
    p.add_argument("--channel", type=int, help="input channel on a multi-channel interface (0 = first)")
    p.add_argument("--gain", type=float, help="extra input gain")
    p.add_argument("--engine", choices=["auto", "mlx", "faster-whisper", "mock"], help="speech engine")
    p.add_argument("--model", help="Whisper model name or folder")
    p.add_argument("--translator", help="translation backends to use, comma separated (e.g. ollama,deepseek)")
    p.add_argument("--no-translate", action="store_true", help="captions in the spoken language only")
    p.add_argument("--host", help="web server address (default from config)")
    p.add_argument("--port", type=int, help="web server port (default 8765)")
    p.add_argument("--open", action="store_true", help="open the caption screen and control page in the browser")
    p.add_argument("--no-open", action="store_true", help="don't open the browser (the .exe opens it by default)")
    p.add_argument("--seconds", type=float, default=8.0, help="check: how long to record (default 8)")
    p.add_argument("--exit-when-done", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--version", action="version", version=f"szoss-live-captions {__version__}")
    return p


def apply_overrides(cfg: dict, a) -> dict:
    if a.lang:
        cfg["asr"]["language"] = a.lang
    if a.device is not None:
        cfg["audio"]["device"] = a.device
    if a.channel is not None:
        cfg["audio"]["channel"] = a.channel
    if a.gain is not None:
        cfg["audio"]["gain"] = a.gain
    if a.engine:
        cfg["asr"]["engine"] = a.engine
    if a.model:
        cfg["asr"]["model"] = a.model
    if a.translator:
        cfg["translation"]["order"] = [t.strip() for t in a.translator.split(",") if t.strip()]
    if a.no_translate:
        cfg["translation"]["order"] = []
    if a.host:
        cfg["server"]["host"] = a.host
    if a.port:
        cfg["server"]["port"] = a.port
    return cfg


# --------------------------------------------------------------------------- devices

def cmd_devices(cfg):
    from livecaptions.audio import list_input_devices

    devs = list_input_devices()
    if not devs:
        print("No audio inputs found. Is a microphone connected, and does Python have microphone permission?")
        return
    print("Audio inputs (use the number or part of the name as audio.device in config.toml):\n")
    for d in devs:
        mark = "  <- system default" if d["default"] else ""
        print(f"  {d['index']:>3}  {d['name']}  [{d['channels']} ch, {d['samplerate']} Hz, {d['hostapi']}]{mark}")


# --------------------------------------------------------------------------- download

def cmd_download(cfg):
    from livecaptions.asr import default_model, local_model_dir, pick_engine

    engine = pick_engine(cfg["asr"]["engine"])
    model = cfg["asr"]["model"]
    if model == "auto":
        model = default_model(engine, cfg["asr"].get("device", "auto"))
    target = local_model_dir(engine, model)
    ok = True
    print(f"Speech model: {engine} / {model}\n  -> {target}")
    if os.environ.get("HF_ENDPOINT"):
        print(f"  (using HF_ENDPOINT={os.environ['HF_ENDPOINT']})")
    try:
        if engine == "mlx":
            from huggingface_hub import snapshot_download

            snapshot_download(model, local_dir=str(target))
        else:
            from faster_whisper import download_model

            download_model(model, output_dir=str(target))
        print("  done")
    except Exception as e:
        ok = False
        print(f"  FAILED: {e}")
        print("  In mainland China, try again with: HF_ENDPOINT=https://hf-mirror.com python captions.py download")

    backends = cfg["translation"].get("backends", {})
    wanted = [n for n in cfg["translation"].get("order", []) if backends.get(n, {}).get("type") == "ollama"]
    for name in wanted:
        m = backends[name].get("model")
        print(f"\nTranslation model: Ollama / {m}")
        if not shutil.which("ollama"):
            ok = False
            print("  Ollama is not installed. Get it from https://ollama.com, then run this again.")
            continue
        if subprocess.run(["ollama", "pull", m]).returncode != 0:
            ok = False
    if ok:
        print("\nAll set. Run `python captions.py check` to test.")
    else:
        sys.exit("\nSome downloads failed (see above).")


# --------------------------------------------------------------------------- check

def cmd_check(cfg, args):
    import numpy as np

    from livecaptions.asr import make_asr
    from livecaptions.audio import SAMPLE_RATE, FileSource, MicSource
    from livecaptions.segmenter import make_vad
    from livecaptions.translate import TranslationError, Translator

    print("Loading the speech model…")
    t0 = time.monotonic()
    asr = make_asr(cfg["asr"])
    print(f"  loaded in {time.monotonic() - t0:.1f} s ({asr.engine}, {asr.model_name}, {asr.device})")

    q: queue.Queue = queue.Queue()
    a = cfg["audio"]
    if args.file:
        src = FileSource(q, args.file, speed=100, gain=a["gain"])
    else:
        src = MicSource(q, a["device"], a["channel"], a["gain"], hostapi=a.get("device_api", ""), strict=False)
    src.start()
    secs = args.seconds
    if not args.file:
        print(f"\nSpeak now for {secs:.0f} seconds (English or Chinese)…")
    chunks, got, peak = [], 0, -90.0
    deadline = time.monotonic() + secs + 5
    while got < secs * SAMPLE_RATE and time.monotonic() < deadline:
        try:
            c = q.get(timeout=1)
        except queue.Empty:
            continue
        if c is None:
            break
        chunks.append(c)
        got += len(c)
        db = 20 * np.log10(np.sqrt(np.mean(c**2)) + 1e-9)
        peak = max(peak, db)
        if not args.file:
            bar = "#" * int(max(0, db + 60) / 2)
            print(f"\r  level {db:6.1f} dB |{bar:<30}|", end="", flush=True)
    src.stop()
    audio = np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE, dtype=np.float32)
    print(f"\n  recorded {len(audio) / SAMPLE_RATE:.1f} s, loudest moment {peak:.0f} dB")
    if peak < -40:
        print("  !! Very quiet. Check the mic is selected (python captions.py devices) and raise audio.gain.")

    vad = make_vad()
    probs = [vad(audio[i : i + 512]) for i in range(0, len(audio) - 511, 512)]
    speech = sum(p > 0.5 for p in probs) * 0.032
    print(f"  voice detected for {speech:.1f} s ({vad.name})")

    print("\nRecognising…")
    res = asr.transcribe(audio[: 29 * SAMPLE_RATE], cfg["asr"].get("language", "auto"), None, final=True)
    rtf = res.elapsed_s / max(res.audio_s, 0.1)
    print(f"  [{res.lang}, {res.lang_prob:.0%}] {res.text or '(nothing recognised)'}")
    print(f"  took {res.elapsed_s:.2f} s for {res.audio_s:.1f} s of audio ({rtf:.2f}x real time)")
    if rtf > 0.5:
        print("  !! Slow for live use. Try a smaller model (asr.model = \"small\" or \"base\") or a faster computer.")

    tr = Translator(cfg["translation"], cfg.get("glossary", {}))
    if tr.enabled and res.text:
        print("\nTranslating…")
        other = "zh" if res.lang == "en" else "en"
        try:
            out, name, s = tr.translate(res.text, res.lang, other, [])
            print(f"  [{other}] {out}\n  took {s:.2f} s via {name}")
        except TranslationError as e:
            print(f"  FAILED: {e}")
    elif not tr.enabled:
        print("\nTranslation is off (translation.order is empty).")
    tr.close()
    print("\nDone.")


# --------------------------------------------------------------------------- run

def cmd_run(cfg, args):
    from livecaptions.audio import FileSource, MicSource
    from livecaptions.hub import Hub
    from livecaptions.pipeline import Pipeline
    from livecaptions.server import CaptionServer

    d = cfg["display"]
    hub = Hub(
        {
            "mode": cfg["asr"].get("language", "auto"),
            "paused": False,
            "message": "",
            "title": d.get("title", ""),
            "note": d.get("note", ""),
            "order": d.get("order", ["zh", "en"]),
            "lines": int(d.get("lines", 3)),
        }
    )
    a = cfg["audio"]
    if args.file:
        factory = lambda q: FileSource(q, args.file, speed=args.speed, gain=a["gain"])
    else:
        factory = lambda q: MicSource(q, a["device"], a["channel"], a["gain"], hostapi=a.get("device_api", ""), strict=False)
    log_folder = resolve_path(cfg["log"]["folder"]) if cfg["log"].get("folder") else None
    pipeline = Pipeline(cfg, hub, factory, log_folder)

    port = int(cfg["server"]["port"])
    try:
        server = CaptionServer(hub, pipeline, cfg["server"]["host"], port, cfg.get("_path"))
    except OSError as e:
        if FROZEN and not args.no_open:
            webbrowser.open(f"http://127.0.0.1:{port}/control")
        sys.exit(
            f"Can't start the web server on port {port}: {e}.\n"
            "Captions are probably already running in another window. Close that one first, "
            "or set a different port in config.toml."
        )
    server.start()
    urls = server.urls()

    print(f"SZOSS live captions {__version__}. Loading models, please wait…")
    pipeline.start()
    print("\n" + "=" * 64)
    print(f"  Caption screen:  {urls['display']}")
    print(f"  Control page:    {urls['control']}")
    if "lan" in urls:
        print(f"  Phones on Wi-Fi: {urls['lan']}")
        print(f"  Remote control:  {urls['lan_control']}")
    if pipeline.log:
        print(f"  Transcript:      {pipeline.log.txt_path}")
    print("  Keys on the caption screen: A auto · E English · C Chinese · Space pause · ? help")
    print("  Stop with Ctrl+C")
    print("=" * 64 + "\n")
    if args.open or (FROZEN and not args.no_open and not args.file):
        webbrowser.open(urls["control"])
        webbrowser.open(urls["display"])

    try:
        while True:
            time.sleep(0.5)
            if args.exit_when_done and pipeline.source_done.is_set() and pipeline.idle():
                time.sleep(0.5)
                if pipeline.idle():
                    break
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        pipeline.stop()
        server.stop()


def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    if args.command == "devices":
        cmd_devices(cfg)
    elif args.command == "download":
        cmd_download(cfg)
    elif args.command == "check":
        cmd_check(cfg, args)
    else:
        cmd_run(cfg, args)


def pause_before_closing():
    """When the .exe was double-clicked, keep the window open so the message can be read."""
    if FROZEN and sys.stdin and sys.stdin.isatty():
        try:
            input("\nPress Enter to close this window…")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    if FROZEN:
        import multiprocessing

        multiprocessing.freeze_support()
    setup_windows_console()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"\n{e.code}" if isinstance(e.code, str) else f"\nStopped (code {e.code}).", file=sys.stderr)
            pause_before_closing()
            sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        print("\nSomething went wrong (details above).", file=sys.stderr)
        pause_before_closing()
        sys.exit(1)
