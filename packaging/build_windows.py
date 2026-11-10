"""Build, test and publish the Windows package. Runs on a Windows machine (GitHub Actions).

    python packaging/build_windows.py build     fetch models + llama.cpp, PyInstaller, zip
    python packaging/build_windows.py test      unzip the package and run it on real speech
    python packaging/build_windows.py release   publish the zip as a GitHub release

Settings via environment variables (all optional):
    WHISPER_REPO       Hugging Face repo of the CTranslate2 Whisper model (Systran/faster-whisper-small)
    TRANSLATION_GGUF   "repo:filename-pattern" list, comma separated, tried in order
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build-out"
APP_NAME = "SZOSS Captions"
APP = OUT / APP_NAME
ZIP = OUT / "SZOSS-Captions-win64.zip"
UPDATE_ZIP = OUT / "SZOSS-Captions-update.zip"  # program files only, to unzip over an existing folder
LOGS = OUT / "test-logs"
WHISPER_REPO = os.environ.get("WHISPER_REPO", "Systran/faster-whisper-small")
GGUF_CHOICES = os.environ.get(
    "TRANSLATION_GGUF",
    "Qwen/Qwen3-1.7B-GGUF:*Q4_K_M.gguf,unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q4_K_M.gguf,"
    "bartowski/Qwen_Qwen3-1.7B-GGUF:*Q4_K_M.gguf,Qwen/Qwen2.5-1.5B-Instruct-GGUF:*q4_k_m.gguf",
)
MAX_ZIP = 1_990_000_000  # GitHub release assets must be under 2 GiB


def log(msg: str):
    print(f"\n=== {msg}", flush=True)


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


# =========================================================================== build

def fetch_whisper(models: Path):
    from huggingface_hub import snapshot_download

    target = models / "faster-whisper-small"
    log(f"Whisper model {WHISPER_REPO} -> {target}")
    snapshot_download(
        WHISPER_REPO,
        local_dir=str(target),
        allow_patterns=["*.bin", "*.json", "*.txt"],
    )
    for junk in target.glob(".cache"):
        shutil.rmtree(junk, ignore_errors=True)
    print(f"  {size_mb(target):.0f} MB: {sorted(p.name for p in target.iterdir())}")


def fetch_gguf(models: Path) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for choice in [c.strip() for c in GGUF_CHOICES.split(",") if c.strip()]:
        repo, pattern = choice.split(":", 1)
        try:
            files = api.list_repo_files(repo)
        except Exception as e:
            print(f"  {repo}: {e}")
            continue
        match = sorted(f for f in files if fnmatch.fnmatch(f.split("/")[-1], pattern))
        if not match:
            print(f"  {repo}: no file matching {pattern}")
            continue
        fname = match[0]
        log(f"Translation model {repo}/{fname}")
        path = Path(hf_hub_download(repo, fname, local_dir=str(models)))
        final = models / Path(fname).name
        if path != final:
            shutil.move(str(path), final)
        shutil.rmtree(models / ".cache", ignore_errors=True)
        print(f"  {size_mb(final):.0f} MB")
        return final.name
    raise SystemExit("Could not download any translation model")


CPU_RX = r"bin-win-cpu-x64\.zip$"


def gh_json(*args):
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def find_llama_release(dl: Path) -> dict:
    """The newest llama.cpp release that has Windows binaries (the 'latest' one may not)."""
    repo = ["--repo", "ggml-org/llama.cpp"]
    candidates = []
    latest = gh_json("release", "view", *repo, "--json", "tagName,assets")
    if any(re.search(CPU_RX, a["name"]) for a in latest["assets"]):
        return latest
    print(f"  latest release {latest['tagName']} has no Windows build: {[a['name'] for a in latest['assets']]}")
    if any(a["name"] == "nightly-tag.txt" for a in latest["assets"]):
        subprocess.run(["gh", "release", "download", latest["tagName"], *repo, "--pattern", "nightly-tag.txt",
                        "--dir", str(dl), "--clobber"], check=False)
        hint = (dl / "nightly-tag.txt")
        if hint.exists():
            tag = hint.read_text(encoding="utf-8", errors="replace").strip().split()[0] if hint.read_text().strip() else ""
            print(f"  nightly-tag.txt says: {tag!r}")
            if tag:
                candidates.append(tag)
    releases = gh_json("release", "list", *repo, "--limit", "40", "--json", "tagName,publishedAt")
    print("  recent releases:", [r["tagName"] for r in releases[:15]])
    candidates += [r["tagName"] for r in releases]
    seen, looked = set(), []
    for tag in candidates:
        if tag in seen:
            continue
        seen.add(tag)
        try:
            info = gh_json("release", "view", tag, *repo, "--json", "tagName,assets")
        except subprocess.CalledProcessError:
            continue
        if any(re.search(CPU_RX, a["name"]) for a in info["assets"]):
            return info
        looked.append((tag, [a["name"] for a in info["assets"]][:12]))
    for tag, names in looked[:4]:
        print(f"  {tag}: {names}")
    raise SystemExit("No llama.cpp release with Windows binaries found")


def fetch_llama(dest: Path) -> str:
    """llama.cpp: CPU and Vulkan builds of llama-server for Windows x64."""
    dl = OUT / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    info = find_llama_release(dl)
    tag = info["tagName"]
    log(f"llama.cpp {tag}")
    wanted = {"cpu": CPU_RX, "vulkan": r"bin-win-vulkan-x64\.zip$"}
    for kind, rx in wanted.items():
        asset = next((a["name"] for a in info["assets"] if re.search(rx, a["name"])), None)
        if not asset:
            names = ", ".join(a["name"] for a in info["assets"])
            if kind == "cpu":
                raise SystemExit(f"No CPU Windows build in llama.cpp {tag}: {names}")
            print(f"  no {kind} build in this release; skipping")
            continue
        run(["gh", "release", "download", tag, "--repo", "ggml-org/llama.cpp", "--pattern", asset, "--dir", dl, "--clobber"])
        tmp = dl / kind
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(dl / asset) as z:
            z.extractall(tmp)
        exe = next(tmp.rglob("llama-server.exe"))
        target = dest / kind
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        # keep the server and the libraries it loads; skip the other tools
        for f in exe.parent.iterdir():
            if f.is_file() and (f.suffix.lower() == ".dll" or f.name in ("llama-server.exe", "LICENSE")):
                shutil.copy2(f, target / f.name)
        print(f"  {kind}: {size_mb(target):.0f} MB")
    (dest / "VERSION.txt").write_text(f"llama.cpp {tag}\n", encoding="utf-8")
    return tag


def windows_config(gguf_name: str) -> str:
    """config.toml for the packaged app, derived from the main one so the glossary stays in sync."""
    s = (ROOT / "config.toml").read_text(encoding="utf-8")

    def sub(old: str, new: str):
        nonlocal s
        if old not in s:
            raise SystemExit(f"windows_config: can't find {old!r}")
        s = s.replace(old, new, 1)

    sub(
        "# SZOSS live captions: settings\n# Edit this file, then restart the program. Most things can also be set on the\n# command line (python captions.py --help).",
        "# SZOSS live captions: settings for the Windows app\n# Edit this file with Notepad, then restart SZOSS Captions.exe.",
    )
    sub(
        '# Which microphone / audio interface to listen to. "" = the system default input.\n'
        '# Use part of the name ("Scarlett", "USB Audio") or the number shown by:\n'
        "#   python captions.py devices\n"
        'device = ""\n',
        '# Which microphone / audio interface to listen to. "" = the Windows default input.\n'
        "# Easiest: pick it from the list on the control page; the choice is saved here.\n"
        'device = ""\n'
        'device_api = ""\n',
    )
    sub('engine = "auto"            # auto | mlx | faster-whisper', 'engine = "faster-whisper"')
    sub(
        '# faster-whisper: tiny | base | small | medium | large-v3 | large-v3-turbo, or a folder.\n'
        "# mlx: a Hugging Face repo such as mlx-community/whisper-large-v3-turbo, or a folder.\n"
        '# "auto" picks large-v3-turbo on Apple Silicon or an NVIDIA GPU, and small on CPU only.\n'
        'model = "auto"\n',
        "# The bundled model is in models/faster-whisper-small. Other faster-whisper models can be\n"
        "# put in the models folder as models/faster-whisper-<name> and named here.\n"
        'model = "small"\n',
    )
    sub('device = "auto"            # faster-whisper only: auto | cpu | cuda', 'device = "cpu"')
    sub('compute_type = "default"   # faster-whisper only: default | int8 | float16 | int8_float16', 'compute_type = "int8"')
    sub("beam_size = 5              # faster-whisper only; 1 is faster, 5 is more accurate",
        "beam_size = 4              # 1 is fastest, 5 is most accurate")
    sub('order = ["ollama"]', 'order = ["builtin"]')
    sub("timeout_s = 8", "timeout_s = 20")
    sub(
        "# Local model, runs on this computer, no internet needed.",
        "# Built-in translator: a Qwen3 model run by llama.cpp, both included in this folder.\n"
        "# Works offline. gpu = \"auto\" tries the graphics card first (Vulkan), then the CPU;\n"
        "# set gpu = \"off\" if the graphics card misbehaves.\n"
        "[translation.backends.builtin]\n"
        'type = "llamacpp"\n'
        f'model = "models/{gguf_name}"\n'
        'server = "llama"\n'
        'gpu = "auto"\n'
        "threads = 0                # 0 = half the CPU threads\n"
        "context = 4096\n"
        "context_lines = 0          # earlier sentences sent along; small models do better without\n\n"
        "# Ollama, if installed separately (not needed for this app).",
    )
    return s


README_TXT = """SZOSS LIVE CAPTIONS for Windows
===============================

Live English <-> Mandarin captions. Everything needed is in this folder:
no Python, no installs, no internet.

START
  Double-click "SZOSS Captions.exe". A black window opens (that is the app;
  closing it stops the captions), the models load (~10-30 s) and your browser
  opens two pages:
    Control page    http://127.0.0.1:8765/control  (pick the mic, language, pause)
    Caption screen  http://127.0.0.1:8765          (drag to the projector, press F)

FIRST TIME ON A PC
  * "Windows protected your PC": click "More info" -> "Run anyway".
    (The app isn't signed with a paid certificate.)
  * Firewall question: "Allow" lets phones on the same Wi-Fi see the captions.
    "Cancel" is fine if you only use this PC's screen.
  * Microphone: on the control page, pick the input from the list and watch the
    level meter. If it says "No sound from this input", also check Windows
    Settings -> Privacy & security -> Microphone -> "Let desktop apps access
    your microphone".

ON THE DAY
  * Best audio: a feed from the venue's mixer into a USB audio interface, or a
    wireless mic receiver plugged into the PC.
  * Before each talk, on the caption screen press E (English speaker) or C
    (Chinese speaker). Press A (auto-detect) for Q&A.
  * Space = pause, X = clear the screen, +/- = text size, Q = QR code for phones,
    ? = all keys.
  * The control page can show a banner ("Break, back at 16:30") and hide a
    wrong caption.
  * Every caption is saved in the "transcripts" folder.

REHEARSE
  Drag a recording of a talk (mp3, mp4, wav...) onto
  "Rehearse with a recording.bat" to see the captions it would produce.
  "Test microphone.bat" records 8 seconds and shows what it heard.

SETTINGS
  config.toml (open with Notepad): glossary of fixed translations, speech
  hints, text order, online translation backups (DeepSeek / Alibaba Qwen).

IF SOMETHING GOES WRONG
  * Translations missing: the black window shows why; logs\\llama-server.log has
    details. In config.toml set gpu = "off" under [translation.backends.builtin]
    if the graphics card is the problem.
  * "port already in use": the app is already running in another window.
"""

THIRD_PARTY = """Third-party components in this package
* Whisper "small" speech recognition model, OpenAI, MIT licence (CTranslate2 conversion by Systran).
* Qwen3 translation model, Alibaba Qwen team, Apache 2.0 licence.
* llama.cpp (llama-server), ggml authors, MIT licence. See llama\\cpu\\LICENSE.
* faster-whisper, CTranslate2 (MIT), Silero VAD (MIT), ONNX Runtime (MIT), PyAV / FFmpeg (LGPL),
  PortAudio / python-sounddevice (MIT), NumPy (BSD), zhconv (MIT), qrcode (BSD), soxr (LGPL),
  Python (PSF licence).
"""

BAT_TEST = '@echo off\r\ncd /d "%~dp0"\r\n"SZOSS Captions.exe" check\r\necho.\r\npause\r\n'
BAT_REHEARSE = (
    '@echo off\r\ncd /d "%~dp0"\r\n'
    'if "%~1"=="" (\r\n  echo Drag an audio or video file onto this .bat file to rehearse with it.\r\n  pause\r\n  exit /b\r\n)\r\n'
    '"SZOSS Captions.exe" --file "%~1" --open\r\npause\r\n'
)


def cmd_build():
    shutil.rmtree(APP, ignore_errors=True)
    APP.mkdir(parents=True)
    models = APP / "models"
    models.mkdir()
    fetch_whisper(models)
    gguf = fetch_gguf(models)
    llama_tag = fetch_llama(APP / "llama")

    log("PyInstaller")
    work = OUT / "pyi"
    shutil.rmtree(work, ignore_errors=True)
    icon = ROOT / "packaging" / "windows" / "icon.ico"
    run(
        [
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--name", APP_NAME, "--console",
            *(["--icon", icon] if icon.exists() else []),
            "--distpath", work / "dist", "--workpath", work / "build", "--specpath", work,
            "--add-data", f"{ROOT / 'livecaptions' / 'web'}{os.pathsep}livecaptions/web",
            "--collect-submodules", "livecaptions",
            "--collect-data", "faster_whisper",
            "--collect-all", "ctranslate2",
            "--collect-data", "zhconv",
            "--collect-submodules", "qrcode",
            "--exclude-module", "tkinter", "--exclude-module", "matplotlib", "--exclude-module", "torch",
            "--exclude-module", "IPython", "--exclude-module", "pytest",
            ROOT / "captions.py",
        ],
        cwd=ROOT,
    )
    built = work / "dist" / APP_NAME
    for item in built.iterdir():
        dst = APP / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    (APP / "config.toml").write_text(windows_config(gguf), encoding="utf-8")
    (APP / "README.txt").write_text(README_TXT.replace("\n", "\r\n"), encoding="utf-8")
    (APP / "THIRD-PARTY.txt").write_text(THIRD_PARTY.replace("\n", "\r\n"), encoding="utf-8")
    (APP / "Test microphone.bat").write_text(BAT_TEST, encoding="ascii")
    (APP / "Rehearse with a recording.bat").write_text(BAT_REHEARSE, encoding="ascii")

    log("Zip")
    ZIP.unlink(missing_ok=True)
    seven = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"
    if Path(seven).exists():
        run([seven, "a", "-tzip", "-mx=5", "-mmt=on", ZIP, APP_NAME], cwd=OUT)
    else:
        shutil.make_archive(str(ZIP.with_suffix("")), "zip", OUT, APP_NAME)
    # small update package: the program without models, llama.cpp or settings
    UPDATE_ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(UPDATE_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(APP.rglob("*")):
            rel = f.relative_to(APP)
            if f.is_file() and rel.parts[0] not in ("models", "llama", "config.toml", "transcripts", "logs"):
                z.write(f, Path(APP_NAME) / rel)
    print(f"update zip: {size_mb(UPDATE_ZIP):.0f} MB")
    sizes = {p.name: round(size_mb(p)) for p in APP.iterdir()}
    print(json.dumps(sizes, indent=1))
    print(f"zip: {size_mb(ZIP):.0f} MB")
    (OUT / "build-info.json").write_text(
        json.dumps({"gguf": gguf, "whisper": WHISPER_REPO, "llama": llama_tag, "zip_mb": round(size_mb(ZIP)), "parts_mb": sizes}),
        encoding="utf-8",
    )
    if ZIP.stat().st_size > MAX_ZIP:
        raise SystemExit("zip is over GitHub's 2 GiB release limit")


# =========================================================================== test

SPEECH_EN = "Tiny Tapeout lets anyone put their own design on a real chip. Today we will talk about wafers and chip packaging."
SPEECH_ZH = "大家好，欢迎来到深圳开源芯片交流会。今天我们聊一聊开源芯片、流片和封装。"


def tts(text: str, lang: str, path: Path) -> bool:
    try:
        from gtts import gTTS

        gTTS(text, lang=lang).save(str(path))
        return path.exists() and path.stat().st_size > 1000
    except Exception as e:
        print(f"  gTTS {lang} failed: {e}")
        return False


def download(url: str, path: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, path)
        return path.stat().st_size > 1000
    except Exception as e:
        print(f"  download {url} failed: {e}")
        return False


def write_wav(path: Path, audio):
    import numpy as np

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def prepare_audio(folder: Path) -> dict:
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from livecaptions.audio import decode_file  # faster_whisper.decode_audio breaks on newer PyAV

    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    if download("https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac", folder / "jfk.flac"):
        files["jfk"] = folder / "jfk.flac"
    if tts(SPEECH_EN, "en", folder / "en.mp3"):
        files["en"] = folder / "en.mp3"
    if tts(SPEECH_ZH, "zh-CN", folder / "zh.mp3") or tts(SPEECH_ZH, "zh", folder / "zh.mp3"):
        files["zh"] = folder / "zh.mp3"
    elif download("https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", folder / "zh.wav"):
        files["zh"] = folder / "zh.wav"
    gap = np.zeros(int(16000 * 1.5), np.float32)
    parts = [decode_file(str(files[k])) for k in ("en", "zh", "jfk") if k in files]
    talk = np.concatenate([x for p in parts for x in (gap, p)] + [gap])
    write_wav(folder / "talk.wav", talk)
    files["talk"] = folder / "talk.wav"
    print({k: f"{v.name} {size_mb(v):.2f} MB" for k, v in files.items()})
    return files


def run_app(exe: Path, args: list, name: str, timeout: int = 900) -> str:
    t0 = time.monotonic()
    p = subprocess.run(
        [str(exe), *map(str, args)],
        cwd=exe.parent, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
    )
    out = p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")
    (LOGS / f"{name}.txt").write_text(out, encoding="utf-8")
    print(f"\n----- {name} (exit {p.returncode}, {time.monotonic() - t0:.0f} s)\n{out}", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"{name}: exit code {p.returncode}")
    return out


def llama_processes() -> int:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"], capture_output=True, text=True).stdout
    return out.lower().count("llama-server.exe")


def cmd_test():
    LOGS.mkdir(parents=True, exist_ok=True)
    test_dir = OUT / "test"
    shutil.rmtree(test_dir, ignore_errors=True)
    log("Unzip the package")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(test_dir)
    exe = test_dir / APP_NAME / f"{APP_NAME}.exe"
    assert exe.exists(), exe
    audio = prepare_audio(OUT / "audio")
    failures = []

    def expect(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    cjk = re.compile(r"[\u4e00-\u9fff]")
    run_app(exe, ["--version"], "version", timeout=120)

    if "jfk" in audio:
        out = run_app(exe, ["check", "--file", audio["jfk"], "--lang", "en", "--seconds", "12"], "check-en")
        expect("country" in out.lower(), "English speech recognised (JFK clip)")
        zh_line = next((l for l in out.splitlines() if l.strip().startswith("[zh]")), "")
        expect(bool(cjk.search(zh_line)), f"English -> Chinese translation: {zh_line.strip()}")
    if "zh" in audio:
        out = run_app(exe, ["check", "--file", audio["zh"], "--seconds", "15"], "check-zh-auto")
        heard = next((l for l in out.splitlines() if l.strip().startswith("[zh,")), "")
        expect(bool(cjk.search(heard)), f"Chinese speech recognised with auto-detect: {heard.strip()}")
        en_line = next((l for l in out.splitlines() if l.strip().startswith("[en]")), "")
        expect(bool(re.search(r"[A-Za-z]{3}", en_line)) and not cjk.search(en_line), f"Chinese -> English translation: {en_line.strip()}")
    expect(llama_processes() == 0, "llama-server stopped after check")

    log("Full pipeline on a mixed English/Chinese recording")
    run_app(exe, ["--file", audio["talk"], "--exit-when-done", "--no-open", "--port", "18765"], "run-talk")
    tdir = test_dir / APP_NAME / "transcripts"
    jsonl = sorted(tdir.glob("*.jsonl"))
    caps = [json.loads(l) for l in jsonl[-1].read_text(encoding="utf-8").splitlines()] if jsonl else []
    txt = sorted(tdir.glob("*.txt"))
    if txt:
        shutil.copy2(txt[-1], LOGS / "transcript.txt")
        print(txt[-1].read_text(encoding="utf-8"))
    done = [c for c in caps if c.get("status") == "done"]
    expect(len(caps) >= 3, f"{len(caps)} captions produced")
    expect(len(done) >= max(2, len(caps) - 1), f"{len(done)} of {len(caps)} captions translated")
    expect({"en", "zh"} <= {c["lang"] for c in caps}, "both languages detected in auto mode")
    if caps:
        asr = sorted(c["asr_ms"] / 1000 / max(c["audio_s"], 0.1) for c in caps)
        mt = sorted(c.get("mt_ms", 0) for c in done) or [0]
        print(f"speech: median {asr[len(asr) // 2]:.2f}x real time · translation: median {mt[len(mt) // 2]} ms, max {mt[-1]} ms")
        (LOGS / "timing.json").write_text(json.dumps({"asr_rtf": asr, "mt_ms": mt}), encoding="utf-8")
    time.sleep(2)
    expect(llama_processes() == 0, "llama-server stopped after the app exited")

    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))
    print("\nAll tests passed.")


# =========================================================================== release

def cmd_release():
    info = json.loads((OUT / "build-info.json").read_text(encoding="utf-8"))
    run_no = os.environ.get("GITHUB_RUN_NUMBER", time.strftime("%Y%m%d%H%M"))
    tag = f"windows-build-{run_no}"
    timing = ""
    t = LOGS / "timing.json"
    if t.exists():
        d = json.loads(t.read_text())
        timing = f"\nOn the GitHub test machine (4 CPU cores, no GPU): speech {d['asr_rtf'][len(d['asr_rtf']) // 2]:.2f}x real time, translation ~{d['mt_ms'][len(d['mt_ms']) // 2]} ms per caption.\n"
    transcript = (LOGS / "transcript.txt").read_text(encoding="utf-8") if (LOGS / "transcript.txt").exists() else ""
    notes = (
        "Windows build of SZOSS live captions. Download **SZOSS-Captions-win64.zip**, unzip it, "
        "and double-click **SZOSS Captions.exe**. No Python, installs or internet needed.\n\n"
        "Already have an earlier build? **SZOSS-Captions-update.zip** holds just the program: close the app, "
        "unzip it over your existing folder (replace files) and start it again. Models and your config.toml stay as they are.\n\n"
        f"Includes Whisper small, {info['gguf']} and {info['llama']}.\n{timing}\n"
        f"Test transcript from the automatic check:\n```\n{transcript.strip()}\n```\n"
    )
    (OUT / "notes.md").write_text(notes, encoding="utf-8")
    run(["gh", "release", "create", tag, ZIP, UPDATE_ZIP, "--title", f"SZOSS Captions for Windows (build {run_no})",
         "--notes-file", OUT / "notes.md", "--latest"], cwd=ROOT)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": cmd_build, "test": cmd_test, "release": cmd_release}[cmd]()
