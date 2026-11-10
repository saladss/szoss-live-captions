"""Runs the bundled llama.cpp server (llama-server) with a local translation model.

Layout next to the app (the Windows build ships both):
    llama/vulkan/llama-server.exe   uses the graphics card through Vulkan
    llama/cpu/llama-server.exe      plain CPU build
A folder with llama-server directly inside it also works.

With gpu = "auto" the Vulkan build is tried first and the CPU build is used if
that doesn't start or answer. The server process is tied to this app, so it
is stopped when the app exits, even if the console window is just closed.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from .config import resolve_path

EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _kill_with_parent(proc: subprocess.Popen):
    """Windows: put the child in a job object that is killed when this process ends."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC),
                ("IoInfo", IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # ExtendedLimitInformation
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            return None
        return job  # keep the handle open for the life of the app
    except Exception as e:  # pragma: no cover - Windows only
        print(f"[translate] note: could not tie llama-server to the app ({e})")
        return None


def find_model(spec: str) -> Path | None:
    p = resolve_path(spec) if spec else None
    if p and p.is_file():
        return p
    folder = p.parent if p else resolve_path("models")
    for cand in sorted(folder.glob("*.gguf")) if folder.is_dir() else []:
        return cand
    return None


def find_servers(spec: str, gpu: str) -> list[tuple[str, Path]]:
    """Candidate (mode, executable) pairs in the order to try."""
    base = resolve_path(spec or "llama")
    if base.is_file():
        return [("custom", base)]
    found = []
    if gpu != "off" and (base / "vulkan" / EXE).is_file():
        found.append(("gpu", base / "vulkan" / EXE))
    if (base / "cpu" / EXE).is_file():
        found.append(("cpu", base / "cpu" / EXE))
    if (base / EXE).is_file():
        found.append(("cpu" if gpu == "off" else "auto", base / EXE))
    return found


class LlamaServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = find_model(cfg.get("model", ""))
        self.candidates = find_servers(cfg.get("server", "llama"), str(cfg.get("gpu", "auto")).lower())
        threads = int(cfg.get("threads", 0) or 0)
        self.threads = threads if threads > 0 else max(2, (os.cpu_count() or 4) // 2)
        self.context = int(cfg.get("context", 4096))
        self.load_timeout = float(cfg.get("load_timeout_s", 180))
        self.proc: subprocess.Popen | None = None
        self.mode = ""
        self.gpu_name = ""
        self.url = ""
        self._log_path: Path | None = None
        self.error = ""
        self._job = None
        self._log = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._starter: threading.Thread | None = None
        atexit.register(self.stop)

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        """Start loading in the background; wait_ready() blocks until it answers."""
        with self._lock:
            if self._starter and self._starter.is_alive():
                return
            self._ready.clear()
            self._starter = threading.Thread(target=self._start_any, name="llama-start", daemon=True)
            self._starter.start()

    def wait_ready(self, timeout: float) -> bool:
        if self._ready.is_set() and self.alive():
            return True
        if not (self._starter and self._starter.is_alive()):
            self.start()
        self._starter.join(timeout)
        return self._ready.is_set() and self.alive()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        p, self.proc = self.proc, None
        self._ready.clear()
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if self._log:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None

    # -- internals ------------------------------------------------------------

    def where(self) -> str:
        if self.mode == "gpu":
            return f"graphics card ({self.gpu_name})" if self.gpu_name else "graphics card"
        return {"cpu": "CPU"}.get(self.mode, self.mode)

    def _log_file(self):
        for folder in (resolve_path("logs"), Path(tempfile.gettempdir())):
            try:
                folder.mkdir(parents=True, exist_ok=True)
                self._log_path = folder / "llama-server.log"
                return open(self._log_path, "w", encoding="utf-8", errors="replace")
            except OSError:
                continue
        self._log_path = None
        return subprocess.DEVNULL

    def _actual_mode(self, mode: str) -> str:
        """The Vulkan build quietly runs on the CPU when there is no usable graphics card."""
        if mode == "cpu" or not self._log_path:
            return mode
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return mode
        import re

        gpu = re.search(r"ggml_vulkan: \d+ = ([^|\r\n(]+)", text)
        off = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
        if gpu:
            self.gpu_name = gpu.group(1).strip()
        if off and int(off[-1][0]) > 0:
            return "gpu"
        return "cpu" if off or not gpu else mode

    def _start_any(self):
        if self.model is None:
            self.error = "no translation model (.gguf) found in the models folder"
            print(f"[translate] {self.error}")
            return
        if not self.candidates:
            self.error = f"llama-server not found (looked in {resolve_path(self.cfg.get('server', 'llama'))})"
            print(f"[translate] {self.error}")
            return
        for mode, exe in self.candidates:
            t0 = time.monotonic()
            if self._launch(mode, exe) and self._probe():
                self.mode = self._actual_mode(mode)
                self.error = ""
                self._ready.set()
                print(f"[translate] built-in translator ready on the {self.where()} in {time.monotonic() - t0:.1f} s ({self.model.name})")
                return
            self.stop()
            print(f"[translate] llama-server ({mode}) didn't work: {self.error}; trying the next option")
        print(f"[translate] built-in translator could not start: {self.error}")

    def _launch(self, mode: str, exe: Path) -> bool:
        port = _free_port()
        args = [
            str(exe),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(self.context),
            "-t", str(self.threads),
            "-np", "1",
            "-ngl", "0" if mode == "cpu" else "99",
            "--jinja",
        ]
        args += [str(a) for a in self.cfg.get("extra_args", [])]
        self._log = self._log_file()
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        try:
            self.proc = subprocess.Popen(
                args,
                cwd=str(exe.parent),
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        except OSError as e:
            self.error = f"could not run {exe}: {e}"
            return False
        self._job = _kill_with_parent(self.proc)
        self.url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + self.load_timeout
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.error = f"llama-server exited with code {self.proc.returncode} (see logs/llama-server.log)"
                return False
            try:
                with opener.open(self.url + "/health", timeout=2) as r:
                    if r.status == 200 and json.loads(r.read() or b"{}").get("status", "ok") == "ok":
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        self.error = f"llama-server didn't load the model within {self.load_timeout:.0f} s"
        return False

    def _probe(self) -> bool:
        """One tiny completion, to catch a GPU driver that starts but can't compute."""
        body = {
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 8,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=60) as r:
                res = json.loads(r.read())
            res["choices"][0]["message"]
            return True
        except Exception as e:
            self.error = f"test request failed: {e}"
            return False
