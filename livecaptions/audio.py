"""Audio sources: a live microphone, or an audio file played back in real time.

Both deliver mono float32 chunks at 16 kHz into a queue.Queue. A final ``None``
on the queue means the source has ended.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time

import numpy as np

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- resampling

class Resampler:
    """Streaming resampler to 16 kHz. Uses soxr when installed, else a simple
    low-pass + linear interpolation that is good enough for speech."""

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE):
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self._soxr = None
        self._pos = 0.0
        self._prev = np.zeros(0, dtype=np.float32)
        if self.in_rate == self.out_rate:
            return
        try:
            import soxr

            self._soxr = soxr.ResampleStream(self.in_rate, self.out_rate, 1, dtype="float32")
        except Exception:
            self._soxr = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.in_rate == self.out_rate:
            return x
        if self._soxr is not None:
            return self._soxr.resample_chunk(x)
        return self._fallback(x)

    def _fallback(self, x: np.ndarray) -> np.ndarray:
        ratio = self.in_rate / self.out_rate
        k = max(1, int(round(ratio)))  # moving-average anti-alias filter length
        buf = np.concatenate([self._prev, x])
        smooth = np.convolve(buf, np.ones(k, dtype=np.float32) / k, mode="same") if k > 1 else buf
        # only interpolate where the filter has seen enough future samples
        limit = len(buf) - 1 - k
        if limit < self._pos:
            self._prev = buf
            return np.zeros(0, dtype=np.float32)
        n_out = int((limit - self._pos) // ratio) + 1
        positions = self._pos + ratio * np.arange(n_out)
        out = np.interp(positions, np.arange(len(buf)), smooth).astype(np.float32)
        next_pos = self._pos + ratio * n_out
        keep_from = max(0, int(next_pos) - 2 * k)  # history for the filter
        self._prev = buf[keep_from:]
        self._pos = next_pos - keep_from
        return out


# --------------------------------------------------------------------------- devices

def _sd():
    try:
        import sounddevice

        return sounddevice
    except OSError as e:
        raise SystemExit(
            f"Can't use the microphone: {e}.\n"
            "On Linux install PortAudio (sudo apt install libportaudio2); on macOS/Windows "
            "reinstall with: pip install --force-reinstall sounddevice"
        )


def list_input_devices() -> list[dict]:
    sd = _sd()

    devices = []
    default_in = None
    try:
        default_in = sd.default.device[0]
    except Exception:
        pass
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "samplerate": int(dev.get("default_samplerate") or 0),
                    "hostapi": hostapis[dev["hostapi"]]["name"] if hostapis else "",
                    "default": idx == default_in,
                }
            )
    return devices


def find_device(spec, hostapi: str = "") -> int | None:
    """Turn a device number or (part of) a name into a sounddevice index.
    Exact name matches beat partial ones; a matching host API (MME, WASAPI…) is preferred."""
    if spec in (None, "", "default"):
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        return int(spec)
    want = str(spec).strip().lower()
    devs = list_input_devices()
    matches = [d for d in devs if d["name"].strip().lower() == want] or [d for d in devs if want in d["name"].lower()]
    if not matches:
        names = ", ".join(sorted({d["name"] for d in devs}))
        raise SystemExit(f"No input device matches '{spec}'. Available: {names}")
    if hostapi:
        preferred = [d for d in matches if d["hostapi"] == hostapi]
        matches = preferred or matches
    return matches[0]["index"]


# --------------------------------------------------------------------------- sources

class MicSource:
    def __init__(self, out: queue.Queue, device="", channel: int = 0, gain: float = 1.0, hostapi: str = "", strict: bool = True):
        self.out = out
        try:
            self.device = find_device(device, hostapi)
        except SystemExit as e:
            if strict:
                raise
            print(f"[audio] {e}\n[audio] using the default input instead")
            self.device = None
        self.channel = int(channel)
        self.gain = float(gain)
        self.stream = None
        self.name = ""
        self.status_errors = 0
        self._raw: queue.Queue = queue.Queue(maxsize=500)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        sd = _sd()
        info = sd.query_devices(self.device, "input")
        self.name = info["name"]
        rate = int(info["default_samplerate"]) or 48000
        channels = max(1, min(self.channel + 1, int(info["max_input_channels"])))
        if self.channel >= channels:
            print(f"[audio] channel {self.channel} not available on {self.name}; using channel 0")
            self.channel = 0
        self.resampler = Resampler(rate, SAMPLE_RATE)

        def callback(indata, frames, time_info, status):
            if status:
                self.status_errors += 1
            try:
                self._raw.put_nowait(indata[:, self.channel].copy())
            except queue.Full:
                pass

        self.stream = sd.InputStream(
            device=self.device,
            channels=channels,
            samplerate=rate,
            dtype="float32",
            blocksize=int(rate * 0.032),
            callback=callback,
        )
        self.stream.start()
        self._thread = threading.Thread(target=self._pump, name="mic-pump", daemon=True)
        self._thread.start()
        print(f"[audio] listening on: {self.name} ({rate} Hz, channel {self.channel})")

    def _pump(self):
        while not self._stop.is_set():
            try:
                block = self._raw.get(timeout=0.5)
            except queue.Empty:
                continue
            chunk = self.resampler(block)
            if self.gain != 1.0:
                chunk = np.clip(chunk * self.gain, -1.0, 1.0)
            if len(chunk):
                self.out.put(chunk)

    def stop(self):
        self._stop.set()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def decode_file(path: str) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32."""
    try:
        import av  # installed with faster-whisper

        out = []
        with av.open(path) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(stream):
                for f in resampler.resample(frame):
                    out.append(f.to_ndarray().reshape(-1))
            for f in resampler.resample(None):
                out.append(f.to_ndarray().reshape(-1))
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-nostdin", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()
    import soundfile  # last resort

    data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return Resampler(rate)(data[:, 0])


class FileSource:
    """Plays a file into the pipeline at real-time speed (or faster), for rehearsals."""

    def __init__(self, out: queue.Queue, path: str, speed: float = 1.0, gain: float = 1.0):
        self.out = out
        self.path = path
        self.speed = max(0.1, float(speed))
        self.gain = float(gain)
        self.name = f"file: {path}"
        self.status_errors = 0
        self._stop = threading.Event()

    def start(self):
        audio = decode_file(self.path)
        if self.gain != 1.0:
            audio = np.clip(audio * self.gain, -1.0, 1.0)
        # trailing silence so the last caption gets finished
        self.audio = np.concatenate([audio, np.zeros(SAMPLE_RATE * 2, dtype=np.float32)])
        print(f"[audio] playing {self.path} ({len(audio) / SAMPLE_RATE:.1f} s) at {self.speed}x")
        threading.Thread(target=self._run, name="file-source", daemon=True).start()

    def _run(self):
        step = int(SAMPLE_RATE * 0.032)
        t0 = time.monotonic()
        for i in range(0, len(self.audio), step):
            if self._stop.is_set():
                break
            self.out.put(self.audio[i : i + step])
            due = t0 + (i + step) / SAMPLE_RATE / self.speed
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.out.put(None)

    def stop(self):
        self._stop.set()
