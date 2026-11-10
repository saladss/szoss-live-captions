"""Speech recognition with Whisper: faster-whisper (any OS) or MLX (Apple Silicon).

Both backends restrict language detection to English vs Mandarin, so a short or
accented sentence is never captioned as Japanese or Cantonese, and they hold on
to the current speaker's language unless the evidence for a switch is strong.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import APP_DIR
from .textfix import clean_text, has_cjk, is_hallucination

SAMPLE_RATE = 16000
MODELS_DIR = APP_DIR / "models"
LANGS = ("en", "zh")


@dataclass
class ASRResult:
    text: str
    lang: str
    lang_prob: float
    audio_s: float
    elapsed_s: float
    reran: bool = False


def choose_language(probs: dict, current: str | None, duration: float) -> tuple[str, float]:
    """Pick en or zh from Whisper's language probabilities, with hysteresis."""
    en = float(probs.get("en", 0.0))
    zh = float(probs.get("zh", 0.0)) + float(probs.get("yue", 0.0))
    total = en + zh
    if total <= 1e-9:
        return (current or "en"), 0.0
    share = {"en": en / total, "zh": zh / total}
    best = max(share, key=share.get)
    if current in share and best != current:
        need = 0.8 if duration < 3.0 else 0.65
        if share[best] < need:
            return current, share[current]
    return best, share[best]


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def pick_engine(engine: str) -> str:
    if engine in ("mlx", "faster-whisper", "mock"):
        return engine
    if is_apple_silicon() and importlib.util.find_spec("mlx_whisper"):
        return "mlx"
    return "faster-whisper"


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def default_model(engine: str, device: str) -> str:
    if engine == "mlx":
        return "mlx-community/whisper-large-v3-turbo"
    if device == "cuda" or (device == "auto" and cuda_available()):
        return "large-v3-turbo"
    return "small"


def local_model_dir(engine: str, model: str) -> Path:
    """Where `captions.py download` puts a model."""
    if engine == "mlx":
        return MODELS_DIR / model.rstrip("/").split("/")[-1]
    return MODELS_DIR / ("faster-whisper-" + model.rstrip("/").split("/")[-1])


def resolve_model(engine: str, model: str) -> str:
    p = Path(model).expanduser()
    if p.is_dir():
        return str(p)
    local = local_model_dir(engine, model)
    if local.is_dir() and any(local.iterdir()):
        return str(local)
    return model


def _join(parts: list[str], lang: str) -> str:
    parts = [p.strip() for p in parts if p and p.strip()]
    if lang == "zh":
        return "".join(parts)
    return " ".join(parts)


def _segment_ok(no_speech: float, avg_logprob: float, compression: float) -> bool:
    if no_speech > 0.6 and avg_logprob < -1.0:
        return False
    if compression > 2.6:
        return False
    return True


# --------------------------------------------------------------------------- faster-whisper

class FasterWhisperASR:
    engine = "faster-whisper"

    def __init__(self, cfg: dict):
        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        compute = cfg.get("compute_type", "default")
        if compute in ("default", "auto", ""):
            compute = "float16" if device == "cuda" else "int8"
        model = cfg.get("model", "auto")
        if model == "auto":
            model = default_model("faster-whisper", device)
        path = resolve_model("faster-whisper", model)
        self.model_name = model
        self.device = device
        self.beam = max(1, int(cfg.get("beam_size", 5)))
        self.prompts = {"en": cfg.get("prompt_en") or None, "zh": cfg.get("prompt_zh") or None}
        t0 = time.monotonic()
        self.model = self._load(path, model, device, compute, int(cfg.get("cpu_threads", 0) or 0))
        if device == "cuda":
            try:  # a GPU without the CUDA libraries only fails on first use
                list(self.model.transcribe(np.zeros(16000, np.float32), language="en", beam_size=1)[0])
            except Exception as e:
                print(f"[asr] GPU not usable ({str(e).splitlines()[0]}); using the CPU instead")
                device, compute = "cpu", "int8"
                self.model = self._load(path, model, device, compute, int(cfg.get("cpu_threads", 0) or 0))
        self.device = device
        print(f"[asr] faster-whisper {model} on {device}/{compute} loaded in {time.monotonic() - t0:.1f} s")

    @staticmethod
    def _load(path, model, device, compute, threads):
        from faster_whisper import WhisperModel

        if device == "cpu" and threads <= 0:
            threads = min(8, os.cpu_count() or 4)
        kw = dict(device=device, compute_type=compute, cpu_threads=threads if device == "cpu" else 0)
        try:
            return WhisperModel(path, local_files_only=True, **kw)
        except Exception:
            print(f"[asr] {model} not found in the models folder; trying to download it (needs Hugging Face access)")
            return WhisperModel(path, **kw)

    def _run(self, audio, lang, prompt, final):
        opts = dict(
            beam_size=self.beam if final else 1,
            temperature=[0.0, 0.2, 0.4, 0.6] if final else 0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            initial_prompt=prompt,
        )
        return self.model.transcribe(audio, language=lang, **opts)

    def transcribe(self, audio: np.ndarray, mode: str, current: str | None, final: bool = True) -> ASRResult:
        t0 = time.monotonic()
        dur = len(audio) / SAMPLE_RATE
        reran = False
        if mode in LANGS:
            lang, prob = mode, 1.0
            segments, info = self._run(audio, lang, self.prompts[lang], final)
        else:
            guess = current if current in LANGS else "en"
            segments, info = self._run(audio, None, self.prompts[guess], final)
            probs = dict(info.all_language_probs or [(info.language, info.language_probability)])
            lang, prob = choose_language(probs, current, dur)
            if lang != info.language or lang != guess:
                # detected something else, or the hint prompt was in the wrong language
                segments, info = self._run(audio, lang, self.prompts[lang], final)
                reran = True
        parts = [
            s.text
            for s in segments
            if _segment_ok(s.no_speech_prob, s.avg_logprob, s.compression_ratio)
        ]
        text = clean_text(_join(parts, lang), lang)
        if text and is_hallucination(text):
            text = ""
        return ASRResult(text, lang, prob, dur, time.monotonic() - t0, reran)


# --------------------------------------------------------------------------- MLX (Apple Silicon)

class MlxWhisperASR:
    engine = "mlx"

    def __init__(self, cfg: dict):
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper import audio as mla
        from mlx_whisper.transcribe import ModelHolder

        self.mx, self.mlx_whisper, self.mla = mx, mlx_whisper, mla
        model = cfg.get("model", "auto")
        if model == "auto":
            model = default_model("mlx", "auto")
        path = resolve_model("mlx", model)
        if not Path(path).is_dir():
            try:
                from huggingface_hub import snapshot_download

                path = snapshot_download(path, local_files_only=True)
            except Exception:
                print(f"[asr] {model} not found locally; trying to download it (needs Hugging Face access)")
        self.path = path
        self.model_name = model
        self.device = "apple-gpu"
        self.prompts = {"en": cfg.get("prompt_en") or None, "zh": cfg.get("prompt_zh") or None}
        t0 = time.monotonic()
        self.model = ModelHolder.get_model(self.path, mx.float16)
        print(f"[asr] mlx-whisper {model} loaded in {time.monotonic() - t0:.1f} s")

    def _detect(self, audio: np.ndarray) -> dict:
        mla, mx = self.mla, self.mx
        mel = mla.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels, padding=mla.N_SAMPLES)
        seg = mla.pad_or_trim(mel, mla.N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = self.model.detect_language(seg)
        if isinstance(probs, list):
            probs = probs[0]
        return dict(probs)

    def transcribe(self, audio: np.ndarray, mode: str, current: str | None, final: bool = True) -> ASRResult:
        t0 = time.monotonic()
        dur = len(audio) / SAMPLE_RATE
        if mode in LANGS:
            lang, prob = mode, 1.0
        else:
            try:
                lang, prob = choose_language(self._detect(audio), current, dur)
            except Exception as e:  # pragma: no cover - depends on mlx internals
                print(f"[asr] language detection failed ({e}); letting Whisper decide")
                lang, prob = None, 0.0
        result = self.mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.path,
            language=lang,
            initial_prompt=self.prompts.get(lang) if lang else None,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6) if final else 0.0,
            word_timestamps=False,
            verbose=None,
        )
        if lang is None:
            detected = result.get("language", "en")
            lang = "zh" if detected in ("zh", "yue", "ja") or has_cjk(result.get("text", "")) else "en"
        parts = [
            s.get("text", "")
            for s in result.get("segments", [])
            if _segment_ok(s.get("no_speech_prob", 0), s.get("avg_logprob", 0), s.get("compression_ratio", 1))
        ]
        text = clean_text(_join(parts, lang), lang)
        if text and is_hallucination(text):
            text = ""
        return ASRResult(text, lang, prob, dur, time.monotonic() - t0)


# --------------------------------------------------------------------------- mock (tests / demo)

class MockASR:
    """No model: returns placeholder sentences. For testing the display and network."""

    engine = "mock"
    model_name = "mock"
    device = "none"

    EN = [
        "Welcome to the first Shenzhen open source silicon meetup.",
        "Tiny Tapeout lets anyone put their design on a real chip.",
        "Today we will talk about wafers, PDKs and chip-on-board packaging.",
    ]
    ZH = ["欢迎大家来到深圳开源芯片交流会。", "我们今天聊一聊开源工艺设计套件。", "有问题的话可以随时提问。"]

    def __init__(self, cfg: dict):
        self.n = 0

    def transcribe(self, audio, mode, current, final=True) -> ASRResult:
        dur = len(audio) / SAMPLE_RATE
        lang = mode if mode in LANGS else (current or "en")
        pool = self.EN if lang == "en" else self.ZH
        text = pool[self.n % len(pool)]
        if final:
            self.n += 1
        else:
            words = re.split(r"(?<=\s)", text) if lang == "en" else list(text)
            text = "".join(words[: max(1, len(words) // 2)])
        time.sleep(0.05)
        return ASRResult(text, lang, 1.0, dur, 0.05)


def make_asr(cfg: dict):
    engine = pick_engine(cfg.get("engine", "auto"))
    if engine == "mlx":
        return MlxWhisperASR(cfg)
    if engine == "mock":
        return MockASR(cfg)
    return FasterWhisperASR(cfg)
