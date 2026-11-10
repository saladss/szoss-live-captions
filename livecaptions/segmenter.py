"""Voice activity detection and caption segmentation.

Audio arrives in arbitrary-sized 16 kHz chunks. It is cut into 32 ms frames, each
frame gets a speech probability from Silero VAD, and frames are grouped into
utterances that end at a pause (or are force-split if the speaker never pauses).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms at 16 kHz
FRAME_S = FRAME / SAMPLE_RATE


# --------------------------------------------------------------------------- VAD

class SileroVAD:
    """Streaming Silero VAD using the ONNX model that ships with faster-whisper."""

    CONTEXT = 64

    def __init__(self):
        import onnxruntime

        path = None
        try:
            import faster_whisper

            assets = os.path.join(os.path.dirname(faster_whisper.__file__), "assets")
            for name in sorted(os.listdir(assets), reverse=True):
                if name.startswith("silero") and name.endswith(".onnx"):
                    path = os.path.join(assets, name)
                    break
        except Exception:
            pass
        if path is None:
            raise RuntimeError("Silero VAD model not found (install faster-whisper)")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.input_names = [i.name for i in self.session.get_inputs()]
        if self.input_names != ["input", "h", "c"]:
            raise RuntimeError(f"Unexpected Silero model inputs: {self.input_names}")
        self.name = f"silero ({os.path.basename(path)})"
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.ctx, frame.reshape(1, -1)], axis=1).astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = x[:, -self.CONTEXT :]
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Fallback if Silero can't load: adaptive energy threshold."""

    name = "energy (fallback)"

    def __init__(self):
        self.noise = 1e-3

    def reset(self):
        pass

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame**2)) + 1e-9)
        if rms < self.noise * 2:
            self.noise = 0.95 * self.noise + 0.05 * rms
        else:
            self.noise = 0.999 * self.noise + 0.001 * rms
        snr_db = 20 * np.log10(rms / max(self.noise, 1e-6))
        return float(np.clip((snr_db - 6) / 12, 0, 1))


def make_vad():
    try:
        return SileroVAD()
    except Exception as e:  # pragma: no cover
        print(f"[vad] Silero VAD unavailable ({e}); using energy VAD")
        return EnergyVAD()


# --------------------------------------------------------------------------- segmenter

@dataclass
class Utterance:
    uid: int
    audio: np.ndarray
    final: bool
    start_s: float  # stream time of the first sample
    created: float = field(default=0.0)  # wall-clock when emitted

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class Segmenter:
    def __init__(
        self,
        on_utterance: Callable[[Utterance], None],
        vad=None,
        threshold: float = 0.5,
        end_silence_ms: int = 600,
        min_speech_ms: int = 300,
        max_caption_s: float = 12.0,
        preroll_ms: int = 300,
        partial_interval_s: float = 1.0,
        clock: Callable[[], float] | None = None,
    ):
        import time

        self.on_utterance = on_utterance
        self.vad = vad or make_vad()
        self.threshold = float(threshold)
        self.neg_threshold = max(0.05, self.threshold - 0.15)
        self.end_silence = max(1, int(end_silence_ms / 1000 / FRAME_S))
        # short fragments wait a little longer for the sentence to continue
        self.end_silence_short = int(self.end_silence * 1.5)
        self.min_speech = max(1, int(min_speech_ms / 1000 / FRAME_S))
        self.max_frames = max(int(3 / FRAME_S), int(max_caption_s / FRAME_S))
        self.preroll = max(0, int(preroll_ms / 1000 / FRAME_S))
        self.partial_interval = float(partial_interval_s)
        self.clock = clock or time.monotonic
        self.start_frames = 2  # consecutive speech frames needed to open an utterance
        self.next_uid = 1
        self.frame_count = 0
        self.level_db = -90.0
        self.last_prob = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        self._reset_state(keep_vad=True)

    # -- state -------------------------------------------------------------

    def _reset_state(self, keep_vad: bool = True):
        self.in_speech = False
        self.frames: list[np.ndarray] = []
        self.probs: list[float] = []
        self.speech_count = 0
        self.silence_count = 0
        self.start_count = 0
        self.history: deque = deque(maxlen=self.preroll + self.start_frames)
        self.frames_since_partial = 0
        self.utt_start_frame = 0
        if not keep_vad:
            self.vad.reset()

    def reset(self):
        """Drop any half-finished utterance (used when pausing)."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._reset_state(keep_vad=False)

    # -- input -------------------------------------------------------------

    def feed(self, chunk: np.ndarray):
        data = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float32)])
        n = len(data) // FRAME
        for i in range(n):
            self._frame(data[i * FRAME : (i + 1) * FRAME])
        self._pending = data[n * FRAME :]

    def flush(self):
        """End of stream: finish whatever is in progress."""
        if self.in_speech:
            self._emit_final(len(self.frames))
        self._reset_state()

    def _frame(self, frame: np.ndarray):
        self.frame_count += 1
        prob = self.vad(frame)
        self.last_prob = prob
        rms = float(np.sqrt(np.mean(frame**2)) + 1e-9)
        db = max(-90.0, 20 * np.log10(rms))
        # meter: fast attack, slow release
        self.level_db = db if db > self.level_db else 0.9 * self.level_db + 0.1 * db

        if not self.in_speech:
            self.history.append((frame, prob))
            self.start_count = self.start_count + 1 if prob >= self.threshold else 0
            if self.start_count >= self.start_frames:
                self.in_speech = True
                self.frames = [f for f, _ in self.history]
                self.probs = [p for _, p in self.history]
                self.history.clear()
                self.speech_count = self.start_count
                self.silence_count = 0
                self.frames_since_partial = 0
                self.utt_start_frame = self.frame_count - len(self.frames)
            return

        self.frames.append(frame)
        self.probs.append(prob)
        self.frames_since_partial += 1
        if prob >= self.neg_threshold:
            self.silence_count = 0
            if prob >= self.threshold:
                self.speech_count += 1
        else:
            self.silence_count += 1

        need = self.end_silence if len(self.frames) * FRAME_S >= 2.5 else self.end_silence_short
        if self.silence_count >= need:
            keep_tail = min(self.silence_count, int(0.2 / FRAME_S))
            end = len(self.frames) - self.silence_count + keep_tail
            self._emit_final(end)
            self._reset_state()
            return

        if len(self.frames) >= self.max_frames:
            self._force_split()
            return

        if (
            self.partial_interval > 0
            and self.frames_since_partial * FRAME_S >= self.partial_interval
            and len(self.frames) * FRAME_S >= 0.8
        ):
            self.frames_since_partial = 0
            self._emit(np.concatenate(self.frames), final=False)

    # -- output ------------------------------------------------------------

    def _force_split(self):
        """Speaker hasn't paused: cut at the quietest point of the last few seconds."""
        n = len(self.frames)
        lo = max(int(n * 0.5), n - int(3.0 / FRAME_S))
        hi = n - int(0.5 / FRAME_S)
        probs = np.convolve(np.array(self.probs), np.ones(3) / 3, mode="same")
        cut = lo + int(np.argmin(probs[lo:hi])) if hi > lo else n
        rest_frames, rest_probs = self.frames[cut:], self.probs[cut:]
        self._emit_final(cut)
        # carry the remainder into a new utterance
        self.frames, self.probs = rest_frames, rest_probs
        self.speech_count = sum(1 for p in rest_probs if p >= self.threshold)
        self.silence_count = 0
        self.frames_since_partial = 0
        self.utt_start_frame = self.frame_count - len(self.frames)

    def _emit_final(self, end: int):
        if self.speech_count >= self.min_speech and end > 0:
            self._emit(np.concatenate(self.frames[:end]), final=True)
            self.next_uid += 1
        else:
            # too short to be speech; tell listeners to drop any partial for it
            self.on_utterance(
                Utterance(self.next_uid, np.zeros(0, dtype=np.float32), True, self.utt_start_frame * FRAME_S, self.clock())
            )
            self.next_uid += 1

    def _emit(self, audio: np.ndarray, final: bool):
        self.on_utterance(Utterance(self.next_uid, audio, final, self.utt_start_frame * FRAME_S, self.clock()))
