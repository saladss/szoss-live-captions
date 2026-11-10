"""Wires it together: audio -> segmenter -> Whisper -> translator -> hub (web pages) + transcript."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from .asr import make_asr
from .segmenter import Segmenter, Utterance, make_vad
from .translate import TranslationError, Translator

OTHER = {"en": "zh", "zh": "en"}


class TranscriptLog:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.txt_path = folder / f"captions-{stamp}.txt"
        self.jsonl_path = folder / f"captions-{stamp}.jsonl"
        self.lock = threading.Lock()

    def write(self, c: dict):
        when = datetime.fromtimestamp(c["t"]).strftime("%H:%M:%S")
        src, tgt = c["lang"], OTHER[c["lang"]]
        lines = [f"[{when}] {src.upper()}: {c['src']}"]
        if c.get("tgt"):
            lines.append(f"           {tgt.upper()}: {c['tgt']}")
        elif c.get("status") == "failed":
            lines.append("           (translation failed)")
        with self.lock:
            with open(self.txt_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n\n")
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")


class Pipeline:
    def __init__(self, cfg: dict, hub, source_factory, log_folder: Path | None):
        self.cfg = cfg
        self.hub = hub
        self.source_factory = source_factory
        self.log = TranscriptLog(log_folder) if log_folder else None
        self.audio_q: queue.Queue = queue.Queue(maxsize=2000)
        self.trans_q: queue.Queue = queue.Queue()
        self.cond = threading.Condition()
        self.finals: deque[Utterance] = deque()
        self.partial_job: Utterance | None = None
        self.partials_on = float(cfg["segmenter"].get("partial_interval_s", 1.0)) > 0
        self.utt_lang: dict[int, str] = {}
        self.current_lang: str | None = None
        lang = cfg["asr"].get("language", "auto")
        if lang in ("en", "zh"):
            self.current_lang = lang
        self.translator = Translator(cfg["translation"], cfg.get("glossary", {}))
        self.context_lines = int(cfg["translation"].get("context_lines", 3))
        self.ready = threading.Event()
        self.source_done = threading.Event()
        self.failed: str | None = None
        self.running = True
        self.source = None
        self.segmenter: Segmenter | None = None
        self.stats = {
            "asr_ms": 0,
            "asr_rtf": 0.0,
            "mt_ms": 0,
            "mt_backend": "",
            "captions": 0,
            "mt_failed": 0,
            "partials_disabled": False,
        }

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        threading.Thread(target=self._asr_loop, name="asr", daemon=True).start()
        threading.Thread(target=self._translate_loop, name="translate", daemon=True).start()
        threading.Thread(target=self._status_loop, name="status", daemon=True).start()
        self.ready.wait()
        if self.failed:
            raise SystemExit(self.failed)
        if self.translator.enabled:
            self._warm_translator()
        seg_cfg = self.cfg["segmenter"]
        self.segmenter = Segmenter(
            self._on_utterance,
            vad=make_vad(),
            threshold=seg_cfg["vad_threshold"],
            end_silence_ms=seg_cfg["end_silence_ms"],
            min_speech_ms=seg_cfg["min_speech_ms"],
            max_caption_s=seg_cfg["max_caption_s"],
            preroll_ms=seg_cfg["preroll_ms"],
            partial_interval_s=seg_cfg["partial_interval_s"] if self.partials_on else 0,
        )
        print(f"[vad] {self.segmenter.vad.name}")
        threading.Thread(target=self._segment_loop, name="segmenter", daemon=True).start()
        self.source = self.source_factory(self.audio_q)
        try:
            self.source.start()
        except Exception as e:
            from .audio import MicSource

            if not isinstance(self.source, MicSource) or self.source.device is None:
                raise SystemExit(f"Can't open the microphone: {e}") from None
            print(f"[audio] couldn't open the configured input ({e}); using the default input")
            a = self.cfg["audio"]
            self.source = MicSource(self.audio_q, "", a.get("channel", 0), a.get("gain", 1.0))
            try:
                self.source.start()
            except Exception as e2:
                raise SystemExit(f"Can't open the microphone: {e2}") from None

    def stop(self):
        self.running = False
        if self.source:
            self.source.stop()
        with self.cond:
            self.cond.notify_all()
        self.translator.close()

    # ------------------------------------------------------------------ microphone switching

    def switch_device(self, spec, hostapi: str = "") -> str:
        """Listen to a different input. Returns the new device name; raises on failure."""
        from .audio import MicSource

        if not isinstance(self.source, MicSource):
            raise RuntimeError("playing a file, not listening to a microphone")
        a = self.cfg["audio"]
        old = self.source
        try:
            new = MicSource(self.audio_q, spec, a.get("channel", 0), a.get("gain", 1.0), hostapi=hostapi)
        except BaseException as e:
            raise RuntimeError(str(e)) from None
        old.stop()
        try:
            new.start()
        except BaseException as e:
            try:
                old = MicSource(self.audio_q, old.device if old.device is not None else "", a.get("channel", 0), a.get("gain", 1.0))
                old.start()
                self.source = old
            except BaseException:
                pass
            raise RuntimeError(str(e)) from None
        self.source = new
        self._last_signal = time.monotonic()
        return new.name

    def idle(self) -> bool:
        with self.cond:
            busy = bool(self.finals) or self._asr_busy
        return not busy and self.trans_q.unfinished_tasks == 0

    def _warm_translator(self):
        t0 = time.monotonic()
        if any(type(b).__name__ == "LlamaCppBackend" for b in self.translator.backends):
            print("[translate] loading the built-in translation model…")
        try:
            out, name, _ = self.translator.translate("Hello, everyone.", "en", "zh", [])
            print(f"[translate] {name} ready ({time.monotonic() - t0:.1f} s): Hello, everyone. -> {out}")
        except TranslationError as e:
            print(f"[translate] WARNING: translation not working yet: {e}")
        for b in self.translator.backends:
            b.failures, b.skip_until = 0, 0.0

    # ------------------------------------------------------------------ audio -> segments

    def _segment_loop(self):
        was_paused = False
        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                self.segmenter.flush()
                self.source_done.set()
                return
            if self.hub.settings.get("paused"):
                if not was_paused:
                    self.segmenter.reset()
                    self.hub.clear_partial()
                was_paused = True
                continue
            was_paused = False
            self.segmenter.feed(chunk)

    def _on_utterance(self, u: Utterance):
        with self.cond:
            if u.final:
                self.finals.append(u)
                if self.partial_job and self.partial_job.uid == u.uid:
                    self.partial_job = None
            elif self.partials_on:
                self.partial_job = u
            self.cond.notify()

    # ------------------------------------------------------------------ speech recognition

    _asr_busy = False

    def _asr_loop(self):
        try:
            self.asr = make_asr(self.cfg["asr"])
            noise = (np.random.default_rng(0).standard_normal(16000) * 1e-3).astype(np.float32)
            t0 = time.monotonic()
            self.asr.transcribe(noise, "en", None, final=True)
            print(f"[asr] warm-up done in {time.monotonic() - t0:.1f} s")
        except SystemExit as e:
            self.failed = str(e)
        except Exception as e:
            self.failed = (
                f"Could not load the speech recognition model: {e}\n"
                "Download it first with `python captions.py download` (in mainland China add "
                "HF_ENDPOINT=https://hf-mirror.com in front)."
            )
        finally:
            self.ready.set()
        if self.failed:
            return
        while self.running:
            with self.cond:
                while self.running and not self.finals and self.partial_job is None:
                    self.cond.wait(timeout=1.0)
                if not self.running:
                    return
                if self.finals:
                    job = self.finals.popleft()
                else:
                    job, self.partial_job = self.partial_job, None
                self._asr_busy = True
            try:
                if job.final:
                    self._do_final(job)
                else:
                    self._do_partial(job)
            except Exception as e:  # keep going whatever happens
                print(f"[asr] error: {e}")
            finally:
                self._asr_busy = False

    def _mode(self) -> str:
        return self.hub.settings.get("mode", "auto")

    def _do_partial(self, u: Utterance):
        if not self.partials_on or self.hub.settings.get("paused"):
            return
        if time.monotonic() - u.created > 1.5:
            return  # stale; a newer one is coming
        mode = self._mode()
        if mode == "auto" and u.uid in self.utt_lang:
            mode = self.utt_lang[u.uid]
        res = self.asr.transcribe(u.audio, mode, self.current_lang, final=False)
        if res.elapsed_s > 1.5:
            self.partials_on = False
            self.stats["partials_disabled"] = True
            print(f"[asr] live preview took {res.elapsed_s:.1f} s, too slow on this computer; turning it off")
        self.utt_lang.setdefault(u.uid, res.lang)
        if res.text:
            self.hub.set_partial(u.uid, res.lang, res.text)

    def _do_final(self, u: Utterance):
        self.utt_lang.pop(u.uid, None)
        if len(u.audio) == 0 or self.hub.settings.get("paused"):
            self.hub.clear_partial(u.uid)
            return
        res = self.asr.transcribe(u.audio, self._mode(), self.current_lang, final=True)
        self.stats["asr_ms"] = int(res.elapsed_s * 1000)
        self.stats["asr_rtf"] = round(res.elapsed_s / max(res.audio_s, 0.1), 2)
        if not res.text:
            self.hub.clear_partial(u.uid)
            return
        self.current_lang = res.lang
        caption = {
            "id": u.uid,
            "lang": res.lang,
            "src": res.text,
            "tgt": None,
            "status": "translating" if self.translator.enabled else "untranslated",
            "t": time.time(),
            "audio_s": round(res.audio_s, 2),
            "asr_ms": int(res.elapsed_s * 1000),
            "lag_ms": int((time.monotonic() - u.created) * 1000),
        }
        self.stats["captions"] += 1
        self.hub.upsert(caption)
        print(f"[{res.lang}] {res.text}   ({res.audio_s:.1f} s audio, asr {res.elapsed_s:.2f} s)")
        if self.translator.enabled:
            self.trans_q.put(u.uid)
        elif self.log:
            self.log.write(caption)

    # ------------------------------------------------------------------ translation

    def _translate_loop(self):
        while self.running:
            try:
                uid = self.trans_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._translate_one(uid)
            except Exception as e:
                print(f"[translate] error: {e}")
            finally:
                self.trans_q.task_done()

    def _translate_one(self, uid: int):
        c = self.hub.get(uid)
        if c is None:
            return  # deleted from the control page before translation
        context = [(p["src"], p["tgt"]) for p in self.hub.recent(self.context_lines, uid, c["lang"])]
        backlog = self.trans_q.qsize()
        if backlog > 3:
            context = []  # falling behind: keep prompts short
        t0 = time.monotonic()
        try:
            out, backend, secs = self.translator.translate(c["src"], c["lang"], OTHER[c["lang"]], context)
            c.update(tgt=out, status="done", backend=backend, mt_ms=int(secs * 1000))
            self.stats["mt_ms"] = int(secs * 1000)
            self.stats["mt_backend"] = backend
            print(f"[{OTHER[c['lang']]}] {out}   ({backend}, {secs:.2f} s)")
        except TranslationError as e:
            c.update(status="failed", error=str(e)[:200], mt_ms=int((time.monotonic() - t0) * 1000))
            self.stats["mt_failed"] += 1
            print(f"[translate] failed: {e}")
        if self.hub.get(uid) is not None:  # not deleted meanwhile
            self.hub.upsert(c)
        if self.log:
            self.log.write(c)

    # ------------------------------------------------------------------ status for the control page

    _last_signal = 0.0

    def _status_loop(self):
        from .audio import MicSource

        self._last_signal = time.monotonic()
        while self.running:
            time.sleep(0.25)
            seg = self.segmenter
            now = time.monotonic()
            # true digital silence: wrong input, muted, or Windows microphone privacy blocking the app
            if not seg or seg.level_db > -85 or not isinstance(self.source, MicSource):
                self._last_signal = now
            status = {
                "ready": self.ready.is_set() and not self.failed,
                "source": getattr(self.source, "name", ""),
                "is_mic": isinstance(self.source, MicSource),
                "no_signal": now - self._last_signal > 4,
                "level_db": round(seg.level_db, 1) if seg else -90,
                "vad": round(seg.last_prob, 2) if seg else 0,
                "speaking": bool(seg and seg.in_speech),
                "asr": {
                    "engine": getattr(getattr(self, "asr", None), "engine", ""),
                    "model": getattr(getattr(self, "asr", None), "model_name", ""),
                    "device": getattr(getattr(self, "asr", None), "device", ""),
                },
                "current_lang": self.current_lang,
                "queue": {"asr": len(self.finals), "translate": self.trans_q.qsize()},
                "translators": self.translator.status(),
                "viewers": self.hub.viewers,
                "audio_errors": getattr(self.source, "status_errors", 0),
                **self.stats,
            }
            self.hub.set_status(status)
