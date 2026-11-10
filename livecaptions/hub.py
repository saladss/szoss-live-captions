"""Shared caption state, and publish/subscribe to the web pages (Server-Sent Events)."""

from __future__ import annotations

import copy
import queue
import threading
import time
from collections import OrderedDict


class Hub:
    def __init__(self, settings: dict, keep: int = 60):
        self.lock = threading.RLock()
        self.keep = keep
        self.captions: "OrderedDict[int, dict]" = OrderedDict()
        self.partial: dict | None = None
        self.settings = dict(settings)  # mode, paused, message, title, order, lines
        self.status: dict = {}
        self._subs: list[tuple[queue.Queue, bool]] = []

    # -- subscribers ----------------------------------------------------------

    def subscribe(self, status: bool = False) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self.lock:
            self._subs.append((q, status))
            q.put_nowait(self.snapshot())
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            self._subs = [(s, st) for s, st in self._subs if s is not q]

    @property
    def viewers(self) -> int:
        with self.lock:
            return len(self._subs)

    def publish(self, event: dict, status_only: bool = False):
        with self.lock:
            subs = list(self._subs)
        for q, wants_status in subs:
            if status_only and not wants_status:
                continue
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # a stuck client; it will resync on reconnect

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "type": "snapshot",
                "captions": [copy.deepcopy(c) for c in self.captions.values()],
                "partial": copy.deepcopy(self.partial),
                "settings": dict(self.settings),
            }

    # -- captions -------------------------------------------------------------

    def get(self, uid: int) -> dict | None:
        with self.lock:
            c = self.captions.get(uid)
            return copy.deepcopy(c) if c else None

    def upsert(self, caption: dict):
        with self.lock:
            self.captions[caption["id"]] = copy.deepcopy(caption)
            while len(self.captions) > self.keep:
                self.captions.popitem(last=False)
            if self.partial and self.partial.get("id") == caption["id"]:
                self.partial = None
        self.publish({"type": "caption", "caption": caption})

    def recent(self, n: int, before: int, lang: str) -> list[dict]:
        with self.lock:
            out = [
                c for c in self.captions.values() if c["id"] < before and c["lang"] == lang and c.get("tgt")
            ]
        return [copy.deepcopy(c) for c in out[-n:]] if n > 0 else []

    def set_partial(self, uid: int, lang: str, text: str):
        p = {"id": uid, "lang": lang, "src": text, "t": time.time()}
        with self.lock:
            if uid in self.captions:
                return  # final already arrived
            self.partial = p
        self.publish({"type": "partial", "partial": p})

    def clear_partial(self, uid: int | None = None):
        with self.lock:
            if self.partial is None or (uid is not None and self.partial.get("id") != uid):
                return
            self.partial = None
        self.publish({"type": "partial", "partial": None})

    def remove(self, uid: int) -> bool:
        with self.lock:
            found = self.captions.pop(uid, None) is not None
        if found:
            self.publish({"type": "remove", "id": uid})
        return found

    def clear(self):
        with self.lock:
            self.captions.clear()
            self.partial = None
        self.publish({"type": "clear"})

    # -- settings / status ----------------------------------------------------

    def update_settings(self, **changes):
        with self.lock:
            self.settings.update(changes)
            s = dict(self.settings)
        self.publish({"type": "settings", "settings": s})
        return s

    def set_status(self, status: dict):
        with self.lock:
            self.status = status
        self.publish({"type": "status", "status": status}, status_only=True)
