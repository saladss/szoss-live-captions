"""Translation backends: Ollama (local) and any OpenAI-compatible API
(DeepSeek, Alibaba Qwen / Model Studio, LM Studio, OpenAI...).

Backends are tried in the configured order. One that fails twice in a row is
skipped for 30 seconds so a dead network connection doesn't add a timeout to
every caption.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .textfix import has_cjk, to_simplified

LANG_NAME = {"en": "English", "zh": "Simplified Chinese"}
MT_LANG = {"en": "English", "zh": "Chinese"}


class TranslationError(Exception):
    pass


def _opener_for(url: str):
    host = urlparse(url).hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        # never send local traffic through a system proxy / VPN
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with _opener_for(url).open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise TranslationError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TranslationError(str(getattr(e, "reason", e))) from e


def matching_terms(text: str, src: str, glossary: dict) -> list[tuple[str, str]]:
    """Glossary entries that occur in this sentence, as (source term, target term)."""
    found, seen = [], set()
    for en, zh in glossary.items():
        a, b = (en, zh) if src == "en" else (zh, en)
        if a in seen:
            continue
        if src == "en" or not has_cjk(a):
            hit = re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?:s|es)?(?![A-Za-z])", text, re.I)
        else:
            hit = a in text
        if hit:
            seen.add(a)
            found.append((a, b))
    return found


def build_messages(text: str, src: str, tgt: str, context: list[tuple[str, str]], glossary: dict, topic: str):
    terms = [f"{a} = {b}" if a != b else f"{a} (keep as is)" for a, b in matching_terms(text, src, glossary)]
    system = (
        f"You translate live captions at {topic}. "
        f"Translate the speaker's new sentence from {LANG_NAME[src]} into {LANG_NAME[tgt]}. "
        "Output only the translation, on one line: no notes, no quotes, no pinyin, no explanations. "
        "Keep it natural and concise, like subtitles. The text comes from speech recognition, so it may "
        "contain small errors or be an unfinished fragment: translate what the speaker meant, and keep a "
        "fragment as a fragment. Keep names of people, companies, projects and products in their original form."
    )
    if terms:
        system += " Use these term translations: " + "; ".join(terms) + "."
    user = ""
    if context:
        lines = "\n".join(f"- {s}" for s, _ in context)
        user += f"Earlier sentences, for context only (do not translate them):\n{lines}\n\n"
    user += f"New sentence:\n{text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_output(out: str, tgt: str) -> str:
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip()
    out = re.sub(r"^\s*(translation|译文|翻译)\s*[:：]\s*", "", out, flags=re.I)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if not lines:
        return ""
    out = " ".join(lines) if len(lines) <= 2 else lines[0]
    if len(out) >= 2 and out[0] in "\"'“「『" and out[-1] in "\"'”」』":
        out = out[1:-1].strip()
    if tgt == "zh":
        out = to_simplified(out)
    return out


class OllamaBackend:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.url = cfg.get("url", "http://127.0.0.1:11434").rstrip("/")
        self.model = cfg.get("model", "qwen3:8b")
        self.send_think = True

    def describe(self) -> str:
        return f"{self.name} (Ollama {self.model})"

    def translate(self, messages, src, tgt, text, context, glossary, topic, timeout) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "3h",
            "options": {"temperature": 0.2, "num_predict": 300},
        }
        if self.send_think:
            body["think"] = False  # Qwen3 and similar: answer directly, no reasoning
        try:
            res = _post_json(f"{self.url}/api/chat", body, {}, timeout)
        except TranslationError as e:
            if self.send_think and "think" in str(e).lower():
                self.send_think = False  # model doesn't support the flag
                body.pop("think", None)
                res = _post_json(f"{self.url}/api/chat", body, {}, timeout)
            else:
                raise
        return (res.get("message") or {}).get("content", "")


class OpenAIBackend:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        base = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.model = cfg.get("model", "")
        key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", ""), "")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.is_qwen_mt = self.model.startswith("qwen-mt")
        if not key and not self.url.startswith(("http://127.", "http://localhost")):
            print(f"[translate] warning: no API key for '{name}' (set {cfg.get('api_key_env', 'api_key')})")

    def describe(self) -> str:
        return f"{self.name} ({self.model})"

    def translate(self, messages, src, tgt, text, context, glossary, topic, timeout) -> str:
        if self.is_qwen_mt:
            # Alibaba's dedicated translation models take the raw text plus options
            terms = [{"source": a, "target": b} for a, b in matching_terms(text, src, glossary)]
            options = {
                "source_lang": MT_LANG[src],
                "target_lang": MT_LANG[tgt],
                "domains": f"Live speech captions from {topic}. Translate naturally and concisely, like subtitles.",
            }
            if terms:
                options["terms"] = terms
            if context:
                options["tm_list"] = [{"source": s, "target": t} for s, t in context if t]
            body = {"model": self.model, "messages": [{"role": "user", "content": text}], "translation_options": options}
        else:
            body = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 300, "stream": False}
        res = _post_json(self.url, body, self.headers, timeout)
        try:
            return res["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise TranslationError(f"unexpected response: {str(res)[:200]}") from e


class LlamaCppBackend:
    """Translation model bundled with the app, served by llama.cpp (no install, no internet)."""

    def __init__(self, name: str, cfg: dict):
        from .llamaserver import LlamaServer

        self.name = name
        self.server = LlamaServer(cfg)
        self.model = self.server.model.name if self.server.model else "(no model)"
        low = self.model.lower()
        # Qwen3 hybrid models think before answering unless told not to
        self.no_think = "qwen3" in low and "instruct" not in low
        self.first_wait = float(cfg.get("load_timeout_s", 180)) * 2
        self.server.start()

    def describe(self) -> str:
        where = {"gpu": "graphics card", "cpu": "CPU"}.get(self.server.mode, "loading")
        return f"{self.name} (built-in {self.model}, {where})"

    def translate(self, messages, src, tgt, text, context, glossary, topic, timeout) -> str:
        if not self.server.wait_ready(self.first_wait if not self.server.mode else 1.0):
            raise TranslationError(self.server.error or "built-in translator is still loading")
        msgs = [dict(m) for m in messages]
        if self.no_think:
            msgs[-1]["content"] += " /no_think"
        body = {
            "messages": msgs,
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 300,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        res = _post_json(self.server.url + "/v1/chat/completions", body, {}, timeout)
        try:
            return res["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise TranslationError(f"unexpected response: {str(res)[:200]}") from e

    def close(self):
        self.server.stop()


class MockBackend:
    """For tests: tags the text instead of translating it."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.delay = float(cfg.get("delay", 0.1))

    def describe(self) -> str:
        return f"{self.name} (mock)"

    def translate(self, messages, src, tgt, text, context, glossary, topic, timeout) -> str:
        time.sleep(self.delay)
        return f"[{tgt}] {text}"


BACKEND_TYPES = {"ollama": OllamaBackend, "openai": OpenAIBackend, "llamacpp": LlamaCppBackend, "mock": MockBackend}


class Translator:
    def __init__(self, cfg: dict, glossary: dict):
        self.timeout = float(cfg.get("timeout_s", 8))
        self.topic = cfg.get("topic", "a technical meetup")
        self.glossary = {str(k): str(v) for k, v in (glossary or {}).items()}
        self.backends = []
        defs = cfg.get("backends", {})
        for name in cfg.get("order", []):
            if name not in defs:
                print(f"[translate] '{name}' is in translation.order but has no [translation.backends.{name}] section")
                continue
            kind = defs[name].get("type", "openai")
            if kind not in BACKEND_TYPES:
                print(f"[translate] unknown backend type '{kind}' for '{name}'")
                continue
            b = BACKEND_TYPES[kind](name, defs[name])
            b.failures, b.skip_until, b.last_error = 0, 0.0, ""
            self.backends.append(b)

    @property
    def enabled(self) -> bool:
        return bool(self.backends)

    def close(self):
        for b in self.backends:
            if hasattr(b, "close"):
                try:
                    b.close()
                except Exception:
                    pass

    def status(self) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "name": b.describe(),
                "ok": b.failures == 0,
                "skipping": b.skip_until > now,
                "error": b.last_error,
            }
            for b in self.backends
        ]

    def translate(self, text: str, src: str, tgt: str, context: list[tuple[str, str]]) -> tuple[str, str, float]:
        """Returns (translation, backend name, seconds). Raises TranslationError."""
        messages = build_messages(text, src, tgt, context, self.glossary, self.topic)
        errors = []
        now = time.monotonic()
        candidates = [b for b in self.backends if b.skip_until <= now] or self.backends
        for b in candidates:
            t0 = time.monotonic()
            try:
                raw = b.translate(messages, src, tgt, text, context, self.glossary, self.topic, self.timeout)
                out = clean_output(raw, tgt)
                if not out:
                    raise TranslationError("empty reply")
                if tgt == "zh" and len(text) > 25 and not has_cjk(out):
                    raise TranslationError(f"reply is not Chinese: {out[:60]}")
                if tgt == "en" and len(re.findall(r"[一-鿿]", out)) > len(out) * 0.5:
                    raise TranslationError(f"reply is not English: {out[:60]}")
                b.failures, b.last_error = 0, ""
                return out, b.name, time.monotonic() - t0
            except Exception as e:
                b.failures += 1
                b.last_error = str(e)[:200]
                if b.failures >= 2:
                    b.skip_until = time.monotonic() + 30
                errors.append(f"{b.name}: {b.last_error}")
        raise TranslationError("; ".join(errors) or "no translation backend configured")
