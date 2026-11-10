"""Text clean-up: Simplified Chinese, Whisper hallucinations, repeated phrases."""

from __future__ import annotations

import re
import unicodedata

try:
    import zhconv

    def to_simplified(text: str) -> str:
        # zh-cn: Simplified characters and mainland vocabulary (晶片 -> 芯片, 軟體 -> 软件)
        return zhconv.convert(text, "zh-cn")

except ImportError:  # pragma: no cover
    try:
        import opencc

        _cc = opencc.OpenCC("t2s")

        def to_simplified(text: str) -> str:
            return _cc.convert(text)

    except Exception:

        def to_simplified(text: str) -> str:
            return text


CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# Phrases Whisper is known to invent on silence, music or noise (it learned them
# from subtitled videos). A caption that is only one of these is dropped.
HALLUCINATIONS = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "transcribed by",
    "transcription by",
    "amara.org",
    "see you in the next video",
    "字幕由",
    "字幕提供",
    "字幕志愿者",
    "请不吝点赞",
    "订阅转发",
    "打赏支持明镜",
    "明镜与点点栏目",
    "优优独播剧场",
    "yoyo television series",
    "谢谢观看",
    "感谢观看",
    "多谢收看",
    "请订阅",
    "中文字幕",
]

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def _norm(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text).lower())


def is_hallucination(text: str) -> bool:
    n = _norm(text)
    if not n:
        return True
    for phrase in HALLUCINATIONS:
        p = _norm(phrase)
        if p and p in n and len(n) <= len(p) + 12:
            return True
    return False


_REPEAT = re.compile(r"(.{2,30}?)(?:[\s,，、.。]*\1){3,}")


def collapse_repeats(text: str) -> str:
    """'好的好的好的好的好的' -> '好的'; Whisper sometimes loops on a phrase."""
    return _REPEAT.sub(r"\1", text)


def clean_text(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = collapse_repeats(text)
    if lang == "zh":
        text = to_simplified(text)
        # Whisper sometimes puts spaces between Chinese characters
        text = re.sub(r"(?<=[一-鿿，。！？、]) (?=[一-鿿，。！？、])", "", text)
    return text.strip()


def has_cjk(text: str) -> bool:
    return bool(CJK.search(text or ""))
