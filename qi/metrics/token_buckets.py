"""Token bucketing for the per-token error analysis (plan E1.4): class by regex on the decoded token,
unigram-frequency bucket, and position-in-context bucket."""
from __future__ import annotations

import re
import unicodedata

POSITION_EDGES = (256, 1024, 4096)
POSITION_LABELS = ("0-256", "256-1024", "1024-4096", "4096+")
FREQ_EDGES = (10, 100, 1_000, 10_000)
FREQ_LABELS = ("<10", "10-100", "100-1k", "1k-10k", "10k+")
CLASSES = ("numeral", "numeral_mixed", "punctuation", "code_symbol", "whitespace", "word", "non_latin", "other")

_NUM = re.compile(r"^\s*[+-]?\d+(?:[.,]\d+)*\s*$")
_WS = re.compile(r"^\s*$")
_PUNCT = re.compile(r"^\s*[^\w\s]+\s*$")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_DIGIT = re.compile(r"\d")
_CODE_CHARS = set("{}[]()<>;=+*/\\|&^%$#@~`")


def token_class(tok: str) -> str:
    """Classify a decoded token string. SentencePiece's word-start marker is treated as a space."""
    t = tok.replace("▁", " ")
    if _WS.match(t):
        return "whitespace"
    if _NUM.match(t):
        return "numeral"
    if _PUNCT.match(t):
        return "code_symbol" if any(ch in _CODE_CHARS for ch in t) else "punctuation"
    if any(ord(ch) > 0x024F and unicodedata.category(ch).startswith("L") for ch in t):
        return "non_latin"
    if any(ch.isdigit() for ch in t):
        return "numeral_mixed"
    if _LATIN.search(t):
        return "word"
    return "other"


def position_bucket(pos: int) -> str:
    for edge, label in zip(POSITION_EDGES, POSITION_LABELS):
        if pos < edge:
            return label
    return POSITION_LABELS[-1]


def freq_bucket(count: int) -> str:
    for edge, label in zip(FREQ_EDGES, FREQ_LABELS):
        if count < edge:
            return label
    return FREQ_LABELS[-1]
