"""Deterministic source spans; no normalization, model calls or overlapping chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass


SPLITTER_VERSION = "source-lines-and-sentences-v1"
_END = re.compile(r'[.!?。！？]+[\"\'”’»）)\]]*')
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "fig", "no", "dept", "inc", "ltd", "approx", "cf",
})


@dataclass(frozen=True)
class SourceUnit:
    offset: int
    end_offset: int
    text: str
    unit_type: str


def _unit(text: str, begin: int, end: int, kind: str):
    while begin < end and text[begin].isspace():
        begin += 1
    while end > begin and text[end - 1].isspace():
        end -= 1
    if begin < end:
        return SourceUnit(begin, end, text[begin:end], kind)
    return None


def split_sentences(text: str) -> list[SourceUnit]:
    """Split physical lines and conservative sentence endings, retaining offsets.

    Lists, headings and logs without a sentence ending remain ``line`` units.
    Abbreviation handling is deliberately finite; this is not a linguistic parser.
    Long units stay intact so the caller can audit encoder-window overflow.
    """
    units = []
    line_offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        begin = 0
        structured = ("\t" in line or line.lstrip().startswith(("|", "```")))
        if not structured:
            for ending in _END.finditer(line):
                end = ending.end()
                mark = ending.group()
                # Latin punctuation needs a separator; this protects decimals,
                # versions, domain names and the interior of email/URL tokens.
                if end < len(line) and not line[end].isspace() and not any(c in mark for c in "。！？"):
                    continue
                if mark.startswith("."):
                    if re.fullmatch(r"\s*(?:[-*]\s+)?\d{1,3}\.", line[:ending.start() + 1]):
                        continue  # A numbered-list marker belongs to its item.
                    token = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$", line[:ending.start() + 1])
                    if token:
                        value = token.group(1)
                        if (value.lower() in _ABBREVIATIONS or
                                (len(value) == 1 and value.isupper()) or
                                re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", value)):
                            continue
                unit = _unit(text, line_offset + begin, line_offset + end, "sentence")
                if unit:
                    units.append(unit)
                begin = end
        tail = _unit(text, line_offset + begin, line_offset + len(line), "line")
        if tail:
            units.append(tail)
        line_offset += len(raw)
    return units
