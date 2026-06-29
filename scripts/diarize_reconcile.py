#!/usr/bin/env python3
"""Pure-stdlib reconciliation of Gemini transcripts against acoustic diarization.

No torch/pyannote here — this module is importable and testable with system Python.
It maps each Gemini speaker turn to the acoustically-dominant speaker over that turn's
time span, relabels by first-appearance order, and flags contested/merged turns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# [HH:MM:SS] Speaker N (g)?: text   — g is one of m/f/? when --voice-attrs was used.
TURN_RE = re.compile(
    r"^\[(\d{1,2}):(\d{2}):(\d{2})\]\s+"
    r"Speaker\s+(\d+)"
    r"(?:\s*\(([mf?])\))?"
    r"\s*:\s*(.*)$"
)


def _fmt_ts(total: float) -> str:
    total = int(round(max(0.0, total)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


@dataclass
class ParsedLine:
    raw: str
    ts: float | None
    label: str | None
    label_num: int | None
    gender: str | None
    text: str | None


def parse_lines(body: str) -> list[ParsedLine]:
    out: list[ParsedLine] = []
    for raw in body.splitlines():
        m = TURN_RE.match(raw)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            out.append(ParsedLine(
                raw=raw,
                ts=float(h * 3600 + mi * 60 + s),
                label=f"Speaker {m.group(4)}",
                label_num=int(m.group(4)),
                gender=m.group(5),
                text=m.group(6),
            ))
        else:
            out.append(ParsedLine(raw=raw, ts=None, label=None,
                                  label_num=None, gender=None, text=None))
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def winner_for_span(start: float, end: float, segments: list[dict]):
    totals: dict[str, float] = {}
    for seg in segments:
        ov = overlap(start, end, seg["start"], seg["end"])
        if ov > 0:
            totals[seg["speaker"]] = totals.get(seg["speaker"], 0.0) + ov
    if not totals:
        return None, 0.0, []
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    dur = max(1e-9, end - start)
    return ranked[0][0], ranked[0][1] / dur, ranked
