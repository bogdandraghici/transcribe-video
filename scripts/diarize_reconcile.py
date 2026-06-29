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
