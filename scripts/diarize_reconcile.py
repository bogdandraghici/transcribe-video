#!/usr/bin/env python3
"""Pure-stdlib reconciliation of Gemini transcripts against acoustic diarization.

No torch/pyannote here — this module is importable and testable with system Python.
It maps each Gemini speaker turn to the acoustically-dominant speaker over that turn's
time span, relabels by first-appearance order, and flags contested/merged turns.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
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


def winner_for_span(start: float, end: float, segments: list[dict]) -> tuple[str | None, float, list[tuple[str, float]]]:
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


def assign_spans(lines: list[ParsedLine], media_duration: float) -> list[tuple[int, float, float]]:
    ts_idx = [i for i, ln in enumerate(lines) if ln.ts is not None]
    spans: list[tuple[int, float, float]] = []
    for k, i in enumerate(ts_idx):
        start = lines[i].ts
        end = lines[ts_idx[k + 1]].ts if k + 1 < len(ts_idx) else media_duration
        if end <= start:
            end = start + 1.0
        spans.append((i, start, end))
    return spans


def _compute_winners(lines: list[ParsedLine], spans, segments: list[dict]):
    return {i: winner_for_span(s, e, segments) for (i, s, e) in spans}


def home_clusters(lines: list[ParsedLine], winners: dict) -> dict[str, str]:
    by_label: dict[str, Counter] = defaultdict(Counter)
    for i, (raw, conf, ranked) in winners.items():
        if raw is not None:
            by_label[lines[i].label][raw] += 1
    return {label: ctr.most_common(1)[0][0] for label, ctr in by_label.items()}


def first_appearance_map(segments: list[dict]) -> dict[str, int]:
    seen: list[str] = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        if seg["speaker"] not in seen:
            seen.append(seg["speaker"])
    return {raw: n + 1 for n, raw in enumerate(seen)}


MIXED_TURN_THRESHOLD = 0.60


def _render_line(ln: ParsedLine, new_num: int, flags: list[str]) -> str:
    gender = f" ({ln.gender})" if ln.gender else ""
    flag = f" ‹{'; '.join(flags)}›" if flags else ""
    return f"{_fmt_ts(ln.ts)} Speaker {new_num}{gender}{flag}: {ln.text}"


def reconcile(body: str, segments: list[dict], media_duration: float) -> str:
    lines = parse_lines(body)
    spans = assign_spans(lines, media_duration)
    winners = _compute_winners(lines, spans, segments)
    homes = home_clusters(lines, winners)
    cmap = first_appearance_map(segments)

    out: list[str] = []
    for idx, ln in enumerate(lines):
        if ln.ts is None or idx not in winners:
            out.append(ln.raw)
            continue
        raw_speaker, conf, ranked = winners[idx]
        if raw_speaker is None:
            out.append(ln.raw)  # no acoustic coverage — leave Gemini's label
            continue

        new_num = cmap[raw_speaker]
        flags: list[str] = []

        home = homes.get(ln.label)
        if home is not None and raw_speaker != home:
            flags.append(f"reattr gemini=S{ln.label_num} conf={conf:.2f}")

        if conf < MIXED_TURN_THRESHOLD and len(ranked) >= 2:
            top2 = "/".join(f"S{cmap[r]}" for r, _ in ranked[:2] if r in cmap)
            if "/" in top2:
                flags.append(f"mixed {top2}")

        out.append(_render_line(ln, new_num, flags))
    return "\n".join(out)
