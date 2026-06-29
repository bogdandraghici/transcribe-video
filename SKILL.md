---
name: transcribe-video
description: >
  Transcribe a video or audio recording into a diarized, timestamped, verbatim
  transcript using Google Gemini. Built for Romanian-primary sessions with English
  technical words mixed in (kept as spoken, not translated), but works for any
  language. Output is formatted to drop straight into the meetinginsights skill's
  "Session transcript" intake. Use whenever the user has a recording (.mp4, .mov,
  .mkv, .m4a, .mp3, .wav, etc.) and wants a transcript — triggers on "transcribe
  this video/recording", "get the transcript from this", "extract the transcript",
  "I have a recording of a user-testing/meeting session", or when meetinginsights
  needs a transcript but only a recording exists. Requires ffmpeg and a Gemini API
  key.
---

# Transcribe Video

Turns a recording into a diarized, timestamped, verbatim transcript via Gemini. The
heavy lifting is in `scripts/transcribe.py` — a self-contained CLI. The job here is to
run it correctly and report the result.

## What it produces

A `.txt` next to the source file, one speaker turn per line:

```
# Transcript of <name> — generated <date> via gemini-2.5-pro
# Verbatim, diarized, timestamped. Romanian as spoken; English words kept as English.

[00:00:04] Speaker 1: text spus în română with some English words
[00:00:19] Speaker 2: ...
```

This is exactly the format the `meetinginsights` skill ingests. Keep it **raw** — do
not clean, translate, or summarize it here; meetinginsights repairs the Romanian and
maps `Speaker N` → `P/F/B/O` itself.

## Prerequisites — check before running

1. **ffmpeg + ffprobe** on PATH (`which ffmpeg ffprobe`). Install with `brew install ffmpeg` if missing.
2. **A Gemini API key.** The script reads `$GEMINI_API_KEY`, or accepts `--api-key`.
   If the user only has it as `$GOOGLE_API_KEY` or pasted inline, pass it via `--api-key`.
3. **The recording must be a real file on disk** — get its absolute path. A cloud/Drive
   link cannot be read.

`google-genai` is installed automatically into a local `.venv` (next to the script) on
first run — no manual pip step needed. (It prints a redundant "Installing…" line on every
run; that's cosmetic.)

## How to run

```bash
python3 scripts/transcribe.py "/abs/path/to/recording.mp4" --api-key "$GEMINI_API_KEY"
```
Or, if `GEMINI_API_KEY` is exported, drop `--api-key`.

Useful flags:
- `--model gemini-2.5-pro` (default; best multilingual + diarization accuracy)
- `--chunk-minutes 20` (recordings longer than this are split, transcribed, and stitched
  with continuous absolute timestamps)
- `-o /path/out.txt` (override output path)
- `--keep-audio` (keep the extracted audio chunks for debugging)
- `--speakers "Speaker 1 = Bogdan (designer); Speaker 2 = Ana (PM)"` — feed the **known
  roster + roles** so the model assigns turns by who-would-say-it (role/expertise) instead
  of acoustic guess, and doesn't invent phantom speakers. The biggest lever for accurate
  diarization on **peer meetings** where same-gender voices sound alike — pass it whenever
  you know who was in the room. The roster is echoed into the transcript header. Speaker
  labels stay `Speaker N` (meetinginsights maps them); the roster only steers *which* voice
  is which. `meetinginsights` passes this automatically when it invokes the transcriber.
  **Caveat (tested):** the roster reliably curbs *phantom speakers* and helps *role-clear*
  turns, but it does **not** make diarization trustworthy at the line level for two
  *similar voices* (e.g. two men) — successive runs still disagree on who said a given
  reflective line. Treat a roster-steered re-transcription as a cross-check against content
  cues, not as ground truth for same-voice attribution; fall back to listening or human
  confirmation for the lines that matter.
- `--voice-attrs` — tag each line with **perceived voice gender** after the label:
  `[00:00:04] Speaker 1 (m): …`, values `(m)`/`(f)`/`(?)`. Judged **per turn, independent
  of the speaker label**, so a `(f)` on a line labeled as a male speaker flags a likely
  diarization slip. In a **cross-gender cast** this is the single biggest reliability win —
  it pins every `(f)` line to the one woman in the room and vice-versa, which is exactly the
  confusion same-voice diarization gets wrong. It does **not** separate two same-gender
  speakers (use timing + content for that). Pair it with `--speakers`. meetinginsights
  requests it automatically for cross-gender casts and uses the tag as a reconciliation prior.

For a **cheap sanity check** on a long/unfamiliar recording, transcribe just the start
first: trim with `ffmpeg -y -i in.mp4 -t 300 -c copy clip.mp4`, run the script on
`clip.mp4`, eyeball the result, then run the full file.

## How it works (so you can reason about failures)

- Extracts audio only (16 kHz mono FLAC) before sending to Gemini — cheaper and avoids
  blowing the context window on video frames. meetinginsights handles video frames
  separately when it needs screen context.
- Long recordings are chunked (~20 min); each chunk is uploaded via the Gemini Files API
  and transcribed, then timestamps are re-based to absolute time and concatenated. The
  script disambiguates Gemini's inconsistent timestamp formats (MM:SS / HH:MM:SS /
  MM:SS:mmm) using the known chunk length — don't "simplify" `parse_rel_seconds`.
- **Speaker numbers may reset between chunks** (Gemini renumbers per call). The output
  header flags this; meetinginsights' speaker reconciliation handles it downstream. For a
  single-chunk (≤20 min) recording, labels are consistent throughout.

## After transcribing

Tell the user the output path and the time range covered. If they want insights/cards
from it, hand the `.txt` (and the original recording, for frames) to the `meetinginsights`
skill.
